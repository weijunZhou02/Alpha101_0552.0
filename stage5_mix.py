#!/usr/bin/env python3
"""Stage 5: the three-signal mix (mix_lists.py, traded on the demo since 2026-11-03),
rebuilt with the lab's accounting and judged against (v).

    mix = average of three percentile ranks inside the day's 80-coin universe:
          A101_055, minus 7-day funding (>= 3 days with a payment), 30-day return
    BUY the 16 highest, SHORT the 16 lowest (ties: alphabetical), equal weights.
  (mix)   as traded on the demo, no volatility scaling
  (mix-v) with (v)'s volatility scaling from its own unscaled results
Before judging, the lab's lists must equal mix_lists.py's own functions on 30
sampled days. Rule A after costs (3.5 bp, funding included) on trade days
2021-03-29 to 2023-12-31; declared in README ("Stage 5").

    .venv/bin/python stage5_mix.py       # writes results/stage5/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import a101_055 as live
import backtest as bt
import experiment as ex
import stage2 as s2
import stage3_blend as s3
import stage3b_portfolio as p3

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage5"
MAIN = HERE.parent / "A101_055-main"
FUNDING_DAYS, FUNDING_MIN_DAYS, MOMENTUM_DAYS = 7, 3, 30
CHECK_DAYS, CHECK_SEED = 30, 0
JUDGED = ["mix", "mix-v"]
TITLES = {"v": "(v) working version", "a": "(a) today's A101_055 (information)",
          "mix": "(mix) three-signal mix, as on the demo", "mix-v": "(mix-v) mix with (v)'s scaling"}
WINDOWS = {"All": ("2021-03-29", "2026-10-24"), "Train": ex.JUDGE,
           "2024": ("2024-01-01", "2024-12-31"), "2025-26": ("2025-01-01", "2026-10-24"),
           "Box 1": ex.INFO["Box 1"], "Box 2": ex.INFO["Box 2"], "Dip 3": ex.INFO["Dip 3"],
           "Drawdown 1": s3.WINDOWS["Drawdown 1"], "Drawdown 2": s3.WINDOWS["Drawdown 2"]}
VIEWS = {"before costs": lambda d: d["gross"], "no fees, funding in": lambda d: d["gross"] + d["funding"],
         "3.5 bp": lambda d: s3.net(d, 3.5), "7.5 bp": lambda d: s3.net(d, 7.5)}


def mix_panels(panels, funding):
    """7-day funding (missing with fewer than 3 days of payments) and 30-day return."""
    f7 = funding.rolling(FUNDING_DAYS, min_periods=1).sum()
    f7 = f7.where(funding.rolling(FUNDING_DAYS, min_periods=1).count() >= FUNDING_MIN_DAYS)
    close = panels["close"]
    return f7, close / close.shift(MOMENTUM_DAYS) - 1


def mix_lists(table, f7, mom, day):
    """BUY and SHORT lists of the mix for one signal day, or None."""
    ranks = pd.DataFrame({"a101": table["a101_055"].rank(pct=True),
                          "funding": (-f7.loc[day].reindex(table.index)).rank(pct=True),
                          "mom": mom.loc[day].reindex(table.index).rank(pct=True)})
    score = ranks.mean(axis=1, skipna=False).round(live.TIE_DECIMALS)
    valid = score.dropna()
    if len(valid) < 2 * live.LIST_SIZE:
        return None
    buy = sorted(valid.index, key=lambda s: (-valid[s], s))[:live.LIST_SIZE]
    short = sorted(valid.index, key=lambda s: (valid[s], s))[:live.LIST_SIZE]
    return buy, short


def check_against_live(panels, days, lists):
    """The lab's lists must equal mix_lists.py's own functions on sampled days."""
    sys.path.insert(0, str(MAIN))
    import mix_lists                                   # uses the identical a101_055.py
    raw = pd.read_csv(HERE / "data" / "funding.csv.gz")
    events = {s: g[["funding_time", "funding_rate"]].to_numpy() for s, g in raw.groupby("symbol")}
    has_candle = panels["close"].notna()
    pool = [d for d, *_ in days if lists.get(d) is not None and d >= pd.Timestamp(ex.JUDGE[0])]
    sample = sorted(np.random.default_rng(CHECK_SEED).choice(len(pool), CHECK_DAYS, replace=False))
    for i in sample:
        day = pool[i]
        trading = bt.trading_at_open(has_candle, day)
        columns = trading.index[trading]
        start = day - pd.Timedelta(days=bt.LIVE_CANDLES - 1)
        sliced = {f: panels[f].loc[start:day, columns] for f in ("high", "low", "close", "qvol")}
        universes, _, _ = live.compute_universes(
            sliced, pd.date_range(end=day, periods=live.UNIVERSE_DAYS, freq="D"))
        lo = (day - pd.Timedelta(days=FUNDING_DAYS)).value // 1_000_000
        hi = (day + pd.Timedelta(days=1)).value // 1_000_000 + mix_lists.HOUR_MS // 2
        funding = {}
        for s in universes[day]:
            e = events.get(s, np.empty((0, 2)))
            e = e[(e[:, 0] >= lo) & (e[:, 0] <= hi)]
            funding[s] = mix_lists.funding_by_day([(int(t), float(r)) for t, r in e], day)
        table = mix_lists.compute_mix(sliced, universes, day, funding)
        _, buy, short = mix_lists.select_lists(table)
        if (buy, short) != tuple(lists[day]):
            raise SystemExit(f"CHECK FAILED on {day:%Y-%m-%d}: lab lists differ from mix_lists.py")
    print(f"  check: lab lists equal mix_lists.py on {CHECK_DAYS} sampled days")


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    trend = panels["close"] / panels["close"].shift(s2.TREND_DAYS) - 1
    f7, mom = mix_panels(panels, funding)
    lists = {day: (None if table is None else mix_lists(table, f7, mom, day))
             for day, table, *_ in days}
    check_against_live(panels, days, lists)

    a_spec = ex.VERSIONS[0]
    target = {day: (None if l is None else ex.weights(a_spec, day, *l, ctx)) for day, l in lists.items()}
    results = {"v": s2.run(s2.VERSIONS[-1], days, ctx, trend),
               "a": s2.run(s2.VERSIONS[0], days, ctx, trend),
               "mix": s2.simulate(target, days, ctx),
               "mix-v": p3.scaled(target, days, ctx)}

    judged = pd.DataFrame({n: s3.judge(results[n], results["v"]) for n in JUDGED}).T
    passers = [n for n in JUDGED if judged.at[n, "verdict"] == "PASS"]
    if passers:
        best = max(passers, key=lambda n: judged.at[n, "Sharpe"])
        note = f"({best}) passes and has the highest Sharpe after costs: it is the final version."
    else:
        note = "Neither mix version passes: (v) stays the final version."

    rows = []
    for name, d in results.items():
        for view, fn in VIEWS.items():
            r = fn(d)
            for period, (start, end) in WINDOWS.items():
                g = r.loc[start:end]
                equity = (1 + g).cumprod()
                years = len(g) / bt.DAYS_PER_YEAR
                rows.append({"version": name, "view": view, "period": period,
                             "total %": (equity.iloc[-1] - 1) * 100,
                             "per year %": (equity.iloc[-1] ** (1 / years) - 1) * 100,
                             "Sharpe": s3.sharpe(g),
                             "worst drawdown %": (equity / equity.cummax() - 1).min() * 100})
    info = pd.DataFrame(rows)
    split = pd.concat({n: pd.DataFrame({
        "total % at 3.5 bp": s3.net(d, 3.5).groupby(d.index.year).apply(lambda g: ((1 + g).prod() - 1) * 100),
        "price part %": d["gross"].groupby(d.index.year).sum() * 100,
        "funding part %": d["funding"].groupby(d.index.year).sum() * 100,
        "fees at 3.5 bp %": -(d["turnover"] * 3.5 / 1e4).groupby(d.index.year).sum() * 100})
        for n, d in results.items()}, axis=1)
    sl = slice(*WINDOWS["All"])
    risk = pd.DataFrame({n: {
        "turnover %": d.loc[sl, "turnover"].mean() * 100,
        "worst day % (3.5 bp)": s3.net(d, 3.5).loc[sl].min() * 100,
        "corr with (v), daily (3.5 bp)": s3.net(d, 3.5).loc[sl].corr(s3.net(results["v"], 3.5).loc[sl]),
        "average size": d.loc[sl, "scale"].mean()} for n, d in results.items()}).T

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    f2 = lambda x: f"{x:.2f}"
    print(f"\nJUDGED against (v): after costs at {s3.COST} bp, funding included, "
          f"trade days {ex.JUDGE[0]} to {ex.JUDGE[1]}")
    print(judged.to_string(float_format=f2))
    print(f"\n{note}")
    for view in VIEWS:
        print(f"\nINFORMATION: per year % ({view})")
        print(info[info["view"] == view].pivot(index="version", columns="period", values="per year %")
              [list(WINDOWS)].reindex(list(results)).to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nINFORMATION: Sharpe, whole period, by view")
    print(info[info["period"] == "All"].pivot(index="version", columns="view", values="Sharpe")
          [list(VIEWS)].reindex(list(results)).to_string(float_format=f2))
    print("\nINFORMATION: worst drawdown %, whole period, by view")
    print(info[info["period"] == "All"].pivot(index="version", columns="view", values="worst drawdown %")
          [list(VIEWS)].reindex(list(results)).to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nINFORMATION: by year, total and its parts (price and funding are simple sums)")
    print(split.to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nINFORMATION: risk and size, 2021-03-29 to 2026-10-24")
    print(risk.to_string(float_format=f2))

    OUT.mkdir(parents=True, exist_ok=True)
    judged.to_csv(OUT / "judged.csv", float_format="%.6g")
    info.to_csv(OUT / "information.csv", index=False, float_format="%.6g")
    split.to_csv(OUT / "by_year_parts.csv", float_format="%.6g")
    risk.to_csv(OUT / "risk.csv", float_format="%.6g")
    (OUT / "final.txt").write_text(note + "\n")
    chart(results, judged, OUT / "mix_vs_v.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/")


def chart(results, judged, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ink, ink2, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    blue, orange, wash, surface = "#2a78d6", "#eb6834", "#f0efec", "#fcfcfb"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface,
    })
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, height_ratios=(3, 2))
    start = pd.Timestamp(WINDOWS["All"][0])
    for ax in (top, bottom):
        ax.axvspan(start, pd.Timestamp(ex.JUDGE[1]), color=wash, lw=0, zorder=0)
        for s, e in (s3.WINDOWS["Drawdown 1"], s3.WINDOWS["Drawdown 2"]):
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=grid, lw=0, zorder=0)

    style = {"a": (ink2, 1.1, ":"), "v": (ink, 1.9, "-"), "mix": (blue, 1.9, "-"), "mix-v": (orange, 1.4, "-")}
    ends = []
    for name in ("a", "v", "mix-v", "mix"):
        eq = 10_000 * (1 + s3.net(results[name], s3.COST).loc[start:]).cumprod()
        color, width, ls = style[name]
        top.plot(eq.index, eq, color=color, lw=width, ls=ls, label=TITLES[name])
        ends.append((eq.iloc[-1], eq.index[-1], color))
    top.set_yscale("log")
    low, high = top.get_ylim()
    gap, previous = 0.045 * np.log(high / low), None
    for last, when, color in sorted(ends, reverse=True):
        y = last if previous is None else min(last, previous * np.exp(-gap))
        previous = y
        top.annotate(f"${last:,.0f}", (when, last), xytext=(6, y), textcoords=("offset points", "data"),
                     va="center", color=color, fontsize=8.5)
    top.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 3, 5)))
    top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    top.yaxis.set_minor_formatter(ticker.NullFormatter())
    top.set_ylabel("Value of $10,000")
    parts = [f"({n}) {'PASS' if judged.at[n, 'verdict'] == 'PASS' else 'fail'}, Sharpe 2021-23 "
             f"{judged.at[n, 'Sharpe']:.2f}" for n in JUDGED]
    top.set_title(" · ".join(parts) + f"  vs (v) {judged['Sharpe of (v)'].iloc[0]:.2f}",
                  loc="left", color=ink, fontsize=10)
    top.legend(frameon=False, loc="upper left", labelcolor=ink2, fontsize=8.5)

    for name, color in (("mix", blue), ("v", ink)):
        d = results[name].loc[start:]
        bottom.plot(d.index, d["gross"].cumsum() * 100, color=color, lw=1.6,
                    label=f"{TITLES[name].split(' ', 1)[0]} price part")
        bottom.plot(d.index, d["funding"].cumsum() * 100, color=color, lw=1.6, ls="--",
                    label=f"{TITLES[name].split(' ', 1)[0]} funding part")
    bottom.axhline(0, color=axis, lw=0.8)
    bottom.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    bottom.set_ylabel("Running sum, % of the account")
    bottom.set_title("Where the money came from, before fees (simple running sums of daily returns)",
                     loc="left", color=ink, fontsize=10)
    bottom.legend(frameon=False, loc="upper left", labelcolor=ink2, fontsize=8.5, ncol=2)
    bottom.xaxis.set_major_locator(mdates.YearLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Stage 5: the three-signal mix against (v), after costs ({s3.COST} bp per dollar traded, "
                 "funding included)", x=0.01, ha="left", color=ink, fontsize=12.5)
    fig.text(0.01, 0.948, "Light shading = judged period 2021-03-29 to 2023-12-31. Darker bands = "
             "Drawdown 1 and 2 of (v). Later years are information only.", color=ink2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
