#!/usr/bin/env python3
"""Stage 6: sell earlier in the day (declared in README, "Stage 6").

All versions trade the (mix) lists of stage 5, as on the demo, and buy at
00:00 UTC.
  (h24)                 hold 24 hours, today's way: the yardstick
  (h4) ... (h20)        sell everything at 04:00 ... 20:00 UTC, then hold
                        nothing until the next 00:00
Hourly candles come from get_hourly.py (data_hourly/). The buy price is the
00:00 price (the daily open, as in stage 5); the sell price is the close of the
hour ending at the selling time, or the last hourly close before it. Funding
counts each payment after 00:00 and up to the selling time.

Check first: (h24) rebuilt from hourly candles must equal stage 5's (mix).
Judged after costs (3.5 bp, funding included) on trade days 2021-03-29 to
2023-12-31: a higher Sharpe than (h24) and t >= 2 on the daily difference.

    .venv/bin/python stage6_timing.py       # writes results/stage6/
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import experiment as ex
import stage2 as s2
import stage3_blend as s3
import stage5_mix as s5

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage6"
HOURLY = HERE / "data_hourly" / "klines_1h.parquet"
HOUR_MS = 3_600_000
HOURS = (4, 8, 12, 16, 20)
NAMES = ["h24"] + [f"h{h}" for h in HOURS]
FIRST, LAST = "2021-03-29", "2026-10-24"
WINDOWS = {"Train": ex.JUDGE, "2024": ex.INFO["2024"], "2025-26": ex.INFO["2025-26"],
           "All": (FIRST, LAST)}
VIEWS = {"before costs": lambda d: d["gross"],
         "no fees, funding in": lambda d: d["gross"] + d["funding"],
         "3.5 bp": lambda d: s3.net(d, 3.5), "7.5 bp": lambda d: s3.net(d, 7.5)}
CHECK_BP, COST = 0.1, s3.COST


def title(name):
    return "(h24) hold 24 hours" if name == "h24" else f"({name}) sell at {name[1:]:0>2}:00 UTC"


def hourly_grid(trades, columns):
    """(days, 24, coins) hourly closes at the end of hours 1..24 of each trade day,
    forward-filled inside the day, and the mask of candles that exist."""
    k = pd.read_parquet(HOURLY)
    close = k.pivot(index="open_time", columns="symbol", values="close")
    start = int(trades[0].value // 1_000_000)
    hours = start + HOUR_MS * np.arange(len(trades) * 24, dtype=np.int64)
    grid = close.reindex(index=hours, columns=columns).to_numpy().reshape(len(trades), 24, len(columns))
    exists = ~np.isnan(grid)
    filled = grid.copy()
    for h in range(1, 24):
        filled[:, h] = np.where(np.isnan(filled[:, h]), filled[:, h - 1], filled[:, h])
    return filled, exists


def funding_grid(trades, columns):
    """(days, 24, coins) funding rate summed over payments after 00:00 and up to the end of
    hour 1..24 of each trade day (the 00:00 payment of the next day is hour 24)."""
    f = pd.read_csv(HERE / "data" / "funding.csv.gz")
    hour = pd.to_datetime(f["funding_time"], unit="ms").dt.round("h")
    f["day"] = (hour - pd.Timedelta(milliseconds=1)).dt.normalize()
    f["h"] = ((hour - f["day"]) / pd.Timedelta(hours=1)).astype(int)        # 1..24
    f = f[f["day"].isin(trades) & f["symbol"].isin(columns)]
    g = f.groupby(["day", "h", "symbol"])["funding_rate"].sum()
    out = np.zeros((len(trades), 24, len(columns)))
    di = trades.get_indexer(g.index.get_level_values("day"))
    ci = pd.Index(columns).get_indexer(g.index.get_level_values("symbol"))
    np.add.at(out, (di, g.index.get_level_values("h").to_numpy() - 1, ci), g.to_numpy())
    return np.cumsum(out, axis=1)


def weight_grid(target, days, trades, columns):
    """(days, coins) weights held on each trade day (0 where there are no lists)."""
    w = np.zeros((len(trades), len(columns)))
    col = pd.Index(columns)
    for day, *_ in days:
        nxt = day + pd.Timedelta(days=1)
        t = target.get(day)
        if nxt in trades and t is not None and len(t):
            w[trades.get_loc(nxt), col.get_indexer(t.index)] = t.to_numpy()
    return w


def early(h, W, R, FU, trades):
    """Daily results of selling at hour h: buy the whole book at 00:00, sell it at h."""
    r = np.nan_to_num(R[:, h - 1])
    gross = (W * r).sum(axis=1)
    funding = -(W * FU[:, h - 1]).sum(axis=1)
    turnover = np.abs(W).sum(axis=1) + np.abs(W * (1 + r)).sum(axis=1)
    return pd.DataFrame({"gross": gross, "turnover": turnover, "funding": funding,
                         "scale": 1.0}, index=pd.DatetimeIndex(trades, name="trade_date"))


def judge(d, base):
    row = s3.judge(d, base)
    row["Sharpe of (h24)"] = row.pop("Sharpe of (v)")
    row["vs (h24) bp/day"] = row.pop("vs (v) bp/day")
    row["t vs (h24)"] = row.pop("t vs (v)")
    failed = [k for k, ok in {"Sharpe above (h24)": row["Sharpe"] > row["Sharpe of (h24)"],
                              "t >= 2": row["t vs (h24)"] >= 2}.items() if not ok]
    row["verdict"] = "PASS" if not failed else "fail: " + ", ".join(failed)
    return row


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    f7, mom = s5.mix_panels(panels, funding)
    lists = {day: (None if table is None else s5.mix_lists(table, f7, mom, day))
             for day, table, *_ in days}
    target = {day: (None if l is None else ex.weights(ex.VERSIONS[0], day, *l, ctx))
              for day, l in lists.items()}
    stage5 = s2.simulate(target, days, ctx)                  # stage 5's (mix), daily candles

    trades = pd.date_range(FIRST, LAST, freq="D")
    columns = list(panels["close"].columns)
    filled, exists = hourly_grid(trades, columns)
    W = weight_grid(target, days, trades, columns)
    opens = panels["open"].reindex(index=trades, columns=columns).to_numpy()
    R = filled / opens[:, None, :] - 1                       # return from 00:00 to the end of hour h
    FU = funding_grid(trades, columns)
    held = W != 0

    # ---- check: (h24) from hourly candles equals stage 5's (mix)
    r24 = pd.DataFrame(R[:, 23], index=trades, columns=columns)
    f24 = pd.DataFrame(FU[:, 23], index=trades, columns=columns)
    hourly_ctx = dataclasses.replace(ctx, ret=ctx.ret.copy(), funding=ctx.funding.copy())
    hourly_ctx.ret.loc[trades, columns] = r24.where(held, ctx.ret.loc[trades, columns])
    hourly_ctx.funding.loc[trades, columns] = f24.where(held, ctx.funding.loc[trades, columns])
    rebuilt = s2.simulate(target, days, hourly_ctx).loc[FIRST:LAST]
    base = stage5.loc[FIRST:LAST]
    gap_bp = (s3.net(rebuilt, COST) - s3.net(base, COST)).abs() * 1e4
    sh_daily, sh_hourly = s3.sharpe(s3.net(base.loc[slice(*ex.JUDGE)], COST)), \
        s3.sharpe(s3.net(rebuilt.loc[slice(*ex.JUDGE)], COST))
    print(f"CHECK (h24) hourly vs stage 5 (mix): average daily gap {gap_bp.mean():.4f} bp, "
          f"largest {gap_bp.max():.2f} bp; Sharpe 2021-23 {sh_hourly:.3f} vs {sh_daily:.3f}")
    if gap_bp.mean() >= CHECK_BP or round(sh_hourly, 2) != round(sh_daily, 2):
        worst = gap_bp.sort_values().tail(5)
        raise SystemExit(f"CHECK FAILED; largest gaps (bp):\n{worst}")

    # ---- missing hourly candles at each selling hour, among held coin-days
    missing = {}
    for h in (*HOURS, 24):
        at_h = held & ~exists[:, h - 1]
        none = held & np.isnan(filled[:, h - 1])
        missing[f"h{h}"] = {"held coin-days": int(held.sum()), "no candle at the hour": int(at_h.sum()),
                            "no candle that day so far (return 0)": int(none.sum())}
    missing = pd.DataFrame(missing).T

    results = {"h24": base}
    for h in HOURS:
        results[f"h{h}"] = early(h, W, R, FU, trades)

    judged = pd.DataFrame({n: judge(results[n], base) for n in NAMES[1:]}).T
    passers = [n for n in judged.index if judged.at[n, "verdict"] == "PASS"]
    if passers:
        best = max(passers, key=lambda n: judged.at[n, "Sharpe"])
        note = f"({best}) passes and has the highest Sharpe after costs: it goes to the Forward Test."
    else:
        note = "No early version passes: (h24), holding 24 hours, stays."

    rows = []
    for name, d in results.items():
        for view, fn in VIEWS.items():
            r = fn(d)
            for period, (start, end) in WINDOWS.items():
                g = r.loc[start:end]
                equity = (1 + g).cumprod()
                rows.append({"version": name, "view": view, "period": period,
                             "per year %": (equity.iloc[-1] ** (bt.DAYS_PER_YEAR / len(g)) - 1) * 100,
                             "Sharpe": s3.sharpe(g),
                             "worst drawdown %": (equity / equity.cummax() - 1).min() * 100})
    info = pd.DataFrame(rows)
    parts = pd.DataFrame([{"version": n, "period": p,
                           "price bp/day": d.loc[s:e, "gross"].mean() * 1e4,
                           "funding bp/day": d.loc[s:e, "funding"].mean() * 1e4,
                           "fees at 3.5 bp, bp/day": -(d.loc[s:e, "turnover"] * 3.5).mean(),
                           "net at 3.5 bp, bp/day": s3.net(d.loc[s:e], 3.5).mean() * 1e4,
                           "turnover %": d.loc[s:e, "turnover"].mean() * 100}
                          for n, d in results.items() for p, (s, e) in WINDOWS.items()])

    # ---- the average path through the day (information only), bp of the account
    on = held.any(axis=1)
    price_path = pd.DataFrame((W[:, None, :] * np.nan_to_num(R)).sum(axis=2), index=trades,
                              columns=range(1, 25))[on]
    fund_path = pd.DataFrame(-(W[:, None, :] * FU).sum(axis=2), index=trades,
                             columns=range(1, 25))[on]
    path = pd.concat({p: pd.DataFrame({"price bp": price_path.loc[s:e].mean() * 1e4,
                                       "funding bp": fund_path.loc[s:e].mean() * 1e4})
                      for p, (s, e) in WINDOWS.items()}, axis=1)
    path.index.name = "hour UTC"

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    f2 = lambda x: f"{x:.2f}"
    print(f"\nJUDGED against (h24): after costs at {COST} bp, funding included, "
          f"trade days {ex.JUDGE[0]} to {ex.JUDGE[1]}")
    print(judged.to_string(float_format=f2))
    print(f"\n{note}")
    print("\nMissing hourly candles among held coin-days")
    print(missing.to_string())
    for view in VIEWS:
        print(f"\nINFORMATION: per year % ({view})")
        print(info[info["view"] == view].pivot(index="version", columns="period", values="per year %")
              [list(WINDOWS)].reindex(NAMES).to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nINFORMATION: Sharpe at 3.5 bp")
    print(info[info["view"] == "3.5 bp"].pivot(index="version", columns="period", values="Sharpe")
          [list(WINDOWS)].reindex(NAMES).to_string(float_format=f2))
    print("\nINFORMATION: parts, bp a day")
    print(parts.set_index(["period", "version"]).loc[list(WINDOWS)].to_string(float_format=f2))
    print("\nINFORMATION: average path through the day, bp of the account since 00:00")
    print(path.loc[[*HOURS, 24]].to_string(float_format=lambda x: f"{x:.1f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    judged.to_csv(OUT / "judged.csv", float_format="%.6g")
    info.to_csv(OUT / "information.csv", index=False, float_format="%.6g")
    parts.to_csv(OUT / "parts.csv", index=False, float_format="%.6g")
    path.to_csv(OUT / "day_path.csv", float_format="%.6g")
    missing.to_csv(OUT / "missing_candles.csv")
    pd.DataFrame({n: s3.net(d, COST) for n, d in results.items()}).to_csv(
        OUT / "daily_net_3_5bp.csv", float_format="%.8g")
    (OUT / "final.txt").write_text(note + "\n")
    chart(results, judged, path, OUT / "timing.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/")


def chart(results, judged, path, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ink, ink2, grid, axis, wash, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec", "#fcfcfb"
    ramp = dict(zip([f"h{h}" for h in HOURS], ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]))
    periods = {"Train": "#2a78d6", "2024": "#eb6834", "2025-26": "#1baf7a"}
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface, "lines.linewidth": 2,
    })
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9.5), height_ratios=(3, 2))

    top.axvspan(pd.Timestamp(FIRST), pd.Timestamp(ex.JUDGE[1]), color=wash, lw=0, zorder=0)
    ends = []
    for name in reversed(NAMES):
        eq = 10_000 * (1 + s3.net(results[name], COST).loc[FIRST:LAST]).cumprod()
        color = ink if name == "h24" else ramp[name]
        top.plot(eq.index, eq, color=color, lw=2.2 if name == "h24" else 1.6, label=title(name))
        ends.append((eq.iloc[-1], eq.index[-1], color, name))
    top.set_yscale("log")
    low, high = top.get_ylim()
    gap, previous = 0.05 * np.log(high / low), None
    for last, when, color, name in sorted(ends, key=lambda e: e[0], reverse=True):
        y = last if previous is None else min(last, previous * np.exp(-gap))
        previous = y
        top.annotate(f"({name}) ${last:,.0f}", (when, last), xytext=(6, y),
                     textcoords=("offset points", "data"), va="center", color=ink2, fontsize=8.5)
    top.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 3, 5)))
    top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    top.yaxis.set_minor_formatter(ticker.NullFormatter())
    top.xaxis.set_major_locator(mdates.YearLocator())
    top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    top.set_ylabel("Value of $10,000")
    passed = [n for n in judged.index if judged.at[n, "verdict"] == "PASS"]
    top.set_title(f"After costs ({COST} bp per dollar traded, funding included). Judged 2021-03-29 to "
                  f"2023-12-31 (shaded): {'pass: ' + ', '.join(passed) if passed else 'none passes'}",
                  loc="left", color=ink, fontsize=10)
    handles = dict(zip(top.get_legend_handles_labels()[1], top.get_legend_handles_labels()[0]))
    top.legend([handles[title(n)] for n in NAMES], [title(n) for n in NAMES],
               frameon=False, loc="upper left", labelcolor=ink2, fontsize=8.5)
    top.margins(x=0.01)

    hours = [0, *path.index]
    for p, color in periods.items():
        total = (path[(p, "price bp")] + path[(p, "funding bp")]).to_list()
        bottom.plot(hours, [0, *total], color=color, lw=2)
        bottom.annotate(p, (24, total[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=ink2, fontsize=8.5)
    for h in (*HOURS, 24):
        bottom.axvline(h, color=grid, lw=0.8, zorder=0)
    bottom.axhline(0, color=axis, lw=0.8)
    bottom.set_xlim(0, 24)
    bottom.set_xticks(range(0, 25, 4))
    bottom.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):02d}:00"))
    bottom.set_xlabel("Hour of the trade day (UTC); bought at 00:00")
    bottom.set_ylabel("bp of the account since 00:00")
    bottom.set_title("Average price + funding income through the day, before fees (information only)",
                     loc="left", color=ink, fontsize=10)
    bottom.legend([plt.Line2D([], [], color=c, lw=2) for c in periods.values()], list(periods),
                  frameon=False, loc="upper left", labelcolor=ink2, fontsize=8.5)
    fig.suptitle("Stage 6: sell earlier in the day, the three-signal mix", x=0.01, ha="left",
                 color=ink, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    fig.savefig(file, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
