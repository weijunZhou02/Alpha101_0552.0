#!/usr/bin/env python3
"""Stage 3b, step A2: helpers keep A101_055's best 10 of 16, judged after costs against (v).

A101_055 makes its usual 16 + 16 lists; a helper keeps the best 10 per side (BUY:
highest helper score, SHORT: lowest; ties to the stronger A101_055 signal; a missing
value counts as the list median; fewer than 10 values -> keep all 16 that day).
  (p10)  CRYPTO_HMRW_PREMIUM keeps the best 10 + 10
  (p16)  half today's 16 + 16, half (p10)
  (i10)  Q158_IMIN5 keeps the best 10 + 10
  (i16)  half today's 16 + 16, half (i10)
  (pi10) both helpers' percentile ranks within the list (missing = middle), averaged
  (a10)  information only: A101_055 keeps its own 10 strongest
Each with (v)'s volatility scaling from its own unscaled results. Rule A after costs
(3.5 bp per dollar traded, funding included) on trade days 2021-03-29 to 2023-12-31;
declared in README ("Stage 3b, step A2").

    .venv/bin/python stage3b_portfolio.py       # writes results/stage3b/portfolio_*
"""

from pathlib import Path

import numpy as np
import pandas as pd

import a101_055 as live
import backtest as bt
import experiment as ex
import stage2 as s2
import stage3_blend as s3
from screen_money import spread

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage3b"
KEEP, MIN_VALUES = 10, 10
PREMIUM, IMIN = "CRYPTO_HMRW_PREMIUM", "Q158_IMIN5"
VERSIONS = {
    "p10": ((PREMIUM,), 1.0),
    "p16": ((PREMIUM,), 0.5),
    "i10": ((IMIN,), 1.0),
    "i16": ((IMIN,), 0.5),
    "pi10": ((PREMIUM, IMIN), 1.0),
    "a10": (("a101_055",), 1.0),
}
JUDGED = ["p10", "p16", "i10", "i16", "pi10"]
TITLES = {"v": "(v) working version", "p10": "(p10) premium keeps best 10 + 10",
          "p16": "(p16) half 16 + 16, half (p10)", "i10": "(i10) IMIN5 keeps best 10 + 10",
          "i16": "(i16) half 16 + 16, half (i10)", "pi10": "(pi10) both helpers keep best 10 + 10",
          "a10": "(a10) A101_055 keeps its own top 10 (information)"}
PANELS = [("p10", "p16"), ("i10", "i16"), ("pi10", "a10")]


def keep(coins, score, strength, keep_high):
    """The 10 coins of one list the score likes best (ties: stronger A101_055 signal first)."""
    if len(coins) <= KEEP:
        return list(coins)
    v = score.reindex(coins)
    if v.notna().sum() < MIN_VALUES:
        return list(coins)
    v = v.fillna(v.median())
    tie = strength[coins].rank(ascending=not keep_high) / 100
    key = v.rank(method="min") - tie if keep_high else v.rank(method="min") + tie
    return list(key.sort_values(ascending=not keep_high).index[:KEEP])


def list_score(coins, names, values):
    """One score per coin of a list: the helper itself, or the average within-list
    percentile rank of several helpers (missing = middle)."""
    if len(names) == 1:
        return values[names[0]].reindex(coins)
    if all(values[n].reindex(coins).notna().sum() < MIN_VALUES for n in names):
        return pd.Series(np.nan, index=coins)
    return sum(values[n].reindex(coins).rank(pct=True).fillna(0.5) for n in names) / len(names)


def books(names, share, days, ctx, helpers):
    """Target weights per signal day: share in the kept 10 + 10, the rest in today's 16 + 16."""
    a_spec = ex.VERSIONS[0]
    target = {}
    for day, table, buy, short, problem in days:
        if problem:
            target[day] = None
            continue
        if share == 0:
            target[day] = ex.weights(a_spec, day, list(buy), list(short), ctx)
            continue
        strength = table["a101_055"]
        values = {"a101_055": strength}
        for n in names:
            if n != "a101_055":
                values[n] = (helpers[n].loc[day] if day in helpers[n].index
                             else pd.Series(dtype=float))
        kb = keep(list(buy), list_score(list(buy), names, values), strength, keep_high=True)
        ks = keep(list(short), list_score(list(short), names, values), strength, keep_high=False)
        w10 = ex.weights(a_spec, day, kb, ks, ctx)
        if share < 1:
            w16 = ex.weights(a_spec, day, list(buy), list(short), ctx)
            w10 = s2.add(w16 * (1 - share), w10 * share)
        target[day] = w10
    return target


def helper_by_year(days, ctx, helpers):
    """Information: the screen's kept-10-minus-dropped-6 spread of each helper, by year, all years."""
    rows, dates = [], []
    for day, table, buy, short, problem in days:
        nxt = day + pd.Timedelta(days=1)
        if problem or table is None or nxt < pd.Timestamp(ex.JUDGE[0]):
            continue
        u, a = table.index, table["a101_055"]
        X = pd.DataFrame({n: (h.loc[day].reindex(u) if day in h.index else np.nan)
                          for n, h in helpers.items()}, index=u)
        r, f = ctx.ret.loc[nxt, u].fillna(0.0), ctx.funding.loc[nxt, u].fillna(0.0)
        buy, short = list(buy), list(short)
        rows.append(pd.concat([spread(X.loc[buy], a[buy], (r - f)[buy], keep_high=True),
                               spread(X.loc[short], a[short], (f - r)[short], keep_high=False)],
                              axis=1).mean(axis=1))
        dates.append(nxt)
    s = pd.DataFrame(rows, index=dates)
    return (s.groupby(s.index.year).mean() * 1e4).T


def scaled(target, days, ctx):
    unscaled = s2.simulate(target, days, ctx)
    return s2.simulate(target, days, ctx, scale=s2.vol_scale(unscaled["gross"]))


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    trend = panels["close"] / panels["close"].shift(s2.TREND_DAYS) - 1
    raw = pd.read_csv(HERE / "data" / "shortlist_3b_values.csv.gz", parse_dates=["date"])
    helpers = {a: raw.pivot(index="date", columns="symbol", values=a) for a in (PREMIUM, IMIN)}

    results = {"v": s2.run(s2.VERSIONS[-1], days, ctx, trend)}
    check = scaled(books((), 0.0, days, ctx, helpers), days, ctx)
    gap = (check["gross"] - results["v"]["gross"]).abs().max()
    assert gap < 1e-12, f"the 16 + 16 book through this script differs from (v) (gap {gap:.2g})"
    print(f"  check: today's 16 + 16 through this script reproduces (v) (gap {gap:.1g})")
    for name, (names, share) in VERSIONS.items():
        results[name] = scaled(books(names, share, days, ctx, helpers), days, ctx)
        print(f"  ({name}) done")

    judged = pd.DataFrame({n: s3.judge(results[n], results["v"]) for n in JUDGED}).T
    sl = slice(*ex.JUDGE)
    extra = pd.DataFrame({n: {
        "worst day %": s3.net(d, s3.COST).loc[sl].min() * 100,
        "worst 1% of days, avg %": s3.net(d, s3.COST).loc[sl].nsmallest(
            max(1, int(0.01 * len(d.loc[sl])))).mean() * 100,
        "average size": d.loc[sl, "scale"].mean(),
    } for n, d in results.items()}).T
    later = {}
    for n, d in results.items():
        row = {}
        for label, (start, end) in {"2021-23": ex.JUDGE, "2024-26": s3.WINDOWS["2024-26"]}.items():
            for cost in (0.0, s3.COST):
                ret = (lambda x: x["gross"]) if cost == 0 else (lambda x: s3.net(x, s3.COST))
                diff = (ret(d) - ret(results["v"])).loc[start:end]       # before costs = no fees, no funding
                tag = "before costs" if cost == 0 else f"at {cost:g} bp"
                row[f"{label} vs (v) {tag}, bp/day"] = diff.mean() * 1e4
                row[f"{label} t {tag}"] = diff.mean() / diff.std() * np.sqrt(len(diff)) if n != "v" else np.nan
            row[f"{label} turnover %"] = d.loc[start:end, "turnover"].mean() * 100
        later[n] = row
    later = pd.DataFrame(later).T
    passers = [n for n in JUDGED if judged.at[n, "verdict"] == "PASS"]
    if passers:
        best = max(passers, key=lambda n: judged.at[n, "Sharpe"])
        note = f"({best}) passes and has the highest Sharpe after costs: it is the final version."
    else:
        note = "No version passes: (v) stays the final version."
    a10 = s3.judge(results["a10"], results["v"])

    rows = []
    for name, d in results.items():
        for cost in (*s3.INFO_COSTS, s3.COST):
            r = s3.net(d, cost)
            for period, (start, end) in {"Train": ex.JUDGE, **s3.WINDOWS}.items():
                g = r.loc[start:end]
                equity = (1 + g).cumprod()
                rows.append({"version": name, "cost bp": cost, "period": period,
                             "Sharpe": s3.sharpe(g), "total %": (equity.iloc[-1] - 1) * 100,
                             "worst drawdown %": (equity / equity.cummax() - 1).min() * 100})
    info = pd.DataFrame(rows)
    yearly = pd.DataFrame({n: s3.net(d, s3.COST).groupby(s3.net(d, s3.COST).index.year)
                           .apply(lambda g: ((1 + g).prod() - 1) * 100) for n, d in results.items()})

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    f = lambda x: f"{x:.2f}"
    print(f"\nJUDGED against (v): after costs at {s3.COST} bp, funding included, "
          f"trade days {ex.JUDGE[0]} to {ex.JUDGE[1]}")
    print(judged.to_string(float_format=f))
    print(f"\n{note}")
    print("\nINFORMATION: (a10) judged the same way (not part of the decision)")
    print(pd.Series(a10).to_string())
    print(f"\nINFORMATION: risk and size, 2021-2023, after costs at {s3.COST} bp")
    print(extra.to_string(float_format=f))
    by_year = helper_by_year(days, ctx, helpers)
    print("\nINFORMATION: each helper's kept-10-minus-dropped-6 spread inside the picks, bp/day, by year")
    print(by_year.to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nINFORMATION: day-by-day difference from (v), judged period and 2024-2026")
    print(later.to_string(float_format=f))
    for cost in (s3.COST, 7.5, 0.0):
        print(f"\nINFORMATION: total return % at {cost} bp")
        print(info[info["cost bp"] == cost].pivot(index="version", columns="period", values="total %")
              [["Train", *s3.WINDOWS]].reindex(list(results)).to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"\nINFORMATION: total return % by year at {s3.COST} bp")
    print(yearly.to_string(float_format=lambda x: f"{x:.1f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([judged, pd.DataFrame({"a10": a10}).T.assign(verdict="information only")]) \
        .join(extra).to_csv(OUT / "portfolio_judged.csv", float_format="%.6g")
    info.to_csv(OUT / "portfolio_information.csv", index=False, float_format="%.6g")
    yearly.to_csv(OUT / "portfolio_yearly.csv", float_format="%.6g")
    later.to_csv(OUT / "portfolio_vs_v_by_period.csv", float_format="%.6g")
    by_year.to_csv(OUT / "portfolio_helper_by_year.csv", float_format="%.6g")
    (OUT / "portfolio_final.txt").write_text(note + "\n")
    chart(results, judged, a10, OUT / "portfolio_after_costs.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/portfolio_*")


def chart(results, judged, a10, path):
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
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 3.6 * len(PANELS)), sharex=True)
    base = 10_000 * (1 + s3.net(results["v"], s3.COST)).cumprod()
    for ax, pair in zip(axes, PANELS):
        ax.axvspan(base.index[0], pd.Timestamp(ex.JUDGE[1]), color=wash, lw=0, zorder=0)
        for start, end in (s3.WINDOWS["Drawdown 1"], s3.WINDOWS["Drawdown 2"]):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=grid, lw=0, zorder=0)
        lines = [(base, TITLES["v"], ink2, 1.3)]
        lines += [(10_000 * (1 + s3.net(results[n], s3.COST)).cumprod(), TITLES[n], c, w)
                  for n, c, w in zip(pair, (blue, orange), (1.9, 1.5))]
        placed = []
        for series, label, color, width in lines:
            ax.plot(series.index, series, color=color, lw=width, label=label)
            end = series.iloc[-1]
            nudge = -9 if any(abs(np.log(end / q)) < 0.04 for q in placed) else 0
            placed.append(end)
            ax.annotate(f"${end:,.0f}", (series.index[-1], end), xytext=(6, nudge),
                        textcoords="offset points", va="center", color=ink, fontsize=8.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 1.5, 2, 3, 5)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        parts = []
        for n in pair:
            row = judged.loc[n] if n in judged.index else pd.Series(a10)
            verdict = ("PASS" if row["verdict"] == "PASS" else "fail") if n in judged.index else "info"
            parts.append(f"({n}) {verdict}, Sharpe {row['Sharpe']:.2f}")
        ax.set_title(f"{' · '.join(parts)}  vs (v) {judged['Sharpe of (v)'].iloc[0]:.2f}  "
                     f"(after costs, 2021-23)", loc="left", color=ink, fontsize=10)
        ax.legend(frameon=False, loc="upper left", labelcolor=ink2)
        ax.set_ylabel("Value of $10,000")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Stage 3b A2: helpers keep A101_055's best 10 of 16, against (v), after costs "
                 f"({s3.COST} bp per dollar traded, funding included)",
                 x=0.01, ha="left", color=ink, fontsize=12.5)
    fig.text(0.01, 0.955, "Light shading = judged period 2021-03-29 to 2023-12-31. Darker bands = "
             "Drawdown 1 and 2. Later years are information only.", color=ink2, fontsize=9)
    fig.text(0.01, 0.943, "Every gap over (v) was built in 2021-2023. From 2024 the lines run parallel: "
             "after costs, each judged version gained about zero on (v) (-0.2 to +0.1 bp/day).", color=ink, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
