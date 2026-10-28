#!/usr/bin/env python3
"""Stage 3, step 5: blend A101_055 with the shortlisted helpers, judged after costs against (v).

Blends of daily percentile ranks among the day's 80 coins:
  (b1) 0.7 A101_055 + 0.3 Q158_MIN10
  (b2) 0.7 A101_055 + 0.3 GTJA_034
  (b3) 0.6 A101_055 + 0.2 Q158_MIN10 + 0.2 GTJA_034
then the usual 16 + 16 lists, equal weights, and (v)'s volatility scaling computed
from each blend's own unscaled results. Rule A after costs (3.5 bp per dollar
traded, funding included) on trade days 2021-03-29 to 2023-12-31; declared in
README ("Stage 3, step 5").

    .venv/bin/python stage3_blend.py       # writes results/stage3/blend_*
"""

from pathlib import Path

import numpy as np
import pandas as pd

import a101_055 as live
import backtest as bt
import experiment as ex
import stage2 as s2

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage3"
COST = 3.5
INFO_COSTS = (0.0, 7.5)
YEARS = (2021, 2022, 2023)
BLENDS = {
    "b1": {"Q158_MIN10": 0.3},
    "b2": {"GTJA_034": 0.3},
    "b3": {"Q158_MIN10": 0.2, "GTJA_034": 0.2},
}
TITLES = {"v": "(v) working version", "b1": "(b1) + 0.3 Q158_MIN10",
          "b2": "(b2) + 0.3 GTJA_034", "b3": "(b3) + 0.2 each"}
WINDOWS = {
    "2024-26": ("2024-01-01", "2026-10-24"),
    "Drawdown 1": ("2023-11-18", "2024-10-27"),
    "Drawdown 2": ("2025-07-22", "2026-06-07"),
    "Box 1": ex.INFO["Box 1"], "Box 2": ex.INFO["Box 2"], "Dip 3": ex.INFO["Dip 3"],
}


def blend_books(weights, days, ctx, helpers):
    """Target weights per signal day for one blend (None on days without lists)."""
    a_spec = ex.VERSIONS[0]
    target = {}
    for day, table, buy, short, problem in days:
        if problem:
            target[day] = None
            continue
        t = table[["a101_055"]].copy()
        score = (1 - sum(weights.values())) * t["a101_055"].rank(pct=True)
        for alpha, w in weights.items():
            values = (helpers[alpha].loc[day].reindex(t.index) if day in helpers[alpha].index
                      else pd.Series(np.nan, index=t.index))
            score = score + w * values.rank(pct=True).fillna(0.5)
        t["a101_055"] = score                          # missing A101_055 score stays missing
        try:
            _, b, s = live.select_lists(t)
        except live.ComputeError:
            target[day] = None
            continue
        target[day] = ex.weights(a_spec, day, b, s, ctx)
    return target


def net(d, cost):
    return d["gross"] - d["turnover"] * cost / 1e4 + d["funding"]


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(bt.DAYS_PER_YEAR)


def judge(d, base):
    sl = slice(*ex.JUDGE)
    g, b = net(d, COST).loc[sl], net(base, COST).loc[sl]
    diff = g - b
    years = sum(sharpe(g[g.index.year == y]) > sharpe(b[b.index.year == y]) for y in YEARS)
    equity = (1 + g).cumprod()
    row = {"Sharpe": sharpe(g), "Sharpe of (v)": sharpe(b), "bp/day": g.mean() * 1e4,
           "vs (v) bp/day": diff.mean() * 1e4,
           "t vs (v)": diff.mean() / diff.std() * np.sqrt(len(diff)),
           "years better": years,
           "worst drawdown %": (equity / equity.cummax() - 1).min() * 100,
           "turnover %": d.loc[sl, "turnover"].mean() * 100,
           "before costs Sharpe": sharpe(d.loc[sl, "gross"])}
    checks = {"Sharpe above (v)": row["Sharpe"] > row["Sharpe of (v)"],
              "t >= 2": row["t vs (v)"] >= 2, "2 of 3 years better": years >= 2}
    failed = [k for k, ok in checks.items() if not ok]
    row["verdict"] = "PASS" if not failed else "fail: " + ", ".join(failed)
    return row


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    trend = panels["close"] / panels["close"].shift(s2.TREND_DAYS) - 1
    raw = pd.read_csv(HERE / "data" / "shortlist_values.csv.gz", parse_dates=["date"])
    helpers = {a: raw.pivot(index="date", columns="symbol", values=a) for a in ("Q158_MIN10", "GTJA_034")}

    results = {"v": s2.run(s2.VERSIONS[-1], days, ctx, trend)}
    for name, weights in BLENDS.items():
        target = blend_books(weights, days, ctx, helpers)
        unscaled = s2.simulate(target, days, ctx)
        results[name] = s2.simulate(target, days, ctx, scale=s2.vol_scale(unscaled["gross"]))
        print(f"  ({name}) done")

    judged = pd.DataFrame({n: judge(results[n], results["v"]) for n in BLENDS}).T
    passers = [n for n in BLENDS if judged.at[n, "verdict"] == "PASS"]
    if passers:
        best = max(passers, key=lambda n: judged.at[n, "Sharpe"])
        note = f"({best}) passes and has the highest Sharpe after costs: it is the final version."
    else:
        note = "No blend passes: (v) stays the final version."

    rows = []
    for name, d in results.items():
        for cost in (*INFO_COSTS, COST):
            r = net(d, cost)
            for period, (start, end) in {"Train": ex.JUDGE, **WINDOWS}.items():
                g = r.loc[start:end]
                equity = (1 + g).cumprod()
                rows.append({"version": name, "cost bp": cost, "period": period,
                             "Sharpe": sharpe(g), "total %": (equity.iloc[-1] - 1) * 100,
                             "worst drawdown %": (equity / equity.cummax() - 1).min() * 100})
    info = pd.DataFrame(rows)
    yearly = pd.DataFrame({n: net(d, COST).groupby(net(d, COST).index.year)
                           .apply(lambda g: ((1 + g).prod() - 1) * 100) for n, d in results.items()})

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    f = lambda x: f"{x:.2f}"
    print(f"\nJUDGED against (v): after costs at {COST} bp, funding included, "
          f"trade days {ex.JUDGE[0]} to {ex.JUDGE[1]}")
    print(judged.to_string(float_format=f))
    print(f"\n{note}")
    for cost in (COST, 7.5, 0.0):
        print(f"\nINFORMATION: total return % at {cost} bp")
        print(info[info["cost bp"] == cost].pivot(index="version", columns="period", values="total %")
              [["Train", *WINDOWS]].reindex(list(results)).to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"\nINFORMATION: total return % by year at {COST} bp")
    print(yearly.to_string(float_format=lambda x: f"{x:.1f}"))

    judged.to_csv(OUT / "blend_judged.csv", float_format="%.6g")
    info.to_csv(OUT / "blend_information.csv", index=False, float_format="%.6g")
    yearly.to_csv(OUT / "blend_yearly.csv", float_format="%.6g")
    (OUT / "blend_final.txt").write_text(note + "\n")
    chart(results, judged, OUT / "blend_after_costs.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/blend_*")


def chart(results, judged, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ink, ink2, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    blue, wash, surface = "#2a78d6", "#f0efec", "#fcfcfb"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface,
    })
    fig, axes = plt.subplots(len(BLENDS), 1, figsize=(11, 3.4 * len(BLENDS)), sharex=True)
    base = 10_000 * (1 + net(results["v"], COST)).cumprod()
    for ax, name in zip(axes, BLENDS):
        ax.axvspan(base.index[0], pd.Timestamp(ex.JUDGE[1]), color=wash, lw=0, zorder=0)
        for start, end in (WINDOWS["Drawdown 1"], WINDOWS["Drawdown 2"]):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=grid, lw=0, zorder=0)
        eq = 10_000 * (1 + net(results[name], COST)).cumprod()
        for series, label, color, width in ((base, TITLES["v"], ink2, 1.3),
                                            (eq, TITLES[name], blue, 1.9)):
            ax.plot(series.index, series, color=color, lw=width, label=label)
            ax.annotate(f"${series.iloc[-1]:,.0f}", (series.index[-1], series.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", color=ink,
                        fontsize=8.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 1.5, 2, 3, 5)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        verdict = "PASS" if judged.at[name, "verdict"] == "PASS" else "fail"
        ax.set_title(f"{TITLES[name]}: {verdict} (Sharpe after costs 2021-23: "
                     f"{judged.at[name, 'Sharpe']:.2f} vs {judged.at[name, 'Sharpe of (v)']:.2f})",
                     loc="left", color=ink, fontsize=10)
        ax.legend(frameon=False, loc="upper left", labelcolor=ink2)
        ax.set_ylabel("Value of $10,000")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Stage 3 blends against (v), after costs ({COST} bp per dollar traded, funding included)",
                 x=0.01, ha="left", color=ink, fontsize=13)
    fig.text(0.01, 0.948, "Light shading = judged period 2021-03-29 to 2023-12-31. Darker bands = "
             "Drawdown 1 and 2. Later years are information only.", color=ink2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
