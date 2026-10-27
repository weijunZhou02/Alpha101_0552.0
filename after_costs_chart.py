#!/usr/bin/env python3
"""Today's version (a) against (v), volatility scaling, after costs. Describes only.

Value of $10,000 after fees, slippage and funding at two cost levels: 3.5 bp per
dollar traded (mostly limit orders + BNB) and 7.5 bp (market orders + BNB, 3 bp
slippage), plus (v)'s position size over time. All years are shown; only
2021-2023 was used to choose (v).

    .venv/bin/python after_costs_chart.py      # writes results/stage2/after_costs.png and .csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import experiment as ex
import stage2 as s2

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage2"
COSTS = {3.5: "mostly limit orders + BNB", 7.5: "market orders + BNB, 3 bp slippage"}
START_MONEY = 10_000


def net(d, cost_bp):
    return d["gross"] - d["turnover"] * cost_bp / 1e4 + d["funding"]


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    trend = panels["close"] / panels["close"].shift(s2.TREND_DAYS) - 1
    a = s2.run(s2.VERSIONS[0], days, ctx, trend)
    v = s2.run(s2.VERSIONS[-1], days, ctx, trend)

    rows = []
    for cost in COSTS:
        for name, d in (("(a) today's version", a), ("(v) volatility scaling", v)):
            r = net(d, cost)
            equity = (1 + r).cumprod()
            row = {"cost bp": cost, "version": name,
                   "a year %": (equity.iloc[-1] ** (bt.DAYS_PER_YEAR / len(r)) - 1) * 100,
                   "Sharpe": r.mean() / r.std() * np.sqrt(bt.DAYS_PER_YEAR),
                   "worst drawdown %": (equity / equity.cummax() - 1).min() * 100,
                   f"${START_MONEY:,} becomes": START_MONEY * equity.iloc[-1],
                   "turnover %": d["turnover"].mean() * 100}
            for y, g in r.groupby(r.index.year):
                row[str(y)] = ((1 + g).prod() - 1) * 100
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["cost bp", "version"])
    pd.set_option("display.width", 250)
    print(table.to_string(float_format=lambda x: f"{x:,.1f}"))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "after_costs_a_vs_v.csv", float_format="%.6g")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ink, ink2, muted, grid, axis = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue, wash, surface = "#2a78d6", "#f0efec", "#fcfcfb"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": muted,
        "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface,
    })
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 2.2, 1.2]})
    judge_end = pd.Timestamp(ex.JUDGE[1])
    for ax, cost in zip(axes[:2], COSTS):
        ax.axvspan(a.index[0], judge_end, color=wash, lw=0, zorder=0)
        for name, d, color, width in (("(a) today's version", a, ink2, 1.4),
                                      ("(v) volatility scaling", v, blue, 2.0)):
            equity = START_MONEY * (1 + net(d, cost)).cumprod()
            ax.plot(equity.index, equity, color=color, lw=width, label=name)
            ax.annotate(f"${equity.iloc[-1]:,.0f}", (equity.index[-1], equity.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        color=ink, fontsize=9)
        ax.axhline(START_MONEY, color=axis, lw=0.8)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_ylabel(f"Value of ${START_MONEY:,} (log)")
        ax.set_title(f"After costs at {cost} bp per dollar traded ({COSTS[cost]}), "
                     f"funding included", loc="left", color=ink, fontsize=10)
        ax.legend(frameon=False, loc="upper left", labelcolor=ink2)

    ax = axes[2]
    ax.axvspan(a.index[0], judge_end, color=wash, lw=0, zorder=0)
    ax.axhline(1.0, color=axis, lw=0.8)
    ax.plot(v.index, v["scale"], color=blue, lw=1.4)
    ax.set_ylim(0.4, 1.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_ylabel("(v) position size")
    ax.set_title("(v) position size: usual volatility / last 30 days' volatility, "
                 "kept between 0.5x and 1.5x", loc="left", color=ink, fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Today's version vs (v) volatility scaling, after costs",
                 x=0.01, ha="left", color=ink, fontsize=13)
    fig.text(0.01, 0.955, "Shaded = 2021-03-29 to 2023-12-31, the only years used to choose (v). "
             "Later years were already seen: information only.", color=ink2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "after_costs.png", dpi=150)
    plt.close(fig)
    print(f"Saved {(OUT / 'after_costs.png').relative_to(HERE)} and after_costs_a_vs_v.csv")


if __name__ == "__main__":
    main()
