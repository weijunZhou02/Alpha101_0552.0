#!/usr/bin/env python3
"""Information view: the stage 3b A2 versions from 2023-07-01 to the end of the data.

Same daily results as stage3b_portfolio.py (the versions trade exactly as before,
volatility scaling included); each line is only rebased to $10,000 on 2023-07-01.
Describes only; chooses nothing. 2023-07-01 to 2023-12-31 is still inside the judged
period, everything after is information.

    .venv/bin/python view_from_2023h2.py      # writes results/stage3b/from_2023h2.*
"""

from pathlib import Path

import pandas as pd

import backtest as bt
import experiment as ex
import stage2 as s2
import stage3_blend as s3
import stage3b_portfolio as p3

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage3b"
START = "2023-07-01"
COSTS = (0.0, 3.5, 7.5)          # 0 = before costs: no fees and no funding (the lab's convention)
SHOW = ["a", "v", "p10", "p16", "i10", "i16", "pi10"]
TITLES = {"a": "(a) today's version", **p3.TITLES}


def daily(d, cost):
    return d["gross"] if cost == 0 else s3.net(d, cost)


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    trend = panels["close"] / panels["close"].shift(s2.TREND_DAYS) - 1
    raw = pd.read_csv(HERE / "data" / "shortlist_3b_values.csv.gz", parse_dates=["date"])
    helpers = {a: raw.pivot(index="date", columns="symbol", values=a) for a in (p3.PREMIUM, p3.IMIN)}
    results = {"a": s2.run(s2.VERSIONS[0], days, ctx, trend), "v": s2.run(s2.VERSIONS[-1], days, ctx, trend)}
    for name in SHOW[2:]:
        names, share = p3.VERSIONS[name]
        results[name] = p3.scaled(p3.books(names, share, days, ctx, helpers), days, ctx)

    rows = []
    for name in SHOW:
        for cost in COSTS:
            g = daily(results[name], cost).loc[START:]
            equity = (1 + g).cumprod()
            years = len(g) / bt.DAYS_PER_YEAR
            rows.append({"version": name, "cost bp": cost, "total %": (equity.iloc[-1] - 1) * 100,
                         "per year %": (equity.iloc[-1] ** (1 / years) - 1) * 100,
                         "Sharpe": s3.sharpe(g),
                         "worst drawdown %": (equity / equity.cummax() - 1).min() * 100})
    table = pd.DataFrame(rows)
    end = results["v"].index[-1]
    print(f"From {START} to {end:%Y-%m-%d} ({len(results['v'].loc[START:])} trade days)")
    for cost in COSTS:
        print(f"\n{'before costs (no fees, no funding)' if cost == 0 else f'after costs at {cost:g} bp, funding included'}")
        print(table[table["cost bp"] == cost].set_index("version").drop(columns="cost bp")
              .to_string(float_format=lambda x: f"{x:.1f}"))
    table.to_csv(OUT / "from_2023h2.csv", index=False, float_format="%.6g")
    chart(results, table, end, OUT / "from_2023h2.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/from_2023h2.*")


def chart(results, table, end, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ink, ink2, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    blue, orange, wash, surface, faint = "#2a78d6", "#eb6834", "#f0efec", "#fcfcfb", "#b9b8b1"
    style = {"a": (ink2, 1.1, ":"), "v": (ink, 1.8, "-"), "p10": (blue, 2.0, "-"),
             "p16": (faint, 1.0, "-"), "i10": (faint, 1.0, "-"), "i16": (faint, 1.0, "-"),
             "pi10": (orange, 1.4, "-")}
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface,
    })
    fig, axes = plt.subplots(len(COSTS), 1, figsize=(11, 3.7 * len(COSTS)), sharex=True)
    for ax, cost in zip(axes, COSTS):
        ax.axvspan(pd.Timestamp(START), pd.Timestamp(ex.JUDGE[1]), color=wash, lw=0, zorder=0)
        for s, e in (s3.WINDOWS["Drawdown 1"], s3.WINDOWS["Drawdown 2"]):
            ax.axvspan(max(pd.Timestamp(s), pd.Timestamp(START)), pd.Timestamp(e), color=grid, lw=0, zorder=0)
        ax.axhline(10_000, color=axis, lw=0.8)
        ends = []
        for name in reversed(SHOW):                      # (v) and (p10) drawn on top
            eq = 10_000 * (1 + daily(results[name], cost).loc[START:]).cumprod()
            color, width, ls = style[name]
            ax.plot(eq.index, eq, color=color, lw=width, ls=ls, label=TITLES[name])
            if name in ("a", "v", "p10", "pi10"):
                ends.append((eq.iloc[-1], eq.index[-1], ink2 if name == "a" else color))
        low, high = ax.get_ylim()
        gap, previous = 0.045 * (high - low), None
        for last, when, color in sorted(ends, reverse=True):   # keep the end labels apart
            y = last if previous is None else min(last, previous - gap)
            previous = y
            ax.annotate(f"${last:,.0f}", (when, last), xytext=(6, y),
                        textcoords=("offset points", "data"), va="center", color=color, fontsize=8.5)
        row = table[(table["cost bp"] == cost)].set_index("version")
        label = "Before costs and funding" if cost == 0 else f"After costs: {cost:g} bp per dollar traded, funding included"
        ax.set_title(f"{label}.  Per year: (v) {row.at['v', 'per year %']:+.0f}%, "
                     f"(p10) {row.at['p10', 'per year %']:+.0f}%, (a) {row.at['a', 'per year %']:+.0f}%",
                     loc="left", color=ink, fontsize=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_ylabel("Value of $10,000")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], frameon=False, loc="upper left", labelcolor=ink2,
                   fontsize=8, ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(f"A101_055 versions from {START} to {end:%Y-%m-%d}, $10,000 at the start",
                 x=0.01, ha="left", color=ink, fontsize=13)
    fig.text(0.01, 0.952, "Light shading = last half-year of the judged period (to 2023-12-31). Darker bands = "
             "Drawdown 1 and 2. Lines trade exactly as in step A2, volatility scaling included.",
             color=ink2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
