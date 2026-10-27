#!/usr/bin/env python3
"""The two big after-cost drawdowns of today's version, month by month. Describes only.

Uses the original backtest's daily results (baseline/daily_main.csv) priced at
7.5 bp per dollar traded (market orders + BNB, 3 bp slippage) plus funding.
Finds the drawdowns deeper than 25%, then splits each one, month by month, into:
long side against the market, short side against the market, fees, funding.
Also: how much of each drawdown came from its 20 worst days, and the average
position size (v) would have used. Writes results/drawdowns/.

    .venv/bin/python drawdowns.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import experiment as ex
import stage2 as s2

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "drawdowns"
COST_BP = 7.5
DEEP = -0.25
WORST_DAYS = 20
PARTS = {  # column -> (label, colour)
    "long": ("Long side vs market", "#2a78d6"),
    "short": ("Short side vs market", "#eb6834"),
    "fees": ("Fees + slippage", "#1baf7a"),
    "funding": ("Funding", "#eda100"),
}


def biggest_fall(equity):
    """(first day down, bottom, depth) of the largest peak-to-bottom fall."""
    dd = equity / equity.cummax() - 1
    bottom = dd.idxmin()
    peak = equity.loc[:bottom].idxmax()
    return equity.index[equity.index.get_loc(peak) + 1], bottom, dd.min()


def windows(net):
    """Falls deeper than 25%: the largest one, then the largest one after its bottom, and so on.
    (At 7.5 bp the 2025 high stays below the 2023 high, so both falls sit inside one long
    stretch below the old peak; each fall is measured from its own peak.)"""
    equity = (1 + net).cumprod()
    found, rest = [], equity
    while len(rest) > 1:
        start, bottom, depth = biggest_fall(rest)
        if depth > DEEP:
            break
        found.append((start, bottom, depth))
        rest = equity.loc[bottom:]
    return found


def main():
    d = pd.read_csv(HERE / "baseline" / "daily_main.csv", index_col=0, parse_dates=True)
    parts = pd.DataFrame({
        "long": 0.5 * (d["long_ret"] - d["ew_universe_ret"]),
        "short": 0.5 * (d["short_ret"] + d["ew_universe_ret"]),
        "fees": -d["turnover"] * COST_BP / 1e4,
        "funding": d["long_funding"] + d["short_funding"],
        "market": d["ew_universe_ret"],
    })
    parts["net"] = parts[["long", "short", "fees", "funding"]].sum(axis=1)

    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    trend = panels["close"] / panels["close"].shift(s2.TREND_DAYS) - 1
    v = s2.run(s2.VERSIONS[-1], days, ctx, trend)
    a = s2.run(s2.VERSIONS[0], days, ctx, trend)
    v_net = v["gross"] - v["turnover"] * COST_BP / 1e4 + v["funding"]

    tables, summary = {}, []
    for i, (start, bottom, depth) in enumerate(windows(parts["net"]), 1):
        w = parts.loc[start:bottom]
        name = f"Drawdown {i}: {start:%Y-%m-%d} to {bottom:%Y-%m-%d}"
        monthly = w.groupby(w.index.to_period("M"))[["market", "long", "short", "fees",
                                                     "funding", "net"]].sum() * 100
        monthly["days"] = w.groupby(w.index.to_period("M")).size()
        monthly["(v) size"] = v.loc[start:bottom, "scale"].groupby(
            v.loc[start:bottom].index.to_period("M")).mean()
        monthly.loc["Total"] = monthly.sum()
        monthly.loc["Total", "(v) size"] = v.loc[start:bottom, "scale"].mean()
        tables[name] = monthly
        worst = w["net"].nsmallest(WORST_DAYS)
        summary.append({
            "drawdown": name, "depth (compounded) %": depth * 100,
            "days": len(w),
            "long side %": w["long"].sum() * 100, "short side %": w["short"].sum() * 100,
            "fees %": w["fees"].sum() * 100, "funding %": w["funding"].sum() * 100,
            "net, simple sum %": w["net"].sum() * 100,
            f"{WORST_DAYS} worst days, simple sum %": worst.sum() * 100,
            "all other days %": (w["net"].sum() - worst.sum()) * 100,
            "market over the window %": ((1 + w["market"]).prod() - 1) * 100,
            "(v) over the window %": ((1 + v_net.loc[start:bottom]).prod() - 1) * 100,
            "(v) average size": v.loc[start:bottom, "scale"].mean(),
        })
    summary = pd.DataFrame(summary).set_index("drawdown")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Today's version after costs at {COST_BP} bp per dollar traded, funding included")
    print(summary.T.to_string(float_format=lambda x: f"{x:.1f}"))
    for name, t in tables.items():
        print(f"\n{name}  (% of equity, simple sums per month)")
        print(t.to_string(float_format=lambda x: f"{x:.1f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", float_format="%.6g")
    for i, (name, t) in enumerate(tables.items(), 1):
        t.to_csv(OUT / f"drawdown_{i}_monthly.csv", float_format="%.6g")
    chart(tables, OUT / "drawdowns.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/")


def chart(tables, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, muted, grid, axis, surface = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                            "#c3c2b7", "#fcfcfb")
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": muted,
        "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface, "axes.axisbelow": True,
    })
    fig, axes = plt.subplots(len(tables), 1, figsize=(11, 4.6 * len(tables)))
    axes = np.atleast_1d(axes)
    for ax, (name, t) in zip(axes, tables.items()):
        m = t.drop(index="Total")
        x = np.arange(len(m))
        pos_base, neg_base = np.zeros(len(m)), np.zeros(len(m))
        for col, (label, color) in PARTS.items():
            vals = m[col].to_numpy()
            up, down = np.clip(vals, 0, None), np.clip(vals, None, 0)
            ax.bar(x, up, 0.62, bottom=pos_base, color=color, label=label,
                   edgecolor=surface, linewidth=1)
            ax.bar(x, down, 0.62, bottom=neg_base, color=color, edgecolor=surface, linewidth=1)
            pos_base += up
            neg_base += down
        ax.plot(x, m["net"], color=ink, lw=0, marker="o", ms=6, label="Net for the month",
                markeredgecolor=surface, markeredgewidth=1.5, zorder=4)
        for xi, val in zip(x, m["net"]):
            ax.annotate(f"{val:+.1f}%", (xi, val), xytext=(0, -12 if val < 0 else 8),
                        textcoords="offset points", ha="center", color=ink, fontsize=8)
        ax.axhline(0, color=axis, lw=0.8)
        ax.set_xticks(x, [str(p) for p in m.index])
        ax.set_ylabel("% of equity in the month")
        total = t.loc["Total"]
        ax.set_title(f"{name}: net {total['net']:+.1f}% (simple sum), "
                     f"of which fees {total['fees']:+.1f}% and funding {total['funding']:+.1f}%",
                     loc="left", color=ink, fontsize=10)
        ax.legend(frameon=False, ncols=5, loc="upper left", labelcolor=ink2, fontsize=8)
        lo, hi = min(neg_base.min(), m["net"].min()), max(pos_base.max(), m["net"].max())
        ax.set_ylim(lo * 1.25, hi * 1.25 + (hi - lo) * 0.18)
    fig.suptitle(f"Where the two big after-cost drawdowns came from "
                 f"(costs {COST_BP} bp per dollar traded, funding included)",
                 x=0.01, ha="left", color=ink, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
