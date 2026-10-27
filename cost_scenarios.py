#!/usr/bin/env python3
"""What today's A101_055 would have kept under Binance fees. Describes only.

Re-prices the original backtest (baseline/daily_main.csv) with different costs
per dollar traded: Binance regular USDⓈ-M futures fees (maker 0.02%, taker
0.05%), the 10% BNB discount, and slippage. Funding is real money and is kept
in every scenario except the "before costs and funding" reference line.
Nothing is tested or chosen.

    .venv/bin/python cost_scenarios.py      # writes results/costs/
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "costs"
DAYS_PER_YEAR = 365
START_MONEY = 10_000

MAKER, TAKER, BNB = 2.0, 5.0, 0.9           # bp; BNB pays 90% of the fee
MAKER_BNB, TAKER_BNB = MAKER * BNB, TAKER * BNB
PAPER_SLIPPAGE = 3.0                         # bp, measured by the paper account (1 day so far)

# name -> (cost per dollar traded in bp, funding included)
SCENARIOS = {
    "Before costs and funding": (0.0, False),
    "No fees, funding paid": (0.0, True),
    "Limit orders (maker) + BNB, no slippage": (MAKER_BNB, True),
    "70% maker / 30% taker + BNB, 3 bp slippage on taker": (
        0.7 * MAKER_BNB + 0.3 * (TAKER_BNB + PAPER_SLIPPAGE), True),
    "Market orders (taker) + BNB, 3 bp slippage": (TAKER_BNB + PAPER_SLIPPAGE, True),
    "Backtest as run: taker, no BNB, 2 bp slippage": (TAKER + 2.0, True),
    "Taker, no BNB, 3 bp slippage": (TAKER + PAPER_SLIPPAGE, True),
    "Market orders (taker) + BNB, 5 bp slippage": (TAKER_BNB + 5.0, True),
}
# lines on the chart: the reference in grey, then light to dark blue as the cost grows
CHART = {
    "Before costs and funding": ("#898781", "--"),
    "Limit orders (maker) + BNB, no slippage": ("#86b6ef", "-"),
    "70% maker / 30% taker + BNB, 3 bp slippage on taker": ("#3987e5", "-"),
    "Market orders (taker) + BNB, 3 bp slippage": ("#1c5cab", "-"),
    "Market orders (taker) + BNB, 5 bp slippage": ("#0d366b", "-"),
}
SHORT_NAMES = {
    "Before costs and funding": "Before costs and funding",
    "Limit orders (maker) + BNB, no slippage": "Maker + BNB",
    "70% maker / 30% taker + BNB, 3 bp slippage on taker": "70/30 maker/taker + BNB",
    "Market orders (taker) + BNB, 3 bp slippage": "Taker + BNB, 3 bp slip",
    "Market orders (taker) + BNB, 5 bp slippage": "Taker + BNB, 5 bp slip",
}


def scenario(daily, cost_bp, with_funding):
    funding = daily["long_funding"] + daily["short_funding"]
    return daily["gross_ret"] - daily["turnover"] * cost_bp / 1e4 + (funding if with_funding else 0)


def summary(daily, returns):
    rows = {}
    years = len(daily) / DAYS_PER_YEAR
    for name, r in returns.items():
        equity = (1 + r).cumprod()
        cost_bp, with_funding = SCENARIOS[name]
        row = {
            "cost per $ traded, bp": cost_bp,
            "funding": "yes" if with_funding else "no",
            "fees + slippage, bp/day": (daily["turnover"] * cost_bp).mean(),
            "kept, bp/day": r.mean() * 1e4,
            "a year, % (compounded)": (equity.iloc[-1] ** (1 / years) - 1) * 100,
            "Sharpe": r.mean() / r.std() * np.sqrt(DAYS_PER_YEAR),
            "worst drawdown, %": (equity / equity.cummax() - 1).min() * 100,
            f"${START_MONEY:,} becomes": START_MONEY * equity.iloc[-1],
        }
        for year, g in r.groupby(r.index.year):
            row[str(year)] = ((1 + g).prod() - 1) * 100
        rows[name] = row
    return pd.DataFrame(rows).T


def chart(returns, table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ink, ink2, muted, grid, axis, surface = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                            "#c3c2b7", "#fcfcfb")
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": muted,
        "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface,
    })
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9.5),
                                      gridspec_kw={"height_ratios": [3, 1.6]})

    for name, (color, style) in CHART.items():
        equity = START_MONEY * (1 + returns[name]).cumprod()
        top.plot(equity.index, equity, color=color, ls=style, lw=2 if style == "-" else 1.4,
                 solid_capstyle="round", label=SHORT_NAMES[name])
        top.annotate(f"${equity.iloc[-1]:,.0f}", (equity.index[-1], equity.iloc[-1]),
                     xytext=(6, 0), textcoords="offset points", va="center", color=ink,
                     fontsize=9)
    top.set_yscale("log")
    top.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    top.yaxis.set_minor_formatter(ticker.NullFormatter())
    top.axhline(START_MONEY, color=axis, lw=0.8)
    top.set_ylabel(f"Value of ${START_MONEY:,} (log scale)")
    top.legend(frameon=False, loc="upper left", labelcolor=ink2)
    top.set_title("A101_055 backtest under Binance fees", loc="left", color=ink,
                  fontsize=14, pad=26)
    top.text(0, 1.03, "Top 16 long, bottom 16 short, rebalanced daily. Fees: maker 0.02%, "
             "taker 0.05%, BNB discount 10%. Every blue line includes funding.",
             transform=top.transAxes, color=ink2, fontsize=9)

    names = list(CHART)
    values = [table.at[n, "a year, % (compounded)"] for n in names]
    colors = [CHART[n][0] for n in names]
    y = np.arange(len(names))[::-1] * 1.0
    bottom.barh(y, values, height=0.42, color=colors)
    bottom.axvline(0, color=axis, lw=0.8)
    for yi, v, n in zip(y, values, names):
        cost = SCENARIOS[n][0]
        # value at the bar's end; a negative bar keeps its value right of zero, clear of the bar
        bottom.text(v + 0.6 if v >= 0 else 0.6, yi, f"{v:+.1f}% a year", va="center",
                    ha="left", color=ink, fontsize=9)
        bottom.text(0, yi + 0.36, f"{SHORT_NAMES[n]}  ({cost:.1f} bp per $ traded)",
                    transform=bottom.get_yaxis_transform(), va="bottom", ha="left",
                    color=ink2, fontsize=9, zorder=3,
                    bbox=dict(facecolor=surface, edgecolor="none", pad=1))
    bottom.set_yticks([])
    bottom.set_axisbelow(True)
    bottom.spines["left"].set_visible(False)
    bottom.set_ylim(y.min() - 0.4, y.max() + 0.85)
    bottom.grid(axis="x", color=grid, lw=0.6)
    bottom.grid(axis="y", visible=False)
    bottom.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    bottom.set_xlim(min(0, min(values)) - 3, max(values) + 9)
    bottom.set_title("Kept per year, compounded", loc="left", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    daily = pd.read_csv(HERE / "baseline" / "daily_main.csv", index_col=0, parse_dates=True)
    returns = {name: scenario(daily, c, f) for name, (c, f) in SCENARIOS.items()}
    table = summary(daily, returns)
    funding = (daily["long_funding"] + daily["short_funding"]).mean()
    breakeven = (daily["gross_ret"].mean() + funding) / daily["turnover"].mean() * 1e4

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Backtest {daily.index[0]:%Y-%m-%d} to {daily.index[-1]:%Y-%m-%d}: turnover "
          f"{daily['turnover'].mean():.0%} of equity a day; funding {funding * 1e4:+.2f} bp/day; "
          f"break-even cost {breakeven:.1f} bp per dollar traded.")
    print(table.to_string(float_format=lambda v: f"{v:,.1f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "scenarios.csv", float_format="%.6g")
    chart(returns, table, OUT / "costs.png")
    print(f"Saved {OUT.relative_to(HERE)}/scenarios.csv and costs.png")


if __name__ == "__main__":
    main()
