#!/usr/bin/env python3
"""Stage 2b: the funding guard, judged after costs.

When the lists are made, a coin whose average daily funding over the last 7
days is below -0.1% cannot be shorted, and one above +0.1% cannot be bought;
each list is refilled from the next scores. Versions: (h) the guard on today's
version, (h+v) the guard plus stage 2's volatility scaling. Judged after costs
(3.5 bp per dollar traded, funding included) on trade days 2021-03-29 to
2023-12-31 with the insurance rule declared in README ("Stage 2b").

    .venv/bin/python stage2b.py          # writes results/stage2b/
"""

from pathlib import Path

import numpy as np
import pandas as pd

import a101_055 as live
import backtest as bt
import experiment as ex
import stage2 as s2

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage2b"
GUARD = 0.001                      # 0.1% a day
FUND_DAYS, FUND_MIN = 7, 4
COST = 3.5                         # bp per dollar traded, judged
INFO_COSTS = (0.0, 7.5)
WINDOWS = {
    "2024-26": ("2024-01-01", "2026-10-24"),
    "Drawdown 1": ("2023-11-18", "2024-10-27"),
    "Drawdown 2": ("2025-07-22", "2026-06-07"),
    "Box 1": ex.INFO["Box 1"], "Box 2": ex.INFO["Box 2"], "Dip 3": ex.INFO["Dip 3"],
}


def guarded_lists(day, table, buy, short, f7):
    """The lists with the funding guard, and how many coins it replaced on each side."""
    t = table[["a101_055"]]
    f = f7.loc[day, t.index]
    no_short, no_buy = f < -GUARD, f > GUARD           # a missing funding value is not guarded
    if not (no_short[short].any() or no_buy[buy].any()):
        return buy, short, 0, 0
    _, new_buy, _ = live.select_lists(t[~no_buy])
    _, _, new_short = live.select_lists(t[~no_short])
    if set(new_buy) & set(new_short):
        raise live.ComputeError("the funding guard made the lists overlap")
    return new_buy, new_short, len(set(buy) - set(new_buy)), len(set(short) - set(new_short))


def books(guard, days, ctx, f7):
    a_spec = ex.VERSIONS[0]
    target, replaced = {}, []
    for day, table, buy, short, problem in days:
        if problem:
            target[day] = None
            continue
        b, s, rb, rs = buy, short, 0, 0
        if guard:
            try:
                b, s, rb, rs = guarded_lists(day, table, buy, short, f7)
            except live.ComputeError:
                target[day] = None
                continue
        target[day] = ex.weights(a_spec, day, b, s, ctx)
        replaced.append({"trade_date": day + pd.Timedelta(days=1), "longs replaced": rb,
                         "shorts replaced": rs, "longs": len(b), "shorts": len(s)})
    return target, pd.DataFrame(replaced).set_index("trade_date")


def run(guard, volscale, days, ctx, f7):
    target, replaced = books(guard, days, ctx, f7)
    d = s2.simulate(target, days, ctx)
    if volscale:
        d = s2.simulate(target, days, ctx, scale=s2.vol_scale(d["gross"]))
    return d, replaced


def net(d, cost):
    return d["gross"] - d["turnover"] * cost / 1e4 + d["funding"]


def insurance(name, d, base, base_name):
    """After-cost measures on 2021-2023 and the insurance verdict against one yardstick."""
    sl = slice(*ex.JUDGE)
    g, b = net(d, COST).loc[sl], net(base, COST).loc[sl]
    m = ex.measures(g, b)
    if (g - b).abs().max() == 0:
        verdict = "untestable: same days as the yardstick"
    else:
        verdict = "PASS" if m["t vs (a)"] > -1 else "fail: clearly worse (t <= -1)"
    return {"version": name, "against": base_name, "Sharpe": m["Sharpe"], "bp/day": m["bp/day"],
            "worst drawdown %": m["worst drawdown %"], "worst 1% of days %": m["worst 1% of days %"],
            "difference bp/day": m["vs (a) bp/day"], "t": m["t vs (a)"],
            "turnover %": d.loc[sl, "turnover"].mean() * 100, "verdict": verdict}


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    f7 = funding.rolling(FUND_DAYS, min_periods=FUND_MIN).mean()

    results, replaced = {}, {}
    for name, guard, volscale in (("a", False, False), ("v", False, True),
                                  ("h", True, False), ("h+v", True, True)):
        results[name], replaced[name] = run(guard, volscale, days, ctx, f7)
        print(f"  ({name}) done")
    original = pd.read_csv(HERE / "baseline" / "daily_main.csv", index_col=0, parse_dates=True)
    a = results["a"]
    assert a.index.equals(original.index) and np.allclose(
        a["gross"], original["gross_ret"], rtol=1e-7, atol=1e-10), "(a) differs from the original"
    print("Version (a) reproduces the original backtest.")

    judged = pd.DataFrame([
        insurance("h", results["h"], results["a"], "(a)"),
        insurance("h+v", results["h+v"], results["a"], "(a)"),
        insurance("h+v", results["h+v"], results["v"], "(v)"),
        insurance("v", results["v"], results["a"], "(a) [information]"),
    ])
    h_ok = judged.iloc[0]["verdict"] == "PASS"
    hv_ok = (judged.iloc[1]["verdict"] == "PASS") and (judged.iloc[2]["verdict"] == "PASS")
    if hv_ok:
        note = "(h+v) passes: it replaces (v) as the final version."
    elif h_ok:
        note = "Only (h) passes: the guard is noted, but (v) stays the final version."
    else:
        note = "Neither passes: (v) stays the final version."

    rows = []
    for name, d in results.items():
        for cost in (*INFO_COSTS, COST):
            r = net(d, cost)
            for period, (start, end) in {"Train": ex.JUDGE, **WINDOWS}.items():
                g = r.loc[start:end]
                equity = (1 + g).cumprod()
                rows.append({"version": name, "cost bp": cost, "period": period,
                             "Sharpe": g.mean() / g.std() * np.sqrt(bt.DAYS_PER_YEAR),
                             "total %": (equity.iloc[-1] - 1) * 100,
                             "worst drawdown %": (equity / equity.cummax() - 1).min() * 100,
                             "funding bp/day": d.loc[start:end, "funding"].mean() * 1e4})
    info = pd.DataFrame(rows)
    swaps = []
    for period, (start, end) in {"Train": ex.JUDGE, **WINDOWS}.items():
        x = replaced["h"].loc[start:end]
        swaps.append({"period": period,
                      "shorts replaced, % of short slots": x["shorts replaced"].sum() / x["shorts"].sum() * 100,
                      "longs replaced, % of long slots": x["longs replaced"].sum() / x["longs"].sum() * 100})
    swaps = pd.DataFrame(swaps).set_index("period")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    f = lambda x: f"{x:.2f}"
    print(f"\nJUDGED: after costs at {COST} bp per dollar traded, funding included, "
          f"trade days {ex.JUDGE[0]} to {ex.JUDGE[1]}")
    print(judged.to_string(index=False, float_format=f))
    print(f"\n{note}")
    print("\nHOW OFTEN THE GUARD ACTED")
    print(swaps.to_string(float_format=lambda x: f"{x:.1f}"))
    for cost in (*INFO_COSTS, COST):
        print(f"\nINFORMATION: total return % at {cost} bp (funding included)")
        print(info[info["cost bp"] == cost].pivot(index="version", columns="period",
                                                   values="total %")
              [["Train", *WINDOWS]].reindex(list(results)).to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"\nINFORMATION: Sharpe at {COST} bp")
    print(info[info["cost bp"] == COST].pivot(index="version", columns="period", values="Sharpe")
          [["Train", *WINDOWS]].reindex(list(results)).to_string(float_format=f))

    OUT.mkdir(parents=True, exist_ok=True)
    judged.to_csv(OUT / "judged.csv", index=False, float_format="%.6g")
    info.to_csv(OUT / "information.csv", index=False, float_format="%.6g")
    swaps.to_csv(OUT / "guard_activity.csv", float_format="%.6g")
    pd.DataFrame({n: net(d, COST) for n, d in results.items()}).to_csv(
        OUT / f"daily_net_{COST}bp.csv", float_format="%.8g", date_format="%Y-%m-%d")
    (OUT / "final.txt").write_text(note + "\n")
    chart(results, OUT / "guard.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/")


def chart(results, path):
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
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    judge_end = pd.Timestamp(ex.JUDGE[1])
    pairs = ((("a", "(a) today's version"), ("h", "(h) with the funding guard")),
             (("v", "(v) volatility scaling"), ("h+v", "(h+v) guard + volatility scaling")))
    for ax, ((n0, l0), (n1, l1)) in zip(axes, pairs):
        ax.axvspan(results["a"].index[0], judge_end, color=wash, lw=0, zorder=0)
        for name, label, color, width in ((n0, l0, ink2, 1.4), (n1, l1, blue, 2.0)):
            equity = 10_000 * (1 + net(results[name], COST)).cumprod()
            ax.plot(equity.index, equity, color=color, lw=width, label=label)
            ax.annotate(f"${equity.iloc[-1]:,.0f}", (equity.index[-1], equity.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", color=ink,
                        fontsize=9)
        for start, end in (WINDOWS["Drawdown 1"], WINDOWS["Drawdown 2"]):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=grid, lw=0, zorder=0)
        ax.axhline(10_000, color=axis, lw=0.8)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_ylabel("Value of $10,000 (log)")
        ax.legend(frameon=False, loc="upper left", labelcolor=ink2)
    axes[0].set_title(f"Without volatility scaling", loc="left", color=ink, fontsize=10)
    axes[1].set_title(f"With volatility scaling", loc="left", color=ink, fontsize=10)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Stage 2b: the funding guard, after costs ({COST} bp per dollar traded, "
                 f"funding included)", x=0.01, ha="left", color=ink, fontsize=13)
    fig.text(0.01, 0.945, "Light shading = judged period (2021-03-29 to 2023-12-31). "
             "Darker bands = Drawdown 1 and Drawdown 2. Later years are information only.",
             color=ink2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
