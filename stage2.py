#!/usr/bin/env python3
"""Stage 2: trend guard, momentum book, holding longer and volatility scaling.

Each version is tested alone against today's version (a), before costs, on
trade days 2021-03-29 to 2023-12-31, with the same accounting as stage 1.
After-cost results (3.5 and 7.5 bp per dollar traded, funding included) and
later years are shown as information only. Declared in README ("Stage 2").

    .venv/bin/python stage2.py          # writes results/stage2/
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import a101_055 as live
import backtest as bt
import experiment as ex

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage2"

TREND_DAYS, TREND_SHARE = 14, 0.20     # (t1), (t2)
MOM_DAYS, MOM_SHARE = 14, 0.25         # (s): 25% of the money in the momentum book
OVERLAP = 3                            # (c): today's and the two previous days' lists
VOL_DAYS, VOL_HISTORY = 30, 90         # (v)
SCALE_MIN, SCALE_MAX = 0.5, 1.5
INFO_COSTS = (3.5, 7.5)                # bp per dollar traded, information only
ALL_LATER = ("2024-01-01", "2026-10-24")


@dataclass(frozen=True)
class Version:
    name: str
    title: str
    trend: str = ""          # "", "short" or "both"
    overlap: bool = False
    sleeve: bool = False
    volscale: bool = False


VERSIONS = [
    Version("a", "today's version"),
    Version("t1", "trend guard on shorts", trend="short"),
    Version("t2", "trend guard, both sides", trend="both"),
    Version("s", "75% A101_055 + 25% momentum book", sleeve=True),
    Version("c", "3 overlapping baskets", overlap=True),
    Version("v", "scaled by its own volatility", volscale=True),
]
RULE = {"t1": "A", "t2": "A", "c": "A", "s": "B", "v": "B"}


# ------------------------------------------------------------------ books

def lists(v, day, table, buy, short, trend):
    """BUY and SHORT lists for one signal day, with the trend guard if asked."""
    if not v.trend:
        return buy, short
    t = table[["a101_055"]]
    pct = trend.loc[day, t.index].rank(pct=True)
    no_short = pct > 1 - TREND_SHARE                   # strongest 14-day runs
    no_buy = (pct <= TREND_SHARE) if v.trend == "both" else pd.Series(False, index=t.index)
    _, buy, _ = live.select_lists(t[~no_buy])
    _, _, short = live.select_lists(t[~no_short])
    if set(buy) & set(short):
        raise live.ComputeError("the trend guard made the lists overlap")
    return buy, short


def momentum_book(day, table, trend):
    m = trend.loc[day, table.index].dropna()
    if len(m) < 2 * live.LIST_SIZE:
        return pd.Series(dtype=float)
    top, bottom = m.nlargest(live.LIST_SIZE).index, m.nsmallest(live.LIST_SIZE).index
    return pd.concat([pd.Series(bt.SIDE_WEIGHT / len(top), index=top),
                      pd.Series(-bt.SIDE_WEIGHT / len(bottom), index=bottom)])


def add(*parts):
    """Sum signed weight Series (a coin in several parts nets out)."""
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.Series(dtype=float)
    out = parts[0]
    for p in parts[1:]:
        out = out.add(p, fill_value=0.0)
    return out


def books(v, days, ctx, trend):
    """Target weights per signal day (None on days without lists)."""
    a_spec = ex.VERSIONS[0]                                   # equal weights, as today
    daily = {}
    for day, table, buy0, short0, problem in days:
        if problem:
            daily[day] = None
            continue
        try:
            buy, short = lists(v, day, table, buy0, short0, trend)
        except live.ComputeError:
            daily[day] = None
            continue
        daily[day] = ex.weights(a_spec, day, buy, short, ctx)
    order = [d for d, *_ in days]
    out = {}
    for i, (day, table, *_rest) in enumerate(days):
        w = daily[day]
        if w is None:
            out[day] = None
            continue
        if v.overlap:
            recent = [daily[order[j]] for j in range(max(0, i - OVERLAP + 1), i + 1)]
            w = add(*[r for r in recent if r is not None]) / OVERLAP
        if v.sleeve:
            w = add(w * (1 - MOM_SHARE), momentum_book(day, table, trend) * MOM_SHARE)
        out[day] = w
    return out


def vol_scale(gross):
    """usual / recent volatility of the unscaled version, known the day before."""
    recent = gross.rolling(VOL_DAYS, min_periods=VOL_DAYS).std().shift(1)
    usual = recent.expanding().median()
    scale = (usual / recent).clip(SCALE_MIN, SCALE_MAX)
    history = pd.Series(np.arange(len(gross)), index=gross.index)   # earlier days available
    return scale.where(history >= VOL_HISTORY, 1.0).fillna(1.0)


def simulate(target, days, ctx, scale=None):
    """Same accounting as experiment.simulate, for any target weights."""
    rows, w_prev = [], pd.Series(dtype=float)
    for day, table, _buy, _short, _problem in days:
        nxt = day + pd.Timedelta(days=1)
        w = target[day]
        if w is None and not rows:
            continue                                         # still warming up
        if w is None:
            w = pd.Series(dtype=float)
        s = float(scale.get(nxt, 1.0)) if scale is not None else 1.0
        w = w * s
        r = ctx.ret.loc[nxt, w.index].fillna(0.0)
        fr = ctx.funding.loc[nxt, w.index].fillna(0.0)
        union = w.index.union(w_prev.index)
        turnover = (w.reindex(union, fill_value=0.0)
                    - w_prev.reindex(union, fill_value=0.0)).abs().sum()
        gross = float((w * r).sum())
        funding = float(-(w * fr).sum())                     # longs pay positive funding
        net = gross - turnover * ex.DRIFT_COST_RATE + funding
        rows.append({"trade_date": nxt, "gross": gross, "turnover": turnover,
                     "funding": funding, "scale": s, "stops": 0, "braked": False})
        w_prev = w * (1 + r) / (1 + net)
    return pd.DataFrame(rows).set_index("trade_date")


def run(v, days, ctx, trend):
    target = books(v, days, ctx, trend)
    if not v.volscale:
        return simulate(target, days, ctx)
    shadow = simulate(target, days, ctx)
    return simulate(target, days, ctx, scale=vol_scale(shadow["gross"]))


# ------------------------------------------------------------------ judging and reports

def after_costs(results, base):
    rows = []
    for name, d in results.items():
        for cost in INFO_COSTS:
            net = d["gross"] - d["turnover"] * cost / 1e4 + d["funding"]
            for period, (start, end) in (("Train", ex.JUDGE), ("2024-26", ALL_LATER)):
                g = net.loc[start:end]
                equity = (1 + g).cumprod()
                rows.append({"version": name, "cost bp": cost, "period": period,
                             "Sharpe": g.mean() / g.std() * np.sqrt(bt.DAYS_PER_YEAR),
                             "a year %": (equity.iloc[-1] ** (bt.DAYS_PER_YEAR / len(g)) - 1) * 100,
                             "worst drawdown %": (equity / equity.cummax() - 1).min() * 100})
    return pd.DataFrame(rows)


def final_version(table, results, days, ctx, trend, base):
    passers = [n for n in RULE if table.at[n, "verdict"] == "PASS"]
    if not passers:
        return "Nothing passed: today's version stays.", None
    if "t1" in passers and "t2" in passers:
        drop = "t1" if table.at["t2", "Sharpe"] >= table.at["t1", "Sharpe"] else "t2"
        passers.remove(drop)
    if len(passers) == 1:
        return f"One version passed: ({passers[0]}) is the final version.", None
    by = {v.name: v for v in VERSIONS}
    combo = Version("final", "all passing versions together: " + ", ".join(passers),
                    trend=next((by[p].trend for p in passers if by[p].trend), ""),
                    overlap="c" in passers, sleeve="s" in passers, volscale="v" in passers)
    results["final"] = run(combo, days, ctx, trend)
    rules = sorted({RULE[p] for p in passers})
    verdicts = {r: ex.judge("final", results["final"], base, rule=r) for r in rules}
    row = dict(verdicts[rules[0]])
    ok = all(x["verdict"] == "PASS" for x in verdicts.values())
    row["rule"] = "+".join(rules)
    row["verdict"] = "PASS" if ok else "; ".join(
        f"rule {r} {x['verdict']}" for r, x in verdicts.items() if x["verdict"] != "PASS")
    if ok:
        return f"The combination of {', '.join(passers)} passes: it is the final version.", (combo, row)
    best = max(passers, key=lambda p: table.at[p, "Sharpe"])
    return (f"The combination fails, so the final version is ({best}), "
            f"the passer with the highest Sharpe."), (combo, row)


def chart(results, table, path):
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
    base = results["a"]["gross"]
    judge_end = pd.Timestamp(ex.JUDGE[1])
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, v in zip(axes.flat, VERSIONS[1:]):
        gain = (results[v.name]["gross"] - base).cumsum() * 100
        ax.axvspan(base.index[0], judge_end, color=wash, lw=0, zorder=0)
        ax.axhline(0, color=axis, lw=0.8)
        ax.plot(gain.index, gain, color=blue, lw=1.6)
        ax.annotate(f"{gain.iloc[-1]:+.0f}%", (gain.index[-1], gain.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=ink, fontsize=8)
        verdict = str(table.at[v.name, "verdict"])
        ax.set_title(f"({v.name}) {v.title}: {'PASS' if verdict == 'PASS' else 'fail'}",
                     loc="left", color=ink, fontsize=10)
        ax.set_ylabel("gain over (a), % of equity")
    ax = axes.flat[-1]
    equity = (1 + base).cumprod()
    ax.axvspan(base.index[0], judge_end, color=wash, lw=0, zorder=0)
    for name in ("Box 1", "Box 2", "Dip 3"):
        start, end = ex.INFO[name]
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=grid, lw=0, zorder=0)
    ax.plot(equity.index, equity, color=ink2, lw=1.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_title("(a) today's version: value of $1, before costs", loc="left", color=ink,
                 fontsize=10)
    ax.text(0.02, 0.92, "darker bands: Box 1, Box 2, Dip 3", transform=ax.transAxes,
            color=ink2, fontsize=8)
    for a in axes[-1]:
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Stage 2: each change against today's version, before costs",
                 x=0.01, ha="left", color=ink, fontsize=13)
    fig.text(0.01, 0.945, "Shaded = judged period (2021-03-29 to 2023-12-31). "
             "After 2023 is information only. A line going up = better than today's version.",
             color=ink2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"Saved {path.relative_to(HERE)}"


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    trend = panels["close"] / panels["close"].shift(TREND_DAYS) - 1

    results = {}
    for v in VERSIONS:
        results[v.name] = run(v, days, ctx, trend)
        if v.name == "a":
            original = pd.read_csv(HERE / "baseline" / "daily_main.csv", index_col=0,
                                   parse_dates=True)
            a = results["a"]
            gap = (a["gross"] - original["gross_ret"]).abs().max()
            assert a.index.equals(original.index) and np.allclose(
                a["gross"], original["gross_ret"], rtol=1e-7, atol=1e-10) and np.allclose(
                a["turnover"], original["turnover"], rtol=1e-7, atol=1e-10), \
                f"version (a) differs from the original backtest (largest gap {gap:.2g})"
            print(f"Version (a) reproduces the original backtest: {len(a)} days, "
                  f"largest gap {gap:.1g}.")
        print(f"  ({v.name}) {v.title}: done")

    base = results["a"]
    table = pd.DataFrame({v.name: ex.judge(v.name, results[v.name], base,
                                           rule=RULE.get(v.name)) for v in VERSIONS}).T
    table.insert(0, "change", [v.title for v in VERSIONS])
    note, combo = final_version(table, results, days, ctx, trend, base)
    if combo is not None:
        spec, row = combo
        table.loc["final"] = pd.Series({"change": spec.title, **row})

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    show = ["change", "rule", "Sharpe", "bp/day", "worst drawdown %", "worst 1% of days %",
            "vs (a) bp/day", "t vs (a)", "years better", "turnover %", "verdict"]
    print(f"\nJUDGED: trade days {ex.JUDGE[0]} to {ex.JUDGE[1]}, before costs")
    print(table[show].to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\n{note}")

    t_info = ex.info(results, base)
    print("\nINFORMATION ONLY: before costs, already seen")
    for metric in ("Sharpe", "worst drawdown %", "vs (a) bp/day"):
        wide = t_info.pivot(index="version", columns="period", values=metric)[list(ex.INFO)]
        print(f"\n{metric}")
        print(wide.reindex(list(results)).to_string(float_format=lambda x: f"{x:.2f}"))
    t_cost = after_costs(results, base)
    print("\nINFORMATION ONLY: after costs (funding included)")
    wide = t_cost.pivot_table(index="version", columns=["cost bp", "period"],
                              values=["Sharpe", "a year %"])
    print(wide.reindex(list(results)).to_string(float_format=lambda x: f"{x:.2f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    table[show].to_csv(OUT / "judged.csv", float_format="%.6g")
    t_info.to_csv(OUT / "information.csv", index=False, float_format="%.6g")
    t_cost.to_csv(OUT / "after_costs.csv", index=False, float_format="%.6g")
    pd.DataFrame({n: d["gross"] for n, d in results.items()}).to_csv(
        OUT / "daily_gross.csv", float_format="%.8g", date_format="%Y-%m-%d")
    (OUT / "final.txt").write_text(note + "\n")
    print(chart(results, table, OUT / "versions.png"))
    print(f"Saved results to {OUT.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
