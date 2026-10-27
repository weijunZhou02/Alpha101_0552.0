#!/usr/bin/env python3
"""Stage 1: test 7 changes to A101_055, each alone against today's version (a).

Judged before costs (no fees, no funding) on trade days 2021-03-29 to
2023-12-31. Later years and the bad windows are shown as information only.
The versions and rules are declared in README ("Stage 1").

    .venv/bin/python experiment.py

Version (a) must reproduce the original backtest exactly (before-cost return
and turnover on every day) before anything is compared. The daily lists of the
original are cached in out/stage1_lists.pkl (rebuilt when the data or the
original scripts change). Writes results/stage1/.
"""

import hashlib
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import a101_055 as live
import backtest as bt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage1"
CACHE = HERE / "out" / "stage1_lists.pkl"

# Costs are not judged. They are kept only so that weights drift exactly as in
# the original backtest, which makes version (a) reproduce it day for day.
DRIFT_COST_RATE = (bt.DEFAULT_FEE_BPS + bt.DEFAULT_SLIPPAGE_BPS) / 1e4

JUDGE = ("2021-03-29", "2023-12-31")
YEARS = (2021, 2022, 2023)
INFO = {
    "2024": ("2024-01-01", "2024-12-31"),
    "2025-26": ("2025-01-01", "2026-10-24"),
    "Box 1": ("2024-01-02", "2024-10-17"),
    "Box 2": ("2025-10-01", "2026-01-23"),
    "Dip 3": ("2026-04-24", "2026-06-07"),
}

LIQUID_TOP = 50            # (f) only the 50 most-traded coins
WILD_SHARE = 0.20          # (g) leave out the wildest 20%
VOL_DAYS, VOL_MIN = 30, 20
STOP_UP, STOP_FILL = 0.30, 0.31   # (n) close a short at +30% within the day; count +31%
BRAKE_DD, BRAKE_SCALE = 0.15, 0.5  # (k) half size while 15%+ below the peak
HELPER_ONE, HELPER_TWO = 0.3, 0.2  # (l), (m); 0.6 / 0.2 / 0.2 if both are combined
WORST_SHARE = 0.01


@dataclass(frozen=True)
class Spec:
    name: str
    title: str
    liquid: bool = False
    calm: bool = False
    invvol: bool = False
    stop: bool = False
    brake: bool = False
    helpers: tuple = ()


VERSIONS = [
    Spec("a", "today's version"),
    Spec("f", "only the 50 most-traded coins", liquid=True),
    Spec("g", "leave out the wildest 20%", calm=True),
    Spec("d", "1/volatility weights", invvol=True),
    Spec("n", "stop-loss on shorts at +30%", stop=True),
    Spec("k", "half size 15%+ below the peak", brake=True),
    Spec("l", "helper: perp premium", helpers=("CRYPTO_HMRW_PREMIUM",)),
    Spec("m", "helper: GTJA_095", helpers=("GTJA_095",)),
]
RULE = {"f": "B", "g": "B", "d": "B", "n": "B", "k": "B", "l": "A", "m": "A"}


# ------------------------------------------------------------------ inputs

def fingerprint():
    h = hashlib.sha256()
    for name in ("data/klines_1d.csv.gz", "data/funding.csv.gz", "a101_055.py", "backtest.py"):
        h.update((HERE / name).read_bytes())
    return h.hexdigest()


def original_lists(panels):
    """(day, table, buy, short, problem) for every signal day, exactly as backtest.run makes them."""
    key = fingerprint()
    if CACHE.exists():
        cached = pickle.loads(CACHE.read_bytes())
        if cached["key"] == key:
            return cached["days"]
    print("Building the original daily lists (about 2 minutes, cached afterwards) ...")
    has_candle = panels["close"].notna()
    index = panels["close"].index
    first = index[0] + pd.Timedelta(days=bt.LIVE_CANDLES - 1)
    days = []
    for day in index[index >= first]:
        if day + pd.Timedelta(days=1) not in index:
            break
        table, buy, short, problem = bt.live_lists(panels, day, bt.trading_at_open(has_candle, day))
        days.append((day, table, buy, short, problem))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps({"key": key, "days": days}))
    return days


@dataclass
class Context:
    ret: pd.DataFrame
    high_move: pd.DataFrame      # day's high / open - 1
    funding: pd.DataFrame
    vol: pd.DataFrame            # 30-day volatility, known at the signal day's close
    adv: pd.DataFrame            # 30-day average quote volume
    helpers: dict                # alpha -> date x symbol, higher = better


def context(panels, funding):
    close = panels["close"]
    raw = pd.read_csv(HERE / "data" / "helpers.csv.gz", parse_dates=["date"])
    helpers = {a: raw.pivot(index="date", columns="symbol", values=a)
               for a in ("CRYPTO_HMRW_PREMIUM", "GTJA_095")}
    return Context(
        ret=close / panels["open"] - 1,
        high_move=panels["high"] / panels["open"] - 1,
        funding=funding,
        vol=close.pct_change(fill_method=None).rolling(VOL_DAYS, min_periods=VOL_MIN).std(),
        adv=panels["qvol"].rolling(live.ADV_WINDOW, min_periods=live.ADV_MIN_DAYS).mean(),
        helpers=helpers,
    )


# ------------------------------------------------------------------ one version

def choose(spec, day, table, buy, short, ctx):
    """BUY and SHORT lists for one signal day (the original lists when nothing changes them)."""
    if not (spec.liquid or spec.calm or spec.helpers):
        return buy, short
    members = table.index
    t = table[["a101_055"]].copy()
    if spec.helpers:
        w_helper = HELPER_ONE if len(spec.helpers) == 1 else HELPER_TWO
        blend = (1 - w_helper * len(spec.helpers)) * t["a101_055"].rank(pct=True)
        for alpha in spec.helpers:
            frame = ctx.helpers[alpha]
            values = (frame.loc[day].reindex(members) if day in frame.index
                      else pd.Series(np.nan, index=members))
            blend = blend + w_helper * values.rank(pct=True).fillna(0.5)
        t["a101_055"] = blend                    # a coin without an A101_055 score stays missing
    keep = pd.Series(True, index=members)
    if spec.liquid:
        keep &= ctx.adv.loc[day, members].rank(ascending=False, method="first") <= LIQUID_TOP
    if spec.calm:
        keep &= ~(ctx.vol.loc[day, members].rank(pct=True) > 1 - WILD_SHARE)
    _, buy, short = live.select_lists(t[keep])
    return buy, short


def weights(spec, day, buy, short, ctx):
    """Signed weights: BUY coins first, then SHORT coins, each side summing to 50%."""
    def side(coins):
        if not spec.invvol:
            return pd.Series(bt.SIDE_WEIGHT / len(coins), index=coins)
        v = ctx.vol.loc[day, coins]
        v = v.fillna(v.median()).fillna(1.0).clip(lower=1e-6)
        return bt.SIDE_WEIGHT * (1 / v) / (1 / v).sum()
    parts = [side(buy) if buy else None, -side(short) if short else None]
    w = pd.concat(parts) if any(p is not None for p in parts) else pd.Series(dtype=float)
    return w if len(w) else pd.Series(dtype=float)


def brake_flags(gross):
    """True on trade days when the version's before-cost equity, up to the day before,
    is 15% or more below its peak."""
    equity = (1 + gross).cumprod()
    drawdown = equity / np.maximum(equity.cummax(), 1.0) - 1
    return (drawdown <= -BRAKE_DD).shift(1, fill_value=False)


def simulate(spec, days, ctx, flags=None):
    """Daily results of one version, with the same accounting as backtest.run."""
    if spec.brake and flags is None:
        flags = brake_flags(simulate(replace(spec, brake=False), days, ctx)["gross"])
    rows, w_prev = [], pd.Series(dtype=float)
    for day, table, buy0, short0, problem in days:
        nxt = day + pd.Timedelta(days=1)
        buy, short = [], []
        if problem is None:
            try:
                buy, short = choose(spec, day, table, buy0, short0, ctx)
            except live.ComputeError:
                problem = "lists"
        if problem and not rows:
            continue                              # still warming up: no trading yet
        if problem:
            buy, short = [], []
        w = weights(spec, day, buy, short, ctx)
        scale = BRAKE_SCALE if spec.brake and flags.get(nxt, False) else 1.0
        w = w * scale

        r = ctx.ret.loc[nxt, w.index].fillna(0.0)
        fr = ctx.funding.loc[nxt, w.index].fillna(0.0)
        stopped = []
        if spec.stop and short:
            move = ctx.high_move.loc[nxt, short]
            stopped = list(move.index[move >= STOP_UP])
            r[stopped] = STOP_FILL
        union = w.index.union(w_prev.index)
        turnover = (w.reindex(union, fill_value=0.0)
                    - w_prev.reindex(union, fill_value=0.0)).abs().sum()
        turnover += sum(abs(w[s]) * (1 + STOP_FILL) for s in stopped)   # buying back stopped shorts
        gross = float((w * r).sum())
        long_funding = float(-(w[buy] * fr[buy]).sum()) if buy else 0.0
        short_funding = float(-(w[short] * fr[short]).sum()) if short else 0.0
        net = gross - turnover * DRIFT_COST_RATE + long_funding + short_funding
        universe = table.index if table is not None else []
        rows.append({
            "trade_date": nxt, "gross": gross, "turnover": turnover,
            "ew": ctx.ret.loc[nxt, universe].mean() if len(universe) else np.nan,
            "stops": len(stopped), "braked": scale < 1, "n_long": len(buy), "n_short": len(short),
        })
        w_prev = w * (1 + r) / (1 + net)
        w_prev[stopped] = 0.0
    return pd.DataFrame(rows).set_index("trade_date")


# ------------------------------------------------------------------ judging

def measures(g, base):
    """Sharpe, mean, worst drawdown, worst 1% of days, and the difference from (a)."""
    equity = (1 + g).cumprod()
    diff = g - base
    k = max(1, round(WORST_SHARE * len(g)))
    return {
        "Sharpe": g.mean() / g.std() * np.sqrt(bt.DAYS_PER_YEAR),
        "bp/day": g.mean() * 1e4,
        "worst drawdown %": (equity / np.maximum(equity.cummax(), 1.0) - 1).min() * 100,
        "worst 1% of days %": g.nsmallest(k).mean() * 100,
        "vs (a) bp/day": diff.mean() * 1e4,
        "t vs (a)": diff.mean() / diff.std() * np.sqrt(len(diff)) if diff.std() > 0 else np.nan,
    }


def judge(name, d, base, rule=None):
    """Measures on the judging period and the verdict under the version's rule."""
    sl = slice(*JUDGE)
    g, b = d.loc[sl, "gross"], base.loc[sl, "gross"]
    m = measures(g, b)
    a = measures(b, b)
    m["turnover %"] = d.loc[sl, "turnover"].mean() * 100
    better_years = sum(
        measures(g[g.index.year == y], b[b.index.year == y])["Sharpe"]
        > measures(b[b.index.year == y], b[b.index.year == y])["Sharpe"] for y in YEARS)
    m["years better"] = better_years
    m["stops"] = int(d.loc[sl, "stops"].sum())
    m["braked days"] = int(d.loc[sl, "braked"].sum())
    rule = rule or RULE.get(name)
    if rule is None:
        m["rule"], m["verdict"] = "-", "yardstick"
        return m
    m["rule"] = rule
    if (g - b).abs().max() == 0:
        m["verdict"] = "untestable: same as (a) on 2021-2023"
        return m
    if rule == "A":
        checks = {"Sharpe above (a)": m["Sharpe"] > a["Sharpe"], "t >= 2": m["t vs (a)"] >= 2,
                  "2 of 3 years better": better_years >= 2}
    else:
        checks = {"smaller worst drawdown": m["worst drawdown %"] > a["worst drawdown %"],
                  "smaller worst 1% days": m["worst 1% of days %"] > a["worst 1% of days %"],
                  "t above -1": m["t vs (a)"] > -1}
    failed = [k for k, ok in checks.items() if not ok]
    m["verdict"] = "PASS" if not failed else "fail: " + ", ".join(failed)
    return m


def info(results, base):
    rows = []
    for period, (start, end) in INFO.items():
        b = base.loc[start:end, "gross"]
        for name, d in results.items():
            g = d.loc[start:end, "gross"]
            m = measures(g, b)
            rows.append({"period": period, "version": name, **{
                k: m[k] for k in ("Sharpe", "bp/day", "worst drawdown %", "vs (a) bp/day")}})
    return pd.DataFrame(rows)


def combined_spec(passers):
    helpers = tuple(h for p in passers for h in next(v for v in VERSIONS if v.name == p).helpers)
    return Spec("final", "all passing versions together: " + ", ".join(passers),
                liquid="f" in passers, calm="g" in passers, invvol="d" in passers,
                stop="n" in passers, brake="k" in passers, helpers=helpers)


# ------------------------------------------------------------------ chart

def chart(results, table, path):
    """Small multiples: each version's running gain over (a), before costs, plus (a) itself."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return "matplotlib is not installed; no chart."
    ink, ink2, muted, grid, axis = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue, wash, surface = "#2a78d6", "#f0efec", "#fcfcfb"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": muted,
        "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface,
    })
    names = [s.name for s in VERSIONS if s.name != "a"]
    titles = {s.name: s.title for s in VERSIONS}
    base = results["a"]["gross"]
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    judge_end = pd.Timestamp(JUDGE[1])
    for ax, name in zip(axes.flat, names):
        gain = (results[name]["gross"] - base).cumsum() * 100
        ax.axvspan(base.index[0], judge_end, color=wash, lw=0, zorder=0)
        ax.axhline(0, color=axis, lw=0.8)
        ax.plot(gain.index, gain, color=blue, lw=1.6, solid_joinstyle="round")
        ax.annotate(f"{gain.iloc[-1]:+.0f}%", (gain.index[-1], gain.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=ink, fontsize=8)
        verdict = str(table.at[name, "verdict"])
        short = "PASS" if verdict == "PASS" else ("untestable" if verdict.startswith("untest")
                                                  else "fail")
        ax.set_title(f"({name}) {titles[name]}: {short}", loc="left", color=ink, fontsize=10)
        ax.set_ylabel("gain over (a), % of equity")
    ax = axes.flat[-1]
    equity = (1 + base).cumprod()
    ax.axvspan(base.index[0], judge_end, color=wash, lw=0, zorder=0)
    for start, end in (INFO["Box 1"], INFO["Box 2"], INFO["Dip 3"]):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=grid, lw=0, zorder=0)
    ax.plot(equity.index, equity, color=ink2, lw=1.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("(a) today's version: value of $1, before costs", loc="left", color=ink,
                 fontsize=10)
    ax.text(0.02, 0.92, "darker bands: Box 1, Box 2, Dip 3", transform=ax.transAxes,
            color=ink2, fontsize=8)
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Stage 1: each change against today's version, before costs",
                 x=0.01, ha="left", color=ink, fontsize=13)
    fig.text(0.01, 0.955, "Shaded = judged period (2021-03-29 to 2023-12-31). "
             "After 2023 is information only. A line going up = better than today's version.",
             color=ink2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"Saved {path.relative_to(HERE)}"


# ------------------------------------------------------------------ main

def main():
    panels, funding = bt.load_data(HERE / "data")
    days = original_lists(panels)
    ctx = context(panels, funding)

    results = {}
    for spec in VERSIONS:
        results[spec.name] = simulate(spec, days, ctx)
        if spec.name == "a":
            original = pd.read_csv(HERE / "baseline" / "daily_main.csv", index_col=0,
                                   parse_dates=True)
            a = results["a"]
            same = (a.index.equals(original.index)
                    and np.allclose(a["gross"], original["gross_ret"], rtol=1e-7, atol=1e-10)
                    and np.allclose(a["turnover"], original["turnover"], rtol=1e-7, atol=1e-10))
            gap = (a["gross"] - original["gross_ret"]).abs().max()
            assert same, f"version (a) differs from the original backtest (largest gap {gap:.2g})"
            print(f"Version (a) reproduces the original backtest: {len(a)} days, "
                  f"largest gap {gap:.1g}.")
        print(f"  ({spec.name}) {spec.title}: done")

    base = results["a"]
    table = pd.DataFrame({s.name: judge(s.name, results[s.name], base) for s in VERSIONS}).T
    table.insert(0, "change", [s.title for s in VERSIONS])
    passers = [n for n in RULE if table.at[n, "verdict"] == "PASS"]

    final_note = "Nothing passed: today's version stays."
    if len(passers) == 1:
        final_note = f"One version passed: ({passers[0]}) is the final version."
    elif passers:
        spec = combined_spec(passers)
        results["final"] = simulate(spec, days, ctx)
        rules = sorted({RULE[p] for p in passers})      # it must pass every rule its parts passed
        verdicts = {r: judge("final", results["final"], base, rule=r) for r in rules}
        row = dict(verdicts[rules[0]])
        ok = all(v["verdict"] == "PASS" for v in verdicts.values())
        row["rule"] = "+".join(rules)
        row["verdict"] = "PASS" if ok else "; ".join(
            f"rule {r} {v['verdict']}" for r, v in verdicts.items() if v["verdict"] != "PASS")
        table.loc["final"] = pd.Series({"change": spec.title, **row})
        if ok:
            final_note = f"The combination of {', '.join(passers)} passes: it is the final version."
        else:
            best = max(passers, key=lambda p: table.at[p, "Sharpe"])
            final_note = (f"The combination fails, so the final version is ({best}), "
                          f"the passer with the highest Sharpe.")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    show = ["change", "rule", "Sharpe", "bp/day", "worst drawdown %", "worst 1% of days %",
            "vs (a) bp/day", "t vs (a)", "years better", "turnover %", "stops", "braked days",
            "verdict"]
    print(f"\nJUDGED: trade days {JUDGE[0]} to {JUDGE[1]}, before costs")
    print(table[show].to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\n{final_note}")
    t_info = info(results, base)
    print("\nINFORMATION ONLY (already seen; never used to choose)")
    for metric in ("Sharpe", "worst drawdown %", "vs (a) bp/day"):
        wide = t_info.pivot(index="version", columns="period", values=metric)[list(INFO)]
        print(f"\n{metric}")
        print(wide.reindex(list(results)).to_string(float_format=lambda v: f"{v:.2f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    table[show].to_csv(OUT / "judged.csv", float_format="%.6g")
    t_info.to_csv(OUT / "information.csv", index=False, float_format="%.6g")
    pd.DataFrame({n: d["gross"] for n, d in results.items()}).to_csv(
        OUT / "daily_gross.csv", float_format="%.8g", date_format="%Y-%m-%d")
    (OUT / "final.txt").write_text(final_note + "\n")
    print(chart(results, table, OUT / "versions.png"))
    print(f"Saved results to {OUT.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
