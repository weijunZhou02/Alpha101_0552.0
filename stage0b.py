#!/usr/bin/env python3
"""Stage 0b: post-mortem on the before-cost line (no fees, no funding). Describes only.

The owner chose on 2026-10-27 to judge before costs from now on. Periods come
from the before-cost drawdowns of the original backtest. Same measurements as
stage 0, all before costs, plus one new check: in the bad windows, did the whole
ranking flip, or only the two ends? Nothing is tested or chosen.
Tables go to results/stage0b/.

    .venv/bin/python stage0b.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import stage0 as s0

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage0b"

PERIODS = {                                  # trade days, inclusive (declared in README)
    "Train": ("2021-03-29", "2023-12-31"),
    "Box 1": ("2024-01-02", "2024-10-17"),
    "Box 2": ("2025-10-01", "2026-01-23"),
    "Dip 3": ("2026-04-24", "2026-06-07"),
}
OTHER = "Other"                              # 2024 onwards, outside the three windows
NAMES = [*PERIODS, OTHER]
QUINTILES = 5


def period_masks(index):
    masks = {name: (index >= a) & (index <= b) for name, (a, b) in PERIODS.items()}
    masks[OTHER] = ((index >= "2024-01-01") & ~masks["Box 1"] & ~masks["Box 2"]
                    & ~masks["Dip 3"])
    return masks


def labels(day):
    return " + ".join(n for n, m in period_masks(pd.DatetimeIndex([day])).items() if m[0]) or "-"


def quintiles(scores, ret, index):
    """Each trade day: next-day return of each group of 16 (Q1 = best scores)
    minus the average of the scored coins."""
    fut = ret.shift(-1).reindex(index=scores.index, columns=scores.columns)
    valid = scores.notna() & fut.notna()
    s, f = scores.where(valid), fut.where(valid)
    bucket = np.ceil(s.rank(axis=1, ascending=False, pct=True) * QUINTILES)
    mkt = f.mean(axis=1)
    out = pd.DataFrame({f"Q{q}": f.where(bucket == q).mean(axis=1) - mkt
                        for q in range(1, QUINTILES + 1)})
    out.index = out.index + pd.Timedelta(days=1)                # signal day -> trade day
    return out.reindex(index)


def money(daily, ic, masks):
    rows = {}
    for name, m in masks.items():
        g, i = daily[m], ic[m].dropna()
        rows[name] = {
            "days": len(g),
            "before costs, % (compounded)": s0.compound(g["gross_ret"]) * 100,
            "Sharpe before costs": s0.sharpe(g["gross_ret"]),
            "before costs bp/day": g["gross_ret"].mean() * 1e4,
            "  long side vs market": (0.5 * (g["long_ret"] - g["ew_universe_ret"])).mean() * 1e4,
            "  short side vs market": (0.5 * (g["short_ret"] + g["ew_universe_ret"])).mean() * 1e4,
            "rank IC": i.mean(),
            "rank IC t": i.mean() / i.std() * np.sqrt(len(i)),
            "days with IC > 0, %": (i > 0).mean() * 100,
            "turnover, % of equity a day": g["turnover"].mean() * 100,
        }
    return pd.DataFrame(rows)


def quintile_table(q, masks):
    table = pd.DataFrame({name: q[m].mean() * 1e4 for name, m in masks.items()})
    table.loc["Q1 - Q5"] = table.loc["Q1"] - table.loc[f"Q{QUINTILES}"]
    table.index = [{"Q1": "Q1 best scores (≈ BUY list)",
                    f"Q{QUINTILES}": f"Q{QUINTILES} worst scores (≈ SHORT list)"}.get(i, i)
                   for i in table.index]
    return table


def top_losers(pos, masks_pos, masks, daily):
    rows = []
    for name, m in masks_pos.items():
        per_coin = pos[m].groupby("symbol").agg(
            price=("price", "sum"), days=("price", "size"),
            days_short=("side", lambda s: int((s == "SHORT").sum())))
        worst = per_coin.nsmallest(s0.TOP_LOSERS, "price")
        period_sum = daily.loc[masks[name], "gross_ret"].sum()
        for symbol, r in worst.iterrows():
            rows.append({
                "period": name, "symbol": symbol, "days held": r["days"],
                "of which short": r["days_short"], "price vs market, % of equity": r["price"] * 100,
                "top-10 together, %": worst["price"].sum() * 100,
                "period before costs, simple sum %": period_sum * 100,
            })
    return pd.DataFrame(rows)


def worst_days(daily, pos):
    rows = []
    for day, d in daily.nsmallest(s0.WORST_DAYS, "gross_ret").iterrows():
        g = pos[pos["trade_date"] == day]
        coin = g.loc[g["price"].idxmin()]
        rows.append({
            "date": f"{day:%Y-%m-%d}", "period": labels(day),
            "before costs %": d["gross_ret"] * 100,
            "long side vs market %": 0.5 * (d["long_ret"] - d["ew_universe_ret"]) * 100,
            "short side vs market %": 0.5 * (d["short_ret"] + d["ew_universe_ret"]) * 100,
            "market %": d["ew_universe_ret"] * 100, "BTC %": d["btc_ret"] * 100,
            "worst coin": f"{coin['symbol']} ({coin['side']}, {coin['ret'] * 100:+.0f}%)",
            "its cost %": coin["price"] * 100,
        })
    return pd.DataFrame(rows)


def unusual(daily, ic, masks):
    train = masks["Train"]
    gross12 = s0.rolling_compound(daily["gross_ret"])
    ic12 = ic.rolling(s0.WINDOW, min_periods=s0.WINDOW // 2).mean()
    inside = train & (daily.index >= daily.index[train][0] + pd.Timedelta(days=s0.WINDOW - 1))
    windows = pd.DataFrame({
        "worst 12 months inside Train": {
            "ends": f"{gross12[inside].idxmin():%Y-%m-%d}",
            "before costs %": gross12[inside].min() * 100,
            "rank IC (lowest 12-month average)": ic12[inside].min()},
        "worst 12 months, whole history": {
            "ends": f"{gross12.idxmin():%Y-%m-%d}", "before costs %": gross12.min() * 100,
            "rank IC (lowest 12-month average)": ic12.min()},
    })
    rng = np.random.default_rng(s0.SEED)
    actual = s0.max_drawdowns(daily["gross_ret"].to_numpy()[None, :])[0]
    boot = {}
    for name, x in (("Train days reshuffled", daily.loc[train, "gross_ret"].to_numpy()),
                    ("all days reshuffled (information)", daily["gross_ret"].to_numpy())):
        dd = s0.bootstrap(x, len(daily), rng)
        boot[name] = {
            "median worst drawdown %": np.median(dd) * 100,
            "1 in 4 paths worse than %": np.percentile(dd, 25) * 100,
            "1 in 10 paths worse than %": np.percentile(dd, 10) * 100,
            "1 in 20 paths worse than %": np.percentile(dd, 5) * 100,
            f"paths with a drawdown as deep as {actual * 100:.1f}%, %": (dd <= actual).mean() * 100,
        }
    return windows, pd.DataFrame(boot), actual


def main():
    print("Re-running the original backtest on the frozen data ...")
    panels, funding = bt.load_data(HERE / "data")
    daily, pos, scores, ret, _ = bt.run(panels, funding, pd.Timestamp("2000-01-01"),
                                        pd.Timestamp("2100-01-01"), s0.COST_RATE)
    original = pd.read_csv(HERE / "baseline" / "daily_main.csv", index_col=0)
    assert len(original) == len(daily) and np.allclose(
        original["gross_ret"].to_numpy(), daily["gross_ret"].to_numpy(), rtol=0, atol=1e-9), \
        "the re-run differs from the original backtest"
    print(f"Same as the original: {len(daily)} trade days.")

    fut = ret.shift(-1).reindex(index=scores.index, columns=scores.columns)
    ic = bt.rank_ic(scores, fut)
    ic.index = ic.index + pd.Timedelta(days=1)                  # signal day -> trade day
    ic = ic.reindex(daily.index)
    q = quintiles(scores, ret, daily.index)

    pos = s0.coin_traits(panels, scores, pos)
    pos["ew"] = pos["trade_date"].map(daily["ew_universe_ret"])
    pos["price"] = pos["weight"] * (pos["ret"] - pos["ew"])     # against the market average
    pos["funding"] = 0.0                                        # costs ignored in stage 0b
    pos["total"] = pos["price"]

    masks = period_masks(daily.index)
    masks_pos = period_masks(pd.DatetimeIndex(pos["trade_date"]))
    n_days = {name: int(m.sum()) for name, m in masks.items()}

    t_money = money(daily, ic, masks)
    t_quint = quintile_table(q, masks)
    t_groups = s0.by_group(pos, masks_pos, n_days).drop(columns=["funding bp/day", "total bp/day"])
    t_losers = top_losers(pos, masks_pos, masks, daily)
    t_market = s0.market(daily, ret, funding, scores, masks)
    t_worst = worst_days(daily, pos)
    t_windows, t_boot, actual = unusual(daily, ic, masks)

    s0.show("1. BEFORE COSTS (bp/day unless stated; sides measured against the market average)",
            t_money)
    s0.show("2. DID THE WHOLE RANKING FLIP? next-day return against the market, bp/day, "
            "5 groups of 16 coins", t_quint)
    for trait in ("volatility", "liquidity", "age"):
        g = t_groups[t_groups["trait"] == trait]
        wide = g.pivot_table(index="group", columns=["period", "side"],
                             values="price vs market bp/day")
        s0.show(f"3. BY {trait.upper()}: price vs market, bp/day of the book", wide[NAMES])
    s0.show("3b. BIGGEST LOSING COINS (% of equity over the period, simple sums)",
            t_losers.set_index(["period", "symbol"]))
    s0.show("4. MARKET CONDITIONS", t_market)
    s0.show(f"5. THE {s0.WORST_DAYS} WORST DAYS BEFORE COSTS (%)", t_worst.set_index("date"))
    s0.show("6a. 12-MONTH STRETCHES, before costs", t_windows, fmt="{:.3f}")
    s0.show(f"6b. HOW DEEP BEFORE-COST DRAWDOWNS GET: {s0.PATHS} reshuffled paths of {len(daily)} "
            f"days, {s0.BLOCK_DAYS}-day blocks (actual worst: {actual * 100:.1f}%)", t_boot)

    OUT.mkdir(parents=True, exist_ok=True)
    t_money.to_csv(OUT / "money.csv", float_format="%.6g")
    t_quint.to_csv(OUT / "quintiles.csv", float_format="%.6g")
    t_groups.to_csv(OUT / "groups.csv", index=False, float_format="%.6g")
    t_losers.to_csv(OUT / "top_losers.csv", index=False, float_format="%.6g")
    t_market.to_csv(OUT / "market.csv", float_format="%.6g")
    t_worst.to_csv(OUT / "worst_days.csv", index=False, float_format="%.6g")
    t_windows.to_csv(OUT / "windows.csv")
    t_boot.to_csv(OUT / "bootstrap.csv", float_format="%.6g")
    print(f"\nSaved tables to {OUT.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
