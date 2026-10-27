#!/usr/bin/env python3
"""Stage 0: facts and post-mortem of the original A101_055 backtest. Describes only.

Re-runs the original backtest on the frozen data (and checks it still equals the
original), then measures, for each period (train, the two drawdowns, the
recoveries):
  1. where the money went (long side, short side, fees, funding, rank IC),
  2. which coins (volatility, liquidity and age groups; the biggest losers),
  3. market conditions,
  4. the 20 worst days,
  5. how unusual the drawdowns were (worst 12 months, block bootstrap).
Nothing is tested or chosen. Tables go to results/stage0/.

    .venv/bin/python stage0.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage0"
COST_RATE = (bt.DEFAULT_FEE_BPS + bt.DEFAULT_SLIPPAGE_BPS) / 1e4

PERIODS = {                                  # trade days, inclusive (declared in README)
    "Train": ("2021-03-29", "2023-12-31"),
    "Bad 1": ("2023-11-18", "2024-10-22"),
    "Bad 2": ("2025-07-22", "2026-06-07"),
}
OTHER = "Other"                              # 2024 onwards, outside Bad 1 and Bad 2
NAMES = [*PERIODS, OTHER]

HEAVY_FUNDING = 0.001                        # 0.1% a day
WINDOW = 365                                 # "12 months" of trade days
BLOCK_DAYS, PATHS, SEED = 30, 2000, 0        # block bootstrap
TOP_LOSERS, WORST_DAYS = 10, 20

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def period_masks(index):
    masks = {name: (index >= a) & (index <= b) for name, (a, b) in PERIODS.items()}
    masks[OTHER] = (index >= "2024-01-01") & ~masks["Bad 1"] & ~masks["Bad 2"]
    return masks


def labels(day):
    return " + ".join(n for n, m in period_masks(pd.DatetimeIndex([day])).items() if m[0]) or "-"


def lookup(frame, dates, symbols):
    """frame.loc[date, symbol] for many (date, symbol) pairs; NaN when absent."""
    stacked = frame.stack()
    return stacked.reindex(pd.MultiIndex.from_arrays([dates, symbols])).to_numpy()


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(bt.DAYS_PER_YEAR) if x.std() > 0 else np.nan


def compound(x):
    return float(np.prod(1 + x) - 1)


def show(title, table, fmt="{:.2f}"):
    print(f"\n{title}")
    print(table.to_string(float_format=lambda v: fmt.format(v)))


# ------------------------------------------------------------------ 1. where the money went

def money(daily, ic, masks):
    rows = {}
    for name, m in masks.items():
        g, i = daily[m], ic[m].dropna()
        rows[name] = {
            "days": len(g),
            "net, % (compounded)": compound(g["net_ret"]) * 100,
            "before costs, % (compounded)": compound(g["gross_ret"]) * 100,
            "Sharpe after costs": sharpe(g["net_ret"]),
            "net bp/day": g["net_ret"].mean() * 1e4,
            "  long side vs market": (0.5 * (g["long_ret"] - g["ew_universe_ret"])).mean() * 1e4,
            "  short side vs market": (0.5 * (g["short_ret"] + g["ew_universe_ret"])).mean() * 1e4,
            "  fees + slippage": -g["trading_cost"].mean() * 1e4,
            "  funding, long side": g["long_funding"].mean() * 1e4,
            "  funding, short side": g["short_funding"].mean() * 1e4,
            "rank IC": i.mean(),
            "rank IC t": i.mean() / i.std() * np.sqrt(len(i)),
            "days with IC > 0, %": (i > 0).mean() * 100,
            "turnover, % of equity a day": g["turnover"].mean() * 100,
        }
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. which coins

def coin_traits(panels, scores, pos):
    """Volatility, liquidity and age group of each held coin on its signal day."""
    close, qvol = panels["close"], panels["qvol"]
    vol30 = close.pct_change(fill_method=None).rolling(30, min_periods=20).std()
    adv30 = qvol.rolling(bt.live.ADV_WINDOW, min_periods=bt.live.ADV_MIN_DAYS).mean()
    age = close.notna().cumsum()
    universe = scores.notna()
    grid = dict(index=scores.index, columns=scores.columns)
    vol_pct = vol30.reindex(**grid).where(universe).rank(axis=1, pct=True)
    liq_rank = adv30.reindex(**grid).where(universe).rank(axis=1, ascending=False, method="first")
    days_listed = age.reindex(**grid)

    v = lookup(vol_pct, pos["signal_date"], pos["symbol"])
    q = lookup(liq_rank, pos["signal_date"], pos["symbol"])
    a = lookup(days_listed, pos["signal_date"], pos["symbol"])
    pos = pos.copy()
    pos["volatility"] = np.select([v <= 1 / 3, v <= 2 / 3, v <= 1],
                                  ["1 calm third", "2 middle third", "3 wild third"], "unknown")
    pos["liquidity"] = np.select([q <= 20, q <= 50, q <= 1000],
                                 ["1 volume rank 1-20", "2 rank 21-50", "3 rank 51-80"], "unknown")
    pos["age"] = np.select([a < 180, a < 365, a >= 365],
                           ["1 listed < 6 months", "2 6-12 months", "3 over 1 year"], "unknown")
    return pos


def by_group(pos, masks_pos, n_days):
    rows = []
    for trait in ("volatility", "liquidity", "age"):
        for name, m in masks_pos.items():
            for side in ("BUY", "SHORT"):
                g = pos[m & (pos["side"] == side)]
                total_days = len(g)
                for group, gg in g.groupby(trait):
                    rows.append({
                        "trait": trait, "group": group, "period": name, "side": side,
                        "price vs market bp/day": gg["price"].sum() / n_days[name] * 1e4,
                        "funding bp/day": gg["funding"].sum() / n_days[name] * 1e4,
                        "total bp/day": gg["total"].sum() / n_days[name] * 1e4,
                        "share of side's coin-days %": len(gg) / total_days * 100,
                    })
    return pd.DataFrame(rows)


def top_losers(pos, masks_pos, daily_masks, daily):
    rows = []
    for name, m in masks_pos.items():
        g = pos[m]
        per_coin = g.groupby("symbol").agg(
            total=("total", "sum"), price=("price", "sum"), funding=("funding", "sum"),
            days=("total", "size"),
            days_short=("side", lambda s: int((s == "SHORT").sum())))
        worst = per_coin.nsmallest(TOP_LOSERS, "total")
        net_sum = daily.loc[daily_masks[name], "net_ret"].sum()
        for symbol, r in worst.iterrows():
            rows.append({
                "period": name, "symbol": symbol, "days held": r["days"],
                "of which short": r["days_short"],
                "total, % of equity": r["total"] * 100, "price part": r["price"] * 100,
                "funding part": r["funding"] * 100,
                "top-10 together, %": worst["total"].sum() * 100,
                "period net, simple sum %": net_sum * 100,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. market conditions

def market(daily, ret, funding, scores, masks):
    universe = scores.notna()
    universe.index = universe.index + pd.Timedelta(days=1)     # signal day -> trade day
    universe = universe.reindex(daily.index, fill_value=False)
    grid = dict(index=universe.index, columns=universe.columns)
    r = ret.reindex(**grid).where(universe)
    f = funding.reindex(**grid).where(universe)
    mkt = daily["ew_universe_ret"]
    rows = {}
    for name, m in masks.items():
        rows[name] = {
            "market (top-80 average) bp/day": mkt[m].mean() * 1e4,
            "market up days %": (mkt[m] > 0).mean() * 100,
            "market volatility, % a year": mkt[m].std() * np.sqrt(bt.DAYS_PER_YEAR) * 100,
            "BTC minus market bp/day": (daily["btc_ret"] - mkt)[m].mean() * 1e4,
            "spread of coin returns (daily std) %": r[m].std(axis=1).mean() * 100,
            "average funding %/day": f[m].mean(axis=1).mean() * 100,
            "coins with funding < -0.1%/day, %": (f[m] < -HEAVY_FUNDING).sum(axis=1).div(
                f[m].notna().sum(axis=1)).mean() * 100,
            "coins with funding > +0.1%/day, %": (f[m] > HEAVY_FUNDING).sum(axis=1).div(
                f[m].notna().sum(axis=1)).mean() * 100,
        }
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. worst days

def worst_days(daily, pos):
    rows = []
    for day, d in daily.nsmallest(WORST_DAYS, "net_ret").iterrows():
        g = pos[pos["trade_date"] == day]
        coin = g.loc[g["total"].idxmin()]
        rows.append({
            "date": f"{day:%Y-%m-%d}", "period": labels(day),
            "net %": d["net_ret"] * 100,
            "long side vs market %": 0.5 * (d["long_ret"] - d["ew_universe_ret"]) * 100,
            "short side vs market %": 0.5 * (d["short_ret"] + d["ew_universe_ret"]) * 100,
            "market %": d["ew_universe_ret"] * 100, "BTC %": d["btc_ret"] * 100,
            "worst coin": f"{coin['symbol']} ({coin['side']}, {coin['ret'] * 100:+.0f}%)",
            "its cost %": coin["total"] * 100,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. how unusual

def rolling_compound(x):
    return np.exp(np.log1p(x).rolling(WINDOW).sum()) - 1


def max_drawdowns(paths):
    equity = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0), axis=1)
    return (equity / peak - 1).min(axis=1)


def bootstrap(x, length, rng):
    n_blocks = -(-length // BLOCK_DAYS)
    starts = rng.integers(0, len(x) - BLOCK_DAYS + 1, size=(PATHS, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK_DAYS)).reshape(PATHS, -1)[:, :length]
    return max_drawdowns(x[idx])


def unusual(daily, ic, masks):
    train = masks["Train"]
    net12, gross12 = rolling_compound(daily["net_ret"]), rolling_compound(daily["gross_ret"])
    ic12 = ic.rolling(WINDOW, min_periods=WINDOW // 2).mean()
    inside = train & (daily.index >= daily.index[train][0] + pd.Timedelta(days=WINDOW - 1))
    y2024 = daily.index.year == 2024
    windows = pd.DataFrame({
        "worst 12 months inside Train": {
            "ends": f"{net12[inside].idxmin():%Y-%m-%d}", "net %": net12[inside].min() * 100,
            "before costs %": gross12[net12[inside].idxmin()] * 100,
            "rank IC (lowest 12-month average)": ic12[inside].min()},
        "calendar 2024": {
            "ends": "2024-12-31", "net %": compound(daily.loc[y2024, "net_ret"]) * 100,
            "before costs %": compound(daily.loc[y2024, "gross_ret"]) * 100,
            "rank IC (lowest 12-month average)": ic[y2024].mean()},
        "worst 12 months, whole history": {
            "ends": f"{net12.idxmin():%Y-%m-%d}", "net %": net12.min() * 100,
            "before costs %": gross12[net12.idxmin()] * 100,
            "rank IC (lowest 12-month average)": ic12.min()},
    })

    rng = np.random.default_rng(SEED)
    length = len(daily)
    actual = max_drawdowns(daily["net_ret"].to_numpy()[None, :])[0]
    boot = {}
    for name, x in (("Train days reshuffled", daily.loc[train, "net_ret"].to_numpy()),
                    ("all days reshuffled (information)", daily["net_ret"].to_numpy())):
        dd = bootstrap(x, length, rng)
        boot[name] = {
            "median worst drawdown %": np.median(dd) * 100,
            "1 in 4 paths worse than %": np.percentile(dd, 25) * 100,
            "1 in 10 paths worse than %": np.percentile(dd, 10) * 100,
            "1 in 20 paths worse than %": np.percentile(dd, 5) * 100,
            f"paths with a drawdown as deep as {actual * 100:.1f}%, %": (dd <= actual).mean() * 100,
        }
    return windows, pd.DataFrame(boot), actual


# ------------------------------------------------------------------ main

def main():
    print("Re-running the original backtest on the frozen data ...")
    panels, funding = bt.load_data(HERE / "data")
    daily, pos, scores, ret, _ = bt.run(panels, funding, pd.Timestamp("2000-01-01"),
                                        pd.Timestamp("2100-01-01"), COST_RATE)
    original = pd.read_csv(HERE / "baseline" / "daily_main.csv", index_col=0)
    assert len(original) == len(daily) and np.allclose(
        original["net_ret"].to_numpy(), daily["net_ret"].to_numpy(), rtol=0, atol=1e-9), \
        "the re-run differs from the original backtest"
    print(f"Same as the original: {len(daily)} trade days.")

    fut = ret.shift(-1).reindex(index=scores.index, columns=scores.columns)
    ic = bt.rank_ic(scores, fut)
    ic.index = ic.index + pd.Timedelta(days=1)                  # signal day -> trade day
    ic = ic.reindex(daily.index)

    pos = coin_traits(panels, scores, pos)
    pos["ew"] = pos["trade_date"].map(daily["ew_universe_ret"])
    pos["price"] = pos["weight"] * (pos["ret"] - pos["ew"])     # against the market average
    pos["funding"] = -pos["weight"] * pos["funding_rate_sum"]   # longs pay positive funding
    pos["total"] = pos["price"] + pos["funding"]

    masks = period_masks(daily.index)
    masks_pos = period_masks(pd.DatetimeIndex(pos["trade_date"]))
    n_days = {name: int(m.sum()) for name, m in masks.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    t_money = money(daily, ic, masks)
    t_groups = by_group(pos, masks_pos, n_days)
    t_losers = top_losers(pos, masks_pos, masks, daily)
    t_market = market(daily, ret, funding, scores, masks)
    t_worst = worst_days(daily, pos)
    t_windows, t_boot, actual = unusual(daily, ic, masks)

    show("1. WHERE THE MONEY WENT (bp/day unless stated; sides measured against the market average)",
         t_money)
    for trait in ("volatility", "liquidity", "age"):
        g = t_groups[t_groups["trait"] == trait]
        wide = g.pivot_table(index="group", columns=["period", "side"], values="total bp/day")
        show(f"2. BY {trait.upper()}: price vs market + funding, bp/day of the book", wide[NAMES])
        fund = g[g["side"] == "SHORT"].pivot_table(index="group", columns="period",
                                                   values="funding bp/day")
        show(f"   short side, funding part only (bp/day)", fund[NAMES])
    show("2b. BIGGEST LOSING COINS (% of equity over the period, simple sums)",
         t_losers.set_index(["period", "symbol"]))
    show("3. MARKET CONDITIONS", t_market)
    show(f"4. THE {WORST_DAYS} WORST DAYS (%)", t_worst.set_index("date"))
    show("5a. 12-MONTH STRETCHES", t_windows, fmt="{:.3f}")
    show(f"5b. HOW DEEP DRAWDOWNS GET: {PATHS} reshuffled paths of {len(daily)} days, "
         f"{BLOCK_DAYS}-day blocks (actual worst: {actual * 100:.1f}%)", t_boot)

    t_money.to_csv(OUT / "money.csv", float_format="%.6g")
    t_groups.to_csv(OUT / "groups.csv", index=False, float_format="%.6g")
    t_losers.to_csv(OUT / "top_losers.csv", index=False, float_format="%.6g")
    t_market.to_csv(OUT / "market.csv", float_format="%.6g")
    t_worst.to_csv(OUT / "worst_days.csv", index=False, float_format="%.6g")
    t_windows.to_csv(OUT / "windows.csv")
    t_boot.to_csv(OUT / "bootstrap.csv", float_format="%.6g")
    print(f"\nSaved tables to {OUT.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
