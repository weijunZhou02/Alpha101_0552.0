#!/usr/bin/env python3
"""Stage 0c: why the bad windows happened. Describes only; nothing is tested or chosen.

Before costs, on the original daily lists (cached by experiment.py). Measures:
  1. what the BUY and SHORT coins look like on the day they are picked,
  2. how A101_055 relates to two simple strategies: 1-day reversal (buy
     yesterday's losers, short its winners) and 14-day momentum (buy the
     winners of the last 14 days, short the losers),
  3. what the shorted (and bought) coins did over the next 1, 3, 5 and 10 days,
  4. "reversal mode" vs "momentum mode" of the whole market: each day, the rank
     correlation across coins between yesterday's and today's return,
  5. whether a bad month for A101_055 tends to be followed by another,
  6. crowded shorts: shorted coins already paying heavy negative funding,
  7. examples: the coins that cost most in each bad window.
Writes results/stage0c/.

    .venv/bin/python stage0c.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import experiment as ex

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage0c"
PERIODS = {
    "Train": ("2021-03-29", "2023-12-31"),
    "Box 1": ("2024-01-02", "2024-10-17"),
    "Box 2": ("2025-10-01", "2026-01-23"),
    "Dip 3": ("2026-04-24", "2026-06-07"),
}
OTHER = "Other"
NAMES = [*PERIODS, OTHER]
K = 16                         # coins per side, as in the strategy
HORIZONS = (1, 3, 5, 10)
MOM_DAYS = 14
TRAIL = 30
HEAVY_FUNDING = -0.001         # 7-day average below -0.1% a day
EXAMPLES = 3


def period_of(day):
    for name, (a, b) in PERIODS.items():
        if name != "Train" and pd.Timestamp(a) <= day <= pd.Timestamp(b):
            return name
    if pd.Timestamp(PERIODS["Train"][0]) <= day <= pd.Timestamp(PERIODS["Train"][1]):
        return "Train"
    return OTHER if day >= pd.Timestamp("2024-01-01") else None


def spearman(a, b):
    ok = a.notna() & b.notna()
    if ok.sum() < 10:
        return np.nan
    return a[ok].rank().corr(b[ok].rank())


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    close = panels["close"]
    c2c = close.pct_change(fill_method=None)                       # close to close
    o2c = close / panels["open"] - 1                               # the strategy's daily return
    mom = close / close.shift(MOM_DAYS) - 1
    fund7 = funding.rolling(7, min_periods=4).mean()
    index = close.index

    daily, picks, coin_days = [], [], []
    for day, table, buy, short, problem in days:
        nxt = day + pd.Timedelta(days=1)
        if table is None or problem:
            continue
        u = table.index
        r_next = o2c.loc[nxt, u]
        mkt = r_next.mean()

        def ls(score):                                             # top 16 minus bottom 16, half each side
            s = score.dropna()
            if len(s) < 2 * K:
                return np.nan
            return 0.5 * r_next[s.nlargest(K).index].mean() - 0.5 * r_next[s.nsmallest(K).index].mean()

        row = {"trade_date": nxt,
               "a101_055": 0.5 * r_next[buy].fillna(0).mean() - 0.5 * r_next[short].fillna(0).mean(),
               "reversal_1d": ls(-c2c.loc[day, u]),
               f"momentum_{MOM_DAYS}d": ls(mom.loc[day, u]),
               "market": mkt,
               "reversal_mode": -spearman(c2c.loc[day, u], c2c.loc[nxt, u]),   # + = yesterday's movers reverse
               "reversal_mode_known": (-spearman(c2c.loc[day - pd.Timedelta(days=1), u], c2c.loc[day, u])
                                       if day - pd.Timedelta(days=1) in index else np.nan)}
        for k in HORIZONS:
            end = day + pd.Timedelta(days=k)
            if end in index:
                fwd = close.loc[end, u] / close.loc[day, u] - 1
                fwd = fwd - fwd.mean()
                row[f"short_fwd_{k}d"] = fwd[short].mean()
                row[f"buy_fwd_{k}d"] = fwd[buy].mean()
        daily.append(row)

        for side, coins in (("BUY", buy), ("SHORT", short)):
            picks.append({"trade_date": nxt, "side": side,
                          "price position rank": table.loc[coins, "pos_rank"].mean(),
                          "volume rank": table.loc[coins, "vol_rank"].mean(),
                          "last 3 days vs market %": ((close.loc[day, coins] / close.loc[day - pd.Timedelta(days=3), coins] - 1).mean()
                                                      - (close.loc[day, u] / close.loc[day - pd.Timedelta(days=3), u] - 1).mean()) * 100})
        for s in short:
            coin_days.append({"trade_date": nxt, "symbol": s, "ret": o2c.at[nxt, s], "market": mkt,
                              "funding7": fund7.at[day, s] if s in fund7.columns else np.nan})

    d = pd.DataFrame(daily).set_index("trade_date")
    d["period"] = [period_of(t) for t in d.index]
    p = pd.DataFrame(picks)
    p["period"] = [period_of(t) for t in p["trade_date"]]
    c = pd.DataFrame(coin_days)
    c["period"] = [period_of(t) for t in c["trade_date"]]
    c["vs market"] = c["ret"] - c["market"]

    # 1. what the picks look like
    looks = p.groupby(["period", "side"])[["price position rank", "volume rank",
                                           "last 3 days vs market %"]].mean()
    # 2. strategies by period
    strat_cols = ["a101_055", "reversal_1d", f"momentum_{MOM_DAYS}d"]
    by_period = d.groupby("period")[strat_cols].mean().mul(1e4).reindex(NAMES)
    corr_train = d.loc[d["period"] == "Train", strat_cols].corr()
    corr_all = d[strat_cols].corr()
    # 3. what the coins did next
    fwd_cols = [f"{side}_fwd_{k}d" for side in ("short", "buy") for k in HORIZONS]
    nxt = d.groupby("period")[fwd_cols].mean().mul(1e4).reindex(NAMES)
    # 4. reversal mode
    mode = d.groupby("period")["reversal_mode"].agg(["mean", lambda s: (s > 0).mean() * 100])
    mode.columns = ["reversal mode (avg)", "days in reversal mode %"]
    mode = mode.reindex(NAMES)
    trailing = d["reversal_mode_known"].rolling(TRAIL, min_periods=20).mean()
    forward = d["a101_055"][::-1].rolling(TRAIL, min_periods=20).mean()[::-1].shift(-1)
    train = d["period"] == "Train"
    tercile = pd.qcut(trailing[train], 3, labels=["low (momentum-ish)", "middle", "high (reversal)"])
    by_mode = d.loc[train].groupby(tercile, observed=True)["a101_055"].mean().mul(1e4)
    mode_link = pd.Series({
        f"Train: corr(trailing {TRAIL}-day reversal mode, next {TRAIL} days of A101_055)":
            trailing[train].corr(forward[train]),
        f"All days: same (information)": trailing.corr(forward)})
    # 5. does a bad month follow a bad month?
    monthly = d.groupby([d.index.year, d.index.month])["a101_055"].sum()
    m_train = monthly.loc[[ix for ix in monthly.index if (ix[0], ix[1]) <= (2023, 12)]]
    persist = pd.Series({
        "Train months": len(m_train),
        "Train: next month after a losing month, bp/day": (m_train.shift(-1)[m_train < 0].mean() / 30 * 1e4),
        "Train: next month after a winning month, bp/day": (m_train.shift(-1)[m_train > 0].mean() / 30 * 1e4),
        "Train: corr(month, next month)": m_train.corr(m_train.shift(-1)),
        "All months: corr(month, next month) (information)": monthly.corr(monthly.shift(-1)),
    })
    # 6. crowded shorts
    c["crowded"] = c["funding7"] < HEAVY_FUNDING
    crowd = c.groupby("period").agg(
        crowded_share=("crowded", lambda s: s.mean() * 100),
        crowded_vs_market_bp=("vs market", lambda s: np.nan),
    ).reindex(NAMES)
    crowd["crowded shorts: coin vs market, bp/day"] = (
        c[c["crowded"]].groupby("period")["vs market"].mean().mul(1e4).reindex(NAMES))
    crowd["other shorts: coin vs market, bp/day"] = (
        c[~c["crowded"]].groupby("period")["vs market"].mean().mul(1e4).reindex(NAMES))
    crowd = crowd.drop(columns="crowded_vs_market_bp").rename(
        columns={"crowded_share": "shorts already paying heavy funding, %"})
    # 7. examples
    losers = pd.read_csv(HERE / "results" / "stage0b" / "top_losers.csv")
    ex_rows = []
    for name in ("Box 1", "Box 2", "Dip 3"):
        for symbol in losers.loc[losers["period"] == name, "symbol"].head(EXAMPLES):
            g = c[(c["period"] == name) & (c["symbol"] == symbol)].sort_values("trade_date")
            worst = g.loc[g["ret"].idxmax()]
            ex_rows.append({
                "window": name, "symbol": symbol, "days shorted": len(g),
                "first": f"{g['trade_date'].min():%Y-%m-%d}", "last": f"{g['trade_date'].max():%Y-%m-%d}",
                "coin on shorted days %": ((1 + g["ret"]).prod() - 1) * 100,
                "market on those days %": ((1 + g["market"]).prod() - 1) * 100,
                "up days %": (g["ret"] > 0).mean() * 100,
                "biggest day": f"{worst['trade_date']:%Y-%m-%d} {worst['ret'] * 100:+.0f}%",
                "days with heavy funding": int(g["crowded"].sum()),
            })
    examples = pd.DataFrame(ex_rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    f = lambda v: f"{v:.2f}"
    print("1. WHAT THE PICKS LOOK LIKE ON THE DAY (ranks: 0 = lowest of the 80, 1 = highest)")
    print(looks.to_string(float_format=f))
    print("\n2. A101_055 vs SIMPLE STRATEGIES (bp/day before costs, 16 long / 16 short)")
    print(by_period.to_string(float_format=f))
    print("\n   correlation of daily results, Train")
    print(corr_train.to_string(float_format=f))
    print("\n   correlation of daily results, all days (information)")
    print(corr_all.to_string(float_format=f))
    print("\n3. WHAT THE COINS DID NEXT (return against the market, bp, over the next k days)")
    print(nxt.to_string(float_format=f))
    print("\n4. REVERSAL MODE OF THE MARKET (+ = yesterday's winners lose today)")
    print(mode.to_string(float_format=f))
    print(f"\n   A101_055 in Train by the last {TRAIL} days' reversal mode (bp/day)")
    print(by_mode.to_string(float_format=f))
    print(mode_link.to_string(float_format=f))
    print("\n5. DOES A BAD MONTH FOLLOW A BAD MONTH?")
    print(persist.to_string(float_format=f))
    print("\n6. CROWDED SHORTS")
    print(crowd.to_string(float_format=f))
    print("\n7. EXAMPLES: the coins that cost most in each bad window, on the days they were shorted")
    print(examples.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    looks.to_csv(OUT / "picks.csv", float_format="%.6g")
    by_period.to_csv(OUT / "strategies_bp.csv", float_format="%.6g")
    corr_train.to_csv(OUT / "correlation_train.csv", float_format="%.6g")
    nxt.to_csv(OUT / "next_days_bp.csv", float_format="%.6g")
    mode.to_csv(OUT / "reversal_mode.csv", float_format="%.6g")
    by_mode.to_csv(OUT / "a101_by_reversal_mode_train.csv", float_format="%.6g")
    persist.to_csv(OUT / "month_persistence.csv", float_format="%.6g")
    crowd.to_csv(OUT / "crowded_shorts.csv", float_format="%.6g")
    examples.to_csv(OUT / "examples.csv", index=False, float_format="%.6g")
    d.to_csv(OUT / "daily.csv", float_format="%.8g", date_format="%Y-%m-%d")
    print(f"\nSaved tables to {OUT.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
