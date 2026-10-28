#!/usr/bin/env python3
"""Stage 4 (step B): a machine-learning re-ranker inside A101_055's candidates.

A101_055 makes the first cut (its 16 per side, or its top 30); a model scores each
candidate's next-day money against the other candidates that day, and the strategy
keeps the best 10 per side (buffer: a held coin stays while in the model's best 12).
Models are refit on the 1st of every month on earlier trade days only.
  (R16) linear, 16    (T16) trees, 16    (D16) AlphaForge-style dynamic linear, 16
  (R30) linear, 30    (T30) trees, 30    (T16+) trees with the 158 broad-list alphas, 16
Each with (v)'s volatility scaling. Gate 1: Rule A after costs (3.5 bp, funding
included) on trade days 2021-10-01 to 2023-12-31; Gate 2 (reject only): t > -1 against
(v) on 2024-01-01 to 2026-10-24. Declared in README ("Stage 4").

    .venv/bin/python stage4_ml.py       # writes results/stage4/
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import a101_055 as live
import backtest as bt
import experiment as ex
import stage2 as s2
import stage3_blend as s3
import stage3b_portfolio as p3

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage4"
KEEP, BUFFER, WIDE = 10, 12, 30
TRAIN_START, FIRST_FIT = pd.Timestamp("2021-03-29"), pd.Timestamp("2021-10-01")
GATE1 = ("2021-10-01", "2023-12-31")
GATE2 = ("2024-01-01", "2026-10-24")
GATE1_YEARS, GATE2_T = (2022, 2023), -1.0
DYN_DAYS, DYN_KEEP, RIDGE_SHARE = 182, 8, 0.10
TREES = dict(loss="squared_error", max_depth=3, learning_rate=0.05, max_iter=300,
             min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
COIN = ["a101_score", "a101_in_list", "ret_1", "ret_3", "ret_6", "ret_14", "ret_30", "tv_today",
        "tv_6d", "vol_30", "max_rise_30", "days_since_high", "funding_day", "funding_7d", "premium",
        "adv_30", "age"]
DAY = ["btc_7", "market_7", "market_vol"]
VERSIONS = {"R16": ("ridge", False, False), "T16": ("trees", False, False),
            "D16": ("dynamic", False, False), "R30": ("ridge", True, False),
            "T30": ("trees", True, False), "T16+": ("trees", False, True)}
TITLES = {"v": "(v) working version", "R16": "(R16) linear, best 10 of 16",
          "T16": "(T16) trees, best 10 of 16", "D16": "(D16) dynamic linear, best 10 of 16",
          "R30": "(R30) linear, best 10 of top 30", "T30": "(T30) trees, best 10 of top 30",
          "T16+": "(T16+) trees + 158 alphas, best 10 of 16"}
WINDOWS = {"Gate 1 window": GATE1, "2024": ("2024-01-01", "2024-12-31"),
           "2025-26": ("2025-01-01", "2026-10-24"), "Box 1": ex.INFO["Box 1"],
           "Box 2": ex.INFO["Box 2"], "Dip 3": ex.INFO["Dip 3"],
           "Drawdown 1": s3.WINDOWS["Drawdown 1"], "Drawdown 2": s3.WINDOWS["Drawdown 2"]}


# ------------------------------------------------------------------ inputs

def coin_panels(panels, funding, ctx, broad):
    """Date x symbol panels of the coin inputs (raw values, known at the day's close)."""
    close, high, qvol = panels["close"], panels["high"], panels["qvol"]
    daily = close.pct_change(fill_method=None)
    out = {f"ret_{k}": close / close.shift(k) - 1 for k in (1, 3, 6, 14, 30)}
    tv30 = qvol.rolling(30, min_periods=20).mean()
    out["tv_today"] = qvol / tv30
    out["tv_6d"] = qvol.rolling(6, min_periods=4).mean() / tv30
    out["vol_30"] = daily.rolling(30, min_periods=20).std()
    out["max_rise_30"] = daily.rolling(30, min_periods=20).max()
    top = high.rolling(30, min_periods=20).max()
    since = pd.DataFrame(np.nan, index=high.index, columns=high.columns)
    for k in reversed(range(30)):                       # the smallest k wins: the latest high
        since = since.mask(high.shift(k) == top, float(k))
    out["days_since_high"] = since
    out["funding_day"] = funding
    out["funding_7d"] = funding.rolling(7, min_periods=3).mean()
    out["premium"] = broad["CRYPTO_HMRW_PREMIUM"].unstack().reindex(index=close.index, columns=close.columns)
    out["adv_30"] = ctx.adv
    out["age"] = close.notna().cumsum().where(close.notna())
    return out


def pick(panel, idx):
    """Values of a date x symbol panel at a (date, symbol) index."""
    rows = panel.index.get_indexer(idx.get_level_values(0))
    cols = panel.columns.get_indexer(idx.get_level_values(1))
    values = np.full(len(idx), np.nan)
    ok = (rows >= 0) & (cols >= 0)
    values[ok] = panel.to_numpy(dtype=float)[rows[ok], cols[ok]]
    return pd.Series(values, index=idx)


def centred_rank(values):
    """The day's percentile rank among the universe, centred on 0; missing = 0."""
    return (values.groupby(level=0).rank(pct=True) - 0.5).fillna(0.0)


def features(days, panels, funding, ctx, broad):
    """One row per universe coin-day: the 16 ranked coin inputs (all but a101_in_list),
    the day-level inputs, and the 158 ranked broad-list alphas."""
    idx, a101 = [], []
    for day, table, _b, _s, problem in days:
        if problem is None and table is not None:
            idx += [(day, c) for c in table.index]
            a101.append(table["a101_055"])
    idx = pd.MultiIndex.from_tuples(idx, names=["date", "symbol"])
    raw = coin_panels(panels, funding, ctx, broad)
    f = pd.DataFrame({name: centred_rank(pick(panel, idx)) for name, panel in raw.items()})
    f["a101_score"] = centred_rank(pd.Series(pd.concat(a101).to_numpy(), index=idx))

    close = panels["close"]
    daily = close.pct_change(fill_method=None)
    ret7 = close / close.shift(7) - 1
    dates = idx.get_level_values(0)
    market_7 = pick(ret7, idx).groupby(level=0).mean()
    ew = pick(daily, idx).groupby(level=0).mean()
    day_level = pd.DataFrame({"btc_7": ret7["BTCUSDT"].reindex(market_7.index),
                              "market_7": market_7,
                              "market_vol": ew.rolling(30, min_periods=20).std()})
    f = f.join(day_level.reindex(dates).set_axis(idx))
    alphas = broad.reindex(idx)
    alphas = (alphas.groupby(level=0).rank(pct=True) - 0.5).fillna(0.0)
    return f, alphas


def candidates(table, buy, short, wide):
    """BUY and SHORT candidates: A101_055's lists, or its 30 highest and lowest scores."""
    if not wide:
        return list(buy), list(short)
    score = table["a101_055"].round(live.TIE_DECIMALS)
    valid = score.dropna()
    n = min(WIDE, len(valid) // 2)
    ordered = valid.sort_values(ascending=False)
    b, s = valid[valid >= ordered.iloc[n - 1]], valid[valid <= ordered.iloc[-n]]
    if set(b.index) & set(s.index):
        return list(buy), list(short)
    return sorted(b.index, key=lambda c: (-b[c], c)), sorted(s.index, key=lambda c: (s[c], c))


def candidate_rows(days, ctx, f, alphas, wide, with_alphas):
    """One row per candidate per day: inputs, side, A101_055 strength, and the target."""
    recs = []
    for day, table, buy, short, problem in days:
        if problem is not None or table is None:
            continue
        nxt = day + pd.Timedelta(days=1)
        b, s = candidates(table, buy, short, wide)
        r = ctx.ret.loc[nxt].fillna(0.0)
        fr = ctx.funding.loc[nxt].fillna(0.0)
        for side, coins in ((1, b), (0, s)):
            sign = 1.0 if side else -1.0
            strength = (sign * table["a101_055"][coins]).rank(pct=True)
            bet = sign * (r[coins] - fr[coins])
            for c in coins:
                recs.append((day, c, nxt, side, strength[c], bet[c] - bet.mean()))
    rows = pd.DataFrame(recs, columns=["date", "symbol", "nxt", "side", "a101_in_list", "target"])
    key = pd.MultiIndex.from_frame(rows[["date", "symbol"]])
    rows = pd.concat([rows, f.reindex(key).reset_index(drop=True)], axis=1)
    if with_alphas:
        rows = pd.concat([rows, alphas.reindex(key).reset_index(drop=True)], axis=1)
    return rows


# ------------------------------------------------------------------ models

def ridge(train, test, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0.0, 1.0)
    model = Ridge(alpha=RIDGE_SHARE * len(train)).fit((train[cols] - mu) / sd, train["target"])
    return model.predict((test[cols] - mu) / sd), pd.Series(model.coef_, index=cols)


def daily_corr(df, cols):
    """Average over trade days of the correlation, inside the day's candidates, of each
    input with the target."""
    g = df.groupby("nxt")
    xd = df[cols] - g[cols].transform("mean")
    yd = df["target"] - g["target"].transform("mean")
    num = xd.mul(yd, axis=0).groupby(df["nxt"]).sum()
    den = np.sqrt((xd ** 2).groupby(df["nxt"]).sum().mul((yd ** 2).groupby(df["nxt"]).sum(), axis=0))
    return (num / den.where(den > 0)).mean()


def walk_forward(kind, rows, extra):
    """Out-of-sample scores, refit on the 1st of each month on earlier trade days only."""
    score = pd.Series(np.nan, index=rows.index)
    notes = []
    last = rows["nxt"].max()
    starts = pd.date_range(FIRST_FIT, last, freq="MS")
    tree_cols = COIN + DAY + ["side"] + extra
    for m, m_next in zip(starts, list(starts[1:]) + [last + pd.Timedelta(days=1)]):
        test = (rows["nxt"] >= m) & (rows["nxt"] < m_next)
        if not test.any():
            continue
        past = (rows["nxt"] >= TRAIN_START) & (rows["nxt"] < m)
        if kind == "trees":
            model = HistGradientBoostingRegressor(**TREES).fit(rows.loc[past, tree_cols], rows.loc[past, "target"])
            score[test] = model.predict(rows.loc[test, tree_cols])
            continue
        for side in (1, 0):
            t_side = test & (rows["side"] == side)
            p_side = past & (rows["side"] == side)
            cols = COIN
            if kind == "dynamic":
                p_side &= rows["nxt"] >= m - pd.Timedelta(days=DYN_DAYS)
                corr = daily_corr(rows[p_side], COIN)
                cols = list(corr.abs().sort_values(ascending=False).index[:DYN_KEEP])
            pred, coef = ridge(rows[p_side], rows[t_side], cols)
            score[t_side] = pred
            notes.append({"month": m, "side": "BUY" if side else "SHORT", **coef.to_dict()})
    return score, pd.DataFrame(notes)


# ------------------------------------------------------------------ portfolio

def keep_lists(rows, score, days):
    """Kept BUY and SHORT coins per signal day (today's lists before the first fit)."""
    scored = rows.assign(score=score)
    groups = {k: g for k, g in scored.groupby(["date", "side"])}
    out, held = {}, {1: set(), 0: set()}
    for day, table, buy, short, problem in days:
        if problem is not None:
            out[day], held = None, {1: set(), 0: set()}
            continue
        if day + pd.Timedelta(days=1) < FIRST_FIT:
            out[day] = (list(buy), list(short))
            held = {1: set(buy), 0: set(short)}
            continue
        kept = {}
        for side in (1, 0):
            g = groups[(day, side)].sort_values(["score", "a101_in_list"], ascending=False)
            order = g["symbol"].tolist()
            best = set(order[:BUFFER])
            stay = [c for c in order if c in held[side] and c in best]
            kept[side] = (stay + [c for c in order if c not in stay])[:KEEP]
        out[day] = (kept[1], kept[0])
        held = {1: set(kept[1]), 0: set(kept[0])}
    return out


def books(lists, days, ctx):
    a_spec = ex.VERSIONS[0]
    return {day: (None if lists[day] is None else ex.weights(a_spec, day, *lists[day], ctx))
            for day, *_ in days}


def kept_minus_dropped(rows, lists):
    """Information: average target of the kept coins minus the dropped ones, per trade day."""
    out = {}
    for (day, side), g in rows.groupby(["date", "side"]):
        if lists.get(day) is None or day + pd.Timedelta(days=1) < FIRST_FIT:
            continue
        kept = g["symbol"].isin(lists[day][0 if side else 1])
        if kept.all() or not kept.any():
            continue
        out.setdefault(day + pd.Timedelta(days=1), []).append(
            g.loc[kept, "target"].mean() - g.loc[~kept, "target"].mean())
    return pd.Series({d: np.mean(v) for d, v in out.items()}).sort_index()


# ------------------------------------------------------------------ judging

def gate1(d, base):
    sl = slice(*GATE1)
    g, b = s3.net(d, s3.COST).loc[sl], s3.net(base, s3.COST).loc[sl]
    diff = g - b
    better = {y: s3.sharpe(g[g.index.year == y]) > s3.sharpe(b[b.index.year == y]) for y in GATE1_YEARS}
    equity = (1 + g).cumprod()
    row = {"Sharpe": s3.sharpe(g), "Sharpe of (v)": s3.sharpe(b), "bp/day": g.mean() * 1e4,
           "vs (v) bp/day": diff.mean() * 1e4, "t vs (v)": diff.mean() / diff.std() * np.sqrt(len(diff)),
           **{f"better in {y}": better[y] for y in GATE1_YEARS},
           "worst drawdown %": (equity / equity.cummax() - 1).min() * 100,
           "turnover %": d.loc[sl, "turnover"].mean() * 100,
           "before costs Sharpe": s3.sharpe(d.loc[sl, "gross"])}
    checks = {"Sharpe above (v)": row["Sharpe"] > row["Sharpe of (v)"], "t >= 2": row["t vs (v)"] >= 2,
              "better in 2022 and 2023": all(better.values())}
    failed = [k for k, ok in checks.items() if not ok]
    row["Gate 1"] = "PASS" if not failed else "fail: " + ", ".join(failed)
    return row


def gate2(d, base):
    sl = slice(*GATE2)
    diff = (s3.net(d, s3.COST) - s3.net(base, s3.COST)).loc[sl]
    t = diff.mean() / diff.std() * np.sqrt(len(diff))
    gross = (d["gross"] - base["gross"]).loc[sl]
    return {"2024-26 vs (v) bp/day": diff.mean() * 1e4, "2024-26 t": t,
            "2024-26 vs (v) before costs bp/day": gross.mean() * 1e4,
            "2024-26 turnover %": d.loc[sl, "turnover"].mean() * 100,
            "Gate 2": "PASS" if t > GATE2_T else "fail: t <= -1"}


def main():
    t0 = time.time()
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    trend = panels["close"] / panels["close"].shift(s2.TREND_DAYS) - 1
    broad = pd.read_parquet(HERE / "data" / "broad_all_years.parquet")
    f, alphas = features(days, panels, funding, ctx, broad)
    print(f"  inputs built: {len(f):,} coin-days ({time.time() - t0:.0f}s)")

    results = {"v": s2.run(s2.VERSIONS[-1], days, ctx, trend)}
    notes, spreads, all_rows = {}, {}, {}
    for name, (kind, wide, with_alphas) in VERSIONS.items():
        key = (wide, with_alphas)
        if key not in all_rows:
            all_rows[key] = candidate_rows(days, ctx, f, alphas, wide, with_alphas)
        rows = all_rows[key]
        extra = list(alphas.columns) if with_alphas else []
        score, notes[name] = walk_forward(kind, rows, extra)
        lists = keep_lists(rows, score, days)
        results[name] = p3.scaled(books(lists, days, ctx), days, ctx)
        spreads[name] = kept_minus_dropped(rows, lists)
        early = results[name].loc[:FIRST_FIT - pd.Timedelta(days=1), "gross"]
        gap = (early - results["v"]["gross"].reindex(early.index)).abs().max()
        assert gap < 1e-12, f"({name}) differs from (v) before the first fit (gap {gap:.2g})"
        print(f"  ({name}) done: {len(rows):,} candidate rows ({time.time() - t0:.0f}s)")

    names = list(VERSIONS)
    judged = pd.DataFrame({n: {**gate1(results[n], results["v"]), **gate2(results[n], results["v"])}
                           for n in names}).T
    passers = [n for n in names if judged.at[n, "Gate 1"] == "PASS" and judged.at[n, "Gate 2"] == "PASS"]
    if passers:
        best = max(passers, key=lambda n: judged.at[n, "Sharpe"])
        note = f"({best}) passes both gates and has the highest Sharpe after costs: it is the final version."
    else:
        note = "No version passes both gates: (v) stays the final version."

    rows_info = []
    for name, d in results.items():
        for cost in (0.0, s3.COST, 7.5):
            r = d["gross"] if cost == 0 else s3.net(d, cost)
            for period, (start, end) in WINDOWS.items():
                g = r.loc[start:end]
                equity = (1 + g).cumprod()
                rows_info.append({"version": name, "cost bp": cost, "period": period,
                                  "Sharpe": s3.sharpe(g), "total %": (equity.iloc[-1] - 1) * 100,
                                  "worst drawdown %": (equity / equity.cummax() - 1).min() * 100})
    info = pd.DataFrame(rows_info)
    yearly = pd.DataFrame({n: s3.net(d, s3.COST).loc[FIRST_FIT:].groupby(s3.net(d, s3.COST).loc[FIRST_FIT:].index.year)
                           .apply(lambda g: ((1 + g).prod() - 1) * 100) for n, d in results.items()})
    spread_year = pd.DataFrame({n: s.groupby(s.index.year).mean() * 1e4 for n, s in spreads.items()})

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    f2 = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
    print(f"\nJUDGED against (v), after costs at {s3.COST} bp, funding included")
    print(judged.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\n{note}")
    for cost in (s3.COST, 0.0, 7.5):
        label = "before costs (no fees, no funding)" if cost == 0 else f"after costs at {cost:g} bp"
        print(f"\nINFORMATION: total return %, {label}")
        print(info[info["cost bp"] == cost].pivot(index="version", columns="period", values="total %")
              [list(WINDOWS)].reindex(list(results)).to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"\nINFORMATION: total return % by year at {s3.COST} bp (from {FIRST_FIT:%Y-%m-%d})")
    print(yearly.to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nINFORMATION: kept 10 minus dropped candidates, money bp/day, by year")
    print(spread_year.to_string(float_format=lambda x: f"{x:.1f}"))
    for name in ("R16", "D16"):
        n = notes[name].reindex(columns=["month", "side", *COIN])
        n = n.assign(year=n["month"].dt.year)
        if name == "R16":
            print("\nINFORMATION: (R16) average linear weight per input (standardised), by side and year")
            print(n.groupby(["side", "year"])[COIN].mean().T.to_string(float_format=lambda x: f"{x:+.4f}"))
        else:
            print("\nINFORMATION: (D16) share of months each input was picked, by side")
            print(n.groupby("side")[COIN].apply(lambda g: g.notna().mean()).T.to_string(float_format=lambda x: f"{x:.2f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    judged.to_csv(OUT / "judged.csv", float_format="%.6g")
    info.to_csv(OUT / "information.csv", index=False, float_format="%.6g")
    yearly.to_csv(OUT / "yearly.csv", float_format="%.6g")
    spread_year.to_csv(OUT / "kept_minus_dropped_by_year.csv", float_format="%.6g")
    for name, n in notes.items():
        if len(n):
            n.to_csv(OUT / f"weights_{name}.csv", index=False, float_format="%.6g")
    pd.DataFrame({n: s3.net(d, s3.COST) for n, d in results.items()}).to_csv(
        OUT / "daily_net_3p5bp.csv.gz", float_format="%.6g")
    (OUT / "final.txt").write_text(note + "\n")
    chart(results, judged, OUT / "after_costs.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/ ({time.time() - t0:.0f}s)")


def chart(results, judged, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ink, ink2, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    wash, surface = "#f0efec", "#fcfcfb"
    colors = {"R16": "#2a78d6", "T16": "#eb6834", "D16": "#3a9e6e",
              "R30": "#2a78d6", "T30": "#eb6834", "T16+": "#8a5cc2"}
    panels = [("R16", "T16", "D16"), ("R30", "T30", "T16+")]
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface,
    })
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 4.2 * len(panels)), sharex=True)
    for ax, group in zip(axes, panels):
        ax.axvspan(pd.Timestamp(GATE1[0]), pd.Timestamp(GATE1[1]), color=wash, lw=0, zorder=0)
        for s, e in (s3.WINDOWS["Drawdown 1"], s3.WINDOWS["Drawdown 2"]):
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=grid, lw=0, zorder=0)
        ends = []
        for name in ("v", *group):
            eq = 10_000 * (1 + s3.net(results[name], s3.COST).loc[FIRST_FIT:]).cumprod()
            color = ink if name == "v" else colors[name]
            ax.plot(eq.index, eq, color=color, lw=1.9 if name == "v" else 1.4, label=TITLES[name])
            ends.append((eq.iloc[-1], eq.index[-1], color))
        ax.set_yscale("log")
        low, high = ax.get_ylim()
        gap, previous = 0.05 * np.log(high / low), None
        for last, when, color in sorted(ends, reverse=True):
            y = last if previous is None else min(last, previous * np.exp(-gap))
            previous = y
            ax.annotate(f"${last:,.0f}", (when, last), xytext=(6, y), textcoords=("offset points", "data"),
                        va="center", color=color, fontsize=8.5)
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 1.5, 2, 3, 5)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        parts = []
        for n in group:
            g1 = "pass" if judged.at[n, "Gate 1"] == "PASS" else "fail"
            g2 = "pass" if judged.at[n, "Gate 2"] == "PASS" else "fail"
            parts.append(f"({n}) G1 {g1}, G2 {g2}")
        ax.set_title(" · ".join(parts), loc="left", color=ink, fontsize=10)
        ax.legend(frameon=False, loc="upper left", labelcolor=ink2, fontsize=8.5)
        ax.set_ylabel("Value of $10,000")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Stage 4: machine-learning re-ranker against (v), after costs ({s3.COST} bp per dollar traded, "
                 "funding included)", x=0.01, ha="left", color=ink, fontsize=12.5)
    fig.text(0.01, 0.942, f"From {FIRST_FIT:%Y-%m-%d} (first model fit). Light shading = Gate 1 window "
             f"(to {GATE1[1]}); after it = Gate 2 window. Darker bands = Drawdown 1 and 2. "
             "Models refit monthly on past days only.", color=ink2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
