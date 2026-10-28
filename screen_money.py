#!/usr/bin/env python3
"""Stage 3b (step A): money screen inside A101_055's picks, keep 10 of 16.

For each of the 158 broad-list candidates, on trade days 2021-03-29 to 2023-12-31 only:
each day it re-ranks A101_055's 16 BUY coins (keeps its 10 highest) and 16 SHORT
coins (keeps its 10 lowest); ties go to the stronger A101_055 signal. The day value
per side is the average bet return (next-day return in the bet's direction plus
funding) of the 10 kept minus the 6 dropped; the pooled value is the average of the
two sides. A101_055 re-ranking its own lists is shown as a reference row. Pass rule
declared in README ("Stage 3b"). Information columns (not part of the rule): the
price and funding parts of the spread, and the spread with tied coins sharing the
cut evenly instead of going to A101_055's order. Writes results/stage3b/.

    .venv/bin/python screen_money.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import experiment as ex
from screen import daily_ic, tstat

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage3b"
TRADE_START, TRADE_END = pd.Timestamp("2021-03-29"), pd.Timestamp("2023-12-31")
YEARS = (2021, 2022, 2023)
KEEP, MIN_VALUES = 10, 10
MIN_ALL, MIN_LIST = 30, 10
RULE_T, PER_FAMILY, MAX_SHORTLIST = 3.0, 1, 2
REFERENCE = "A101_055 (own top 10)"


def spread(X, strength, bet, keep_high):
    """Kept-10 minus dropped-6 average bet return, one value per column of X.

    X: candidate values for one list (coins x candidates), higher = expected better.
    strength: A101_055 values for the same coins (tie-break: stronger signal first).
    """
    enough = X.notna().sum() >= MIN_VALUES
    X = X.fillna(X.median())
    tie = strength.rank(ascending=keep_high is False).to_numpy()[:, None] / 100
    key = X.rank(method="min") - tie if keep_high else X.rank(method="min") + tie
    kept = key.rank(ascending=not keep_high, method="first") <= KEEP
    b = bet.to_numpy()[:, None]
    out = (np.where(kept, b, 0).sum(0) / KEEP
           - np.where(~kept, b, 0).sum(0) / (len(X) - KEEP))
    return pd.Series(out, index=X.columns).where(enough)


def shared_ties(X, bet, keep_high):
    """Information: like spread(), but coins tied at the cut share the remaining slots evenly."""
    enough = X.notna().sum() >= MIN_VALUES
    V = X.fillna(X.median()).to_numpy()
    better = (V[None, :, :] > V[:, None, :]) if keep_high else (V[None, :, :] < V[:, None, :])
    tied = (V[None, :, :] == V[:, None, :]).sum(1)
    w = np.clip((KEEP - better.sum(1)) / np.maximum(tied, 1), 0, 1)
    b = bet.to_numpy()[:, None]
    out = (w * b).sum(0) / KEEP - ((1 - w) * b).sum(0) / (len(X) - KEEP)
    return pd.Series(out, index=X.columns).where(enough)


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    ctx = ex.context(panels, funding)
    raw = pd.read_csv(HERE / "data" / "broad_2021_2023.csv.gz", parse_dates=["date"])
    raw = raw.set_index(["date", "symbol"]).sort_index()
    assert raw.index.get_level_values("date").max() <= TRADE_END
    names = list(raw.columns)
    raw_days = set(raw.index.get_level_values("date"))
    family = pd.read_csv(HERE / "data" / "broad_families.csv", index_col=0)["family"]

    rec = {k: [] for k in ("buy", "short", "pooled", "price", "funding", "shared", "distinct",
                           "ic_inside", "overlap")}
    dates = []
    for day, table, buy, short, problem in days:
        nxt = day + pd.Timedelta(days=1)
        if problem or table is None or not (TRADE_START <= nxt <= TRADE_END) or day not in raw_days:
            continue
        u = table.index
        a = table["a101_055"]
        X = pd.concat([raw.loc[day].reindex(u), a.rename(REFERENCE)], axis=1)
        r = ctx.ret.loc[nxt, u].fillna(0.0)
        f = ctx.funding.loc[nxt, u].fillna(0.0)
        buy, short = list(buy), list(short)
        s_buy = spread(X.loc[buy], a[buy], (r - f)[buy], keep_high=True)
        s_short = spread(X.loc[short], a[short], (f - r)[short], keep_high=False)
        rec["buy"].append(s_buy)
        rec["short"].append(s_short)
        rec["pooled"].append(pd.concat([s_buy, s_short], axis=1).mean(axis=1))
        for key, (bb, bs) in {"price": (r, -r), "funding": (-f, f)}.items():
            rec[key].append(pd.concat([spread(X.loc[buy], a[buy], bb[buy], keep_high=True),
                                       spread(X.loc[short], a[short], bs[short], keep_high=False)],
                                      axis=1).mean(axis=1))
        rec["shared"].append(pd.concat([shared_ties(X.loc[buy], (r - f)[buy], keep_high=True),
                                        shared_ties(X.loc[short], (f - r)[short], keep_high=False)],
                                       axis=1).mean(axis=1))
        rec["distinct"].append((X.loc[buy].nunique() + X.loc[short].nunique()) / 2)
        rec["ic_inside"].append((daily_ic(X.loc[buy], r[buy], MIN_LIST)
                                 + daily_ic(X.loc[short], r[short], MIN_LIST)) / 2)
        rec["overlap"].append(daily_ic(X, a, MIN_ALL))
        dates.append(nxt)
    frames = {k: pd.DataFrame(v, index=dates) for k, v in rec.items()}
    print(f"Screened {len(names)} candidates on {len(dates)} trade days "
          f"({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}).")

    rows = []
    for c in [REFERENCE, *names]:
        p = frames["pooled"][c]
        rows.append({
            "alpha": c, "family": "reference" if c == REFERENCE else family.get(c, ""),
            "spread bp/day": p.mean() * 1e4, "spread t": tstat(p),
            **{f"spread {y}": p[p.index.year == y].mean() * 1e4 for y in YEARS},
            "BUY spread bp": frames["buy"][c].mean() * 1e4, "BUY t": tstat(frames["buy"][c]),
            "SHORT spread bp": frames["short"][c].mean() * 1e4, "SHORT t": tstat(frames["short"][c]),
            "10+10 gain bp/day": p.mean() * 1e4 * (16 - KEEP) / 16,
            "price part bp": frames["price"][c].mean() * 1e4,
            "funding part bp": frames["funding"][c].mean() * 1e4,
            "ties shared bp": frames["shared"][c].mean() * 1e4, "ties shared t": tstat(frames["shared"][c]),
            "distinct values in 16": frames["distinct"][c].mean(),
            "median day bp": p.median() * 1e4,
            "inside rank IC": frames["ic_inside"][c].mean(), "inside IC t": tstat(frames["ic_inside"][c]),
            "corr with A101_055": frames["overlap"][c].mean(),
            "days": int(p.notna().sum()),
        })
    table = pd.DataFrame(rows).set_index("alpha")
    ref = table.loc[[REFERENCE]]
    table = table.drop(REFERENCE).sort_values("spread t", ascending=False)
    table["passes rule"] = ((table["spread t"] >= RULE_T)
                            & (table[[f"spread {y}" for y in YEARS]] > 0).all(axis=1))
    shortlist = (table[table["passes rule"]].groupby("family", sort=False).head(PER_FAMILY)
                 .head(MAX_SHORTLIST))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["family", "spread bp/day", "spread t", *[f"spread {y}" for y in YEARS], "BUY spread bp",
            "BUY t", "SHORT spread bp", "SHORT t", "inside IC t", "corr with A101_055", "passes rule"]
    f2 = lambda x: f"{x:.2f}"
    print("\nREFERENCE: A101_055 keeping its own 10 strongest picks per side")
    print(ref[show[:-1]].to_string(float_format=f2))
    print("\nTOP 20 BY MONEY t")
    print(table[show].head(20).to_string(float_format=f2))
    print("\nBOTTOM 5 BY MONEY t")
    print(table[show].tail(5).to_string(float_format=f2))
    print(f"\nPassing the rule: {int(table['passes rule'].sum())} of {len(table)}")
    print(f"Money t >= 2: {int((table['spread t'] >= 2).sum())}; money t > 0: "
          f"{int((table['spread t'] > 0).sum())}; money t <= -2: {int((table['spread t'] <= -2).sum())}")
    print(f"Inside rank IC t >= 3: {int((table['inside IC t'] >= 3).sum())} of {len(table)}")
    print(f"Correlation across candidates between inside rank IC t and money t: "
          f"{table['inside IC t'].corr(table['spread t']):.2f}")
    print("\nSHORTLIST:")
    print(shortlist[show].to_string(float_format=f2) if len(shortlist) else "  none")
    extra = ["spread bp/day", "price part bp", "funding part bp", "ties shared bp", "ties shared t",
             "distinct values in 16", "median day bp", "10+10 gain bp/day"]
    print("\nINFORMATION (not part of the rule): passers and reference")
    print(pd.concat([table[table["passes rule"]], ref])[extra].to_string(float_format=f2))
    if len(shortlist) > 1:
        print(f"Correlation of the shortlisted helpers' daily spreads: "
              f"{frames['pooled'][shortlist.index].corr().iloc[0, 1]:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([ref, table]).to_csv(OUT / "screen_money.csv", float_format="%.6g")
    shortlist.to_csv(OUT / "shortlist.csv", float_format="%.6g")
    frames["pooled"].to_csv(OUT / "daily_pooled_spread.csv.gz", float_format="%.6g")
    chart(table, ref, shortlist, OUT / "screen_money.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/")


def chart(table, ref, shortlist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, axis, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface, "axes.axisbelow": True,
    })
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(RULE_T, color=axis, lw=1, ls="--")
    ax.axhline(0, color=axis, lw=0.8)
    ax.axvline(0, color=axis, lw=0.8)
    passed = table["passes rule"]
    ax.scatter(table.loc[~passed, "inside IC t"], table.loc[~passed, "spread t"], s=36, color=blue,
               edgecolors=surface, linewidths=1.1, zorder=3, label="candidate")
    if passed.any():
        ax.scatter(table.loc[passed, "inside IC t"], table.loc[passed, "spread t"], s=48, color=orange,
                   edgecolors=surface, linewidths=1.1, zorder=4, label="passes the money rule")
    r = ref.iloc[0]
    ax.scatter([r["inside IC t"]], [r["spread t"]], marker="D", s=46, color=ink, zorder=5,
               label="A101_055 keeping its own top 10")
    for name, row in pd.concat([shortlist, table.head(3)]).drop_duplicates().iterrows():
        ax.annotate(name, (row["inside IC t"], row["spread t"]), xytext=(6, 4),
                    textcoords="offset points", color=ink, fontsize=8.5)
    ax.text(ax.get_xlim()[1], RULE_T, f"money rule t ≥ {RULE_T:g}  ", va="bottom", ha="right",
            color=ink2, fontsize=8)
    ax.set_xlabel("Ranking skill inside A101_055's picks (rank IC t)")
    ax.set_ylabel("Money: kept 10 minus dropped 6 (t)")
    ax.set_title("Stage 3b: does any alpha pick A101_055's best 10 of 16, in money?",
                 loc="left", color=ink, fontsize=12, pad=24)
    ax.text(0, 1.02, "158 broad-list alphas, trade days 2021-03-29 to 2023-12-31, "
            "bet return includes funding, big jumps not trimmed.", transform=ax.transAxes,
            color=ink2, fontsize=9)
    ax.legend(frameon=False, loc="lower right", labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
