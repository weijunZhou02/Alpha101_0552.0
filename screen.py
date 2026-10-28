#!/usr/bin/env python3
"""Stage 3, steps 1-4: screen 126 trend and reversal alphas as helpers for A101_055.

For each candidate, on trade days 2021-03-29 to 2023-12-31 only:
  1. skill across all 80 coins: daily rank IC with the next-day return,
  2. skill inside A101_055's picks: rank IC within the day's 16 BUY coins and
     within its 16 SHORT coins (day value = their average); also by side and by
     bet type (reversal bets: shorts that were rising + buys that were falling;
     continuation bets: buys that were rising + shorts that were falling),
  3. overlap: daily rank correlation with A101_055, and day-to-day rank stability.
Then the shortlist rule declared in README ("Stage 3"). Writes results/stage3/.

    .venv/bin/python screen.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import experiment as ex

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "stage3"
TRADE_START, TRADE_END = pd.Timestamp("2021-03-29"), pd.Timestamp("2023-12-31")
YEARS = (2021, 2022, 2023)
MOVE_DAYS = 6
TREND_DAYS = 14            # information: does a candidate follow the trend (+) or fade it (-)?
MIN_ALL, MIN_LIST, MIN_GROUP = 30, 10, 5
RULE_T, RULE_CORR, PER_FAMILY = 3.0, 0.3, 1


def daily_ic(X, y, min_n):
    """Spearman correlation of each column of X with y over the rows where both exist
    (ranks taken over exactly those rows, column by column)."""
    ok_y = y.notna().to_numpy()
    X = X[ok_y]
    yv = y[ok_y].to_numpy(dtype=float)
    Y = pd.DataFrame(np.where(X.notna(), yv[:, None], np.nan), index=X.index, columns=X.columns)
    xr, yr = X.rank(), Y.rank()
    xd, yd = xr - xr.mean(), yr - yr.mean()
    den = np.sqrt((xd ** 2).sum() * (yd ** 2).sum())
    ic = (xd * yd).sum() / den.where(den > 0)
    return ic.where(X.notna().sum() >= min_n)


def paired_ic(A, B, min_n):
    """Spearman correlation of each column of A with the same column of B."""
    mask = A.notna() & B.notna()
    ar, br = A.where(mask).rank(), B.where(mask).rank()
    ad, bd = ar - ar.mean(), br - br.mean()
    den = np.sqrt((ad ** 2).sum() * (bd ** 2).sum())
    return ((ad * bd).sum() / den.where(den > 0)).where(mask.sum() >= min_n)


def tstat(s):
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 2 and s.std() > 0 else np.nan


def main():
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    close = panels["close"]
    o2c = close / panels["open"] - 1
    raw = pd.read_csv(HERE / "data" / "candidates_2021_2023.csv.gz", parse_dates=["date"])
    raw = raw.set_index(["date", "symbol"]).sort_index()
    assert raw.index.get_level_values("date").max() <= TRADE_END
    names = list(raw.columns)
    raw_days = set(raw.index.get_level_values("date"))
    family = pd.read_csv(HERE / "data" / "candidate_families.csv", index_col=0)["family"]

    rec = {k: [] for k in ("all", "buy", "short", "inside", "reversal", "continuation",
                           "overlap", "stability", "trend")}
    dates, coverage_hits, coverage_total = [], pd.Series(0.0, index=names), 0
    prev_X = None
    for day, table, buy, short, problem in days:
        nxt = day + pd.Timedelta(days=1)
        if problem or table is None or not (TRADE_START <= nxt <= TRADE_END):
            continue
        u = table.index
        if day not in raw_days:
            continue
        X = raw.loc[day].reindex(u)
        y = o2c.loc[nxt, u]
        move = close.loc[day, u] / close.loc[day - pd.Timedelta(days=MOVE_DAYS), u] - 1
        rising = move - move.mean() > 0
        reversal = [c for c in short if rising.get(c, False)] + [c for c in buy if not rising.get(c, True)]
        continuation = [c for c in buy if rising.get(c, False)] + [c for c in short if not rising.get(c, True)]

        ic_buy, ic_short = daily_ic(X.loc[buy], y[buy], MIN_LIST), daily_ic(X.loc[short], y[short], MIN_LIST)
        rec["all"].append(daily_ic(X, y, MIN_ALL))
        rec["buy"].append(ic_buy)
        rec["short"].append(ic_short)
        rec["inside"].append((ic_buy + ic_short) / 2)
        rec["reversal"].append(daily_ic(X.loc[reversal], y[reversal], MIN_GROUP))
        rec["continuation"].append(daily_ic(X.loc[continuation], y[continuation], MIN_GROUP))
        rec["overlap"].append(daily_ic(X, table["a101_055"], MIN_ALL))
        rec["trend"].append(daily_ic(X, close.loc[day, u] / close.loc[day - pd.Timedelta(days=TREND_DAYS), u] - 1,
                                     MIN_ALL))
        rec["stability"].append(paired_ic(X, prev_X.reindex(u), MIN_ALL) if prev_X is not None
                                else pd.Series(np.nan, index=names))
        picks = list(buy) + list(short)
        coverage_hits += X.loc[picks].notna().sum()
        coverage_total += len(picks)
        dates.append(nxt)
        prev_X = X
    frames = {k: pd.DataFrame(v, index=dates) for k, v in rec.items()}
    print(f"Screened {len(names)} candidates on {len(dates)} trade days "
          f"({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}).")

    rows = []
    for a in names:
        inside = frames["inside"][a]
        by_year = {y: inside[inside.index.year == y].mean() for y in YEARS}
        rows.append({
            "alpha": a, "family": family.get(a, ""),
            "inside IC": inside.mean(), "inside t": tstat(inside),
            **{f"inside {y}": by_year[y] for y in YEARS},
            "BUY-list IC": frames["buy"][a].mean(), "SHORT-list IC": frames["short"][a].mean(),
            "reversal-bets IC": frames["reversal"][a].mean(), "reversal t": tstat(frames["reversal"][a]),
            "continuation-bets IC": frames["continuation"][a].mean(),
            "continuation t": tstat(frames["continuation"][a]),
            "all-80 IC": frames["all"][a].mean(), "all-80 t": tstat(frames["all"][a]),
            "corr with A101_055": frames["overlap"][a].mean(),
            "rank stability": frames["stability"][a].mean(),
            "corr with 14-day return": frames["trend"][a].mean(),
            "coverage of picks %": coverage_hits[a] / coverage_total * 100,
        })
    table = pd.DataFrame(rows).set_index("alpha").sort_values("inside t", ascending=False)
    table["passes rule"] = ((table["inside t"] >= RULE_T)
                            & (table[[f"inside {y}" for y in YEARS]] > 0).all(axis=1)
                            & (table["corr with A101_055"].abs() <= RULE_CORR))
    shortlist = (table[table["passes rule"]].sort_values("inside t", ascending=False)
                 .groupby("family").head(PER_FAMILY))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["family", "inside IC", "inside t", *[f"inside {y}" for y in YEARS], "BUY-list IC",
            "SHORT-list IC", "reversal t", "continuation t", "all-80 t", "corr with A101_055",
            "rank stability", "corr with 14-day return", "passes rule"]
    for fam in table["family"].unique():
        print(f"\nTOP 12 BY INSIDE-PICKS t: {fam}")
        print(table[table["family"] == fam][show].head(12).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nPassing the rule: {int(table['passes rule'].sum())} of {len(table)}")
    follow = table[table["corr with 14-day return"] > RULE_CORR]
    print(f"\nINFORMATION: candidates that FOLLOW the trend (corr with 14-day return > {RULE_CORR:g}): "
          f"{len(follow)}; best by inside t:")
    print(follow[show + ["corr with 14-day return"]].head(8).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nINFORMATION: correlation across candidates between inside t and all-80 t: "
          f"{table['inside t'].corr(table['all-80 t']):.2f}")
    print("\nSHORTLIST (best one per family among those passing):")
    print(shortlist[show].to_string(float_format=lambda x: f"{x:.3f}") if len(shortlist) else "  none")

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "screen.csv", float_format="%.6g")
    shortlist.to_csv(OUT / "shortlist.csv", float_format="%.6g")
    chart(table, shortlist, OUT / "screen.png")
    print(f"\nSaved {OUT.relative_to(HERE)}/")


def chart(table, shortlist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, axis, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    colors = {"Momentum / trend": "#2a78d6", "Short-term reversal": "#eb6834"}
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": surface, "axes.facecolor": surface, "axes.axisbelow": True,
    })
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axvspan(-RULE_CORR, RULE_CORR, color="#f0efec", lw=0, zorder=0)
    ax.axhline(RULE_T, color=axis, lw=1, ls="--")
    ax.axhline(0, color=axis, lw=0.8)
    for fam, color in colors.items():
        t = table[table["family"] == fam]
        ax.scatter(t["corr with A101_055"], t["inside t"], s=42, color=color, label=fam,
                   edgecolors=surface, linewidths=1.2, zorder=3)
    for name, r in shortlist.iterrows():
        ax.annotate(name, (r["corr with A101_055"], r["inside t"]), xytext=(6, 4),
                    textcoords="offset points", color=ink, fontsize=9)
    ax.text(RULE_CORR, RULE_T, f"  shortlist zone: t ≥ {RULE_T:g}, |corr| ≤ {RULE_CORR:g}",
            va="bottom", ha="left", color=ink2, fontsize=8)
    ax.set_xlabel("Daily rank correlation with A101_055 (across the 80 coins)")
    ax.set_ylabel("t of skill inside A101_055's picks")
    ax.set_title("Stage 3 screen: can an existing alpha tell A101_055's good picks from bad ones?",
                 loc="left", color=ink, fontsize=12, pad=24)
    ax.text(0, 1.02, "126 candidates, trade days 2021-03-29 to 2023-12-31. Shaded band = allowed overlap.",
            transform=ax.transAxes, color=ink2, fontsize=9)
    ax.legend(frameon=False, loc="lower right", labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
