#!/usr/bin/env python3
"""Export the stage 3 candidate alphas from the main Alpha-mining project, 2021-2023 only.

Candidates: every alpha in the "Momentum / trend" and "Short-term reversal"
families of the main alpha catalog that has a direction in the main project's
2021-2023 scorecard. Each is multiplied by that direction (higher = expected to
do better). Rows after 2023-12-31 are dropped before saving, so the screening
cannot see later years. Values are NaN outside the main project's daily top 80.

Run with the main project's Python:

    ../../.venv/bin/python make_candidates.py      # writes data/candidates_2021_2023.csv.gz
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
sys.path.insert(0, str(MAIN / "src"))
from alphas import AlphaEngine               # noqa: E402

FAMILIES = ("Momentum / trend", "Short-term reversal")
START, END = "2020-12-01", "2023-12-31"      # a little warm-up before the first trade day
OUT = HERE / "data" / "candidates_2021_2023.csv.gz"


def main():
    catalog = pd.read_csv(MAIN / "alpha_library" / "alpha_catalog.csv").set_index("id")
    card = pd.read_csv(MAIN / "data" / "eval" / "scorecard.csv", index_col="id")
    pool = [a for a in catalog.index[catalog["category"].isin(FAMILIES)] if a in card.index]
    print(f"{len(pool)} candidates")
    eng = AlphaEngine.load()
    columns, t0 = {}, time.time()
    for i, alpha in enumerate(pool, 1):
        columns[alpha] = (eng.compute(alpha).loc[START:END] * int(card.at[alpha, "direction"])).stack()
        if i % 20 == 0:
            print(f"  {i} computed ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(columns).dropna(how="all")
    out.index.names = ["date", "symbol"]
    assert out.index.get_level_values("date").max() <= pd.Timestamp(END)
    out.to_csv(OUT, float_format="%.6g", date_format="%Y-%m-%d")
    families = catalog.loc[pool, "category"].rename("family")
    families.to_csv(HERE / "data" / "candidate_families.csv")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MAIN,
                            capture_output=True, text=True).stdout.strip()
    print(f"Saved {OUT.relative_to(HERE)}: {len(out):,} rows x {out.shape[1]} alphas, "
          f"last date {out.index.get_level_values('date').max():%Y-%m-%d}, main commit {commit}")


if __name__ == "__main__":
    main()
