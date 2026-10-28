#!/usr/bin/env python3
"""Export the 158 broad-list alphas for all years, for the stage 4 models.

Same alphas, directions and values as make_broad.py, but for every day in the main
project's data (the stage 4 models are trained walk-forward, so each prediction
only uses earlier rows). The 2021-2023 part must equal the screening file
data/broad_2021_2023.csv.gz; the script checks this.

Run with the main project's Python:

    ../../.venv/bin/python make_ml_inputs.py      # writes data/broad_all_years.parquet
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
sys.path.insert(0, str(MAIN / "src"))
from alphas import AlphaEngine               # noqa: E402

OUT = HERE / "data" / "broad_all_years.parquet"


def main():
    broad = pd.read_csv(MAIN / "data" / "eval" / "dedupe" / "broad_list.csv", index_col=0)
    card = pd.read_csv(MAIN / "data" / "eval" / "scorecard.csv", index_col="id")
    pool = [a for a in broad.index if a != "A101_055"]
    eng = AlphaEngine.load()
    columns, t0 = {}, time.time()
    for i, alpha in enumerate(pool, 1):
        columns[alpha] = (eng.compute(alpha) * int(card.at[alpha, "direction"])).stack()
        if i % 40 == 0:
            print(f"  {i} computed ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(columns).dropna(how="all")
    out.index.names = ["date", "symbol"]

    screened = pd.read_csv(HERE / "data" / "broad_2021_2023.csv.gz", parse_dates=["date"]) \
        .set_index(["date", "symbol"])[pool]
    both = out.reindex(screened.index)
    gap = np.nanmax(np.abs(both.to_numpy() - screened.to_numpy()) /
                    np.maximum(np.abs(screened.to_numpy()), 1e-12))
    assert gap < 1e-5, f"2021-2023 values differ from the screening file (relative gap {gap:.2g})"
    out.astype("float64").to_parquet(OUT)
    dates = out.index.get_level_values("date")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MAIN,
                            capture_output=True, text=True).stdout.strip()
    print(f"Saved {OUT.relative_to(HERE)}: {len(out):,} rows x {out.shape[1]} alphas, "
          f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}; 2021-2023 matches the screening file "
          f"(relative gap {gap:.1g}); main commit {commit}")


if __name__ == "__main__":
    main()
