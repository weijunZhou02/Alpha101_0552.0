#!/usr/bin/env python3
"""Export the two stage 3 shortlisted helpers for all years (the shortlist is already fixed).

Run with the main project's Python:

    ../../.venv/bin/python make_shortlist_values.py    # writes data/shortlist_values.csv.gz

Each helper is multiplied by its direction from the main project's 2021-2023
scorecard (higher = expected to do better). The 2021-2023 part must equal the
screening file data/candidates_2021_2023.csv.gz; the script checks this.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
sys.path.insert(0, str(MAIN / "src"))
from alphas import AlphaEngine               # noqa: E402

SHORTLIST = ("Q158_MIN10", "GTJA_034")
OUT = HERE / "data" / "shortlist_values.csv.gz"


def main():
    card = pd.read_csv(MAIN / "data" / "eval" / "scorecard.csv", index_col="id")
    eng = AlphaEngine.load()
    out = pd.DataFrame({a: (eng.compute(a) * int(card.at[a, "direction"])).stack() for a in SHORTLIST})
    out = out.dropna(how="all")
    out.index.names = ["date", "symbol"]

    screened = pd.read_csv(HERE / "data" / "candidates_2021_2023.csv.gz", parse_dates=["date"],
                           usecols=["date", "symbol", *SHORTLIST]).set_index(["date", "symbol"])[list(SHORTLIST)]
    both = out.reindex(screened.index)
    gap = np.nanmax(np.abs(both.to_numpy() - screened.to_numpy()) /
                    np.maximum(np.abs(screened.to_numpy()), 1e-12))
    assert gap < 1e-5, f"2021-2023 values differ from the screening file (relative gap {gap:.2g})"
    out.to_csv(OUT, float_format="%.6g", date_format="%Y-%m-%d")
    dates = out.index.get_level_values("date")
    print(f"Saved {OUT.relative_to(HERE)}: {len(out):,} rows, {dates.min():%Y-%m-%d} to "
          f"{dates.max():%Y-%m-%d}; 2021-2023 matches the screening file (relative gap {gap:.1g}).")


if __name__ == "__main__":
    main()
