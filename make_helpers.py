#!/usr/bin/env python3
"""Export the two helper alphas for stage 1 from the main Alpha-mining project.

Run it with the main project's Python (it needs that project's alpha engine and
data panels):

    ../../.venv/bin/python make_helpers.py

Writes data/helpers.csv.gz (date, symbol, CRYPTO_HMRW_PREMIUM, GTJA_095), each
multiplied by its direction from the main project's training scorecard, so
higher = better. Values are NaN outside the main project's daily top 80.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]                       # the Alpha-mining project
sys.path.insert(0, str(MAIN / "src"))
from alphas import AlphaEngine               # noqa: E402

HELPERS = ("CRYPTO_HMRW_PREMIUM", "GTJA_095")


def main():
    card = pd.read_csv(MAIN / "data" / "eval" / "scorecard.csv", index_col="id")
    eng = AlphaEngine.load()
    columns = {}
    for alpha in HELPERS:
        direction = int(card.at[alpha, "direction"])
        columns[alpha] = (eng.compute(alpha) * direction).stack()
        print(f"{alpha}: direction {direction:+d}, {columns[alpha].notna().sum():,} values")
    out = pd.DataFrame(columns).dropna(how="all")
    out.index.names = ["date", "symbol"]
    path = HERE / "data" / "helpers.csv.gz"
    out.to_csv(path, float_format="%.8g", date_format="%Y-%m-%d")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MAIN,
                            capture_output=True, text=True).stdout.strip()
    dates = out.index.get_level_values("date")
    print(f"Saved {path.relative_to(HERE)}: {len(out):,} rows, {dates.min():%Y-%m-%d} to "
          f"{dates.max():%Y-%m-%d}, main project commit {commit}")


if __name__ == "__main__":
    main()
