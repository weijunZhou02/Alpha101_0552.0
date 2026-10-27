#!/usr/bin/env python3
"""Baseline check: the untouched lab copy must reproduce the original A101_055 backtest.

Run it before trusting any experiment, and again after changing a shared file:

    .venv/bin/python check_baseline.py

It checks three things:
  1. the data in data/ is the frozen snapshot (SHA-256 in baseline/data_sha256.txt),
  2. backtest.py, run on that data, prints the same summary as the original
     (baseline/summary_main.txt), apart from the "Saved ..." lines,
  3. its day-by-day results equal the original's (baseline/daily_main.csv).

Prints PASS or FAIL and exits with 0 or 1.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE / "baseline"
OUT = HERE / "out" / "baseline"
LIVE_LIST = BASE / "a101_055_2026-10-24.csv"
TOLERANCE = 1e-9          # daily.csv is written to 8 significant digits


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_is_snapshot():
    ok = True
    for line in (BASE / "data_sha256.txt").read_text().splitlines():
        expected, name = line.split()
        path = HERE / name
        if not path.exists():
            print(f"  missing {name}: copy it from A101_055-main/data/")
            ok = False
        elif sha256(path) != expected:
            print(f"  {name} is not the frozen snapshot (it changed)")
            ok = False
    return ok


def summary_lines(path):
    """The summary without the lines that name output files."""
    return [line for line in path.read_text().splitlines()
            if not line.startswith(("Saved ", "matplotlib is not installed"))]


def main():
    print("1. Data is the frozen snapshot ...")
    if not data_is_snapshot():
        print("FAIL")
        return 1

    print("2. Running backtest.py on it (a few minutes) ...")
    subprocess.run([sys.executable, "backtest.py", "--outdir", str(OUT),
                    "--check", str(LIVE_LIST)], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL)
    ours, theirs = summary_lines(OUT / "summary.txt"), summary_lines(BASE / "summary_main.txt")
    same_summary = ours == theirs
    if not same_summary:
        for a, b in zip(theirs, ours):
            if a != b:
                print(f"  original: {a}\n  lab:      {b}")
        if len(ours) != len(theirs):
            print(f"  original has {len(theirs)} lines, lab has {len(ours)}")
    print(f"   summary: {'same' if same_summary else 'DIFFERENT'}")

    print("3. Comparing day-by-day results ...")
    a = pd.read_csv(BASE / "daily_main.csv", index_col=0)
    b = pd.read_csv(OUT / "daily.csv", index_col=0)
    same_shape = a.index.equals(b.index) and list(a.columns) == list(b.columns)
    numbers = a.select_dtypes("number").columns
    diff = (np.abs(a[numbers] - b[numbers]).max().max() if same_shape else np.inf)
    same_daily = same_shape and diff <= TOLERANCE and a.drop(columns=numbers).fillna("").equals(
        b.drop(columns=numbers).fillna(""))
    print(f"   {len(b)} days, largest difference {diff:.2g}: {'same' if same_daily else 'DIFFERENT'}")

    ok = same_summary and same_daily
    print("PASS: the lab reproduces the original backtest." if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
