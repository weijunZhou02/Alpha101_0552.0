#!/usr/bin/env python3
"""Download hourly candles for stage 6 (sell earlier in the day).

Only the coins the (mix) holds, and only for the months it holds them: every
coin in the mix's BUY or SHORT list on a trade day from 2021-03-29 to
2026-10-24, for that trade day's month.

Source (public, no API key): https://data.binance.vision monthly 1h kline
files. If a monthly file is missing, the daily 1h files of that month are used.
Zips are cached in data_hourly/raw and never downloaded twice. Every run
rebuilds data_hourly/klines_1h.parquet (symbol, open_time, open, close) from
that cache and lists what could not be found in data_hourly/missing.csv.

    .venv/bin/python get_hourly.py
"""

import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import backtest as bt
import experiment as ex
import get_history as gh
import stage5_mix as s5

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_hourly"
RAW = OUT / "raw"
FIRST, LAST = "2021-03-29", "2026-10-24"
THREADS = 16


def mix_lists_by_day():
    """{signal day: (buy, short) or None} for the (mix), exactly as stage 5 builds them."""
    panels, funding = bt.load_data(HERE / "data")
    days = ex.original_lists(panels)
    f7, mom = s5.mix_panels(panels, funding)
    return {day: (None if table is None else s5.mix_lists(table, f7, mom, day))
            for day, table, *_ in days}


def needed(lists):
    """Sorted (symbol, 'YYYY-MM') pairs held on trade days FIRST..LAST."""
    pairs = set()
    for day, l in lists.items():
        trade = day + pd.Timedelta(days=1)
        if l is None or not (pd.Timestamp(FIRST) <= trade <= pd.Timestamp(LAST)):
            continue
        for s in l[0] + l[1]:
            pairs.add((s, f"{trade:%Y-%m}"))
    return sorted(pairs)


def fetch(pair):
    """Keys of the cached zips for one coin-month (monthly file, else daily files)."""
    symbol, month = pair
    key = f"{gh.UM}/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    try:
        gh.download(key, RAW)
        return pair, [key]
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
    daily = gh.zip_keys(f"{gh.UM}/daily/klines/{symbol}/1h/{symbol}-1h-{month}")
    for k in daily:
        gh.download(k, RAW)
    return pair, daily


def main():
    print("Building the (mix) lists ...")
    pairs = needed(mix_lists_by_day())
    print(f"{len(pairs)} coin-months, {len({s for s, _ in pairs})} coins")
    keys, missing = [], []
    with ThreadPoolExecutor(THREADS) as pool:
        for i, (pair, found) in enumerate(pool.map(fetch, pairs), 1):
            keys += found
            if not found:
                missing.append(pair)
            if sys.stderr.isatty() and (i % 100 == 0 or i == len(pairs)):
                print(f"\r  {i}/{len(pairs)}", end="", file=sys.stderr)
    if sys.stderr.isatty():
        print(file=sys.stderr)

    rows = []
    for key in keys:
        symbol = key.split("/")[5]
        for r in gh.zip_rows(RAW / key):
            rows.append((symbol, gh.to_ms(r[0]), float(r[1]), float(r[4])))
    k = (pd.DataFrame(rows, columns=["symbol", "open_time", "open", "close"])
         .drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"]))
    OUT.mkdir(parents=True, exist_ok=True)
    k.to_parquet(OUT / "klines_1h.parquet", index=False)
    pd.DataFrame(missing, columns=["symbol", "month"]).to_csv(OUT / "missing.csv", index=False)
    print(f"Saved {len(k):,} hourly candles to {(OUT / 'klines_1h.parquet').relative_to(HERE)}; "
          f"{len(missing)} coin-months not found (data_hourly/missing.csv)")


if __name__ == "__main__":
    main()
