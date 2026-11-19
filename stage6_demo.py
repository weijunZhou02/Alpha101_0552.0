#!/usr/bin/env python3
"""Stage 6, the demo look (information only; declared in README, "Stage 6").

The same six versions as stage6_timing.py, on the demo account's full trade
days 2026-11-04 to 2026-11-18, from the demo's REAL positions:
  * positions after each run are rebuilt from A101_055-main/demo/orders.csv
    (read only; every executed quantity since the first run);
  * hourly prices and funding come from the demo market's public data
    (demo-fapi.binance.com, no API keys), cached in data_fresh/demo_hourly/;
  * fees use the demo's own average fee per dollar traded.
(h24) is also rebuilt this way and compared with the demo's real daily results
(ledger.csv). Fifteen days cannot prove anything.

    .venv/bin/python stage6_demo.py       # writes results/stage6/demo_*.csv
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "A101_055-main" / "demo"
CACHE = HERE / "data_fresh" / "demo_hourly"
OUT = HERE / "results" / "stage6"
DEMO_URL = "https://demo-fapi.binance.com"
USER_AGENT = "a101-055-lab/1.0 (research, public market data only)"
HOUR_MS = 3_600_000
HOURS = (4, 8, 12, 16, 20, 24)
FIRST_DAY, LAST_DAY = "2026-11-04", "2026-11-18"


def get(path, params):
    url = DEMO_URL + path + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as resp:
        time.sleep(0.2)
        return json.loads(resp.read())


def cached(name, fetch):
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fetch()))
    return json.loads(path.read_text())


def market(symbols, start_ms, end_ms):
    """Hourly closes (index = hour end, ms) and funding events [(symbol, time, rate, mark)]."""
    closes, opens, events = {}, {}, []
    tag = f"{start_ms}_{end_ms}"
    for s in symbols:
        k = cached(f"{s}_1h_{tag}.json", lambda: get("/fapi/v1/klines", {
            "symbol": s, "interval": "1h", "startTime": start_ms, "endTime": end_ms - 1, "limit": 1500}))
        closes[s] = pd.Series({int(r[0]) + HOUR_MS: float(r[4]) for r in k}, dtype=float)
        opens[s] = pd.Series({int(r[0]): float(r[1]) for r in k}, dtype=float)
        f = cached(f"{s}_funding_{tag}.json", lambda: get("/fapi/v1/fundingRate", {
            "symbol": s, "startTime": start_ms, "endTime": end_ms + 60_000, "limit": 1000}))
        events += [(s, int(e["fundingTime"]), float(e["fundingRate"]), float(e["markPrice"] or "nan"))
                   for e in f]
    return pd.DataFrame(closes), pd.DataFrame(opens), pd.DataFrame(
        events, columns=["symbol", "time", "rate", "mark"])


def positions_after_runs(orders, runs):
    """{run time: signed quantity per symbol} after each run's orders."""
    o = orders.copy()
    o["signed"] = np.where(o["side"] == "BUY", 1, -1) * pd.to_numeric(o["executed_qty"]).fillna(0)
    out, held = {}, pd.Series(dtype=float)
    for run in runs:
        delta = o[o["run_time_utc"] == run].groupby("symbol")["signed"].sum()
        held = held.add(delta, fill_value=0)
        out[run] = held[held.abs() > 1e-12].copy()
    return out


def main():
    ledger = pd.read_csv(DEMO / "ledger.csv")
    orders = pd.read_csv(DEMO / "orders.csv")
    runs = list(ledger["run_time_utc"])
    held = positions_after_runs(orders, runs)
    filled = orders[pd.to_numeric(orders["executed_qty"]).fillna(0) > 0]
    fee_rate = filled["fee"].sum() / (filled["executed_qty"] * filled["avg_price"]).sum()

    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
    start_ms = int(days[0].value // 1_000_000)
    end_ms = int((days[-1] + pd.Timedelta(days=1)).value // 1_000_000)
    symbols = sorted({s for d in days for r in runs if r.startswith(f"{d:%Y-%m-%d}") for s in held[r].index})
    closes, opens, events = market(symbols, start_ms, end_ms)
    closes = closes.reindex(start_ms + HOUR_MS * np.arange(1, len(days) * 24 + 1)).ffill()

    rows = []
    for d in days:
        run = next(r for r in runs if r.startswith(f"{d:%Y-%m-%d}"))
        after = runs[runs.index(run) + 1]                  # the run that closes this trade day
        q = held[run]
        d0 = int(d.value // 1_000_000)
        p0 = opens.loc[d0, q.index]
        actual = ledger.set_index("run_time_utc").loc[after]
        equity = float(ledger.set_index("run_time_utc").at[run, "equity_after"])
        row = {"trade_day": d.date(), "equity at 00:05": equity, "coins held": len(q),
               "gross held $": float((q.abs() * p0).sum()),
               "actual change $": float(actual["equity_change"]),
               "actual price $": float(actual["price_pnl"]), "actual funding $": float(actual["funding"]),
               "actual fees $": float(actual["commission"])}
        for h in HOURS:
            ph = closes.loc[d0 + h * HOUR_MS, q.index]
            e = events[events["symbol"].isin(q.index) & (events["time"] > d0 + 60_000)
                       & (events["time"] <= d0 + h * HOUR_MS + 60_000)]
            funding = float(-(e["rate"] * e["mark"] * e["symbol"].map(q)).sum())
            price = float((q * (ph - p0)).sum())
            if h == 24:
                fees = float(actual["commission"])          # (h24): the demo's real fees
            else:
                fees = -fee_rate * float((q.abs() * p0).sum() + (q.abs() * ph).sum())
            row[f"h{h} price $"], row[f"h{h} funding $"], row[f"h{h} fees $"] = price, funding, fees
            row[f"h{h} net $"] = price + funding + fees
        rows.append(row)
    t = pd.DataFrame(rows).set_index("trade_day")

    gap = t["h24 net $"] - t["actual change $"]
    summary = []
    for h in HOURS:
        n = f"h{h}"
        r = t[f"{n} net $"] / t["equity at 00:05"]
        diff = r - t["h24 net $"] / t["equity at 00:05"]
        summary.append({"version": n, "net $": t[f"{n} net $"].sum(),
                        "price $": t[f"{n} price $"].sum(), "funding $": t[f"{n} funding $"].sum(),
                        "fees $": t[f"{n} fees $"].sum(), "bp/day": r.mean() * 1e4,
                        "vs (h24) bp/day": diff.mean() * 1e4,
                        "t vs (h24)": diff.mean() / diff.std() * np.sqrt(len(diff)) if n != "h24" else np.nan})
    summary = pd.DataFrame(summary).set_index("version")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"Demo fee per dollar traded: {fee_rate * 1e4:.2f} bp; {len(days)} trade days, "
          f"{len(symbols)} coins")
    print(f"CHECK (h24) rebuild vs the demo's real daily change: total {t['h24 net $'].sum():.2f} vs "
          f"{t['actual change $'].sum():.2f} $; average daily gap {gap.abs().mean():.2f} $, "
          f"largest {gap.abs().max():.2f} $; daily correlation "
          f"{t['h24 net $'].corr(t['actual change $']):.3f}")
    print(f"  price {t['h24 price $'].sum():.2f} vs {t['actual price $'].sum():.2f}; "
          f"funding {t['h24 funding $'].sum():.2f} vs {t['actual funding $'].sum():.2f}")
    print("\nDEMO LOOK (information only), 15 trade days")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "demo_days.csv", float_format="%.6g")
    summary.to_csv(OUT / "demo_summary.csv", float_format="%.6g")
    print(f"\nSaved {OUT.relative_to(HERE)}/demo_days.csv, demo_summary.csv")


if __name__ == "__main__":
    main()
