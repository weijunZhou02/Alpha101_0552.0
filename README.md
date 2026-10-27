# A101_055 lab

A research lab for alpha **A101_055**: make it more robust and cheaper to trade, analyse it, and try combining it with other alphas.
No trading happens here: no orders, no API keys.

```
A101_055 = -1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))),
                            rank(volume), 6)
```

## Where it came from

- `a101_055.py`, `backtest.py` and `get_history.py` are unchanged copies from the A101_055 repo (`github.com/weijunZhou02/A101_055`, commit `f0cce0a`, 2026-10-26).
- The live demo trading, paper trading, dashboard and schedule stay in that repo.
- Data: a frozen snapshot of Binance daily candles and funding, delisted coins included, up to **2026-10-24**. It is `data/klines_1d.csv.gz` and `data/funding.csv.gz`, 27 MB, not in git. Their checksums are in `baseline/data_sha256.txt`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p data && cp ../A101_055-main/data/*.csv.gz data/
.venv/bin/python check_baseline.py
```

## The baseline check

`check_baseline.py` makes sure the lab still gives the original numbers. It:
1. checks that the data is the frozen snapshot,
2. runs `backtest.py`,
3. compares the summary and every day's result with the original (`baseline/`).

It must say **PASS** before any experiment is trusted. Run it again after changing `a101_055.py` or `backtest.py`.

The original numbers (2021-03-29 to 2026-10-24, 7 bp per unit traded):

| | |
| --- | --- |
| Rank IC, next day | +0.0379 (t 13.6) |
| Sharpe before / after costs | 1.52 / 0.36 |
| Before costs / after costs | +8.7 / +2.1 bp a day |
| Turnover | 79% of equity a day |
| Break-even cost | 9.7 bp per unit traded |

## Rules that keep the results honest

Every year from 2021 to 2026 has already been seen, so:
1. **Choose only on 2021-03-29 to 2023-12-31.** Show 2024 and 2025-01-01 to 2026-10-24 as information only, never to pick anything.
2. **Write down the versions to try before running (about 6 at most) and report all of them.**
3. **"Better"** = a higher Sharpe after costs than the baseline on 2021-2023 **and** t ≥ 2 on the day-by-day difference.
4. **The real test is days after 2026-10-24.** Download them into another folder (`get_history.py --datadir data_fresh`) so the snapshot stays frozen. Declare the test before looking.
5. **Improvements go back to the A101_055 repo by hand**, and only after they pass.
6. **No API keys in this folder.**

## Next: agreed step 1, a cost check and a fairer backtest (not built yet)

Agreed on 2026-10-26, to be built later:
- Fee as a setting (taker 5 bp, maker 2 bp) plus slippage per coin. Also show the result after costs at 5, 7, 10 and 12 bp.
- Declared versions:
  - (a) today's version
  - (b) 3-day score average
  - (c) 3 overlapping baskets, each held 3 days
  - (d) 1 / volatility sizing
  - (e) (c) and (d) together
- Pass rule as in rule 3.
