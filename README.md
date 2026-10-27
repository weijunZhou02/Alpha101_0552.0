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
7. **Post-mortems may suggest ideas, never choose them.** Studying a bad period (for example 2024) can give guesses. Each guess becomes a rule with round numbers fixed in advance. It is still chosen on 2021-2023 and judged on new days. Never tune a rule's numbers to the shape of the bad period.

## How each step is recorded

Every step gets two commits, both pushed to GitHub:
1. **Declare**: the plan is written here before anything runs.
2. **Results**: the numbers and the verdict are added to the research log below.

## Research log

### Stage 0: facts and post-mortem (declared 2026-10-27)

Describes only. Nothing is tested, nothing is chosen, and the strategy does not change.

The script `stage0.py` re-runs the original backtest on the frozen data and saves tables to `results/stage0/`.

**Periods**, fixed from the after-cost drawdown dates of the original backtest:

| Name | Trade days | Role |
| --- | --- | --- |
| Train | 2021-03-29 to 2023-12-31 | Normal life (0A) |
| Bad 1 | 2023-11-18 to 2024-10-22 | Drawdown of −35.6% (peak 2023-11-17) |
| Bad 2 | 2025-07-22 to 2026-06-07 | Drawdown of −34.5% (peak 2025-07-21) |
| Other | 2024-01-01 to 2026-10-24, outside Bad 1 and Bad 2 | Recoveries, information only |

Bad 1 overlaps the last 6 weeks of Train.

**What is measured in each period:**
1. Where the money went: long side and short side (each against the market average), fees, funding by side, rank IC and turnover.
2. Which coins: profit by volatility group, liquidity group and coin age, for each side. Also the 10 coins that lost the most, and their share of the loss.
3. Market conditions: market trend and volatility, BTC against altcoins, how spread out coin returns are, and average funding.
4. The 20 worst days in the whole history: what moved, and which single coins hurt most.
5. How unusual the drawdowns were:
   - the worst 12-month stretch inside Train, against 2024;
   - how deep drawdowns get when Train's daily results are reshuffled in 30-day blocks (2,000 paths).

**Output:** tables below, plus a list of guesses tagged "born from bad periods". These guesses can become rules for stages 1-4 (rule 7).

### Stages 1-4: draft plan (to be finished and approved after stage 0)

Each stage is compared with the best version so far, starting from (a), today's version.

Main cost model: 5 bp fee plus slippage by liquidity group (2 / 3 / 5 bp for coins ranked 1-20 / 21-50 / 51-80 by volume). Also shown: flat 5, 7, 10 and 12 bp, and maker 2 bp as a "what if".

| Stage | Versions |
| --- | --- |
| 1. Trade less | (b) average of the last 3 days' scores · (c) 3 overlapping baskets, each held 3 days |
| 2. Prefilter (signal on the top 80; filter only which coins can be traded; refill each side to 16) | (f) only the 50 most-traded coins · (g) leave out the wildest 20% · (h) funding guard: no short below −0.1%/day, no long above +0.1%/day |
| 3. Sizing | (d) 1/volatility weights · (j) volatility targeting, 0.5x-1.5x · (k) half size while 15%+ below the peak |
| 4. Helper alpha | (l) 0.7 × A101_055 + 0.3 × CRYPTO_HMRW_PREMIUM · (m) 0.7 × A101_055 + 0.3 × GTJA_095 |

- **Rule A** (return tools: b, c, f, h, l, m):
  - a higher Sharpe after costs on 2021-2023;
  - t ≥ 2 on the day-by-day difference;
  - better in at least 2 of the 3 years.
- **Rule B** (risk tools: g, d, j, k):
  - a smaller worst drawdown **and** smaller losses on the worst 1% of days;
  - Sharpe not clearly worse (t of the difference above −1).
