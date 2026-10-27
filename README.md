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

## Files

| File | What it is |
| --- | --- |
| `a101_055.py`, `backtest.py`, `get_history.py` | Unchanged copies of the original scripts |
| `check_baseline.py` | Confirms the lab still reproduces the original backtest |
| `stage0.py` | Stage 0: facts and post-mortem, after costs (writes `results/stage0/`) |
| `stage0b.py` | Stage 0b: post-mortem on the before-cost line (writes `results/stage0b/`) |
| `baseline/` | The original backtest's summary, daily results, a live list, and data checksums |
| `results/` | Tables from each stage (kept in git) |

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

#### Stage 0 results (run 2026-10-27)

The re-run equals the original backtest on all 2,036 days. All tables are in `results/stage0/`.

**1. Where the money went** (bp a day; each side measured against the market average)

| | Train | Bad 1 | Bad 2 | Other |
| --- | --- | --- | --- | --- |
| Days | 1,008 | 340 | 321 | 411 |
| After costs, compounded | +58% | −36% | −35% | +102% |
| Before costs, compounded | +177% | −22% | −9% | +169% |
| Sharpe after costs | 1.10 | −2.54 | −1.45 | 2.69 |
| Long side vs market | +4.2 | −3.1 | −0.8 | +10.2 |
| Short side vs market | +6.3 | −3.7 | −0.9 | +14.8 |
| Fees + slippage | −5.5 | −5.5 | −5.5 | −5.5 |
| Funding (long + short) | 0.0 | −0.3 | **−4.8** (long +4.1, short −8.9) | −1.6 |
| **After costs** | **+4.9** | **−12.5** | **−11.9** | **+17.9** |
| Rank IC (t) | 0.041 (10.6) | 0.017 (2.4) | 0.030 (4.2) | 0.051 (8.7) |

- **Neither drawdown was a crash.** The edge faded (Bad 1: both sides lost a little; Bad 2: both sides about zero) while fees kept costing 5.5 bp every day. Zero edge plus fees is about −20% a year.
- **The ranking never broke.** The IC stayed positive in both bad periods. But the 16 + 16 coins at the two ends didn't pay.
- **Bad 2 added funding:** the shorts paid 8.9 bp a day.

**2. Which coins** (bp a day of the book, price against the market plus funding)

| Group | Train short | Bad 1 short | Bad 2 short | Other short |
| --- | --- | --- | --- | --- |
| Calm third | +1.7 | −0.9 | +0.4 | −1.1 |
| Middle third | +2.6 | +2.2 | +0.2 | +1.8 |
| **Wild third** | +2.6 | −3.4 | **−10.3** (funding −7.9) | **+10.3** |

- **Wild-coin shorts swing the whole result.** In Bad 2 they lost 10.3 bp a day, mostly funding. In the recoveries they earned 10.3.
- **Bad 2 losses were concentrated.** The 10 worst coins cost 43% of equity, more than the whole period's loss (38%, simple sums). FHE cost 8.5% and POWER 6.8%, both recently listed hype coins.
- **Bad 1 was a broad bleed.** The 10 worst coins (SUI, RUNE, ENS, WIF, BONK, …) cost 29% of a 42% loss. On the long side, coins listed over a year lost 5.9 bp a day.
- **Less liquid coins hurt in both bad periods** (places 51-80 by volume). But they also earned well in Train and in the recoveries.

**3. Market conditions**

| | Train | Bad 1 | Bad 2 | Other |
| --- | --- | --- | --- | --- |
| Market (top-80 average) bp/day | +3.2 | +12.1 | −27.7 | +0.7 |
| Market volatility, % a year | 92 | **77** | **73** | 81 |
| BTC minus market bp/day | −0.7 | +9.6 | +11.2 | +21.3 |
| Spread of coin returns (daily std) | 4.2% | 4.3% | 7.0% | 6.2% |
| Coins with funding < −0.1%/day | 4.9% | 2.0% | **12.9%** | 7.2% |
| Coins with funding > +0.1%/day | 7.2% | 11.8% | 4.3% | 4.1% |

- Bad 1 was a calm, BTC-led rally with positive funding. Bad 2 was an altcoin fall full of crowded shorts (negative funding).
- So no single market condition explains both. "BTC beat altcoins" happened even more in the recoveries.
- The one thing both bad periods share: **the market was calmer** than in Train and in the recoveries.

**4. The 20 worst days**
- In 18 of the 20, the coin that hurt most was a **short that jumped**, by +13% to +123% in a day, often costing 2-4% of equity by itself.
- 11 of the 20 are in Bad 2.
- The worst day: −4.9% on 2026-02-11, when BERA (short) rose +89%.

**5. How unusual were the drawdowns?**

| | Worst 12 months inside Train | Calendar 2024 |
| --- | --- | --- |
| After costs | −3% | −29% |
| Before costs | +18% | −13% |
| Rank IC (12-month average) | 0.034 (lowest) | 0.021 |

Reshuffling the daily results in 30-day blocks gives 2,000 paths of 2,036 days each:

| | Train days only | All days (information) |
| --- | --- | --- |
| Typical (median) worst drawdown | −17% | −36% |
| 1 path in 10 is worse than | −24% | −53% |
| Paths with a drawdown as deep as −35.6% | **0.5%** | 50% |

- **2021-2023 badly understates the risk.** If the future looked like Train, a −35% drawdown would happen in 1 path out of 200. It happened twice in 5.6 years.
- Train is also the period A101_055 was picked on, so it flatters the alpha.
- **For planning the money:** expect drawdowns of 35-55% with the current version. Position size has to survive that.

#### Guesses born from bad periods (rule 7: test on 2021-2023 and new days, never tune to 2024-2026)

| # | Guess | Evidence | Possible rule (round numbers) |
| --- | --- | --- | --- |
| G1 | Short squeezes cause most of the worst days | 18 of the 20 worst days; Bad 2's top-10 losers | Stop-loss on each short at +30% within the day (daily highs allow a rough test). Also (g) and (d) |
| G2 | Funding on wild shorts | Bad 2: −7.9 of −10.3 bp a day | Funding guard (h), already planned |
| G3 | The edge is weaker in calm markets | Market volatility 77% and 73% in the bad periods vs 92% in Train | First a fact check inside Train: do calmer months earn less? This matters for (j): volatility targeting makes the book *bigger* in calm markets, which this guess says is the wrong way |
| G4 | Fees cause the bleed when there is no edge | 5.5 bp a day in every period | Stage 1 (trade less), already planned |

### Owner's decision (2026-10-27): judge before costs from now on

- **All comparisons from now on use the before-cost line**: no fees and no funding, the grey dashed line on the chart.
- **Every table still shows turnover**, so we can see if a version wins only because it trades more.
- **Stage 0's tables above are after costs.** Its periods came from the after-cost line. That line made Bad 2 eleven months long, covering months where the signal was fine. As a result, it hid a sharp failure from 2025-10 to 2026-01.

### Stage 0b: post-mortem on the before-cost line (declared 2026-10-27)

Describes only. Nothing is tested or chosen. The script `stage0b.py` writes its tables to `results/stage0b/`.

**Periods**, from the before-cost drawdowns of the original backtest:

| Name | Trade days | Before-cost drawdown |
| --- | --- | --- |
| Train | 2021-03-29 to 2023-12-31 | Normal life |
| Box 1 | 2024-01-02 to 2024-10-17 | −23.7% (peak 2024-01-01) |
| Box 2 | 2025-10-01 to 2026-01-23 | The 10 October crash (−12.5%), then −21.4% from 2025-11-09 |
| Dip 3 | 2026-04-24 to 2026-06-07 | −16.9% (peak 2026-04-23), not circled by the owner; information |
| Other | 2024-01-01 to 2026-10-24, outside the three above | The rest of 2024-2026 |

**Measured**, all before costs:
1. Long and short side (each against the market average), rank IC and turnover.
2. **New: did the whole ranking flip, or only the two ends?** For each period, the average next-day return (against the market) of the 5 groups of 16 coins, from best score to worst.
3. Which coins: volatility, liquidity and age groups, and the biggest losers.
4. Market conditions.
5. The 20 worst days before costs.
6. How unusual: the worst 12 months inside Train, and how deep before-cost drawdowns get when Train's days are reshuffled in 30-day blocks.

#### Stage 0b results (run 2026-10-27, before costs)

The re-run equals the original backtest. All tables are in `results/stage0b/`.

**1. The periods** (bp a day; each side measured against the market average)

| | Train | Box 1 | Box 2 | Dip 3 | Other |
| --- | --- | --- | --- | --- | --- |
| Days | 1,008 | 290 | 115 | 45 | 578 |
| Before costs, compounded | +177% | −24% | −16% | −17% | +256% |
| Sharpe before costs | 2.34 | −1.81 | −1.69 | −4.16 | 3.34 |
| Long side vs market | +4.2 | −3.8 | −6.3 | −13.7 | +9.2 |
| Short side vs market | +6.3 | −5.1 | −8.0 | −25.7 | +13.7 |
| Rank IC (t) | 0.041 (10.6) | 0.017 (2.0) | 0.017 (1.8) | 0.000 (−0.2) | 0.051 (9.8) |
| Turnover, % a day | 79 | 78 | 79 | 79 | 78 |

**2. Did the whole ranking flip?** (next-day return against the market, bp a day, 5 groups of 16 coins)

| | Train | Box 1 | Box 2 | Dip 3 | Other |
| --- | --- | --- | --- | --- | --- |
| Q1, best scores (≈ BUY list) | +7.2 | −4.6 | −10.5 | −33.4 | +20.2 |
| Q2 | +3.7 | +2.6 | +1.7 | −16.3 | +2.5 |
| Q3 | +2.0 | −8.6 | −13.1 | −9.0 | +1.4 |
| Q4 | +1.6 | −2.2 | +9.4 | +29.9 | +4.6 |
| Q5, worst scores (≈ SHORT list) | **−13.6** | **+12.7** | **+14.0** | **+24.4** | **−27.6** |

- **The worst-score group is the swing.** These are coins whose price rose to the top of its 12-day range together with a rush of volume. Normally they fall the next day: −13.6 bp a day in Train and −27.6 in the rest of 2024-2026. Most of the alpha's edge comes from here.
- **In every bad window, those coins kept rising instead** (+12.7 to +24.4). The pumps continued rather than fading.
- In Box 1 and Box 2 the two ends flipped while the middle was mixed, so the IC stayed slightly positive. In Dip 3 the whole ranking flipped.

**3. Which coins**
- **Less-traded coins carried the losses.** In Box 2 and Dip 3 the losses came from coins ranked 21-80 by volume. The 20 most-traded coins made money on both sides in Box 2 (+2.9 long, +4.4 short) and on the long side in Dip 3.
- **Wild coins as well:** shorts of the wildest third lost 5.4 bp a day in Box 1, 6.3 in Box 2 and 28.5 in Dip 3.
- **Concentration:**
  - Box 2: the 10 worst coins together lost 31% of equity (FHE alone 7.4%), against −16% for the whole window (simple sums).
  - Box 1: 27% against −26%.

**4. Market conditions**

| | Train | Box 1 | Box 2 | Dip 3 | Other |
| --- | --- | --- | --- | --- | --- |
| Market (top-80 average) bp/day | +3.2 | −0.1 | −39.9 | −21.9 | −3.6 |
| Market volatility, % a year | 92 | 79 | 82 | 57 | 78 |
| BTC minus market bp/day | −0.7 | +18.5 | +21.2 | −23.2 | +18.1 |
| Spread of coin returns (daily std) | 4.2% | 4.2% | 7.1% | 7.3% | 6.4% |
| Coins with funding < −0.1%/day | 4.9% | 2.1% | 14.0% | 7.4% | 9.0% |

- No market condition separates the bad windows from the good months of 2024-2026 ("Other").
- BTC beat altcoins just as much in Other. Market volatility in Box 1 and Box 2 was the same as in Other.
- **Guess G3 from stage 0 ("calm markets are bad") is not supported on the before-cost line. It is dropped.**

**5. The 20 worst days before costs:** 17 were a shorted coin jumping, from +13% to +123% in a day. 10 of the 20 fall inside the three windows.

**6. How unusual**
- Train never had a losing 12-month stretch before costs; its worst was +18%. The worst in the whole history was −20%, the 12 months to 2024-11-16.
- Reshuffling Train's days gives a drawdown as deep as −23.7% in 1 path out of 1,000. Its typical worst drawdown is −11.5%.
- Using all days (information only), the typical worst drawdown is −20%, and 1 path in 10 is worse than −28%.
- **Plan for before-cost drawdowns of 20-30%**, not the 10-15% that 2021-2023 suggests.

#### Guesses after stage 0b (rule 7: test on 2021-2023 and new days, never tune to 2024-2026)

| # | Guess | Evidence | Possible rule (round numbers) |
| --- | --- | --- | --- |
| G1 | Short squeezes cause the worst days | 17 of the 20 worst days | Stop-loss on each short at +30% within the day; smaller bets on wild coins; leave out the wildest 20% |
| G5 | Less-traded coins carry the losses in bad windows | Box 2 and Dip 3: losses in coins ranked 21-80 by volume, while the top 20 made money | Only the 50 most-traded coins |
| G6 | Bad windows are times when pumps keep going | The SHORT-list group rose in every bad window | A drawdown brake. Watching the short list's own recent results is possible but easy to overfit |
| ~~G3~~ | ~~Calm markets are bad~~ | Not supported before costs | Dropped |
| G2, G4 | Funding, fees | — | Parked: costs are ignored for now |

### Stages 1-4: draft plan (written before the owner chose to ignore costs; to be redone after stage 0b)

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
