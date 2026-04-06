# Trade 1: Q5 Momentum — Full Strategy Specification

**Purpose:** This document contains everything needed to independently replicate the backtest. If you follow every step exactly, you should get 2,875 trades with total P&L of +$35,813,058 over the 2020-2026 period.

---

## 1. Universe

**230 symbols.** The complete list is in `data/manifest.csv`. Use every symbol listed there with `status=ok`.

The universe includes:
- Mega-cap tech (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX, etc.)
- Semiconductors (AMD, AVGO, QCOM, SMCI, ARM, etc.)
- Software/Cloud (CRM, PLTR, CRWD, SNOW, ZM, etc.)
- EV/Clean energy (RIVN, NIO, ENPH, FSLR, etc.)
- Biotech/Pharma (LLY, MRNA, PFE, REGN, etc.)
- Financials (JPM, GS, V, MA, PYPL, etc.)
- Crypto-adjacent (COIN, MSTR, MARA, RIOT, HUT)
- Consumer (WMT, COST, UBER, ABNB, DASH, etc.)
- Energy (XOM, OXY, COP, DVN, etc.)
- Industrials/Defense (BA, LMT, CAT, GE, etc.)
- China ADRs (BABA, PDD, JD, BIDU, etc.)
- Meme/Momentum (GME, AMC, CVNA, IONQ, etc.)
- Airlines/Travel (DAL, UAL, CCL, RCL, etc.)
- Key ETFs (SPY, QQQ, IWM, XLE, XLF, SMH, ARKK, TQQQ, VXX, etc.)

**DO NOT filter by market cap, volume, or any other criteria.** Use all 230 symbols as-is for the entire period. Some symbols IPO'd after 2015 — they simply have fewer data points. That is correct.

---

## 2. Data

**Source:** Polygon.io
**Timeframe:** Daily bars
**Period:** 2015-01-01 through 2026-04-03
**Fields used:** date, open, high, low, close (split-adjusted)

The data files are at `data/{SYMBOL}_daily.csv`. Each row is one trading day.

**IMPORTANT:** The data is split-adjusted. Do not apply additional adjustments.

---

## 3. Signal Construction

The signal determines WHEN to buy. It fires on the close of a trading day. You enter the next morning.

### Step 3a: Compute the 50-day Simple Moving Average

For each symbol on each day:

```
SMA50[t] = mean(close[t], close[t-1], ..., close[t-49])
```

This requires 50 days of closing prices. The first valid SMA50 for a symbol is on its 50th trading day.

### Step 3b: Compute the 10-day slope of the SMA50

```
SMA50_slope[t] = 100 × (SMA50[t] / SMA50[t-10] - 1)
```

This is the percent change in the SMA50 over the last 10 trading days. Positive = SMA is rising. Negative = SMA is falling.

This requires 10 additional days beyond the SMA50 warmup, so the first valid slope value is on the 60th trading day.

### Step 3c: Z-score the slope within each symbol

For each symbol independently:

```
rolling_mean[t] = mean(SMA50_slope[t-1], SMA50_slope[t-2], ..., SMA50_slope[t-252])
rolling_std[t]  = std(SMA50_slope[t-1], SMA50_slope[t-2], ..., SMA50_slope[t-252])
z_signal[t]     = (SMA50_slope[t] - rolling_mean[t]) / rolling_std[t]
```

- The rolling window is **252 trading days** (1 year).
- Minimum periods to start computing: **60 days**.
- The z-score measures how extreme the current slope is relative to THIS SYMBOL'S OWN HISTORY. A z-score of +2 means the SMA50 is rising much faster than usual for this stock.

**CRITICAL:** The z-score is computed per-symbol, NOT across symbols. Each symbol has its own mean and std.

### Step 3d: Cross-sectional quintile rank

On each trading day, take all symbols that have a valid `z_signal` value on that day (not NaN). Rank them into 5 equal-sized groups (quintiles) by z_signal value:

- **Q1:** Lowest z_signal (SMA50 declining fastest relative to history)
- **Q2**
- **Q3**
- **Q4**
- **Q5:** Highest z_signal (SMA50 rising fastest relative to history)

Use `pandas.qcut(z_signal, 5, labels=[1,2,3,4,5])` or equivalent. If there are ties that prevent clean quintile splits, use `duplicates='drop'` and skip that day.

**Minimum symbols required:** If fewer than 20 symbols have valid z_signal on a given day, skip that day entirely (no signal, no trades).

### Step 3e: Identify Q5 entries (the trade signal)

A **Q5 entry** occurs when:
1. A symbol is in Q5 today, AND
2. That same symbol was NOT in Q5 yesterday

This means the stock's SMA50 slope just crossed into the top quintile — it just started accelerating faster than usual.

To compute this:
- Sort the data by (symbol, date)
- For each symbol, track `prev_quintile = quintile on the previous trading day`
- `entered_q5 = (quintile == 5) AND (prev_quintile != 5)`

**The first day a symbol appears in the data has no prev_quintile. Do not generate a signal on the first day.**

---

## 4. Entry

When `entered_q5 = True` on the close of day T:

- **Entry day:** T+1 (the next trading day)
- **Entry price:** The OPEN price of day T+1
- **Position size:** $1,000,000 notional. Compute shares as `int($1,000,000 / open_price)`. Round down to whole shares.
- **Entry cost:** 5 basis points of notional. This is `shares × open_price × 5/10000`.

**If the open price on T+1 is not available** (symbol not trading, or open is NaN/zero), **skip this trade.** Do not fill at any other price.

**Multiple entries on the same day are allowed.** If 5 different symbols enter Q5 on the same day, you open 5 positions, each at $1M.

**You CAN have multiple positions in the same symbol** if it enters Q5, exits, and re-enters on a later date. Each is a separate trade.

---

## 5. Exit

There is exactly ONE exit condition:

- **Time-based exit at the close of day T+20** (20 trading days after entry).
- **Exit price:** The CLOSE price on the 20th trading day after entry.
- **Exit cost:** 5 basis points of notional. This is `shares × close_price × 5/10000`.

**There is no stop loss.**
**There is no take profit.**
**There is no signal-based exit.**
**There is no early exit of any kind.**

You hold for exactly 20 trading days and exit at the close. Period.

If the 20th trading day does not exist (symbol delisted, data ends), exit at the close of the last available trading day for that symbol.

---

## 6. P&L Calculation

For each trade:

```
gross_pnl = shares × (exit_price - entry_price)
entry_cost = shares × entry_price × 5 / 10000
exit_cost = shares × exit_price × 5 / 10000
net_pnl = gross_pnl - entry_cost - exit_cost
pnl_bps = 10000 × (exit_price / entry_price - 1)
```

---

## 7. Backtest Period

**Signal generation starts:** When z_signal has sufficient warmup. In practice this means:
- 50 days for SMA50
- 10 more days for slope
- 60 more days for z-score rolling window (with min_periods=60)
- Total: ~120 trading days of warmup per symbol

**Trade entry starts:** 2020-01-02 (first possible entry date). Signals from 2015-2019 are used only for warmup — no trades taken.

**Trade entry ends:** 2026-04-01 (last date in dataset). Trades entered near the end may not have a full 20-day hold — exit at the last available close.

---

## 8. Expected Results

If you replicate this correctly, you should get approximately:

| Metric | Value |
|---|---|
| Total trades | ~2,875 |
| Total net P&L | ~+$35,813,058 |
| Average P&L per trade | ~+$12,457 |
| Win rate | ~53.5% |
| Profit factor | ~1.29 |
| Max drawdown | ~$14,969,949 |
| Average trades per day | ~2.0 |
| Max concurrent positions | Varies (no limit) |
| Worst single trade | ~-$669,356 |

### Year-by-year:

| Year | Trades | Net P&L | WR | Positive? |
|---|---|---|---|---|
| 2020 | 438 | +$16,710,663 | 58% | YES |
| 2021 | 413 | +$3,686,091 | 54% | YES |
| 2022 | 461 | -$8,068,201 | 45% | **NO** |
| 2023 | 505 | +$9,061,326 | 53% | YES |
| 2024 | 478 | +$5,806,012 | 57% | YES |
| 2025 | 494 | +$7,073,201 | 55% | YES |
| 2026 | 86 | +$1,543,966 | 52% | YES |

---

## 9. Common Mistakes to Avoid

1. **Do not z-score across symbols.** The z-score must be computed within each symbol against its own 252-day rolling history. If you z-score across the cross-section, you will get different quintile assignments.

2. **Do not use the close as entry price.** The signal fires at the close of day T. Entry is at the OPEN of day T+1. Using close of day T as entry will inflate results because the signal uses the close price to compute the signal.

3. **Do not add a stop loss.** We tested stops at -75, -100, -150, -200, -300, -500 bps. Every stop level reduces total P&L. The -75 bps stop flipped the strategy from +$35M to -$526K. The momentum needs time to play out — stopping out on an early dip kills the trade before it works.

4. **Do not add a take profit.** Every take profit level tested reduced total P&L. The winners are large and right-skewed — capping them destroys the edge.

5. **Do not filter the universe.** Use all 230 symbols regardless of volume, market cap, or sector. Filtering introduces bias.

6. **Do not enter on the first day a symbol appears in quintile data.** There must be a valid `prev_quintile` to determine that the stock JUST entered Q5. The transition is the signal.

7. **The quintile assignment uses all symbols with valid z_signal on that day.** This means on early dates (2015-2016), some symbols won't have enough warmup and the quintile sort will use fewer names. That is correct.

8. **`pd.qcut` with `duplicates='drop'` may produce fewer than 5 quantile groups on some days.** If this happens, skip that day — do not assign quintiles or generate signals.

---

## 10. Data Files for Verification

- `analysis_results/trade1_1m_trades.csv` — Full trade log with every entry/exit
- `analysis_results/trade1_1m_daily.csv` — Daily portfolio P&L
- `analysis_results/trade1_all_configs_report.txt` — All configuration comparisons
- `data/manifest.csv` — Complete symbol list

---

## 11. What This Strategy IS and IS NOT

**What it is:** A momentum trade that buys stocks whose 50-day moving average is accelerating faster than its own historical norm, sized at $1M per entry, held for 20 trading days with no risk management at the individual position level.

**What it is not:** A hedged strategy, a market-neutral strategy, or a strategy with defined risk per trade. You are taking naked long exposure to individual stocks for 20 days. The worst single trade lost $669K on a $1M entry (67% loss). Diversification across ~2 concurrent entries per day is the only risk management.

**Regime risk:** 2022 lost $8M. In a sustained bear market, stocks entering Q5 (accelerating uptrend) reverse as the macro environment deteriorates. This strategy has no mechanism to avoid that drawdown.
