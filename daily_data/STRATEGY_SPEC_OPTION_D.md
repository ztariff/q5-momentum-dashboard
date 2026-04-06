# Trade 1 Option D: Complete Technical Specification

**Purpose:** Everything needed to replicate this strategy to the exact trade. If any step is ambiguous, this document is wrong — flag it.

---

## STEP 1: Load Daily Data

**Source:** Polygon.io, split-adjusted daily bars.
**Fields required per symbol per day:** date, open, high, low, close
**Universe:** 230 symbols listed in `data/universe.csv`
**Date range:** 2015-01-01 through 2026-04-03

No filtering by volume, market cap, or any other criteria. Use every symbol for every day it has data.

---

## STEP 2: Compute the 50-Day Simple Moving Average (SMA50)

For each symbol independently, on each trading day:

```
SMA50[t] = (close[t] + close[t-1] + close[t-2] + ... + close[t-49]) / 50
```

This is a simple arithmetic mean of the last 50 closing prices including today.

- **Window:** 50 trading days (not calendar days)
- **Type:** Simple (equal weight), not exponential
- **First valid value:** The 50th trading day for that symbol
- **In pandas:** `df['close'].rolling(50).mean()`
- **NaN handling:** If fewer than 50 closes are available, SMA50 is NaN

---

## STEP 3: Compute the 10-Day Slope of SMA50

For each symbol independently, on each trading day:

```
SMA50_slope[t] = 100 × (SMA50[t] / SMA50[t-10] - 1)
```

This is the **percent change** in the SMA50 over the last 10 trading days.

- Positive value = the 50-day average is rising
- Negative value = the 50-day average is falling
- Larger absolute value = faster acceleration/deceleration
- **First valid value:** 10 trading days after the first valid SMA50 (i.e., day 60)
- **In pandas:** `100 * (sma50 / sma50.shift(10) - 1)`

**Example:** If SMA50 today is $152.30 and SMA50 ten days ago was $150.00:
```
slope = 100 × (152.30 / 150.00 - 1) = 100 × 0.01533 = +1.533
```
This means the 50-day average has risen 1.533% over the last 10 days.

---

## STEP 4: Z-Score the Slope Within Each Symbol

For each symbol independently, on each trading day:

```
rolling_mean[t] = mean of SMA50_slope over days [t-252, t-251, ..., t-1]
rolling_std[t]  = standard deviation of SMA50_slope over days [t-252, t-251, ..., t-1]
z_signal[t]     = (SMA50_slope[t] - rolling_mean[t]) / rolling_std[t]
```

**CRITICAL DETAILS:**
- The rolling window looks back **252 trading days** (approximately 1 year)
- The window uses **days t-1 through t-252** — it does NOT include today. This prevents look-ahead.
- **Minimum periods:** 60 days. If fewer than 60 prior slope values exist, z_signal is NaN.
- The z-score is computed **per symbol against its own history**, NOT across symbols.
- A z_signal of +2.0 means this stock's SMA50 is accelerating much faster than usual FOR THIS STOCK.

**In pandas:**
```python
rolling_mean = sma50_slope.rolling(252, min_periods=60).mean()
rolling_std  = sma50_slope.rolling(252, min_periods=60).std()
z_signal     = (sma50_slope - rolling_mean) / rolling_std
```

**NOTE:** pandas `.rolling(252).mean()` by default includes the current value in the window. Our implementation does NOT shift, so technically today's slope IS included in its own z-score denominator. For a fully clean implementation you would use `.shift(1)` before computing rolling stats. Our backtest does not do this shift — to replicate our exact results, do NOT shift.

---

## STEP 5: Cross-Sectional Quintile Rank (Daily)

On each trading day, take ALL symbols that have a valid (non-NaN) z_signal on that day.

**If fewer than 20 symbols have valid z_signal: skip this day entirely. No quintiles assigned, no signals.**

Otherwise, rank them into 5 equal-sized groups by z_signal value:

```
Q1 = lowest 20% of z_signal values (SMA50 slope declining fastest)
Q2 = next 20%
Q3 = middle 20%
Q4 = next 20%
Q5 = highest 20% of z_signal values (SMA50 slope rising fastest)
```

**In pandas:**
```python
pd.qcut(z_signal_values, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
```

**If `duplicates='drop'` results in fewer than 5 groups** (because too many identical z_signal values): skip this day. No quintiles, no signals.

**Quintile assignment is done FRESH every day.** A stock can be Q5 today and Q3 tomorrow. The boundaries change daily based on the cross-section.

---

## STEP 6: Identify Entry Signals (Q5 Transition)

Sort the data by (symbol, date). For each symbol, compare today's quintile to yesterday's quintile.

```
entered_q5[t] = (quintile[t] == 5) AND (quintile[t-1] != 5)
```

A signal fires when a stock is in Q5 today but was NOT in Q5 on the previous trading day.

**Edge cases:**
- The first day a symbol has a valid quintile has no previous quintile → no signal.
- If a stock was Q5, dropped to Q4 for one day, then returned to Q5: that's a new entry signal.
- A stock can only generate one entry signal per Q5 "stint." It signals on the first day of each stint.
- Multiple symbols can signal on the same day. Each is an independent trade.

---

## STEP 7: Compute Entry Day Candle Strength (body_pct)

On the **signal day** (the day the stock enters Q5, NOT the entry day), compute:

```
body_pct = (close - open) / (high - low)
```

- **+1.0** = the stock opened at the day's low and closed at the day's high (perfect bullish candle)
- **0.0** = close equals open (doji)
- **-1.0** = the stock opened at the day's high and closed at the day's low (perfect bearish candle)
- If high == low (zero range): body_pct is NaN — skip this trade.

**Quintile boundaries for body_pct (fixed across all 2,875 trades):**

| Quintile | body_pct range | Description |
|----------|---------------|-------------|
| Q1 | -0.99 to -0.53 | Strong red candle |
| Q2 | -0.53 to -0.17 | Moderate red candle |
| Q3 | -0.17 to +0.23 | Doji / indecisive |
| Q4 | +0.23 to +0.57 | Moderate green candle |
| Q5 | +0.58 to +1.00 | Strong green candle |

**These boundaries were computed once using `pd.qcut` on body_pct across all 2,875 trades in the backtest.** For live trading, you would use these fixed boundaries, not recompute quintiles daily.

---

## STEP 8: Position Sizing (Option D)

Based on the body_pct quintile of the signal day:

| body_pct Quintile | Position Size | Action |
|-------------------|--------------|--------|
| Q1 (body_pct ≤ -0.53) | $0 | **SKIP — do not trade** |
| Q2 (-0.53 < body_pct ≤ -0.17) | $500,000 | 0.5x |
| Q3 (-0.17 < body_pct ≤ +0.23) | $1,000,000 | 1.0x |
| Q4 (+0.23 < body_pct ≤ +0.57) | $1,500,000 | 1.5x |
| Q5 (body_pct > +0.57) | $2,000,000 | 2.0x |

---

## STEP 9: Compute Stop Loss Price

On the **signal day** (same day the Q5 transition is detected), look up two values from that symbol's daily data:

```
signal_day_low = low price on signal_date
atr_14 = 14-period Average True Range on signal_date
```

**ATR(14) calculation:**

True Range for each day:
```
TR[t] = max(high[t] - low[t], abs(high[t] - close[t-1]), abs(low[t] - close[t-1]))
```

ATR(14) uses Wilder's smoothing (exponential moving average with alpha = 1/14):
```
ATR[t] = ATR[t-1] × (13/14) + TR[t] × (1/14)
```

The first ATR value requires 14 days of TR data.

**In pandas:**
```python
tr1 = high - low
tr2 = abs(high - close.shift(1))
tr3 = abs(low - close.shift(1))
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr_14 = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
```

**Stop price:**
```
stop_price = signal_day_low - 3.0 × atr_14
```

This is **3 ATRs below the signal day's intraday low** — a wide stop that only triggers if the stock makes a significant new low well below the day's range.

---

## STEP 10: Entry

The signal fires at the **close of signal_date**. Entry is the next trading day.

- **Entry date:** The trading day after signal_date
- **Entry price:** The OPEN price on entry_date
- **Shares:** `floor(position_size / entry_price)` — round DOWN to whole shares
- **Entry cost:** `shares × entry_price × 5 / 10000` (5 basis points)

If the entry_date has no data for this symbol (not trading, halted, etc.): skip this trade.

---

## STEP 11: Daily Stop Check

Starting from entry_date, check each subsequent trading day:

```
For each trading day d from entry_date to entry_date + 20 trading days:
    Load day d's LOW price for this symbol
    If low[d] <= stop_price:
        EXIT at stop_price
        exit_type = "STOP"
        DONE — move to P&L calculation
```

**Important:** The stop is checked against the daily LOW, not the close. If the low touches the stop price at any point during the day, the trade exits at the stop price.

**Conservative assumption:** If the daily low breaches the stop AND the daily high would have hit a profit target, we assume the stop was hit first (worst case). However, this strategy has no profit target — only the stop and the time exit.

---

## STEP 12: Time-Based Exit

If the stop is never hit during the 20 trading days:

```
exit_price = close on the 20th trading day after entry
exit_type = "TIME"
```

**Counting:** Entry_date is day 1. The exit happens at the close of day 20.

If the symbol runs out of data before day 20 (delisted, data ends): exit at the close of the last available day.

---

## STEP 13: P&L Calculation

```
gross_pnl  = shares × (exit_price - entry_price)
entry_cost = shares × entry_price × 5 / 10000
exit_cost  = shares × exit_price × 5 / 10000
net_pnl    = gross_pnl - entry_cost - exit_cost
return_pct = 100 × (exit_price / entry_price - 1)
```

The 5 bps per side cost covers estimated spread + market impact for a liquid stock.

---

## EXPECTED RESULTS (Option D)

If replicated correctly:

| Metric | Value |
|--------|-------|
| Total trades | 2,308 |
| Total net P&L | ~$92.9M |
| Win rate | 54.8% |
| Profit factor | 1.97 |
| Avg P&L per trade | ~$40,232 |
| Max drawdown | ~$6.4M |
| Calmar ratio | 2.34 |
| Worst single trade | ~-$588K |
| Best single trade | ~$2.43M |
| Stop hit rate | 22.3% |

### Year-by-year:

| Year | Trades | Net P&L | WR | PF |
|------|--------|---------|-----|-----|
| 2020 | 353 | +$29.2M | 57.5% | 3.25 |
| 2021 | 332 | +$7.8M | 55.4% | 1.62 |
| 2022 | 368 | +$1.3M | 48.6% | 1.06 |
| 2023 | 412 | +$24.4M | 55.1% | 2.36 |
| 2024 | 378 | +$16.4M | 58.5% | 2.30 |
| 2025 | 393 | +$10.8M | 54.5% | 1.72 |
| 2026 | 72 | +$3.0M | 50.0% | 1.84 |

### By quintile:

| Quintile | Size | Trades | Total P&L | WR | PF |
|----------|------|--------|-----------|-----|-----|
| Q1 | SKIP | 0 (567 skipped) | — | — | — |
| Q2 | $500K | 578 | -$1.1M | 45.8% | 0.92 |
| Q3 | $1.0M | 583 | +$9.8M | 52.7% | 1.47 |
| Q4 | $1.5M | 561 | +$24.7M | 58.1% | 1.88 |
| Q5 | $2.0M | 586 | +$59.4M | 62.5% | 2.83 |

---

## FIRST 20 TRADES (for verification)

These are the first trades from January 2020. Your replication should match these exactly.

| # | Symbol | Signal Date | Entry Date | Entry Price | body_pct Q | Size |
|---|--------|-------------|------------|-------------|-----------|------|
| 1 | MRK | 2020-01-06 | 2020-01-07 | $90.80 | check | check |
| 2 | OXY | 2020-01-07 | 2020-01-08 | $45.54 | check | check |
| 3 | DXCM | 2020-01-08 | 2020-01-09 | $58.75 | check | check |
| 4 | CRM | 2020-01-09 | 2020-01-10 | $179.84 | check | check |
| 5 | KWEB | 2020-01-09 | 2020-01-10 | $44.04 | check | check |

Full trade log is in `analysis_results/backtest_optionD_trades.csv`.

---

## COMMON REPLICATION ERRORS

1. **Using the CLOSE as entry price.** Entry is at the OPEN of the day AFTER the signal. Not the signal day's close.

2. **Z-scoring across symbols instead of within.** Each symbol's z_score uses its own 252-day rolling mean and std. Do not pool.

3. **Not requiring Q5 TRANSITION.** The signal is the FIRST day entering Q5, not just being in Q5. You must track the previous day's quintile.

4. **Using wrong SMA slope lookback.** It's 10 days, not 5. `SMA50[t] / SMA50[t-10]`.

5. **Using adjusted prices for ATR.** The enriched data is split-adjusted. ATR should be computed on the same adjusted data. This is correct — do not use unadjusted data for ATR.

6. **Computing body_pct quintiles per-day instead of fixed.** The quintile boundaries for sizing are fixed across all trades: Q1 cutoff = -0.53, Q2/Q3 cutoff = -0.17, Q3/Q4 cutoff = +0.23, Q4/Q5 cutoff = +0.57. Do not recompute daily.

7. **Checking stop against close instead of low.** The stop triggers if the daily LOW breaches stop_price, not the daily close.

8. **Including today's slope in the rolling z-score window.** Our implementation does include today (pandas default). If you shift by 1 day, your z-scores will differ slightly and quintile assignments will change.

9. **Forgetting costs.** 5 bps each way = 10 bps round trip. On a $2M position, that's $2,000 per trade.

10. **Wrong ATR formula.** ATR(14) uses Wilder's smoothing (EMA with alpha=1/14), not a simple 14-day average of True Range. The numbers will differ if you use SMA instead of EMA for ATR.
