#!/usr/bin/env python3
"""
Enrich all daily bar CSVs with technical indicators.
Reads from data/{SYM}_daily.csv, writes to data/{SYM}_enriched.csv

Indicators:
  1. Bollinger Bands (20,2) — upper, lower, %B, bandwidth
  2. RSI (14)
  3. SMAs: 8, 12, 20, 50, 100, 200
  4. EMAs: 8, 12, 20, 50
  5. ATR (14)
  6. Volume metrics — vol_sma20, vol_sma50, relative_volume
  7. MACD (12, 26, 9)
  8. ADX (14)
  9. Stochastic %K/%D (14, 3)
 10. Rate of change (10, 20)
 11. On-balance volume (OBV)
 12. Keltner Channels (20, 1.5 ATR)
 13. Distance from MAs (% terms)
 14. True Range
 15. Daily return & log return
 16. Realized volatility (10d, 20d)
 17. High-low range (bps)
 18. Gap (overnight return in bps)
 19. Close location value (where close sits in day's range)
 20. Cumulative VWAP deviation
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "/home/ubuntu/daily_data/data"


def compute_rsi(close, period=14):
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing (exponential with alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_adx(high, low, close, period=14):
    """Average Directional Index."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """%K and %D."""
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def enrich(df):
    """Add all indicators to a daily bar dataframe."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── 1. Bollinger Bands (20, 2) ──
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_mid"] = bb_mid
    df["bb_pctb"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

    # ── 2. RSI (14) ──
    df["rsi_14"] = compute_rsi(c, 14)

    # ── 3. Simple Moving Averages ──
    for w in [8, 12, 20, 50, 100, 200]:
        df[f"sma_{w}"] = c.rolling(w).mean()

    # ── 4. Exponential Moving Averages ──
    for w in [8, 12, 20, 50]:
        df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    # ── 5. ATR (14) ──
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["true_range"] = tr
    df["atr_14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # ── 6. Volume metrics ──
    df["vol_sma20"] = v.rolling(20).mean()
    df["vol_sma50"] = v.rolling(50).mean()
    df["rvol"] = v / df["vol_sma20"].replace(0, np.nan)  # relative volume

    # ── 7. MACD (12, 26, 9) ──
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── 8. ADX (14) ──
    df["adx_14"], df["plus_di"], df["minus_di"] = compute_adx(h, l, c, 14)

    # ── 9. Stochastic (14, 3) ──
    df["stoch_k"], df["stoch_d"] = compute_stochastic(h, l, c, 14, 3)

    # ── 10. Rate of Change ──
    df["roc_10"] = 100 * (c / c.shift(10) - 1)
    df["roc_20"] = 100 * (c / c.shift(20) - 1)

    # ── 11. On-Balance Volume ──
    obv_sign = np.sign(c.diff()).fillna(0)
    df["obv"] = (v * obv_sign).cumsum()

    # ── 12. Keltner Channels (20, 1.5×ATR) ──
    kc_mid = c.ewm(span=20, adjust=False).mean()
    df["kc_upper"] = kc_mid + 1.5 * df["atr_14"]
    df["kc_lower"] = kc_mid - 1.5 * df["atr_14"]
    df["kc_mid"] = kc_mid

    # ── 13. Distance from MAs (%) ──
    for w in [8, 20, 50, 200]:
        df[f"dist_sma{w}_pct"] = 100 * (c - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # ── 14. Daily return & log return ──
    df["daily_return_pct"] = 100 * c.pct_change()
    df["log_return"] = np.log(c / c.shift(1))

    # ── 15. Realized volatility (annualized, 10d & 20d) ──
    log_ret = df["log_return"]
    df["rvol_10d"] = log_ret.rolling(10).std() * np.sqrt(252) * 100
    df["rvol_20d"] = log_ret.rolling(20).std() * np.sqrt(252) * 100

    # ── 16. High-low range (bps) ──
    df["range_bps"] = 10000 * (h - l) / o

    # ── 17. Gap (overnight return in bps) ──
    df["gap_bps"] = 10000 * (o - c.shift(1)) / c.shift(1)

    # ── 18. Close location value ──
    day_range = (h - l).replace(0, np.nan)
    df["close_location"] = (c - l) / day_range  # 0=closed at low, 1=closed at high

    # ── 19. Candle body ratio ──
    df["body_pct"] = (c - o) / day_range  # positive=green, negative=red, magnitude=body vs wick

    # ── 20. MA slope (momentum of the MA itself, 5-bar pct change) ──
    for w in [20, 50]:
        sma = df[f"sma_{w}"]
        df[f"sma{w}_slope_5d"] = 100 * (sma / sma.shift(5) - 1)

    # ── 21. Squeeze indicator (BB inside Keltner) ──
    df["squeeze"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
    df["squeeze"] = df["squeeze"].astype(int)

    # ── 22. Consecutive up/down days ──
    direction = np.sign(c.diff())
    consec = direction.copy()
    for i in range(1, len(consec)):
        if direction.iloc[i] == direction.iloc[i-1] and direction.iloc[i] != 0:
            consec.iloc[i] = consec.iloc[i-1] + direction.iloc[i]
        else:
            consec.iloc[i] = direction.iloc[i]
    df["consec_days"] = consec  # positive=up streak, negative=down streak

    # ── 23. N-day high/low proximity ──
    df["pct_from_20d_high"] = 100 * (c / h.rolling(20).max() - 1)
    df["pct_from_20d_low"] = 100 * (c / l.rolling(20).min() - 1)
    df["pct_from_52w_high"] = 100 * (c / h.rolling(252).max() - 1)
    df["pct_from_52w_low"] = 100 * (c / l.rolling(252).min() - 1)

    return df


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_daily.csv")))
    if not files:
        print("No daily CSVs found in", DATA_DIR)
        return

    total = len(files)
    print(f"Enriching {total} symbols with indicators...\n")

    col_count = None
    errors = []

    for i, fpath in enumerate(files, 1):
        sym = os.path.basename(fpath).replace("_daily.csv", "")
        try:
            df = pd.read_csv(fpath)
            if len(df) < 30:
                print(f"[{i:3d}/{total}] {sym}: only {len(df)} bars, skipping")
                continue

            df = enrich(df)

            out_path = os.path.join(DATA_DIR, f"{sym}_enriched.csv")
            df.to_csv(out_path, index=False)

            if col_count is None:
                col_count = len(df.columns)

            print(f"[{i:3d}/{total}] {sym}: {len(df)} bars, {len(df.columns)} cols")
        except Exception as e:
            print(f"[{i:3d}/{total}] {sym}: ERROR — {e}")
            errors.append(sym)

    print(f"\n{'='*60}")
    print(f"DONE — {total - len(errors)} enriched, {len(errors)} errors")
    print(f"Columns per file: {col_count}")
    if errors:
        print(f"Errors: {', '.join(errors)}")
    print(f"Output: {DATA_DIR}/{{SYM}}_enriched.csv")
    print(f"{'='*60}")

    # Print column list from last successful file
    if col_count:
        sample = pd.read_csv(os.path.join(DATA_DIR, f"{sym}_enriched.csv"), nrows=1)
        print(f"\nAll {len(sample.columns)} columns:")
        for j, col in enumerate(sample.columns, 1):
            print(f"  {j:2d}. {col}")


if __name__ == "__main__":
    main()
