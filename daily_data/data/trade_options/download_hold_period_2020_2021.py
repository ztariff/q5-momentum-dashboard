#!/usr/bin/env python3
"""
Download hold-period options bar data for 2020-2021 trades.
Trade log: /home/ubuntu/daily_data/analysis_results/backtest_optionD_noReentry_trades.csv
Filtered to: entry_date >= 2020-01-01 AND entry_date < 2022-01-01

For each trade, downloads 4 option contracts:
  1. 30-delta call, ~1-month expiry
  2. 50-delta call, ~1-month expiry
  3. 30-delta call, ~3-month expiry
  4. 50-delta call, ~3-month expiry

Uses unadjusted spot prices for strike computation (critical for AAPL/TSLA splits in 2020).
APPENDS to existing hold_period_options.csv — does NOT overwrite.
Supports resume: skips (trade_symbol, trade_entry_date) already in output file.
"""

import os
import math
import time
import requests
import pandas as pd
from datetime import date, timedelta, datetime, timezone
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY    = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
TRADE_CSV  = "/home/ubuntu/daily_data/analysis_results/backtest_optionD_noReentry_trades.csv"
DATA_DIR   = "/home/ubuntu/daily_data/data"
OUT_FILE   = "/home/ubuntu/daily_data/data/trade_options/hold_period_options.csv"
LOG_FILE   = "/home/ubuntu/options-data/download_log.txt"
SLEEP_SEC  = 0.12
SAVE_EVERY = 50   # flush to disk every N trades

os.makedirs("/home/ubuntu/options-data", exist_ok=True)
os.makedirs("/home/ubuntu/daily_data/data/trade_options", exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── Strike helpers ───────────────────────────────────────────────────────────
def round_to_increment(strike, spot):
    """Round strike to nearest market increment based on spot price."""
    if spot < 25:
        inc = 0.50
    elif spot < 200:
        inc = 1.0
    elif spot < 500:
        inc = 2.50
    else:
        inc = 5.0
    return round(round(strike / inc) * inc, 4)

def compute_strikes(spot, rvol_pct, dte):
    """
    Return (strike_30d, strike_50d).
    rvol_pct is in percent form (e.g., 30.5 means 30.5% annual vol).
    50-delta: ATM (spot)
    30-delta: spot * exp(0.524 * sigma * sqrt(T))
    """
    sigma = rvol_pct / 100.0          # CRITICAL: divide by 100
    T = dte / 365.0
    strike_50 = round_to_increment(spot, spot)
    strike_30 = spot * math.exp(0.524 * sigma * math.sqrt(T))
    strike_30 = round_to_increment(strike_30, spot)
    return strike_30, strike_50

# ── Expiry helpers ───────────────────────────────────────────────────────────
def third_friday(year, month):
    """Return the 3rd Friday of the given year/month."""
    d = date(year, month, 1)
    days_until_friday = (4 - d.weekday()) % 7  # Friday = weekday 4
    first_friday = d + timedelta(days=days_until_friday)
    return first_friday + timedelta(weeks=2)

def nearest_monthly_expiry(entry_dt, target_days, min_days):
    """
    Find 3rd Friday closest to entry_dt + target_days.
    If that expiry is < min_days away from entry, advance to next month.
    """
    target = entry_dt + timedelta(days=target_days)
    for delta_months in range(0, 5):
        year  = target.year  + (target.month - 1 + delta_months) // 12
        month = (target.month - 1 + delta_months) % 12 + 1
        expiry = third_friday(year, month)
        dte = (expiry - entry_dt).days
        if dte >= min_days:
            return expiry
    # Fallback: 5 months out
    year  = entry_dt.year  + (entry_dt.month - 1 + 5) // 12
    month = (entry_dt.month - 1 + 5) % 12 + 1
    return third_friday(year, month)

def option_ticker(symbol, expiry, strike):
    """Format: O:SYMBOL{YYMMDD}C{strike*1000:08d}"""
    yymmdd = expiry.strftime("%y%m%d")
    strike_int = round(strike * 1000)
    return f"O:{symbol}{yymmdd}C{strike_int:08d}"

# ── API calls ────────────────────────────────────────────────────────────────
SESSION = requests.Session()

def get_spot(symbol, dt_str):
    """Get unadjusted close price — MUST use adjusted=false for pre-split accuracy."""
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{dt_str}"
    try:
        r = SESSION.get(url, params={"adjusted": "false", "apiKey": API_KEY}, timeout=30)
        time.sleep(SLEEP_SEC)
        if r.status_code == 200:
            data = r.json()
            return data.get("close")
        log(f"  WARN: spot API {r.status_code} for {symbol} {dt_str}")
    except Exception as e:
        log(f"  ERROR: spot fetch {symbol} {dt_str}: {e}")
        time.sleep(1.0)
    return None

def get_rvol(symbol, entry_date_str):
    """
    Get rvol_20d (%) for symbol on or before entry_date from enriched CSV.
    Returns float in percent form (e.g., 30.5 for 30.5% vol), or None.
    """
    path = os.path.join(DATA_DIR, f"{symbol}_enriched.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=["date", "rvol_20d"])
        df = df.dropna(subset=["rvol_20d"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        entry_d = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        df = df[df["date"] <= entry_d].sort_values("date")
        if df.empty:
            return None
        val = float(df.iloc[-1]["rvol_20d"])
        # Guard: rvol_20d is stored as percentage (e.g. 30.5).
        # If value is < 2.0 it was accidentally stored as decimal — convert.
        if val <= 2.0:
            val = val * 100.0
        return val
    except Exception as e:
        log(f"  rvol read error {symbol}: {e}")
        return None

def get_minute_bars(ticker, start_date, end_date):
    """
    Fetch all 1-minute bars from start_date to end_date in one call.
    Handles pagination via next_url if needed (>50000 bars unlikely for options).
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    50000,
        "apiKey":   API_KEY,
    }
    all_bars = []
    try:
        r = SESSION.get(url, params=params, timeout=60)
        time.sleep(SLEEP_SEC)
        if r.status_code != 200:
            return []
        data = r.json()
        all_bars.extend(data.get("results", []))

        # Pagination — follow next_url if present
        next_url = data.get("next_url")
        while next_url:
            r2 = SESSION.get(next_url, params={"apiKey": API_KEY}, timeout=60)
            time.sleep(SLEEP_SEC)
            if r2.status_code != 200:
                break
            d2 = r2.json()
            all_bars.extend(d2.get("results", []))
            next_url = d2.get("next_url")

    except Exception as e:
        log(f"  ERROR: bars fetch {ticker}: {e}")
        time.sleep(1.0)

    return all_bars

# ── Bar aggregation ──────────────────────────────────────────────────────────
def bars_to_daily(bars):
    """
    Convert list of 1-min bar dicts to daily summary dicts.
    Returns dict keyed by bar_date string (YYYY-MM-DD).

    morning_price: VWAP of first 6 bars at 9:30–9:35 ET (minutes 570–575).
    Falls back to average of all bars if no 9:30–9:36 data.

    DST handling: use UTC-4 (EDT, Mar-Nov) and UTC-5 (EST, Nov-Mar).
    Polygon timestamps are UTC milliseconds.
    """
    if not bars:
        return {}

    daily = defaultdict(list)
    for b in bars:
        t_ms = b["t"]
        dt_utc = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=t_ms)
        # DST: EDT (UTC-4) from 2nd Sun March to 1st Sun November
        # Approximate: month 3-10 = UTC-4, else UTC-5
        month = dt_utc.month
        et_offset = timedelta(hours=-4) if 3 <= month <= 10 else timedelta(hours=-5)
        dt_et = dt_utc + et_offset
        day_str = dt_et.strftime("%Y-%m-%d")
        hour_min = dt_et.hour * 60 + dt_et.minute  # 9:30 ET = 570

        daily[day_str].append({
            "open":      b.get("o"),
            "high":      b.get("h"),
            "low":       b.get("l"),
            "close":     b.get("c"),
            "volume":    b.get("v", 0),
            "minute_et": hour_min,
        })

    result = {}
    for day_str, day_bars in daily.items():
        day_bars_sorted = sorted(day_bars, key=lambda x: x["minute_et"])

        opens   = [b["open"]  for b in day_bars_sorted if b["open"]  is not None]
        highs   = [b["high"]  for b in day_bars_sorted if b["high"]  is not None]
        lows    = [b["low"]   for b in day_bars_sorted if b["low"]   is not None]
        closes  = [b["close"] for b in day_bars_sorted if b["close"] is not None]
        volumes = [b["volume"] for b in day_bars_sorted]

        day_open   = opens[0]   if opens  else None
        day_high   = max(highs) if highs  else None
        day_low    = min(lows)  if lows   else None
        day_close  = closes[-1] if closes else None
        day_volume = sum(volumes)
        n_bars     = len(day_bars_sorted)

        # morning_price: volume-weighted avg of first 6 bars at 9:30–9:35 ET
        # 9:30 = minute 570, 9:35 = minute 575 (6 bars: :30, :31, :32, :33, :34, :35)
        morning_bars = [b for b in day_bars_sorted if 570 <= b["minute_et"] <= 575]
        if morning_bars:
            total_vol = sum(b["volume"] for b in morning_bars)
            if total_vol > 0:
                morning_price = sum(
                    b["close"] * b["volume"]
                    for b in morning_bars if b["close"] is not None
                ) / total_vol
            else:
                # equal-weight if all volumes are zero
                mc = [b["close"] for b in morning_bars if b["close"] is not None]
                morning_price = sum(mc) / len(mc) if mc else None
        else:
            # Fall back to first bar close
            morning_price = closes[0] if closes else None

        result[day_str] = {
            "day_open":      day_open,
            "day_high":      day_high,
            "day_low":       day_low,
            "day_close":     day_close,
            "day_volume":    day_volume,
            "n_bars":        n_bars,
            "morning_price": morning_price,
        }

    return result

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("Starting hold_period_options download: 2020-2021 trades")
    log(f"Trade CSV: {TRADE_CSV}")
    log(f"Output:    {OUT_FILE}")

    # ── Load and filter trades ──
    trades = pd.read_csv(TRADE_CSV)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
    mask = (trades["entry_date"] >= "2020-01-01") & (trades["entry_date"] < "2022-01-01")
    trades = trades[mask].reset_index(drop=True)
    log(f"Loaded {len(trades)} trades for 2020-2021")

    # ── Check resume state ──
    already_done = set()
    out_exists = os.path.exists(OUT_FILE)
    if out_exists:
        try:
            existing = pd.read_csv(OUT_FILE, usecols=["trade_symbol", "trade_entry_date"])
            for _, row in existing.iterrows():
                already_done.add((row["trade_symbol"], str(row["trade_entry_date"])[:10]))
            log(f"Resume: {len(already_done)} (symbol, entry_date) combos already in output")
        except Exception as e:
            log(f"WARN: could not read existing file for resume check: {e}")

    # ── Output columns (must match existing file exactly) ──
    COLS = [
        "trade_symbol", "trade_entry_date", "trade_exit_date",
        "option_type", "option_ticker", "strike", "expiration",
        "bar_date", "day_open", "day_high", "day_low", "day_close",
        "day_volume", "n_bars", "morning_price"
    ]

    # Do NOT write header — we are appending to an existing file
    # If file doesn't exist (shouldn't happen), create with header
    if not out_exists:
        pd.DataFrame(columns=COLS).to_csv(OUT_FILE, index=False)
        log("WARNING: output file did not exist — created new with header")

    buffer = []
    trades_processed = 0
    trades_skipped   = 0

    for idx, trade in trades.iterrows():
        symbol     = trade["symbol"]
        entry_date = trade["entry_date"].date()
        exit_date  = trade["exit_date"].date()
        entry_str  = str(entry_date)
        exit_str   = str(exit_date)

        key = (symbol, entry_str)
        if key in already_done:
            trades_skipped += 1
            continue

        trades_processed += 1
        log(f"[{trades_processed}] {symbol} {entry_str} -> {exit_str}")

        # ── Unadjusted spot price (CRITICAL for pre-split tickers) ──
        spot = get_spot(symbol, entry_str)
        if spot is None or spot <= 0:
            log(f"  WARN: no spot for {symbol} {entry_str} — writing empty rows and skipping")
            for opt_type in ["30d_1m", "50d_1m", "30d_3m", "50d_3m"]:
                buffer.append({
                    "trade_symbol": symbol, "trade_entry_date": entry_str,
                    "trade_exit_date": exit_str, "option_type": opt_type,
                    "option_ticker": "", "strike": None, "expiration": "",
                    "bar_date": "", "day_open": None, "day_high": None,
                    "day_low": None, "day_close": None,
                    "day_volume": None, "n_bars": 0, "morning_price": None,
                })
            already_done.add(key)
            continue

        # ── Realized vol ──
        rvol = get_rvol(symbol, entry_str)
        if rvol is None or rvol <= 0:
            rvol = 30.0
            log(f"  INFO: no rvol found — using default 30% for {symbol}")

        # ── Expiries ──
        expiry_1m = nearest_monthly_expiry(entry_date, target_days=30,  min_days=20)
        expiry_3m = nearest_monthly_expiry(entry_date, target_days=90,  min_days=75)
        dte_1m    = (expiry_1m - entry_date).days
        dte_3m    = (expiry_3m - entry_date).days

        # ── Strikes ──
        strike_30_1m, strike_50_1m = compute_strikes(spot, rvol, dte_1m)
        strike_30_3m, strike_50_3m = compute_strikes(spot, rvol, dte_3m)

        contracts = [
            ("30d_1m", strike_30_1m, expiry_1m),
            ("50d_1m", strike_50_1m, expiry_1m),
            ("30d_3m", strike_30_3m, expiry_3m),
            ("50d_3m", strike_50_3m, expiry_3m),
        ]

        for opt_type, strike, expiry in contracts:
            ticker    = option_ticker(symbol, expiry, strike)
            expiry_str = str(expiry)
            dte_for_log = (expiry - entry_date).days
            log(f"  {opt_type}: {ticker}  spot={spot:.2f}  rvol={rvol:.1f}%  "
                f"strike={strike}  dte={dte_for_log}")

            bars = get_minute_bars(ticker, entry_str, exit_str)

            if not bars:
                buffer.append({
                    "trade_symbol": symbol, "trade_entry_date": entry_str,
                    "trade_exit_date": exit_str, "option_type": opt_type,
                    "option_ticker": ticker, "strike": strike, "expiration": expiry_str,
                    "bar_date": "", "day_open": None, "day_high": None,
                    "day_low": None, "day_close": None,
                    "day_volume": None, "n_bars": 0, "morning_price": None,
                })
                log(f"    -> no bars returned")
            else:
                daily = bars_to_daily(bars)
                log(f"    -> {len(bars)} bars  {len(daily)} days")
                for bar_date, d in sorted(daily.items()):
                    buffer.append({
                        "trade_symbol":     symbol,
                        "trade_entry_date": entry_str,
                        "trade_exit_date":  exit_str,
                        "option_type":      opt_type,
                        "option_ticker":    ticker,
                        "strike":           strike,
                        "expiration":       expiry_str,
                        "bar_date":         bar_date,
                        "day_open":         d["day_open"],
                        "day_high":         d["day_high"],
                        "day_low":          d["day_low"],
                        "day_close":        d["day_close"],
                        "day_volume":       d["day_volume"],
                        "n_bars":           d["n_bars"],
                        "morning_price":    d["morning_price"],
                    })

        already_done.add(key)

        # ── Incremental save ──
        if trades_processed % SAVE_EVERY == 0 and buffer:
            df_buf = pd.DataFrame(buffer, columns=COLS)
            df_buf.to_csv(OUT_FILE, mode="a", header=False, index=False)
            log(f"  >> Saved {len(buffer)} rows (trade #{trades_processed})")
            buffer = []

    # ── Final flush ──
    if buffer:
        df_buf = pd.DataFrame(buffer, columns=COLS)
        df_buf.to_csv(OUT_FILE, mode="a", header=False, index=False)
        log(f"  >> Final flush: {len(buffer)} rows")

    # ── Summary ──
    log("=" * 70)
    log(f"Trades processed this run: {trades_processed}")
    log(f"Trades skipped (already done): {trades_skipped}")
    if os.path.exists(OUT_FILE):
        total = pd.read_csv(OUT_FILE)
        log(f"Output file total rows: {len(total)}")
        combos = total[["trade_symbol", "trade_entry_date"]].drop_duplicates().shape[0]
        log(f"Total (symbol, entry_date) combos in file: {combos}")
    log("=" * 70)

if __name__ == "__main__":
    main()
