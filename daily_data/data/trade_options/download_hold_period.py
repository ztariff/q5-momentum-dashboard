#!/usr/bin/env python3
"""
Download hold-period options bar data for every trade in trade1_20d_trades.csv.
For each trade (filtered to entry_date >= 2022-01-01), downloads 4 option contracts:
  1. 30-delta call, ~1-month expiry
  2. 50-delta call, ~1-month expiry
  3. 30-delta call, ~3-month expiry
  4. 50-delta call, ~3-month expiry

Uses date-range aggs call (one call per contract covers entire hold period).
Saves daily summaries to hold_period_options.csv.
Supports resume by checking existing (trade_symbol, trade_entry_date) combos.
"""

import os
import math
import time
import requests
import pandas as pd
from datetime import date, timedelta, datetime
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY      = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
TRADE_CSV    = "/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv"
DATA_DIR     = "/home/ubuntu/daily_data/data"
OUT_FILE     = "/home/ubuntu/daily_data/data/trade_options/hold_period_options.csv"
LOG_FILE     = "/home/ubuntu/options-data/download_log.txt"
SLEEP_SEC    = 0.12
SAVE_EVERY   = 100   # flush to disk every N trades

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
    """Return (strike_30d, strike_50d). rvol_pct is percent (e.g. 30.5 = 30.5%)."""
    sigma = rvol_pct / 100.0
    T = dte / 365.0
    # 50-delta: ATM
    strike_50 = round_to_increment(spot, spot)
    # 30-delta: OTM call → spot * exp(z * sigma * sqrt(T)), z=0.524
    strike_30 = spot * math.exp(0.524 * sigma * math.sqrt(T))
    strike_30 = round_to_increment(strike_30, spot)
    return strike_30, strike_50

# ── Expiry helpers ───────────────────────────────────────────────────────────
def third_friday(year, month):
    """Return the 3rd Friday of the given year/month."""
    d = date(year, month, 1)
    # find first Friday
    days_until_friday = (4 - d.weekday()) % 7  # Friday = weekday 4
    first_friday = d + timedelta(days=days_until_friday)
    third_fri = first_friday + timedelta(weeks=2)
    return third_fri

def nearest_monthly_expiry(entry_dt, target_days, min_days):
    """
    Find 3rd Friday of month closest to entry_dt + target_days.
    If that expiry is < min_days away, use next month.
    entry_dt: date object
    """
    target = entry_dt + timedelta(days=target_days)
    # Try the month of target, then next month
    for delta_months in range(0, 4):
        year = target.year + (target.month - 1 + delta_months) // 12
        month = (target.month - 1 + delta_months) % 12 + 1
        expiry = third_friday(year, month)
        dte = (expiry - entry_dt).days
        if dte >= min_days:
            return expiry
    # Fallback: 4 months out
    year = target.year + (target.month + 3) // 12
    month = (target.month + 3) % 12 + 1
    return third_friday(year, month)

def option_ticker(symbol, expiry, strike):
    """Format: O:SYMBOL{YYMMDD}C{strike*1000:08d}"""
    yymmdd = expiry.strftime("%y%m%d")
    strike_int = round(strike * 1000)
    return f"O:{symbol}{yymmdd}C{strike_int:08d}"

# ── API calls ────────────────────────────────────────────────────────────────
SESSION = requests.Session()

def get_spot(symbol, dt_str):
    """Get unadjusted close price from open-close endpoint."""
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{dt_str}"
    r = SESSION.get(url, params={"adjusted": "false", "apiKey": API_KEY}, timeout=30)
    time.sleep(SLEEP_SEC)
    if r.status_code == 200:
        data = r.json()
        return data.get("close")
    return None

def get_rvol(symbol, entry_date_str):
    """Get rvol_20d for symbol on or before entry_date from enriched CSV."""
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
        # Guard: if stored as decimal (<2.0 means <200% vol), convert
        if val <= 2.0:
            val = val * 100.0
        return val
    except Exception as e:
        log(f"  rvol read error {symbol}: {e}")
        return None

def get_minute_bars(ticker, start_date, end_date):
    """Fetch all 1-minute bars from start_date to end_date (single call, up to 50000 bars)."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY
    }
    r = SESSION.get(url, params=params, timeout=60)
    time.sleep(SLEEP_SEC)
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("results", [])

# ── Bar aggregation ──────────────────────────────────────────────────────────
def bars_to_daily(bars):
    """
    Convert list of 1-min bar dicts to daily summary dicts.
    Bar fields: t (ms epoch), o, h, l, c, v, vw (optional)
    Returns list of dicts keyed by bar_date (YYYY-MM-DD string).
    """
    if not bars:
        return {}

    # Group by ET date
    # Polygon timestamps are in UTC milliseconds; ET is UTC-5 (UTC-4 EDT)
    # For simplicity, use UTC-5 offset consistently (options market 9:30-16:00 ET)
    from datetime import timezone
    ET_OFFSET = timedelta(hours=-5)  # EST; EDT would be -4 but -5 is conservative

    daily = defaultdict(list)
    for b in bars:
        t_ms = b["t"]
        dt_utc = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=t_ms)
        dt_et = dt_utc + ET_OFFSET
        day_str = dt_et.strftime("%Y-%m-%d")
        hour_min = dt_et.hour * 60 + dt_et.minute  # minutes since midnight ET
        daily[day_str].append({
            "open": b.get("o"),
            "high": b.get("h"),
            "low":  b.get("l"),
            "close": b.get("c"),
            "volume": b.get("v", 0),
            "vw": b.get("vw"),
            "minute_et": hour_min,  # 9:30 ET = 570 min
        })

    result = {}
    for day_str, day_bars in daily.items():
        day_bars_sorted = sorted(day_bars, key=lambda x: x["minute_et"])

        opens   = [b["open"]   for b in day_bars_sorted if b["open"]   is not None]
        highs   = [b["high"]   for b in day_bars_sorted if b["high"]   is not None]
        lows    = [b["low"]    for b in day_bars_sorted if b["low"]    is not None]
        closes  = [b["close"]  for b in day_bars_sorted if b["close"]  is not None]
        volumes = [b["volume"] for b in day_bars_sorted]

        day_open   = opens[0]  if opens  else None
        day_high   = max(highs) if highs else None
        day_low    = min(lows)  if lows  else None
        day_close  = closes[-1] if closes else None
        day_volume = sum(volumes)
        n_bars     = len(day_bars_sorted)

        # VWAP: volume-weighted avg of close
        total_vol = sum(volumes)
        if total_vol > 0:
            day_vwap = sum(b["close"] * b["volume"] for b in day_bars_sorted
                           if b["close"] is not None) / total_vol
        else:
            day_vwap = None

        # Morning price: first 6 bars starting at/after 9:30 ET (min 570)
        # 9:30 = 570, 9:35 = 575 (inclusive), i.e. minutes 570-575
        morning_bars = [b for b in day_bars_sorted if 570 <= b["minute_et"] <= 575]
        if morning_bars:
            morning_closes = [b["close"] for b in morning_bars if b["close"] is not None]
            morning_price = sum(morning_closes) / len(morning_closes) if morning_closes else None
        else:
            # Fall back to first bar
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
    log("Starting hold_period_options download")

    # Load trades
    trades = pd.read_csv(TRADE_CSV)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
    trades = trades[trades["entry_date"] >= "2022-01-01"].reset_index(drop=True)
    log(f"Loaded {len(trades)} trades (filtered to entry_date >= 2022-01-01)")

    # Check resume state
    already_done = set()
    out_exists = os.path.exists(OUT_FILE)
    if out_exists:
        existing = pd.read_csv(OUT_FILE, usecols=["trade_symbol", "trade_entry_date"])
        for _, row in existing.iterrows():
            already_done.add((row["trade_symbol"], str(row["trade_entry_date"])[:10]))
        log(f"Resume: {len(already_done)} (symbol, entry_date) combos already done")

    # Output columns
    COLS = [
        "trade_symbol", "trade_entry_date", "trade_exit_date",
        "option_type", "option_ticker", "strike", "expiration",
        "bar_date", "day_open", "day_high", "day_low", "day_close",
        "day_volume", "n_bars", "morning_price"
    ]

    # Write header if new file
    if not out_exists:
        pd.DataFrame(columns=COLS).to_csv(OUT_FILE, index=False)

    buffer = []
    trades_processed = 0

    for idx, trade in trades.iterrows():
        symbol     = trade["symbol"]
        entry_date = trade["entry_date"].date()
        exit_date  = trade["exit_date"].date()
        entry_str  = str(entry_date)
        exit_str   = str(exit_date)

        key = (symbol, entry_str)
        if key in already_done:
            continue

        trades_processed += 1
        log(f"[{trades_processed}] {symbol} {entry_str} -> {exit_str}")

        # ── Spot price ──
        spot = get_spot(symbol, entry_str)
        if spot is None or spot <= 0:
            log(f"  WARN: no spot for {symbol} {entry_str}, skipping")
            # Mark as done with empty rows for each contract
            for opt_type in ["30d_1m", "50d_1m", "30d_3m", "50d_3m"]:
                buffer.append({
                    "trade_symbol": symbol, "trade_entry_date": entry_str,
                    "trade_exit_date": exit_str, "option_type": opt_type,
                    "option_ticker": "", "strike": None, "expiration": "",
                    "bar_date": "", "day_open": None, "day_high": None,
                    "day_low": None, "day_close": None,
                    "day_volume": None, "n_bars": 0, "morning_price": None
                })
            already_done.add(key)
            continue

        # ── rvol ──
        rvol = get_rvol(symbol, entry_str)
        if rvol is None or rvol <= 0:
            rvol = 30.0  # default 30%
            log(f"  INFO: using default rvol=30% for {symbol}")

        # ── Expiries ──
        expiry_1m = nearest_monthly_expiry(entry_date, target_days=30, min_days=20)
        expiry_3m = nearest_monthly_expiry(entry_date, target_days=90, min_days=75)
        dte_1m = (expiry_1m - entry_date).days
        dte_3m = (expiry_3m - entry_date).days

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
            ticker = option_ticker(symbol, expiry, strike)
            expiry_str = str(expiry)
            log(f"  {opt_type}: {ticker} (spot={spot:.2f} rvol={rvol:.1f}% strike={strike} dte={( expiry - entry_date).days})")

            # ── Download bars ──
            bars = get_minute_bars(ticker, entry_str, exit_str)

            if not bars:
                # No data — write one row with empty prices to record the contract
                buffer.append({
                    "trade_symbol": symbol, "trade_entry_date": entry_str,
                    "trade_exit_date": exit_str, "option_type": opt_type,
                    "option_ticker": ticker, "strike": strike, "expiration": expiry_str,
                    "bar_date": "", "day_open": None, "day_high": None,
                    "day_low": None, "day_close": None,
                    "day_volume": None, "n_bars": 0, "morning_price": None
                })
                log(f"    -> no bars returned")
            else:
                daily = bars_to_daily(bars)
                log(f"    -> {len(bars)} bars across {len(daily)} days")
                for bar_date, d in sorted(daily.items()):
                    buffer.append({
                        "trade_symbol":    symbol,
                        "trade_entry_date": entry_str,
                        "trade_exit_date": exit_str,
                        "option_type":     opt_type,
                        "option_ticker":   ticker,
                        "strike":          strike,
                        "expiration":      expiry_str,
                        "bar_date":        bar_date,
                        "day_open":        d["day_open"],
                        "day_high":        d["day_high"],
                        "day_low":         d["day_low"],
                        "day_close":       d["day_close"],
                        "day_volume":      d["day_volume"],
                        "n_bars":          d["n_bars"],
                        "morning_price":   d["morning_price"],
                    })

        already_done.add(key)

        # ── Incremental save ──
        if trades_processed % SAVE_EVERY == 0 and buffer:
            df_buf = pd.DataFrame(buffer, columns=COLS)
            df_buf.to_csv(OUT_FILE, mode="a", header=False, index=False)
            log(f"  >> Saved {len(buffer)} rows at trade #{trades_processed}")
            buffer = []

    # ── Final flush ──
    if buffer:
        df_buf = pd.DataFrame(buffer, columns=COLS)
        df_buf.to_csv(OUT_FILE, mode="a", header=False, index=False)
        log(f"  >> Final flush: {len(buffer)} rows")

    # ── Summary ──
    if os.path.exists(OUT_FILE):
        total = pd.read_csv(OUT_FILE)
        log(f"Done. Output file: {OUT_FILE}")
        log(f"Total rows: {len(total)}")
        log(f"Trades covered: {total['trade_symbol'].nunique()} symbols, "
            f"{total[['trade_symbol','trade_entry_date']].drop_duplicates().shape[0]} (sym,date) combos")
    log("=" * 70)

if __name__ == "__main__":
    main()
