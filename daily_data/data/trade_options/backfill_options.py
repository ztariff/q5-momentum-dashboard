#!/usr/bin/env python3
"""
Backfill options data for trades missing 30d_1m coverage.
Iterates through expiries and strikes to find ANY liquid call option.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import math
from datetime import date, timedelta, datetime
import logging

API_KEY = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
OUTPUT_PATH = "/home/ubuntu/daily_data/data/trade_options/hold_period_options_backfill.csv"
LOG_PATH = "/home/ubuntu/options-data/download_log.txt"
UNCOVERED_PATH = "/tmp/uncovered_trades.csv"
SLEEP = 0.12

os.makedirs("/home/ubuntu/options-data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='a'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def nth_friday(year, month, n=3):
    """Return the n-th Friday of a given month."""
    d = date(year, month, 1)
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(weeks=n - 1)


def get_strike_increment(spot):
    if spot < 25:
        return 0.5
    elif spot < 200:
        return 1.0
    elif spot < 500:
        return 2.5
    else:
        return 5.0


def round_to_increment(value, increment):
    return round(round(value / increment) * increment, 4)


def make_ticker(symbol, exp_date, strike):
    yymmdd = exp_date.strftime("%y%m%d")
    strike_int = int(round(strike * 1000))
    return f"O:{symbol}{yymmdd}C{strike_int:08d}"


def get_expiry_candidates(entry_date_str):
    """
    Return list of (label, expiry_date) to try in priority order.
    1. Weekly: next 3 Fridays if >= 5 days away
    2. Monthly 3rd Friday: ~30d, ~45d, ~60d, ~90d out
    """
    entry = date.fromisoformat(entry_date_str)
    candidates = []

    # Weekly Fridays
    # Find next Friday
    days_to_friday = (4 - entry.weekday()) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    next_friday = entry + timedelta(days=days_to_friday)

    for i, offset_weeks in enumerate([0, 1, 2]):
        w_friday = next_friday + timedelta(weeks=offset_weeks)
        dte = (w_friday - entry).days
        if dte >= 5:
            candidates.append((f"weekly_{i+1}", w_friday))

    # Monthly 3rd Fridays at ~30, 45, 60, 90 days out
    for target_days in [30, 45, 60, 90]:
        target = entry + timedelta(days=target_days)
        # find 3rd Friday of that month
        fri = nth_friday(target.year, target.month, 3)
        dte = (fri - entry).days
        # avoid duplicate (same date already in candidates)
        if dte >= 5 and fri not in [c[1] for c in candidates]:
            label = f"monthly_{target_days}d"
            candidates.append((label, fri))

    return candidates


def get_strike_candidates(spot, increment):
    strikes = []
    # ATM, OTM2%, ITM2%, OTM5%, ITM5%, ITM10%, OTM10%
    multipliers = [1.0, 1.02, 0.98, 1.05, 0.95, 0.90, 1.10]
    seen = set()
    for m in multipliers:
        s = round_to_increment(spot * m, increment)
        if s > 0 and s not in seen:
            seen.add(s)
            strikes.append(s)
    return strikes


def get_spot(symbol, date_str):
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{date_str}"
    params = {"adjusted": "false", "apiKey": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        time.sleep(SLEEP)
        if r.status_code == 200:
            data = r.json()
            # prefer open price as spot
            return data.get("open") or data.get("close")
    except Exception as e:
        log.warning(f"  Spot fetch error {symbol} {date_str}: {e}")
    return None


def check_bars(ticker, date_str):
    """Quick check: does this ticker have ANY bars on this date? Returns resultsCount."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date_str}/{date_str}"
    params = {"adjusted": "true", "sort": "asc", "limit": 10, "apiKey": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        time.sleep(SLEEP)
        if r.status_code == 200:
            data = r.json()
            return data.get("resultsCount", 0)
    except Exception as e:
        log.warning(f"  Bar check error {ticker}: {e}")
    return 0


def download_full_hold(ticker, entry_date_str, exit_date_str):
    """Download 1-min bars for the entire hold period, with pagination."""
    all_results = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{entry_date_str}/{exit_date_str}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}

    while url:
        try:
            if "polygon.io" in url and "apiKey" not in url:
                params_to_send = params
            else:
                params_to_send = {} if "apiKey" in url else params
            r = requests.get(url, params=params_to_send, timeout=30)
            time.sleep(SLEEP)
            if r.status_code != 200:
                log.warning(f"  Full download HTTP {r.status_code} for {ticker}")
                break
            data = r.json()
            results = data.get("results", [])
            all_results.extend(results)
            url = data.get("next_url")
        except Exception as e:
            log.warning(f"  Full download error {ticker}: {e}")
            break

    return all_results


def aggregate_to_daily(bars, symbol, entry_date_str, exit_date_str,
                        ticker, strike, expiration, option_type,
                        actual_delta, search_attempt):
    """Aggregate 1-min bars to daily rows, returning list of dicts."""
    if not bars:
        return []

    df = pd.DataFrame(bars)
    # timestamp is ms UTC
    df['dt'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert('America/New_York')
    df['bar_date'] = df['dt'].dt.date.astype(str)

    rows = []
    for bar_date, grp in df.groupby('bar_date'):
        grp_sorted = grp.sort_values('dt')
        morning = grp_sorted[grp_sorted['dt'].dt.hour == 9]
        morning = morning[morning['dt'].dt.minute < 36]
        morning_price = morning['o'].iloc[0] if len(morning) > 0 else (
            grp_sorted['o'].iloc[0] if len(grp_sorted) > 0 else None
        )
        rows.append({
            'trade_symbol': symbol,
            'trade_entry_date': entry_date_str,
            'trade_exit_date': exit_date_str,
            'option_type': option_type,
            'option_ticker': ticker,
            'strike': strike,
            'expiration': str(expiration),
            'bar_date': bar_date,
            'day_open': grp_sorted['o'].iloc[0],
            'day_high': grp_sorted['h'].max(),
            'day_low': grp_sorted['l'].min(),
            'day_close': grp_sorted['c'].iloc[-1],
            'day_volume': grp_sorted['v'].sum(),
            'n_bars': len(grp_sorted),
            'morning_price': morning_price,
            'actual_delta': actual_delta,
            'search_attempt': search_attempt,
        })
    return rows


def estimate_delta(spot, strike, dte):
    """Rough delta estimate from moneyness. Not Black-Scholes, just heuristic."""
    if spot <= 0 or strike <= 0:
        return None
    moneyness = spot / strike
    # very rough: deeper ITM = higher delta
    if moneyness > 1.1:
        return 0.80
    elif moneyness > 1.05:
        return 0.70
    elif moneyness > 1.02:
        return 0.60
    elif moneyness > 0.98:
        return 0.50
    elif moneyness > 0.95:
        return 0.40
    elif moneyness > 0.90:
        return 0.30
    else:
        return 0.20


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    uncovered = pd.read_csv(UNCOVERED_PATH)
    log.info(f"Loaded {len(uncovered)} uncovered trades")

    # Load existing backfill output (for resumption)
    already_done = set()
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        for _, row in existing[['trade_symbol', 'trade_entry_date']].drop_duplicates().iterrows():
            already_done.add((row['trade_symbol'], row['trade_entry_date']))
        log.info(f"Resuming: {len(already_done)} trades already in backfill output")

    recovered = 0
    truly_uncovered = 0
    batch = []

    COLUMNS = [
        'trade_symbol', 'trade_entry_date', 'trade_exit_date', 'option_type',
        'option_ticker', 'strike', 'expiration', 'bar_date',
        'day_open', 'day_high', 'day_low', 'day_close', 'day_volume',
        'n_bars', 'morning_price', 'actual_delta', 'search_attempt'
    ]

    def flush_batch():
        if not batch:
            return
        df = pd.DataFrame(batch, columns=COLUMNS)
        write_header = not os.path.exists(OUTPUT_PATH)
        df.to_csv(OUTPUT_PATH, mode='a', index=False, header=write_header)
        batch.clear()
        log.info(f"  Flushed {len(df)} rows to {OUTPUT_PATH}")

    total = len(uncovered)
    for idx, row in enumerate(uncovered.itertuples(), 1):
        symbol = row.symbol
        entry_date = row.entry_date
        exit_date = row.exit_date

        if (symbol, entry_date) in already_done:
            log.info(f"[{idx}/{total}] {symbol} {entry_date} — skip (already done)")
            continue

        log.info(f"[{idx}/{total}] {symbol} {entry_date} → {exit_date}")

        # Get spot price
        spot = get_spot(symbol, entry_date)
        if not spot or spot <= 0:
            log.warning(f"  No spot for {symbol} {entry_date}, skipping")
            truly_uncovered += 1
            continue

        increment = get_strike_increment(spot)
        expiry_candidates = get_expiry_candidates(entry_date)
        strike_candidates = get_strike_candidates(spot, increment)

        found = False
        for exp_label, expiry in expiry_candidates:
            if found:
                break
            for s_idx, strike in enumerate(strike_candidates):
                ticker = make_ticker(symbol, expiry, strike)
                count = check_bars(ticker, entry_date)
                if count > 0:
                    log.info(f"  HIT: {ticker} (exp={exp_label}, strike_idx={s_idx}, spot={spot:.2f}, dte={(expiry - date.fromisoformat(entry_date)).days})")
                    # Download full hold period
                    bars = download_full_hold(ticker, entry_date, exit_date)
                    if bars:
                        dte = (expiry - date.fromisoformat(entry_date)).days
                        actual_delta = estimate_delta(spot, strike, dte)
                        search_attempt = f"{exp_label}_strike{s_idx}"
                        daily_rows = aggregate_to_daily(
                            bars, symbol, entry_date, exit_date,
                            ticker, strike, expiry,
                            'backfill', actual_delta, search_attempt
                        )
                        batch.extend(daily_rows)
                        recovered += 1
                        already_done.add((symbol, entry_date))
                        found = True
                        break
                    else:
                        log.warning(f"  HIT but no bars on hold period for {ticker}")

        if not found:
            log.info(f"  TRULY UNCOVERED: {symbol} {entry_date}")
            truly_uncovered += 1

        if idx % 50 == 0:
            flush_batch()
            log.info(f"  Progress: {idx}/{total} trades processed, {recovered} recovered, {truly_uncovered} truly uncovered")

    flush_batch()
    log.info(f"\nDone. Recovered: {recovered}, Truly uncovered: {truly_uncovered}, Total processed: {total}")


if __name__ == "__main__":
    main()
