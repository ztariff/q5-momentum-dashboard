#!/usr/bin/env python3
"""
Replace weekly-expiry backfill options with monthly alternatives.

For each trade in hold_period_options_backfill.csv that used a weekly expiry,
search for a monthly option (25-45 DTE) using delta-based strikes.

Output: hold_period_options_monthly_replace.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import math
import logging
from datetime import date, timedelta, datetime

API_KEY = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BACKFILL_PATH = "/home/ubuntu/daily_data/data/trade_options/hold_period_options_backfill.csv"
OUTPUT_PATH = "/home/ubuntu/daily_data/data/trade_options/hold_period_options_monthly_replace.csv"
LOG_PATH = "/home/ubuntu/options-data/download_log.txt"
SLEEP = 0.1
MIN_LIQUIDITY = 5  # minimum day_volume on entry day

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

# ── helpers ───────────────────────────────────────────────────────────────────

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


def get_monthly_expiry_candidates(entry_date_str):
    """
    Return list of (label, expiry_date) — MONTHLY 3rd Fridays only.
    Target DTEs: ~30, ~45, ~60 days out.
    Skip any expiry where DTE < 15 (would expire during 13-day hold).
    Also require DTE >= 20.
    """
    entry = date.fromisoformat(entry_date_str)
    candidates = []
    seen = set()

    for target_days in [30, 45, 60]:
        target = entry + timedelta(days=target_days)
        fri = nth_friday(target.year, target.month, 3)
        dte = (fri - entry).days
        if dte >= 20 and fri not in seen:
            seen.add(fri)
            label = f"monthly_{target_days}d"
            candidates.append((label, fri))

    return candidates


def get_delta_strike_candidates(spot, sigma, T, increment):
    """
    Return list of (label, strike) in priority order using delta-based construction.
    sigma: annualized volatility as decimal (rvol_20d / 100)
    T: time in years
    """
    # N^{-1}(delta) values for various deltas (z-score for call delta approximation)
    # For a call: delta ~ N( (ln(S/K) + 0.5*sigma^2*T) / (sigma*sqrt(T)) )
    # Inverting for strike: K = S * exp(-z * sigma * sqrt(T))  where z = N_inv(delta) approx
    # z-scores (quantiles) for target deltas:
    # 30-delta: N_inv(0.30) ~ -0.524  → K = S * exp(0.524 * sigma * sqrt(T))  [OTM call]
    # 25-delta: N_inv(0.25) ~ -0.674
    # 35-delta: N_inv(0.35) ~ -0.385
    # 20-delta: N_inv(0.20) ~ -0.842
    # 40-delta: N_inv(0.40) ~ -0.253

    candidates = []
    seen = set()

    def add(label, raw_strike):
        s = round_to_increment(raw_strike, increment)
        if s > 0 and s not in seen:
            seen.add(s)
            candidates.append((label, s))

    vol_sqrt_t = sigma * math.sqrt(T)

    # Priority order per spec
    add("30d_otm",    spot * math.exp(0.524  * vol_sqrt_t))
    add("25d_otm",    spot * math.exp(0.674  * vol_sqrt_t))
    add("35d_otm",    spot * math.exp(0.385  * vol_sqrt_t))
    add("atm",        spot)
    add("20d_otm",    spot * math.exp(0.842  * vol_sqrt_t))
    add("40d_otm",    spot * math.exp(0.253  * vol_sqrt_t))
    add("slight_itm", spot * 0.97)

    return candidates


def get_spot(symbol, date_str):
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{date_str}"
    params = {"adjusted": "false", "apiKey": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        time.sleep(SLEEP)
        if r.status_code == 200:
            data = r.json()
            return data.get("open") or data.get("close")
    except Exception as e:
        log.warning(f"  Spot fetch error {symbol} {date_str}: {e}")
    return None


def check_bars_with_volume(ticker, date_str):
    """Check bars on entry date. Returns (bar_count, day_volume)."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date_str}/{date_str}"
    params = {"adjusted": "true", "sort": "asc", "limit": 500, "apiKey": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        time.sleep(SLEEP)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                total_vol = sum(b.get("v", 0) for b in results)
                return len(results), total_vol
    except Exception as e:
        log.warning(f"  Bar check error {ticker}: {e}")
    return 0, 0


def download_full_hold(ticker, entry_date_str, exit_date_str):
    """Download 1-min bars for the entire hold period, with pagination."""
    all_results = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{entry_date_str}/{exit_date_str}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}

    while url:
        try:
            if "apiKey" in url:
                r = requests.get(url, timeout=30)
            else:
                r = requests.get(url, params=params, timeout=30)
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
                        ticker, strike, expiration, search_attempt,
                        replaced_from, new_dte, new_volume_entry_day):
    """Aggregate 1-min bars to daily rows."""
    if not bars:
        return []

    df = pd.DataFrame(bars)
    df['dt'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert('America/New_York')
    df['bar_date'] = df['dt'].dt.date.astype(str)

    rows = []
    for bar_date, grp in df.groupby('bar_date'):
        grp_sorted = grp.sort_values('dt')
        morning = grp_sorted[(grp_sorted['dt'].dt.hour == 9) &
                              (grp_sorted['dt'].dt.minute < 36)]
        morning_price = (morning['o'].iloc[0] if len(morning) > 0
                         else grp_sorted['o'].iloc[0])

        rows.append({
            'trade_symbol': symbol,
            'trade_entry_date': entry_date_str,
            'trade_exit_date': exit_date_str,
            'option_type': 'backfill',
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
            'actual_delta': None,   # not computed here
            'search_attempt': search_attempt,
            'replaced_from': replaced_from,
            'new_dte': new_dte,
            'new_volume_entry_day': new_volume_entry_day,
        })
    return rows


def load_rvol(symbol, entry_date_str):
    """Load rvol_20d for symbol on entry_date from enriched CSV."""
    enriched_path = f"/home/ubuntu/daily_data/data/{symbol}_enriched.csv"
    try:
        if os.path.exists(enriched_path):
            df = pd.read_csv(enriched_path, usecols=['date', 'rvol_20d'])
            row = df[df['date'] == entry_date_str]
            if len(row) > 0:
                val = row.iloc[0]['rvol_20d']
                if pd.notna(val) and val > 0:
                    return float(val) / 100.0  # convert % to decimal
    except Exception as e:
        log.warning(f"  rvol load error {symbol} {entry_date_str}: {e}")
    return 0.30  # default 30%


# ── main ──────────────────────────────────────────────────────────────────────

COLUMNS = [
    'trade_symbol', 'trade_entry_date', 'trade_exit_date', 'option_type',
    'option_ticker', 'strike', 'expiration', 'bar_date',
    'day_open', 'day_high', 'day_low', 'day_close', 'day_volume',
    'n_bars', 'morning_price', 'actual_delta', 'search_attempt',
    'replaced_from', 'new_dte', 'new_volume_entry_day',
]


def main():
    log.info("=" * 60)
    log.info("monthly_replace.py — starting")

    # Load backfill
    bf = pd.read_csv(BACKFILL_PATH)
    bf['entry_dt'] = pd.to_datetime(bf['trade_entry_date'])
    bf['expiry_dt'] = pd.to_datetime(bf['expiration'])
    bf['dte_at_entry'] = (bf['expiry_dt'] - bf['entry_dt']).dt.days

    # Get one row per trade (take first row = first bar_date)
    trades = bf.groupby(['trade_symbol', 'trade_entry_date', 'trade_exit_date']).first().reset_index()
    trades['entry_dt'] = pd.to_datetime(trades['trade_entry_date'])
    trades['expiry_dt'] = pd.to_datetime(trades['expiration'])
    trades['dte_at_entry'] = (trades['expiry_dt'] - trades['entry_dt']).dt.days

    # Flag trades to replace
    weekly_mask = trades['search_attempt'].str.contains('weekly', na=False)
    short_dte_mask = trades['dte_at_entry'] < 15
    flagged = trades[weekly_mask | short_dte_mask].copy()

    log.info(f"Total backfill trades: {len(trades)}")
    log.info(f"Flagged for replacement (weekly OR dte<15): {len(flagged)}")
    log.info(f"  - weekly search_attempt: {weekly_mask.sum()}")
    log.info(f"  - dte<15 (even if monthly): {short_dte_mask.sum()}")

    # Load existing output for resumption
    already_done = set()
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        for _, row in existing[['trade_symbol', 'trade_entry_date']].drop_duplicates().iterrows():
            already_done.add((row['trade_symbol'], row['trade_entry_date']))
        log.info(f"Resuming: {len(already_done)} trades already in output")

    n_replaced = 0
    n_no_monthly = 0
    batch = []

    def flush_batch():
        if not batch:
            return
        df_out = pd.DataFrame(batch, columns=COLUMNS)
        write_header = not os.path.exists(OUTPUT_PATH)
        df_out.to_csv(OUTPUT_PATH, mode='a', index=False, header=write_header)
        batch.clear()
        log.info(f"  Flushed {len(df_out)} rows to output")

    total = len(flagged)
    for idx, row in enumerate(flagged.itertuples(), 1):
        symbol = row.trade_symbol
        entry_date = row.trade_entry_date
        exit_date = row.trade_exit_date
        original_search = row.search_attempt

        if (symbol, entry_date) in already_done:
            log.info(f"[{idx}/{total}] {symbol} {entry_date} — skip (already done)")
            n_replaced += 1  # count toward replaced since it was written previously
            continue

        log.info(f"[{idx}/{total}] {symbol} {entry_date} → {exit_date} (was: {original_search})")

        # Get spot price
        spot = get_spot(symbol, entry_date)
        if not spot or spot <= 0:
            log.warning(f"  No spot for {symbol} {entry_date}, skip")
            n_no_monthly += 1
            continue

        # Get rvol (as decimal)
        sigma = load_rvol(symbol, entry_date)
        increment = get_strike_increment(spot)

        # Get monthly expiry candidates
        expiry_candidates = get_monthly_expiry_candidates(entry_date)
        if not expiry_candidates:
            log.warning(f"  No valid monthly expiries for {symbol} {entry_date}")
            n_no_monthly += 1
            continue

        found = False
        for exp_label, expiry in expiry_candidates:
            if found:
                break
            dte = (expiry - date.fromisoformat(entry_date)).days
            T = dte / 365.0
            strike_candidates = get_delta_strike_candidates(spot, sigma, T, increment)

            for s_label, strike in strike_candidates:
                ticker = make_ticker(symbol, expiry, strike)
                n_bars, day_vol = check_bars_with_volume(ticker, entry_date)

                if n_bars > 0 and day_vol > MIN_LIQUIDITY:
                    log.info(f"  HIT: {ticker} exp={exp_label} dte={dte} strike={strike} "
                             f"spot={spot:.2f} sigma={sigma:.3f} vol={day_vol:.0f}")

                    bars = download_full_hold(ticker, entry_date, exit_date)
                    if bars:
                        search_attempt = f"{exp_label}_{s_label}"
                        daily_rows = aggregate_to_daily(
                            bars, symbol, entry_date, exit_date,
                            ticker, strike, expiry,
                            search_attempt, original_search,
                            dte, day_vol
                        )
                        batch.extend(daily_rows)
                        n_replaced += 1
                        already_done.add((symbol, entry_date))
                        found = True
                        break
                    else:
                        log.warning(f"  HIT but empty hold-period bars: {ticker}")

        if not found:
            log.info(f"  NO MONTHLY FOUND for {symbol} {entry_date} — keeping original")
            n_no_monthly += 1

        if idx % 50 == 0:
            flush_batch()
            log.info(f"Progress: {idx}/{total} | replaced={n_replaced} | no_monthly={n_no_monthly}")

    flush_batch()
    log.info(f"\n{'='*60}")
    log.info(f"DONE monthly_replace.py")
    log.info(f"  Weekly/short-DTE trades flagged: {total}")
    log.info(f"  Successfully replaced with monthly: {n_replaced}")
    log.info(f"  Could not replace (no monthly found): {n_no_monthly}")
    log.info(f"  Output: {OUTPUT_PATH}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
