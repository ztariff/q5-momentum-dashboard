#!/usr/bin/env python3
"""
Download options NBBO data for each of the 2,875 trades.

For each trade, on ENTRY DATE and EXIT DATE:
- Find the nearest monthly expiration ~30 days out (liquid options)
- Get the 80, 70, 60, 50, 40, 30, 20, 10 delta calls
- Pull NBBO quotes from 9:31-9:36 AM ET
- Compute average mid price over that window

Approach:
1. For each (symbol, date), find available option contracts
2. Estimate delta from moneyness (Black-Scholes-ish) since Polygon doesn't give greeks
   - Instead: pull strikes around ATM and label by approximate delta using
     strike distance from spot as a proxy
3. Get NBBO snapshots in the 9:31-9:36 window

Since true delta requires IV which requires the options prices themselves (circular),
we'll use a strike-spacing approach:
- ATM = closest strike to spot → ~50 delta
- Map deltas to strike offsets using the stock's recent realized vol
- 80 delta ≈ spot - 0.84σ√T, 20 delta ≈ spot + 0.84σ√T, etc.

Actually — simpler and more accurate: download ALL strikes near ATM, get their prices,
compute implied delta from the prices, then interpolate to exact delta targets.

SIMPLEST CORRECT APPROACH: Just download strikes at fixed moneyness levels that
approximately correspond to standard deltas, get their NBBO, and label them.
The user can compute exact greeks later from the prices.

Delta → approximate moneyness (for ~30 DTE, ~25% vol):
  80 delta ≈ 96% of spot (4% ITM)
  70 delta ≈ 98% of spot (2% ITM)
  60 delta ≈ 99% of spot (1% ITM)
  50 delta ≈ 100% of spot (ATM)
  40 delta ≈ 101% of spot (1% OTM)
  30 delta ≈ 102.5% of spot (2.5% OTM)
  20 delta ≈ 104.5% of spot (4.5% OTM)
  10 delta ≈ 108% of spot (8% OTM)

These are rough. For higher-vol stocks the spacing is wider.
We'll use realized vol to adjust.
"""

import os
import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
TRADES_CSV = "/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv"
OUT_DIR = "/home/ubuntu/daily_data/data/trade_options"
OUT_CSV = os.path.join(OUT_DIR, "trade_options_nbbo.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# Target deltas and their approximate z-scores (standard normal quantile)
# delta ≈ N(d1), so d1 ≈ N_inv(delta)
# For a call: strike = spot × exp(-d1 × σ√T + 0.5σ²T) ≈ spot × (1 - d1 × σ√T) for small σ√T
DELTA_TARGETS = {
    80: 0.842,   # N_inv(0.80)
    70: 0.524,
    60: 0.253,
    50: 0.0,
    40: -0.253,
    30: -0.524,
    20: -0.842,
    10: -1.282,
}


def get_strike_for_delta(spot, delta_target, rvol_annual, dte_years):
    """Estimate strike price for a given delta using BS approximation."""
    z = DELTA_TARGETS[delta_target]
    sigma_sqrt_t = (rvol_annual / 100) * math.sqrt(dte_years)
    # strike ≈ spot × exp(-z × σ√T)  (simplified, ignoring drift)
    strike = spot * math.exp(-z * sigma_sqrt_t)
    return strike


def find_nearest_monthly_expiry(date_str):
    """Find the nearest standard monthly options expiry (3rd Friday) ~25-40 days out."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    target = dt + timedelta(days=30)

    # Find 3rd Friday of target month
    year, month = target.year, target.month
    # First day of month
    first = datetime(year, month, 1)
    # Day of week (0=Mon, 4=Fri)
    # First Friday
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)

    # If third Friday is too close (< 20 days), use next month
    if (third_friday - dt).days < 20:
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        first = datetime(year, month, 1)
        first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)

    return third_friday.strftime("%Y-%m-%d")


def get_option_contracts(symbol, date_str, expiry_str, spot, rvol):
    """Find option contracts near the target strikes for each delta."""
    # Compute target strikes
    dte_days = (datetime.strptime(expiry_str, "%Y-%m-%d") - datetime.strptime(date_str, "%Y-%m-%d")).days
    dte_years = max(dte_days, 1) / 365.0

    target_strikes = {}
    for delta in DELTA_TARGETS:
        target_strikes[delta] = get_strike_for_delta(spot, delta, max(rvol, 15), dte_years)

    # Get all call contracts for this symbol/expiry
    min_strike = min(target_strikes.values()) * 0.95
    max_strike = max(target_strikes.values()) * 1.05

    url = (
        f"https://api.polygon.io/v3/reference/options/contracts"
        f"?underlying_ticker={symbol}"
        f"&expiration_date={expiry_str}"
        f"&contract_type=call"
        f"&strike_price.gte={min_strike:.2f}"
        f"&strike_price.lte={max_strike:.2f}"
        f"&limit=100"
        f"&apiKey={API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
    except Exception as e:
        return {}

    if "results" not in data or not data["results"]:
        return {}

    contracts = data["results"]

    # For each delta target, find the closest strike
    matched = {}
    for delta, target_strike in target_strikes.items():
        best = min(contracts, key=lambda c: abs(c.get("strike_price", 0) - target_strike))
        matched[delta] = {
            "ticker": best["ticker"],
            "strike": best["strike_price"],
            "target_strike": round(target_strike, 2),
            "expiration": best["expiration_date"],
        }

    return matched


def get_nbbo_window(option_ticker, date_str):
    """
    Get NBBO quotes for an option between 9:31 and 9:36 AM ET.
    Returns average bid, ask, mid over the window.
    """
    # Convert to timestamps
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    # 9:31 AM ET in nanoseconds from epoch
    # ET = UTC-5 (EST) or UTC-4 (EDT)
    # Determine DST: roughly Mar second Sun to Nov first Sun
    year = dt.year
    # Simple DST check
    mar_second_sun = datetime(year, 3, 8) + timedelta(days=(6 - datetime(year, 3, 8).weekday()) % 7)
    nov_first_sun = datetime(year, 11, 1) + timedelta(days=(6 - datetime(year, 11, 1).weekday()) % 7)
    is_dst = mar_second_sun <= dt < nov_first_sun
    utc_offset = 4 if is_dst else 5

    start_utc = dt.replace(hour=9 + utc_offset, minute=31, second=0)
    end_utc = dt.replace(hour=9 + utc_offset, minute=36, second=0)

    start_ns = int(start_utc.timestamp() * 1e9)
    end_ns = int(end_utc.timestamp() * 1e9)

    # Use quotes endpoint
    url = (
        f"https://api.polygon.io/v3/quotes/{option_ticker}"
        f"?timestamp.gte={start_ns}"
        f"&timestamp.lte={end_ns}"
        f"&limit=500"
        f"&sort=timestamp"
        f"&apiKey={API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
    except:
        return None

    if "results" not in data or not data["results"]:
        return None

    quotes = data["results"]
    bids = [q["bid_price"] for q in quotes if q.get("bid_price", 0) > 0]
    asks = [q["ask_price"] for q in quotes if q.get("ask_price", 0) > 0]

    if not bids or not asks:
        return None

    avg_bid = np.mean(bids)
    avg_ask = np.mean(asks)
    avg_mid = (avg_bid + avg_ask) / 2

    return {
        "avg_bid": round(avg_bid, 4),
        "avg_ask": round(avg_ask, 4),
        "avg_mid": round(avg_mid, 4),
        "spread": round(avg_ask - avg_bid, 4),
        "n_quotes": len(quotes),
    }


def load_rvol():
    """Load realized vol from enriched files for each (symbol, date)."""
    import glob
    rvol_lookup = {}
    for f in glob.glob(os.path.join("/home/ubuntu/daily_data/data", "*_enriched.csv")):
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f, usecols=["date", "rvol_20d"])
        for _, row in df.iterrows():
            if pd.notna(row["rvol_20d"]):
                rvol_lookup[(sym, row["date"])] = row["rvol_20d"]
    return rvol_lookup


def main():
    print("Loading trades...")
    trades = pd.read_csv(TRADES_CSV)
    print(f"Trades: {len(trades)}")

    print("Loading realized vol lookup...")
    rvol_lookup = load_rvol()
    print(f"Rvol entries: {len(rvol_lookup):,}")

    # Build task list: each trade has 2 dates (entry, exit) × 8 deltas
    tasks = []
    for _, trade in trades.iterrows():
        for date_type in ["entry", "exit"]:
            date_str = trade["entry_date"] if date_type == "entry" else trade["exit_date"]
            price = trade["entry_price"] if date_type == "entry" else trade["exit_price"]
            tasks.append({
                "trade_idx": _,
                "symbol": trade["symbol"],
                "date": date_str,
                "date_type": date_type,
                "spot": price,
                "z_score": trade["z_score"],
                "return_pct": trade["return_pct"],
            })

    print(f"Total tasks: {len(tasks)} (2 dates × {len(trades)} trades)")

    # Deduplicate by (symbol, date) — many trades share the same lookup
    unique_lookups = {}
    for t in tasks:
        key = (t["symbol"], t["date"])
        if key not in unique_lookups:
            unique_lookups[key] = t
    print(f"Unique (symbol, date) lookups: {len(unique_lookups)}")

    # Process
    all_rows = []
    processed = 0
    errors = 0
    header_written = os.path.exists(OUT_CSV)

    for i, ((sym, date_str), task) in enumerate(unique_lookups.items()):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(unique_lookups)}] {sym} {date_str} (rows: {len(all_rows)}, errors: {errors})")

        spot = task["spot"]
        rvol = rvol_lookup.get((sym, date_str), 30)  # default 30% vol

        # Find expiry
        expiry = find_nearest_monthly_expiry(date_str)

        # Get contracts for each delta
        contracts = get_option_contracts(sym, date_str, expiry, spot, rvol)
        time.sleep(0.12)

        if not contracts:
            errors += 1
            continue

        # Get NBBO for each delta
        for delta, contract_info in contracts.items():
            nbbo = get_nbbo_window(contract_info["ticker"], date_str)
            time.sleep(0.12)

            row = {
                "symbol": sym,
                "date": date_str,
                "spot_price": round(spot, 4),
                "rvol_20d": round(rvol, 1),
                "delta_target": delta,
                "option_ticker": contract_info["ticker"],
                "strike": contract_info["strike"],
                "target_strike": contract_info["target_strike"],
                "expiration": contract_info["expiration"],
                "dte": (datetime.strptime(contract_info["expiration"], "%Y-%m-%d") -
                       datetime.strptime(date_str, "%Y-%m-%d")).days,
            }

            if nbbo:
                row.update({
                    "avg_bid": nbbo["avg_bid"],
                    "avg_ask": nbbo["avg_ask"],
                    "avg_mid": nbbo["avg_mid"],
                    "spread": nbbo["spread"],
                    "n_quotes": nbbo["n_quotes"],
                })
            else:
                row.update({"avg_bid": None, "avg_ask": None, "avg_mid": None,
                           "spread": None, "n_quotes": 0})

            all_rows.append(row)

        processed += 1

        # Save every 500 lookups
        if len(all_rows) > 0 and (i + 1) % 500 == 0:
            batch_df = pd.DataFrame(all_rows)
            batch_df.to_csv(OUT_CSV, mode="a", header=not header_written, index=False)
            header_written = True
            print(f"    Saved {len(all_rows)} rows (total on disk)")
            all_rows = []

    # Final save
    if all_rows:
        batch_df = pd.DataFrame(all_rows)
        batch_df.to_csv(OUT_CSV, mode="a", header=not header_written, index=False)

    print(f"\n{'='*60}")
    print(f"DONE: {processed}/{len(unique_lookups)} lookups, {errors} errors")
    print(f"Output: {OUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
