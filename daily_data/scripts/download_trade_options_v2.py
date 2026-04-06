#!/usr/bin/env python3
"""
Download options NBBO for each trade — v2.
Construct option tickers directly instead of using contracts API.

For each (symbol, date), for each delta target:
1. Compute approximate strike from spot + vol + DTE
2. Round to nearest standard strike increment ($1 for most, $5 for high-priced)
3. Build Polygon option ticker: O:SYMBOL{YYMMDD}C{strike*1000:08d}
4. Pull NBBO quotes 9:31-9:36 ET
5. Compute average mid

Find the nearest monthly expiry ~30 days out.
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
OUT_CSV = "/home/ubuntu/daily_data/data/trade_options/trade_options_nbbo.csv"

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

DELTA_Z = {80: 0.842, 70: 0.524, 60: 0.253, 50: 0.0, 40: -0.253, 30: -0.524, 20: -0.842, 10: -1.282}


def find_monthly_expiry(date_str):
    """Find 3rd Friday of the month ~25-40 days out."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    target = dt + timedelta(days=30)
    year, month = target.year, target.month

    first = datetime(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)

    if (third_friday - dt).days < 20:
        month += 1
        if month > 12:
            month = 1
            year += 1
        first = datetime(year, month, 1)
        first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)

    return third_friday


def get_strike_increment(spot):
    """Standard strike increments by price level."""
    if spot < 25:
        return 0.5
    elif spot < 200:
        return 1.0
    elif spot < 500:
        return 2.5
    else:
        return 5.0


def round_to_strike(price, spot):
    """Round to nearest valid strike price."""
    inc = get_strike_increment(spot)
    return round(price / inc) * inc


def build_option_ticker(symbol, expiry_dt, strike, call_put="C"):
    """Build Polygon option ticker: O:SYMBOL{YYMMDD}{C|P}{strike*1000:08d}"""
    date_part = expiry_dt.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    return f"O:{symbol}{date_part}{call_put}{strike_int:08d}"


def get_nbbo_avg(option_ticker, date_str, utc_offset):
    """Get average NBBO between 9:31-9:36 ET."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_hour_utc = 9 + utc_offset
    start_ns = int(dt.replace(hour=start_hour_utc, minute=31).timestamp() * 1e9)
    end_ns = int(dt.replace(hour=start_hour_utc, minute=36).timestamp() * 1e9)

    url = (f"https://api.polygon.io/v3/quotes/{option_ticker}"
           f"?timestamp.gte={start_ns}&timestamp.lte={end_ns}"
           f"&limit=500&sort=timestamp&apiKey={API_KEY}")

    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
    except:
        return None

    quotes = data.get("results", [])
    if not quotes:
        return None

    bids = [q["bid_price"] for q in quotes if q.get("bid_price", 0) > 0]
    asks = [q["ask_price"] for q in quotes if q.get("ask_price", 0) > 0]

    if not bids or not asks:
        return None

    return {
        "avg_bid": round(np.mean(bids), 4),
        "avg_ask": round(np.mean(asks), 4),
        "avg_mid": round((np.mean(bids) + np.mean(asks)) / 2, 4),
        "spread": round(np.mean(asks) - np.mean(bids), 4),
        "n_quotes": len(quotes),
    }


def is_dst(dt):
    year = dt.year
    mar_sun2 = datetime(year, 3, 8) + timedelta(days=(6 - datetime(year, 3, 8).weekday()) % 7)
    nov_sun1 = datetime(year, 11, 1) + timedelta(days=(6 - datetime(year, 11, 1).weekday()) % 7)
    return mar_sun2 <= dt < nov_sun1


def main():
    print("Loading trades...")
    trades = pd.read_csv(TRADES_CSV)
    print(f"Trades: {len(trades)}")

    print("Loading realized vol...")
    import glob
    rvol_lookup = {}
    for f in glob.glob("/home/ubuntu/daily_data/data/*_enriched.csv"):
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f, usecols=["date", "rvol_20d"])
        for _, row in df.iterrows():
            if pd.notna(row["rvol_20d"]):
                rvol_lookup[(sym, row["date"])] = row["rvol_20d"]
    print(f"Rvol entries: {len(rvol_lookup):,}")

    # Build unique (symbol, date, spot) tasks
    task_map = {}
    for _, trade in trades.iterrows():
        for dt_type in ["entry", "exit"]:
            date_str = trade["entry_date"] if dt_type == "entry" else trade["exit_date"]
            spot = trade["entry_price"] if dt_type == "entry" else trade["exit_price"]
            key = (trade["symbol"], date_str)
            if key not in task_map:
                task_map[key] = {"symbol": trade["symbol"], "date": date_str, "spot": spot}

    tasks = list(task_map.values())
    print(f"Unique (symbol, date) lookups: {len(tasks)}")

    # Remove existing output to start fresh
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    all_rows = []
    processed = 0
    errors = 0
    no_data = 0

    for i, task in enumerate(tasks):
        sym = task["symbol"]
        date_str = task["date"]
        spot = task["spot"]

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(tasks)}] {sym} {date_str} "
                  f"(rows: {len(all_rows)}, ok: {processed}, no_data: {no_data}, err: {errors})")

        rvol = rvol_lookup.get((sym, date_str), 30)
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        utc_offset = 4 if is_dst(dt) else 5

        expiry_dt = find_monthly_expiry(date_str)
        dte_days = (expiry_dt - dt).days
        dte_years = max(dte_days, 1) / 365.0
        sigma_sqrt_t = (max(rvol, 15) / 100) * math.sqrt(dte_years)

        got_any = False
        for delta, z in DELTA_Z.items():
            raw_strike = spot * math.exp(-z * sigma_sqrt_t)
            strike = round_to_strike(raw_strike, spot)
            ticker = build_option_ticker(sym, expiry_dt, strike, "C")

            nbbo = get_nbbo_avg(ticker, date_str, utc_offset)
            time.sleep(0.09)

            row = {
                "symbol": sym,
                "date": date_str,
                "spot_price": round(spot, 4),
                "rvol_20d": round(rvol, 1),
                "delta_target": delta,
                "option_ticker": ticker,
                "strike": strike,
                "expiration": expiry_dt.strftime("%Y-%m-%d"),
                "dte": dte_days,
            }

            if nbbo:
                row.update(nbbo)
                got_any = True
            else:
                row.update({"avg_bid": None, "avg_ask": None, "avg_mid": None,
                           "spread": None, "n_quotes": 0})

            all_rows.append(row)

        if got_any:
            processed += 1
        else:
            no_data += 1

        # Save incrementally
        if len(all_rows) >= 4000:
            batch_df = pd.DataFrame(all_rows)
            header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
            batch_df.to_csv(OUT_CSV, mode="a", header=header, index=False)
            print(f"    Flushed {len(all_rows)} rows to disk")
            all_rows = []

    # Final flush
    if all_rows:
        batch_df = pd.DataFrame(all_rows)
        header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
        batch_df.to_csv(OUT_CSV, mode="a", header=header, index=False)

    print(f"\n{'='*60}")
    print(f"DONE: {processed} with data, {no_data} no data, {errors} errors")
    print(f"Output: {OUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
