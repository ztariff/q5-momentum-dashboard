#!/usr/bin/env python3
"""
Download 6-month and 12-month out historical options for each symbol
while it was in the portfolio.

Uses portfolio_daily.csv to know which symbols were held on which dates.
Samples WEEKLY (every Monday or first trading day of week) to keep API calls sane.
Downloads ATM ± 5 strikes for puts and calls at each snapshot.

Source: Polygon.io
"""

import os
import time
import json
import csv
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
OUT_DIR = "/home/ubuntu/daily_data/data/options"
PORTFOLIO_CSV = "/home/ubuntu/daily_data/analysis_results/portfolio_daily.csv"

os.makedirs(OUT_DIR, exist_ok=True)


def get_options_chain(symbol, date_str, exp_from, exp_to, strike_price, n_strikes=5):
    """
    Get options contracts for a symbol near a given strike price,
    with expiration between exp_from and exp_to.
    Returns list of contract snapshots.
    """
    # Find contracts
    url = (
        f"https://api.polygon.io/v3/reference/options/contracts"
        f"?underlying_ticker={symbol}"
        f"&expiration_date.gte={exp_from}"
        f"&expiration_date.lte={exp_to}"
        f"&strike_price.gte={strike_price * 0.90}"
        f"&strike_price.lte={strike_price * 1.10}"
        f"&limit=100"
        f"&apiKey={API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
    except Exception as e:
        return []

    if "results" not in data:
        return []

    contracts = data["results"]
    if not contracts:
        return []

    # Sort by distance from ATM
    for c in contracts:
        c["_dist"] = abs(c.get("strike_price", 0) - strike_price)

    contracts.sort(key=lambda c: c["_dist"])

    # Take nearest n_strikes calls and n_strikes puts
    calls = [c for c in contracts if c.get("contract_type") == "call"][:n_strikes]
    puts = [c for c in contracts if c.get("contract_type") == "put"][:n_strikes]
    selected = calls + puts

    # Get EOD snapshot for each on the given date
    results = []
    for contract in selected:
        ticker = contract.get("ticker", "")
        snap_url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date_str}/{date_str}"
            f"?adjusted=true&apiKey={API_KEY}"
        )
        try:
            snap_resp = requests.get(snap_url, timeout=10)
            snap_data = snap_resp.json()
        except:
            continue

        if "results" in snap_data and snap_data["results"]:
            bar = snap_data["results"][0]
            results.append({
                "date": date_str,
                "symbol": symbol,
                "option_ticker": ticker,
                "contract_type": contract.get("contract_type"),
                "strike": contract.get("strike_price"),
                "expiration": contract.get("expiration_date"),
                "underlying_close": strike_price,
                "option_open": bar.get("o"),
                "option_high": bar.get("h"),
                "option_low": bar.get("l"),
                "option_close": bar.get("c"),
                "option_volume": bar.get("v"),
                "option_vwap": bar.get("vw"),
                "option_transactions": bar.get("n"),
            })
            time.sleep(0.08)  # rate limit

    return results


def main():
    print("Loading portfolio...")
    pdf = pd.read_csv(PORTFOLIO_CSV)
    pdf["date"] = pd.to_datetime(pdf["date"])

    # Sample weekly (every ~5 trading days)
    pdf_weekly = pdf.iloc[::5].copy()
    print(f"Portfolio days: {len(pdf)}, sampling weekly: {len(pdf_weekly)} snapshots\n")

    # Get unique (symbol, date, close_price) tuples
    tasks = []
    for _, row in pdf_weekly.iterrows():
        date = row["date"]
        date_str = date.strftime("%Y-%m-%d")

        # Parse long and short names
        for side, names_col in [("long", "long_names"), ("short", "short_names")]:
            names = str(row[names_col]).split("|") if pd.notna(row[names_col]) else []
            for sym in names:
                if not sym or sym == "nan":
                    continue
                tasks.append({
                    "symbol": sym,
                    "date": date_str,
                    "side": side,
                })

    print(f"Total symbol-date pairs: {len(tasks)}")

    # We need the closing price for each symbol on each date to find ATM strikes
    # Load from enriched files
    print("Loading closing prices...")
    price_lookup = {}
    import glob
    for f in glob.glob(os.path.join("/home/ubuntu/daily_data/data", "*_enriched.csv")):
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f, usecols=["date", "close"])
        for _, r in df.iterrows():
            price_lookup[(sym, r["date"])] = r["close"]

    print(f"Price lookup: {len(price_lookup):,} entries\n")

    # Deduplicate tasks (same symbol can be in portfolio on multiple weeks)
    seen = set()
    unique_tasks = []
    for t in tasks:
        key = (t["symbol"], t["date"])
        if key not in seen:
            seen.add(key)
            price = price_lookup.get(key)
            if price and price > 0:
                t["close"] = price
                unique_tasks.append(t)

    print(f"Unique symbol-date pairs: {len(unique_tasks)}")

    # Only download for 2023+ (holdout period, more recent, more likely to have data)
    unique_tasks = [t for t in unique_tasks if t["date"] >= "2023-01-01"]
    print(f"After filtering to 2023+: {len(unique_tasks)}")

    # Process
    all_results = []
    errors = 0
    processed = 0

    # Save incrementally
    out_path = os.path.join(OUT_DIR, "portfolio_options.csv")
    header_written = False

    for i, task in enumerate(unique_tasks):
        sym = task["symbol"]
        date_str = task["date"]
        close = task["close"]

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(unique_tasks)}] {sym} {date_str} (errors: {errors}, rows: {len(all_results)})")

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        # 6-month window: 150-210 days out
        exp_6m_from = (date_obj + timedelta(days=150)).strftime("%Y-%m-%d")
        exp_6m_to = (date_obj + timedelta(days=210)).strftime("%Y-%m-%d")

        # 12-month window: 330-400 days out
        exp_12m_from = (date_obj + timedelta(days=330)).strftime("%Y-%m-%d")
        exp_12m_to = (date_obj + timedelta(days=400)).strftime("%Y-%m-%d")

        for tenor, ef, et in [("6m", exp_6m_from, exp_6m_to), ("12m", exp_12m_from, exp_12m_to)]:
            results = get_options_chain(sym, date_str, ef, et, close, n_strikes=5)
            for r in results:
                r["tenor"] = tenor
                r["side"] = task["side"]
            all_results.extend(results)
            time.sleep(0.12)

        # Save every 200 symbols
        if len(all_results) > 0 and (i + 1) % 200 == 0:
            batch_df = pd.DataFrame(all_results)
            batch_df.to_csv(out_path, mode="a", header=not header_written, index=False)
            header_written = True
            print(f"    Saved {len(all_results)} rows to {out_path}")
            all_results = []

        processed += 1

    # Final save
    if all_results:
        batch_df = pd.DataFrame(all_results)
        batch_df.to_csv(out_path, mode="a", header=not header_written, index=False)

    print(f"\n{'='*60}")
    print(f"DONE: {processed} symbol-dates processed, {errors} errors")
    print(f"Output: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
