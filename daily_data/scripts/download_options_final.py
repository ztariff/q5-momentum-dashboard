#!/usr/bin/env python3
"""
Final options NBBO downloader. Fixes applied:
- 2022+ only (no NBBO before that)
- Unadjusted prices from Polygon for correct strikes
- Direct ticker construction
- Incremental save with resume support
"""

import os, sys, time, math, json, csv
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
TRADES_CSV = "/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv"
OUT_CSV = "/home/ubuntu/daily_data/data/trade_options/options_nbbo_final.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

DELTA_Z = {80: 0.842, 70: 0.524, 60: 0.253, 50: 0.0, 40: -0.253, 30: -0.524, 20: -0.842, 10: -1.282}


def is_dst(dt):
    yr = dt.year
    mar = datetime(yr, 3, 8) + timedelta(days=(6 - datetime(yr, 3, 8).weekday()) % 7)
    nov = datetime(yr, 11, 1) + timedelta(days=(6 - datetime(yr, 11, 1).weekday()) % 7)
    return mar <= dt < nov


def monthly_expiry(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    target = dt + timedelta(days=30)
    yr, mo = target.year, target.month
    first = datetime(yr, mo, 1)
    fri1 = first + timedelta(days=(4 - first.weekday()) % 7)
    fri3 = fri1 + timedelta(days=14)
    if (fri3 - dt).days < 20:
        mo += 1
        if mo > 12: mo, yr = 1, yr + 1
        first = datetime(yr, mo, 1)
        fri1 = first + timedelta(days=(4 - first.weekday()) % 7)
        fri3 = fri1 + timedelta(days=14)
    return fri3


def round_strike(price, spot):
    if spot < 25: inc = 0.5
    elif spot < 200: inc = 1.0
    elif spot < 500: inc = 2.5
    else: inc = 5.0
    return round(price / inc) * inc


def get_unadjusted_close(symbol, date_str):
    """Get the actual (unadjusted) close price from Polygon."""
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{date_str}?adjusted=false&apiKey={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") == "OK" and data.get("close"):
            return data["close"]
    except:
        pass
    return None


def get_nbbo(ticker, date_str, utc_offset):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    h = 9 + utc_offset
    start = dt.replace(hour=h, minute=31)
    end = dt.replace(hour=h, minute=36)
    sns = int(start.timestamp() * 1e9)
    ens = int(end.timestamp() * 1e9)

    url = (f"https://api.polygon.io/v3/quotes/{ticker}"
           f"?timestamp.gte={sns}&timestamp.lte={ens}&limit=500&sort=timestamp&apiKey={API_KEY}")
    try:
        resp = requests.get(url, timeout=10)
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


def main():
    trades = pd.read_csv(TRADES_CSV)
    trades = trades[trades["entry_date"] >= "2022-01-01"].reset_index(drop=True)
    print(f"Trades (2022+): {len(trades)}")

    # Load rvol lookup
    import glob
    rvol = {}
    for f in glob.glob("/home/ubuntu/daily_data/data/*_enriched.csv"):
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f, usecols=["date", "rvol_20d"])
        for _, r in df.iterrows():
            if pd.notna(r["rvol_20d"]):
                rvol[(sym, r["date"])] = r["rvol_20d"]
    print(f"Rvol entries: {len(rvol):,}")

    # Build unique tasks
    tasks = []
    for _, t in trades.iterrows():
        for dt_type, date_col in [("entry", "entry_date"), ("exit", "exit_date")]:
            tasks.append({"symbol": t["symbol"], "date": t[date_col], "date_type": dt_type})

    # Dedupe
    seen = set()
    unique = []
    for t in tasks:
        key = (t["symbol"], t["date"], t["date_type"])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    tasks = unique
    print(f"Unique tasks: {len(tasks)}")

    # Resume support
    done_keys = set()
    if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
        existing = pd.read_csv(OUT_CSV)
        for _, r in existing.iterrows():
            done_keys.add((r["symbol"], r["date"], r["date_type"]))
        print(f"Already done: {len(done_keys) // 8} lookups")

    # Open file for append
    file_exists = os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0
    outf = open(OUT_CSV, "a", newline="")
    writer = csv.writer(outf)
    if not file_exists:
        writer.writerow(["symbol", "date", "date_type", "spot_unadjusted", "rvol_20d",
                         "delta_target", "option_ticker", "strike", "expiration", "dte",
                         "avg_bid", "avg_ask", "avg_mid", "spread", "n_quotes"])

    ok = 0
    no_data = 0
    total = len(tasks)

    for i, task in enumerate(tasks):
        sym, date_str, dt_type = task["symbol"], task["date"], task["date_type"]

        if (sym, date_str, dt_type) in done_keys:
            continue

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] {sym} {date_str} {dt_type} (ok={ok}, no_data={no_data})")
            outf.flush()

        # Get unadjusted price
        spot = get_unadjusted_close(sym, date_str)
        time.sleep(0.08)

        if spot is None or spot <= 0:
            no_data += 1
            # Write empty rows
            for delta in DELTA_Z:
                writer.writerow([sym, date_str, dt_type, "", "", delta, "", "", "", "", "", "", "", "", 0])
            continue

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        utc_off = 4 if is_dst(dt) else 5
        vol = rvol.get((sym, date_str), 30)
        expiry = monthly_expiry(date_str)
        dte = (expiry - dt).days
        dte_yr = max(dte, 1) / 365.0
        sig_sqrt_t = (max(vol, 15) / 100) * math.sqrt(dte_yr)

        got_any = False
        for delta, z in DELTA_Z.items():
            strike = round_strike(spot * math.exp(-z * sig_sqrt_t), spot)
            ticker = f"O:{sym}{expiry.strftime('%y%m%d')}C{int(strike*1000):08d}"

            nbbo = get_nbbo(ticker, date_str, utc_off)
            time.sleep(0.08)

            if nbbo:
                writer.writerow([sym, date_str, dt_type, spot, round(vol, 1), delta, ticker,
                                strike, expiry.strftime("%Y-%m-%d"), dte,
                                nbbo["avg_bid"], nbbo["avg_ask"], nbbo["avg_mid"],
                                nbbo["spread"], nbbo["n_quotes"]])
                got_any = True
            else:
                writer.writerow([sym, date_str, dt_type, spot, round(vol, 1), delta, ticker,
                                strike, expiry.strftime("%Y-%m-%d"), dte, "", "", "", "", 0])

        if got_any:
            ok += 1
        else:
            no_data += 1

    outf.close()
    print(f"\nDONE: {ok} with data, {no_data} no data, total {total}")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
