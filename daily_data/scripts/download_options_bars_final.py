#!/usr/bin/env python3
"""Options bar download - proper file version with resume and wider time windows."""

import os, sys, time, math, csv, glob
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
TRADES_CSV = "/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv"
OUT_CSV = "/home/ubuntu/daily_data/data/trade_options/options_bars_final.csv"
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

def get_unadj_close(symbol, date_str):
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{date_str}?adjusted=false&apiKey={API_KEY}"
    try:
        r = requests.get(url, timeout=10).json()
        if r.get("status") == "OK" and r.get("close"): return r["close"]
    except: pass
    return None

def get_bars_with_windows(ticker, date_str, utc_offset):
    """Get 1-min bars and try progressively wider windows."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date_str}/{date_str}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}"
    try:
        r = requests.get(url, timeout=10).json()
    except:
        return None, 0, 0, "error"
    bars = r.get("results", [])
    if not bars:
        return None, 0, 0, "no_bars"

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    h = 9 + utc_offset

    # Try progressively wider windows
    windows = [
        ("930-936", int(dt.replace(hour=h, minute=30).timestamp()*1000), int(dt.replace(hour=h, minute=36).timestamp()*1000)),
        ("936-1000", int(dt.replace(hour=h, minute=36).timestamp()*1000), int(dt.replace(hour=h+1, minute=0).timestamp()*1000)),
        ("1000-1100", int(dt.replace(hour=h+1, minute=0).timestamp()*1000), int(dt.replace(hour=h+2, minute=0).timestamp()*1000)),
    ]

    for window_name, start_ms, end_ms in windows:
        window_bars = [b for b in bars if start_ms <= b["t"] < end_ms]
        if window_bars:
            total_vol = sum(b.get("v", 0) for b in window_bars)
            if total_vol > 0:
                avg = sum(b["c"] * b.get("v", 1) for b in window_bars) / total_vol
            else:
                avg = np.mean([b["c"] for b in window_bars])
            return round(avg, 4), len(window_bars), total_vol, window_name

    # Last resort: first bar of day
    b = bars[0]
    return round(b["c"], 4), 1, b.get("v", 0), "first_bar"

# Load
trades = pd.read_csv(TRADES_CSV)
trades = trades[trades["entry_date"] >= "2022-01-01"].reset_index(drop=True)
print(f"Trades (2022+): {len(trades)}", flush=True)

rvol_lookup = {}
for f in glob.glob("/home/ubuntu/daily_data/data/*_enriched.csv"):
    sym = os.path.basename(f).replace("_enriched.csv", "")
    df = pd.read_csv(f, usecols=["date", "rvol_20d"])
    for _, row in df.iterrows():
        if pd.notna(row["rvol_20d"]): rvol_lookup[(sym, row["date"])] = row["rvol_20d"]
print(f"Rvol: {len(rvol_lookup):,}", flush=True)

# Build tasks
tasks = []
for _, t in trades.iterrows():
    for dt_type, dcol in [("entry", "entry_date"), ("exit", "exit_date")]:
        tasks.append({"symbol": t["symbol"], "date": t[dcol], "date_type": dt_type})
seen = set()
unique = []
for t in tasks:
    k = (t["symbol"], t["date"], t["date_type"])
    if k not in seen: seen.add(k); unique.append(t)
tasks = unique
print(f"Unique tasks: {len(tasks)}", flush=True)

# Resume
done_keys = set()
if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 100:
    existing = pd.read_csv(OUT_CSV)
    for _, r in existing.iterrows():
        done_keys.add((str(r["symbol"]), str(r["date"]), str(r["date_type"])))
    print(f"Already done: {len(done_keys)//8}", flush=True)

file_exists = os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 100
outf = open(OUT_CSV, "a", newline="")
writer = csv.writer(outf)
if not file_exists:
    writer.writerow(["symbol", "date", "date_type", "spot_unadjusted", "delta_target",
                     "option_ticker", "strike", "expiration", "dte",
                     "avg_price", "n_bars", "total_volume", "price_window"])

ok = 0; nodata = 0; skipped = 0
for i, task in enumerate(tasks):
    sym, date_str, dt_type = task["symbol"], task["date"], task["date_type"]
    if (sym, date_str, dt_type) in done_keys:
        skipped += 1
        continue
    if (i+1) % 50 == 0:
        print(f"  [{i+1}/{len(tasks)}] {sym} {date_str} {dt_type} (ok={ok} nodata={nodata} skip={skipped})", flush=True)
        outf.flush()

    spot = get_unadj_close(sym, date_str)
    time.sleep(0.08)
    if not spot or spot <= 0:
        for d in DELTA_Z:
            writer.writerow([sym, date_str, dt_type, "", d, "", "", "", "", "", 0, 0, "no_spot"])
        nodata += 1
        continue

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    utc_off = 4 if is_dst(dt) else 5
    vol = rvol_lookup.get((sym, date_str), 30)
    expiry = monthly_expiry(date_str)
    dte = (expiry - dt).days
    sig_sqrt_t = (max(vol, 15)/100) * math.sqrt(max(dte, 1)/365)

    got = False
    for delta, z in DELTA_Z.items():
        strike = round_strike(spot * math.exp(-z * sig_sqrt_t), spot)
        ticker = f"O:{sym}{expiry.strftime('%y%m%d')}C{int(strike*1000):08d}"
        avg_p, nb, tv, window = get_bars_with_windows(ticker, date_str, utc_off)
        time.sleep(0.08)
        if avg_p: got = True
        writer.writerow([sym, date_str, dt_type, spot, delta, ticker, strike,
                        expiry.strftime("%Y-%m-%d"), dte,
                        avg_p if avg_p else "", nb, tv, window])
    if got: ok += 1
    else: nodata += 1

outf.close()
print(f"\nDONE: ok={ok} nodata={nodata} skipped={skipped}", flush=True)
print(f"Output: {OUT_CSV}", flush=True)
