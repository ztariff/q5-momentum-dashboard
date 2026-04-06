#!/usr/bin/env python3
"""
Download daily bars for a broad universe of in-play mid/large cap stocks.
Covers 2015-01-01 through 2026-04-03.
Source: Polygon.io
"""

import os
import time
import requests
import csv
from datetime import datetime

API_KEY = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
START = "2015-01-01"
END = "2026-04-03"
OUT_DIR = "/home/ubuntu/daily_data/data"

# ── Universe: ~200 names across every major tradeable sector ──

SYMBOLS = {
    # ── Mega-cap tech ──
    "mega_tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
        "ORCL", "ADBE", "INTC", "CSCO", "IBM",
    ],
    # ── Semiconductors ──
    "semis": [
        "AMD", "MU", "AVGO", "QCOM", "MRVL", "ON", "AMAT", "LRCX",
        "KLAC", "TSM", "TXN", "ADI", "SMCI", "ARM", "MCHP", "SWKS",
    ],
    # ── Software / Cloud / SaaS ──
    "software": [
        "CRM", "NOW", "SNOW", "PLTR", "NET", "DDOG", "ZS", "CRWD",
        "PANW", "SHOP", "SQ", "TWLO", "U", "ROKU", "TTD", "HUBS",
        "WDAY", "ZM", "DOCU", "OKTA", "MDB", "TEAM", "VEEV", "BILL",
        "PATH", "AI",
    ],
    # ── EV / Clean energy ──
    "ev_clean": [
        "RIVN", "LCID", "NIO", "XPEV", "LI", "ENPH", "SEDG", "FSLR",
        "RUN", "PLUG", "CHPT",
    ],
    # ── Biotech / Pharma ──
    "biotech_pharma": [
        "MRNA", "BNTX", "PFE", "JNJ", "ABBV", "LLY", "NVO", "AMGN",
        "BIIB", "GILD", "REGN", "VRTX", "BMY", "MRK", "AZN",
    ],
    # ── Financials ──
    "financials": [
        "JPM", "GS", "MS", "BAC", "C", "WFC", "SCHW", "BLK", "AXP",
        "COF", "V", "MA", "PYPL",
    ],
    # ── Crypto-adjacent ──
    "crypto": [
        "COIN", "MSTR", "MARA", "RIOT", "HUT",
    ],
    # ── Fintech / Brokers ──
    "fintech": [
        "HOOD", "SOFI", "AFRM", "UPST",
    ],
    # ── Retail / Consumer ──
    "consumer": [
        "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
        "DIS", "ABNB", "UBER", "LYFT", "DASH", "CMG", "LULU",
        "PTON", "W", "ETSY", "BKNG",
    ],
    # ── Energy ──
    "energy": [
        "XOM", "CVX", "OXY", "COP", "SLB", "DVN", "MPC", "VLO",
        "HAL", "PSX", "EOG", "PXD",
    ],
    # ── Metals / Mining / Materials ──
    "materials": [
        "FCX", "NEM", "CLF", "X", "AA", "VALE", "NUE", "MP",
    ],
    # ── Defense / Industrials ──
    "industrials": [
        "LMT", "RTX", "NOC", "BA", "GE", "CAT", "DE", "HON",
        "UPS", "FDX", "MMM",
    ],
    # ── Healthcare / Medtech ──
    "healthcare": [
        "UNH", "ISRG", "DXCM", "TMO", "ABT", "HCA", "ELV", "CI",
        "SYK", "ZTS", "GEHC",
    ],
    # ── China ADRs ──
    "china": [
        "BABA", "JD", "PDD", "BIDU", "BEKE", "KWEB",
    ],
    # ── Meme / Momentum / Speculative ──
    "meme_momentum": [
        "GME", "AMC", "SPCE", "BYND", "DKNG", "LAZR", "QS",
        "CVNA", "CLOV", "IONQ", "RKLB", "JOBY",
    ],
    # ── Airlines / Travel / Leisure ──
    "travel": [
        "DAL", "UAL", "AAL", "LUV", "CCL", "RCL", "NCLH",
        "MAR", "HLT", "EXPE",
    ],
    # ── Telecom / Media / Streaming ──
    "media_telecom": [
        "T", "VZ", "CMCSA", "PARA", "WBD", "SPOT", "RBLX",
    ],
    # ── REITs / Real Estate ──
    "reits": [
        "AMT", "PLD", "SPG", "O", "EQIX",
    ],
    # ── Key ETFs (sector + market) ──
    "etfs": [
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XBI",
        "ARKK", "GLD", "SLV", "TLT", "HYG", "EEM", "XOP", "SMH",
        "KWEB", "SOXX", "XLV", "XLI", "XLP", "XLU", "XLY", "XLC",
        "VXX", "SOXL", "TQQQ",
    ],
}

# Flatten
ALL_SYMBOLS = sorted(set(sym for group in SYMBOLS.values() for sym in group))


def download_daily(symbol):
    """Download daily bars for one symbol, return row count or -1 on failure."""
    url = (
        f"{BASE_URL}/{symbol}/range/1/day/{START}/{END}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"  ERROR {symbol}: {e}")
        return -1

    if data.get("resultsCount", 0) == 0 or "results" not in data:
        print(f"  SKIP {symbol}: no data (status={data.get('status')})")
        return 0

    rows = data["results"]
    out_path = os.path.join(OUT_DIR, f"{symbol}_daily.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "date", "open", "high", "low", "close",
            "volume", "vwap", "transactions"
        ])
        for r in rows:
            ts = r["t"]
            dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            writer.writerow([
                ts, dt,
                r.get("o"), r.get("h"), r.get("l"), r.get("c"),
                r.get("v"), r.get("vw"), r.get("n"),
            ])
    return len(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading daily bars for {len(ALL_SYMBOLS)} symbols")
    print(f"Period: {START} to {END}")
    print(f"Output: {OUT_DIR}/\n")

    results = {}
    failures = []
    total = len(ALL_SYMBOLS)

    for i, sym in enumerate(ALL_SYMBOLS, 1):
        print(f"[{i:3d}/{total}] {sym}...", end=" ", flush=True)
        n = download_daily(sym)
        if n > 0:
            print(f"{n} bars")
            results[sym] = n
        elif n == 0:
            results[sym] = 0
        else:
            failures.append(sym)
            results[sym] = -1
        # Polygon rate limit: ~5/sec on paid plan, be polite
        time.sleep(0.25)

    # Summary
    ok = sum(1 for v in results.values() if v > 0)
    skip = sum(1 for v in results.values() if v == 0)
    fail = len(failures)
    total_bars = sum(v for v in results.values() if v > 0)

    summary = (
        f"\n{'='*60}\n"
        f"DOWNLOAD COMPLETE\n"
        f"  Symbols requested: {total}\n"
        f"  Downloaded:        {ok}\n"
        f"  No data:           {skip}\n"
        f"  Errors:            {fail}\n"
        f"  Total bars:        {total_bars:,}\n"
        f"{'='*60}\n"
    )
    if failures:
        summary += f"  Failed: {', '.join(failures)}\n"

    print(summary)

    # Save manifest
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "bars", "status"])
        for sym in ALL_SYMBOLS:
            n = results[sym]
            status = "ok" if n > 0 else ("no_data" if n == 0 else "error")
            writer.writerow([sym, max(n, 0), status])

    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
