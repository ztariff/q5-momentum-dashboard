#!/usr/bin/env python3
"""
Enrich the trade log with:
1. All 64 enriched columns for the SYMBOL on entry date and exit date
2. SPY context on entry and exit date (where is SPY relative to its MAs, 52w highs, etc.)
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "/home/ubuntu/daily_data/data"
TRADES_CSV = "/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv"
OUT_CSV = "/home/ubuntu/daily_data/analysis_results/trade1_20d_enriched.csv"

# Columns to pull from enriched data (skip raw price/timestamp, keep indicators)
INDICATOR_COLS = [
    "bb_pctb", "bb_bandwidth", "rsi_14",
    "sma_8", "sma_12", "sma_20", "sma_50", "sma_100", "sma_200",
    "ema_8", "ema_12", "ema_20", "ema_50",
    "atr_14", "rvol", "rvol_20d", "rvol_10d",
    "macd", "macd_signal", "macd_hist",
    "adx_14", "plus_di", "minus_di",
    "stoch_k", "stoch_d", "roc_10", "roc_20",
    "dist_sma8_pct", "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
    "daily_return_pct", "range_bps", "gap_bps",
    "close_location", "body_pct",
    "sma20_slope_5d", "sma50_slope_5d",
    "squeeze", "consec_days",
    "pct_from_20d_high", "pct_from_20d_low",
    "pct_from_52w_high", "pct_from_52w_low",
]


def main():
    print("Loading trades...")
    trades = pd.read_csv(TRADES_CSV)
    print(f"Trades: {len(trades)}")

    # Load all enriched data into a lookup: (symbol, date) -> dict of indicators
    print("Loading enriched data for all symbols...")
    sym_lookup = {}
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f, usecols=["date", "close"] + [c for c in INDICATOR_COLS if c != "close"])
        for _, row in df.iterrows():
            sym_lookup[(sym, row["date"])] = row.to_dict()

    print(f"Lookup entries: {len(sym_lookup):,}")

    # Load SPY enriched separately
    print("Loading SPY context...")
    spy = pd.read_csv(os.path.join(DATA_DIR, "SPY_enriched.csv"))
    spy_lookup = {}
    for _, row in spy.iterrows():
        spy_lookup[row["date"]] = row.to_dict()

    # SPY columns to attach
    spy_cols = [
        "close", "rsi_14", "sma_50", "sma_200",
        "dist_sma8_pct", "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
        "pct_from_52w_high", "pct_from_52w_low", "pct_from_20d_high", "pct_from_20d_low",
        "adx_14", "bb_pctb", "rvol_20d", "macd_hist", "squeeze", "daily_return_pct",
        "roc_10", "roc_20", "consec_days",
    ]

    # Build enriched trade log
    print("Enriching trades...")
    enriched_rows = []

    for _, trade in trades.iterrows():
        row = trade.to_dict()

        # Symbol indicators at ENTRY
        entry_data = sym_lookup.get((trade["symbol"], trade["entry_date"]), {})
        for col in INDICATOR_COLS:
            row[f"entry_{col}"] = entry_data.get(col)

        # Symbol indicators at EXIT
        exit_data = sym_lookup.get((trade["symbol"], trade["exit_date"]), {})
        for col in INDICATOR_COLS:
            row[f"exit_{col}"] = exit_data.get(col)

        # SPY context at ENTRY
        spy_entry = spy_lookup.get(trade["entry_date"], {})
        for col in spy_cols:
            row[f"spy_entry_{col}"] = spy_entry.get(col)

        # SPY context at EXIT
        spy_exit = spy_lookup.get(trade["exit_date"], {})
        for col in spy_cols:
            row[f"spy_exit_{col}"] = spy_exit.get(col)

        enriched_rows.append(row)

    enriched = pd.DataFrame(enriched_rows)
    enriched.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")
    print(f"Rows: {len(enriched)}")
    print(f"Columns: {len(enriched.columns)}")

    # Print column summary
    print(f"\nColumn groups:")
    base = [c for c in enriched.columns if not c.startswith("entry_") and not c.startswith("exit_") and not c.startswith("spy_")]
    entry_sym = [c for c in enriched.columns if c.startswith("entry_")]
    exit_sym = [c for c in enriched.columns if c.startswith("exit_")]
    spy_entry = [c for c in enriched.columns if c.startswith("spy_entry_")]
    spy_exit = [c for c in enriched.columns if c.startswith("spy_exit_")]

    print(f"  Base trade fields:           {len(base)}")
    print(f"  Symbol indicators at entry:  {len(entry_sym)}")
    print(f"  Symbol indicators at exit:   {len(exit_sym)}")
    print(f"  SPY context at entry:        {len(spy_entry)}")
    print(f"  SPY context at exit:         {len(spy_exit)}")
    print(f"  TOTAL:                       {len(enriched.columns)}")

    # Quick sample
    print(f"\nSample — trade 0 SPY context:")
    r = enriched.iloc[0]
    print(f"  Entry date: {r['entry_date']}, Symbol: {r['symbol']}")
    print(f"  SPY close at entry:        ${r['spy_entry_close']:.2f}")
    print(f"  SPY dist from 50 SMA:      {r['spy_entry_dist_sma50_pct']:+.1f}%")
    print(f"  SPY dist from 200 SMA:     {r['spy_entry_dist_sma200_pct']:+.1f}%")
    print(f"  SPY pct from 52w high:     {r['spy_entry_pct_from_52w_high']:+.1f}%")
    print(f"  SPY RSI:                   {r['spy_entry_rsi_14']:.1f}")
    print(f"  Symbol RSI at entry:       {r['entry_rsi_14']:.1f}")
    print(f"  Symbol dist from 50 SMA:   {r['entry_dist_sma50_pct']:+.1f}%")
    print(f"  Symbol pct from 52w high:  {r['entry_pct_from_52w_high']:+.1f}%")


if __name__ == "__main__":
    main()
