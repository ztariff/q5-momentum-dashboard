#!/usr/bin/env python3
"""
TRADE 1 DEEP DIVE v2: Memory-efficient version.
Compute path data only for entry rows.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"
MAX_HOLD = 20


def load_base():
    """Load minimal columns, assign quintiles, find entries."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    all_dfs = []
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f, usecols=["date", "open", "high", "low", "close", "volume",
                                      "rsi_14", "adx_14", "bb_bandwidth", "rvol",
                                      "dist_sma50_pct", "daily_return_pct", "close_location",
                                      "gap_bps", "range_bps", "consec_days"])
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = sym
        c = df["close"]

        sma50 = c.rolling(50).mean()
        raw_slope = 100 * (sma50 / sma50.shift(10) - 1)
        roll_m = raw_slope.rolling(252, min_periods=60).mean()
        roll_s = raw_slope.rolling(252, min_periods=60).std()
        df["z_signal"] = (raw_slope - roll_m) / roll_s.replace(0, np.nan)
        df["rvol_20d"] = (np.log(c / c.shift(1))).rolling(20).std() * np.sqrt(252) * 100

        all_dfs.append(df)
    pooled = pd.concat(all_dfs, ignore_index=True)
    return pooled


def assign_q5_entries(pooled):
    pooled["quintile"] = np.nan
    for date, ddf in pooled.groupby("date"):
        valid = ddf["z_signal"].dropna()
        if len(valid) < 20:
            continue
        try:
            q = pd.qcut(valid, 5, labels=[1,2,3,4,5], duplicates="drop")
            pooled.loc[q.index, "quintile"] = q.values
        except ValueError:
            continue

    pooled = pooled.sort_values(["symbol", "date"]).reset_index(drop=True)
    pooled["prev_q"] = pooled.groupby("symbol")["quintile"].shift(1)
    pooled["entered_q5"] = (pooled["quintile"] == 5) & (pooled["prev_q"] != 5)
    return pooled


def build_entry_paths(pooled, entry_mask):
    """For each entry row, look ahead MAX_HOLD days and build the daily path."""
    entries = pooled[entry_mask].copy()
    # Build lookup: (symbol, date) -> row index in pooled
    pooled_indexed = pooled.set_index(["symbol", "date"]).sort_index()

    path_data = []
    for idx, row in entries.iterrows():
        sym = row["symbol"]
        entry_date = row["date"]
        entry_close = row["close"]
        entry_open_next = None
        overnight = None

        if entry_close <= 0 or pd.isna(entry_close):
            continue

        # Get future rows for this symbol
        try:
            sym_data = pooled_indexed.loc[sym]
        except KeyError:
            continue

        future = sym_data[sym_data.index > entry_date].head(MAX_HOLD)
        if len(future) < 3:
            continue

        # Overnight
        if len(future) > 0:
            entry_open_next = future.iloc[0]["open"]
            overnight = 10000 * (entry_open_next / entry_close - 1) if entry_open_next > 0 else np.nan

        # Build path arrays
        day_lows = []
        day_highs = []
        day_closes = []
        for i in range(min(MAX_HOLD, len(future))):
            day_lows.append(10000 * (future.iloc[i]["low"] / entry_close - 1))
            day_highs.append(10000 * (future.iloc[i]["high"] / entry_close - 1))
            day_closes.append(10000 * (future.iloc[i]["close"] / entry_close - 1))

        # Pad if needed
        while len(day_lows) < MAX_HOLD:
            day_lows.append(np.nan)
            day_highs.append(np.nan)
            day_closes.append(np.nan)

        path_row = {
            "idx": idx,
            "symbol": sym,
            "date": entry_date,
            "entry_close": entry_close,
            "overnight": overnight,
            "z_signal": row["z_signal"],
            "rvol_20d": row.get("rvol_20d"),
            "rsi_14": row.get("rsi_14"),
            "gap_bps": row.get("gap_bps"),
            "range_bps": row.get("range_bps"),
            "daily_return_pct": row.get("daily_return_pct"),
            "close_location": row.get("close_location"),
            "bb_bandwidth": row.get("bb_bandwidth"),
            "adx_14": row.get("adx_14"),
            "consec_days": row.get("consec_days"),
            "dist_sma50_pct": row.get("dist_sma50_pct"),
        }
        for d in range(MAX_HOLD):
            path_row[f"d{d+1}_low"] = day_lows[d]
            path_row[f"d{d+1}_high"] = day_highs[d]
            path_row[f"d{d+1}_close"] = day_closes[d]

        path_data.append(path_row)

    return pd.DataFrame(path_data)


def simulate(paths, stop, tp, max_hold):
    """Simulate stop/TP on path data."""
    pnls = []
    for _, row in paths.iterrows():
        exited = False
        for d in range(1, max_hold + 1):
            lo = row.get(f"d{d}_low")
            hi = row.get(f"d{d}_high")
            cl = row.get(f"d{d}_close")
            if pd.isna(lo) or pd.isna(hi):
                break
            if stop is not None and lo <= stop:
                pnls.append(stop)
                exited = True
                break
            if tp is not None and hi >= tp:
                pnls.append(tp)
                exited = True
                break
        if not exited:
            final = row.get(f"d{max_hold}_close")
            if not pd.isna(final):
                pnls.append(final)

    if not pnls:
        return None
    pnls = np.array(pnls)
    sr = pnls.mean() / pnls.std() * np.sqrt(252 / max_hold) if pnls.std() > 0 else 0
    return {
        "n": len(pnls), "mean": round(pnls.mean(), 2), "median": round(np.median(pnls), 2),
        "wr": round(100 * (pnls > 0).mean(), 1), "sharpe": round(sr, 2),
        "worst": round(pnls.min(), 0), "best": round(pnls.max(), 0), "total": round(pnls.sum(), 0),
    }


def main():
    print("Loading base data...")
    pooled = load_base()
    print(f"Rows: {len(pooled):,}")

    print("Assigning quintiles...")
    pooled = assign_q5_entries(pooled)

    val_mask = (pooled["date"].dt.year >= 2020) & (pooled["date"].dt.year <= 2022)
    hold_mask = pooled["date"].dt.year >= 2023

    val_entries = pooled[pooled["entered_q5"] & val_mask]
    hold_entries = pooled[pooled["entered_q5"] & hold_mask]
    print(f"Q5 entries — Val: {len(val_entries):,}, Hold: {len(hold_entries):,}")

    print("Building entry paths (validate)...")
    val_paths = build_entry_paths(pooled, pooled["entered_q5"] & val_mask)
    print(f"Val paths: {len(val_paths):,}")

    print("Building entry paths (holdout)...")
    hold_paths = build_entry_paths(pooled, pooled["entered_q5"] & hold_mask)
    print(f"Hold paths: {len(hold_paths):,}\n")

    report = []
    report.append("=" * 110)
    report.append("TRADE 1 DEEP DIVE: Q5 Momentum — Entry, Hold, Stop, TP from data")
    report.append("=" * 110)

    # 1. ENTRY
    report.append(f"\n{'═'*110}")
    report.append("1. ENTRY — Signal close vs next open")
    report.append(f"{'═'*110}")
    ovn = val_paths["overnight"].dropna()
    report.append(f"  Overnight (close→open): mean={ovn.mean():+.1f} bps, WR={100*(ovn>0).mean():.1f}%, "
                  f"t={ovn.mean()/(ovn.std()/np.sqrt(len(ovn))):.2f}")
    report.append(f"  → {'BUY AT CLOSE' if ovn.mean() > 0 else 'WAIT FOR OPEN'} is better by {abs(ovn.mean()):.1f} bps")

    # 2. HOLD CURVE
    report.append(f"\n{'═'*110}")
    report.append("2. OPTIMAL HOLD — Day-by-day return from entry close")
    report.append(f"{'═'*110}")
    report.append(f"  {'Day':>5} {'Mean':>8} {'Median':>8} {'WR':>5} {'t':>7} {'AnnSR':>7}")
    report.append(f"  {'─'*45}")
    peak_d, peak_m = 0, -999
    for d in range(1, MAX_HOLD + 1):
        col = f"d{d}_close"
        vals = val_paths[col].dropna()
        if len(vals) < 50:
            continue
        m = vals.mean()
        md = vals.median()
        wr = 100 * (vals > 0).mean()
        t = m / (vals.std() / np.sqrt(len(vals)))
        sr = m / vals.std() * np.sqrt(252 / d) if vals.std() > 0 else 0
        if m > peak_m:
            peak_m, peak_d = m, d
        marker = " ← PEAK" if d == peak_d and d > 3 else ""
        report.append(f"  {d:>5} {m:>+8.1f} {md:>+8.1f} {wr:>5.1f} {t:>+7.2f} {sr:>+7.2f}{marker}")
    report.append(f"\n  Peak at day {peak_d}: {peak_m:+.1f} bps")

    # 3. STOP LOSS
    report.append(f"\n{'═'*110}")
    report.append("3. STOP LOSS — Grid (no TP)")
    report.append(f"{'═'*110}")
    stops = [None, -50, -75, -100, -150, -200, -300, -500]
    holds = [5, 7, 10, 15, 20]
    report.append(f"  {'Hold':>5} {'Stop':>6} {'N':>6} {'Mean':>7} {'WR':>5} {'SR':>6} {'Worst':>7}")
    report.append(f"  {'─'*50}")
    for h in holds:
        for s in stops:
            r = simulate(val_paths, s, None, h)
            if r is None: continue
            ss = f"{s}" if s else "none"
            report.append(f"  {h:>5} {ss:>6} {r['n']:>6} {r['mean']:>+7.1f} {r['wr']:>5.1f} {r['sharpe']:>+6.2f} {r['worst']:>+7.0f}")

    # 4. TAKE PROFIT
    report.append(f"\n{'═'*110}")
    report.append("4. TAKE PROFIT — Grid (no stop)")
    report.append(f"{'═'*110}")
    tps = [None, 50, 100, 150, 200, 300, 500]
    report.append(f"  {'Hold':>5} {'TP':>6} {'N':>6} {'Mean':>7} {'WR':>5} {'SR':>6}")
    report.append(f"  {'─'*45}")
    for h in holds:
        for tp in tps:
            r = simulate(val_paths, None, tp, h)
            if r is None: continue
            ts = f"+{tp}" if tp else "none"
            report.append(f"  {h:>5} {ts:>6} {r['n']:>6} {r['mean']:>+7.1f} {r['wr']:>5.1f} {r['sharpe']:>+6.2f}")

    # 5. COMBINED GRID
    report.append(f"\n{'═'*110}")
    report.append("5. COMBINED — Stop × TP × Hold (top 30 by Sharpe)")
    report.append(f"{'═'*110}")
    combo_stops = [-75, -100, -150, -200, -300]
    combo_tps = [100, 150, 200, 300, 500, None]
    combo_holds = [5, 7, 10, 15, 20]
    combos = []
    for h, s, tp in product(combo_holds, combo_stops, combo_tps):
        r = simulate(val_paths, s, tp, h)
        if r is None: continue
        combos.append({"hold": h, "stop": s, "tp": tp, **r})
    combo_df = pd.DataFrame(combos).sort_values("sharpe", ascending=False)

    report.append(f"  {'Hold':>5} {'Stop':>6} {'TP':>6} {'N':>6} {'Mean':>7} {'WR':>5} {'SR':>6} {'Worst':>7} {'Total':>8}")
    report.append(f"  {'─'*65}")
    for _, row in combo_df.head(30).iterrows():
        ts = f"+{int(row['tp'])}" if pd.notna(row['tp']) else "none"
        report.append(f"  {int(row['hold']):>5} {int(row['stop']):>6} {ts:>6} {int(row['n']):>6} "
                      f"{row['mean']:>+7.1f} {row['wr']:>5.1f} {row['sharpe']:>+6.2f} {row['worst']:>+7.0f} {row['total']:>+8.0f}")

    # 6. SIZING
    report.append(f"\n{'═'*110}")
    report.append("6. SIZING — z-score and vol terciles")
    report.append(f"{'═'*110}")
    for split_col, label in [("z_signal", "Z-score"), ("rvol_20d", "Realized Vol")]:
        vals = val_paths[split_col].dropna()
        if len(vals) < 100:
            continue
        try:
            val_paths["_terc"] = pd.qcut(vals, 3, labels=["low", "mid", "high"], duplicates="drop")
        except:
            continue
        report.append(f"\n  By {label}:")
        report.append(f"  {'Terc':<8} {'N':>6} {'5d':>8} {'10d':>8} {'10d SR':>7} {'10d WR':>6}")
        report.append(f"  {'─'*45}")
        for t in ["low", "mid", "high"]:
            sub = val_paths[val_paths["_terc"] == t]
            d5 = sub["d5_close"].dropna()
            d10 = sub["d10_close"].dropna()
            if len(d10) < 20: continue
            sr10 = d10.mean() / d10.std() * np.sqrt(252/10) if d10.std() > 0 else 0
            report.append(f"  {t:<8} {len(d10):>6} {d5.mean():>+8.1f} {d10.mean():>+8.1f} {sr10:>+7.2f} {100*(d10>0).mean():>6.1f}")

    # 7. HOLDOUT
    report.append(f"\n{'═'*110}")
    report.append("7. HOLDOUT (2023-2026) — Top validate combos")
    report.append(f"{'═'*110}")
    report.append(f"  {'Hold':>5} {'Stop':>6} {'TP':>6} {'ValSR':>6} | {'HoldSR':>7} {'HoldMean':>9} {'HoldWR':>7} {'N':>5} {'Worst':>7}")
    report.append(f"  {'─'*75}")

    for i, (_, row) in enumerate(combo_df.head(10).iterrows()):
        tp_val = int(row["tp"]) if pd.notna(row["tp"]) else None
        hr = simulate(hold_paths, int(row["stop"]), tp_val, int(row["hold"]))
        if hr is None: continue
        ts = f"+{int(row['tp'])}" if pd.notna(row['tp']) else "none"
        report.append(f"  {int(row['hold']):>5} {int(row['stop']):>6} {ts:>6} {row['sharpe']:>+6.2f} | "
                      f"{hr['sharpe']:>+7.2f} {hr['mean']:>+9.1f} {hr['wr']:>7.1f} {hr['n']:>5} {hr['worst']:>+7.0f}")

        # Year by year for rank 1
        if i == 0:
            report.append(f"\n  Best combo year-by-year on HOLDOUT:")
            hold_paths_c = hold_paths.copy()
            hold_paths_c["year"] = pd.to_datetime(hold_paths_c["date"]).dt.year
            for yr in sorted(hold_paths_c["year"].unique()):
                yr_p = hold_paths_c[hold_paths_c["year"] == yr]
                yr_r = simulate(yr_p, int(row["stop"]), tp_val, int(row["hold"]))
                if yr_r:
                    report.append(f"    {yr}: N={yr_r['n']:>4}, mean={yr_r['mean']:>+7.1f}, WR={yr_r['wr']:.0f}%, "
                                  f"SR={yr_r['sharpe']:+.2f}, worst={yr_r['worst']:+.0f}")

    # Baseline
    report.append(f"\n  BASELINE (no stop, no TP, 10d hold) on HOLDOUT:")
    bl = simulate(hold_paths, None, None, 10)
    if bl:
        report.append(f"    N={bl['n']}, mean={bl['mean']:+.1f}, WR={bl['wr']}%, SR={bl['sharpe']}, worst={bl['worst']}")

    report.append(f"\n{'═'*110}")
    report_text = "\n".join(report)
    print(report_text)
    with open(os.path.join(OUT_DIR, "trade1_deep_report.txt"), "w") as f:
        f.write(report_text)
    combo_df.to_csv(os.path.join(OUT_DIR, "trade1_grid.csv"), index=False)
    print(f"\nReport: {os.path.join(OUT_DIR, 'trade1_deep_report.txt')}")


if __name__ == "__main__":
    main()
