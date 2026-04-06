#!/usr/bin/env python3
"""
TRADE 1 DEEP DIVE: Trend joining on Q5 entries.

When a stock's SMA50 slope first enters the top quintile, buy it.
Now find the optimal: entry, exit, stop — from data only.

All analysis on VALIDATE (2020-2022).
HOLDOUT (2023-2026) touched ONCE at the end.

Questions:
1. ENTRY: Buy at close? Next open? Wait for a pullback intraday?
   - Close of signal day vs open of next day vs close of next day
   - Does waiting 1 day for "confirmation" help?

2. OPTIMAL HOLD: Return curve by day — when does momentum exhaust?
   - Plot cumulative excess return for each day 1-30
   - Find the peak

3. STOP LOSS: Grid search using actual daily lows (MAE)
   - Test stops from -50 to -500 bps
   - Which maximizes Sharpe, not just mean?
   - Per-trade MAE distribution

4. TAKE PROFIT: Does capping upside help?
   - Test TPs from +50 to +500 bps using actual daily highs (MFE)
   - Diminishing returns?

5. COMBINED: Stop + TP + optimal hold together
   - Grid search all combos
   - Report the Pareto frontier (best Sharpe at each mean PnL level)

6. SIZING: Does z-score, vol, or entry character predict magnitude?

7. YEAR-BY-YEAR: Does the optimal combo work every year?
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    all_dfs = []
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = sym

        c = df["close"]
        o = df["open"]
        h = df["high"]
        l = df["low"]

        # Forward returns from CLOSE (close-to-close)
        for n in range(1, 31):
            df[f"fwd_c2c_{n}d"] = 10000 * (c.shift(-n) / c - 1)

        # Forward returns from OPEN next day (open-to-close of day+n)
        for n in range(1, 31):
            df[f"fwd_o2c_{n}d"] = 10000 * (c.shift(-n) / o.shift(-1) - 1)

        # Overnight return (close to next open)
        df["overnight"] = 10000 * (o.shift(-1) / c - 1)

        # Daily low/high relative to ENTRY CLOSE for MAE/MFE
        for n in range(1, 21):
            # Cumulative min of lows from day+1 to day+n (relative to today's close)
            future_low = l.shift(-1)
            for d in range(2, n + 1):
                future_low = pd.concat([future_low, l.shift(-d)], axis=1).min(axis=1)
            df[f"mae_{n}d"] = 10000 * (future_low / c - 1)

            # Cumulative max of highs
            future_high = h.shift(-1)
            for d in range(2, n + 1):
                future_high = pd.concat([future_high, h.shift(-d)], axis=1).max(axis=1)
            df[f"mfe_{n}d"] = 10000 * (future_high / c - 1)

        # For stop/TP simulation: need day-by-day path
        # Store each day's low and high relative to entry close
        for n in range(1, 21):
            df[f"day{n}_low_bps"] = 10000 * (l.shift(-n) / c - 1)
            df[f"day{n}_high_bps"] = 10000 * (h.shift(-n) / c - 1)
            df[f"day{n}_close_bps"] = 10000 * (c.shift(-n) / c - 1)

        # Signal
        sma50 = c.rolling(50).mean()
        raw_slope = 100 * (sma50 / sma50.shift(10) - 1)
        roll_m = raw_slope.rolling(252, min_periods=60).mean()
        roll_s = raw_slope.rolling(252, min_periods=60).std()
        df["z_signal"] = (raw_slope - roll_m) / roll_s.replace(0, np.nan)

        df["rvol_20d"] = (np.log(c / c.shift(1))).rolling(20).std() * np.sqrt(252) * 100

        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def assign_quintiles_and_entries(pooled):
    pooled["quintile"] = np.nan
    for date, ddf in pooled.groupby("date"):
        valid = ddf["z_signal"].dropna()
        if len(valid) < 20:
            continue
        try:
            q = pd.qcut(valid, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            pooled.loc[q.index, "quintile"] = q.values
        except ValueError:
            continue

    pooled = pooled.sort_values(["symbol", "date"]).reset_index(drop=True)
    pooled["prev_q"] = pooled.groupby("symbol")["quintile"].shift(1)
    pooled["entered_q5"] = (pooled["quintile"] == 5) & (pooled["prev_q"] != 5)
    return pooled


def simulate_stop_tp(entries, stop_bps, tp_bps, max_hold, entry_type="close"):
    """
    Simulate trades with stop and TP using day-by-day low/high.
    Entry at close of signal day.
    Each day, check: did low breach stop? Did high breach TP?
    If both on same day, assume stop hit first (conservative).
    """
    pnls = []

    for _, row in entries.iterrows():
        exited = False
        for d in range(1, max_hold + 1):
            low_col = f"day{d}_low_bps"
            high_col = f"day{d}_high_bps"
            close_col = f"day{d}_close_bps"

            if low_col not in row.index or pd.isna(row[low_col]):
                break

            day_low = row[low_col]
            day_high = row[high_col]

            # Check stop first (conservative)
            if stop_bps is not None and day_low <= stop_bps:
                pnls.append(stop_bps)
                exited = True
                break

            # Check TP
            if tp_bps is not None and day_high >= tp_bps:
                pnls.append(tp_bps)
                exited = True
                break

        if not exited:
            # Exit at close of max_hold day
            close_col = f"day{max_hold}_close_bps"
            if close_col in row.index and not pd.isna(row[close_col]):
                pnls.append(row[close_col])

    if not pnls:
        return None

    pnls = np.array(pnls)
    sr = pnls.mean() / pnls.std() * np.sqrt(252 / max_hold) if pnls.std() > 0 else 0

    return {
        "n": len(pnls),
        "mean": round(pnls.mean(), 2),
        "median": round(np.median(pnls), 2),
        "std": round(pnls.std(), 2),
        "wr": round(100 * (pnls > 0).mean(), 1),
        "sharpe": round(sr, 2),
        "total": round(pnls.sum(), 0),
        "worst": round(pnls.min(), 0),
        "best": round(pnls.max(), 0),
        "pnls": pnls,
    }


def main():
    print("Loading data...")
    pooled = load_data()
    print("Assigning quintiles...")
    pooled = assign_quintiles_and_entries(pooled)

    val = pooled[(pooled["date"].dt.year >= 2020) & (pooled["date"].dt.year <= 2022)]
    hold = pooled[pooled["date"].dt.year >= 2023]

    entries_val = val[val["entered_q5"]].copy()
    entries_hold = hold[hold["entered_q5"]].copy()
    print(f"Q5 entries — Validate: {len(entries_val):,}, Holdout: {len(entries_hold):,}\n")

    report = []
    report.append("=" * 110)
    report.append("TRADE 1 DEEP DIVE: Optimal entry, exit, stop for Q5 momentum entries")
    report.append("Entry = close of day stock first enters Q5 (SMA50 slope accelerating)")
    report.append("=" * 110)

    # ═══════════════════════════════════════════════════════════════════
    # 1. ENTRY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n{'═'*110}")
    report.append("1. ENTRY — Close vs next-day open vs next-day close")
    report.append(f"{'═'*110}")

    report.append(f"\n  Validate (N={len(entries_val):,}):")
    report.append(f"  {'Entry':>20} {'5d bps':>8} {'10d bps':>9} {'20d bps':>9} {'5d WR':>6} {'10d t':>7}")
    report.append(f"  {'─'*65}")

    # Close-to-close
    for h in [5, 10, 20]:
        fwd = entries_val[f"fwd_c2c_{h}d"].dropna()
        t = fwd.mean() / (fwd.std() / np.sqrt(len(fwd))) if fwd.std() > 0 else 0
        if h == 5:
            wr5 = 100 * (fwd > 0).mean()
    fwd5 = entries_val["fwd_c2c_5d"].dropna()
    fwd10 = entries_val["fwd_c2c_10d"].dropna()
    fwd20 = entries_val["fwd_c2c_20d"].dropna()
    t10 = fwd10.mean() / (fwd10.std() / np.sqrt(len(fwd10)))
    report.append(f"  {'Signal close':>20} {fwd5.mean():>+8.1f} {fwd10.mean():>+9.1f} {fwd20.mean():>+9.1f} {100*(fwd5>0).mean():>6.1f} {t10:>+7.2f}")

    # Open-to-close
    fwd5o = entries_val["fwd_o2c_5d"].dropna()
    fwd10o = entries_val["fwd_o2c_10d"].dropna()
    fwd20o = entries_val["fwd_o2c_20d"].dropna()
    t10o = fwd10o.mean() / (fwd10o.std() / np.sqrt(len(fwd10o)))
    report.append(f"  {'Next open':>20} {fwd5o.mean():>+8.1f} {fwd10o.mean():>+9.1f} {fwd20o.mean():>+9.1f} {100*(fwd5o>0).mean():>6.1f} {t10o:>+7.2f}")

    # Overnight
    ovn = entries_val["overnight"].dropna()
    report.append(f"\n  Overnight (close→open): mean={ovn.mean():>+6.1f} bps, WR={100*(ovn>0).mean():.1f}%, t={ovn.mean()/(ovn.std()/np.sqrt(len(ovn))):.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. OPTIMAL HOLD — Day-by-day return curve
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("2. OPTIMAL HOLD — Day-by-day cumulative return from entry close")
    report.append(f"{'═'*110}")

    report.append(f"\n  {'Day':>5} {'Mean bps':>9} {'Median':>8} {'WR%':>5} {'t-stat':>7} {'Sharpe':>7}")
    report.append(f"  {'─'*50}")

    peak_day = 0
    peak_mean = 0
    for d in range(1, 26):
        col = f"fwd_c2c_{d}d"
        if col not in entries_val.columns:
            continue
        fwd = entries_val[col].dropna()
        if len(fwd) < 50:
            continue
        mean = fwd.mean()
        med = fwd.median()
        wr = 100 * (fwd > 0).mean()
        t = mean / (fwd.std() / np.sqrt(len(fwd))) if fwd.std() > 0 else 0
        sr = mean / fwd.std() * np.sqrt(252 / d) if fwd.std() > 0 else 0

        if mean > peak_mean:
            peak_mean = mean
            peak_day = d

        marker = " ← PEAK" if d == peak_day and d > 5 else ""
        report.append(f"  {d:>5} {mean:>+9.1f} {med:>+8.1f} {wr:>5.1f} {t:>+7.2f} {sr:>+7.2f}{marker}")

    report.append(f"\n  Peak mean return at day {peak_day}: {peak_mean:+.1f} bps")

    # ═══════════════════════════════════════════════════════════════════
    # 3. STOP LOSS — Grid using day-by-day simulation
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("3. STOP LOSS — Simulated with daily low checks")
    report.append(f"{'═'*110}")

    # Test various hold periods × stop levels
    stop_levels = [None, -50, -75, -100, -150, -200, -300, -500]
    hold_periods = [5, 7, 10, 15, 20]

    report.append(f"\n  No TP, vary stop and hold:")
    report.append(f"  {'Hold':>5} {'Stop':>6} {'N':>6} {'Mean':>7} {'Median':>7} {'WR':>5} {'SR':>6} {'Worst':>7}")
    report.append(f"  {'─'*60}")

    grid_results = []

    for max_h in hold_periods:
        for stop in stop_levels:
            result = simulate_stop_tp(entries_val, stop, None, max_h)
            if result is None:
                continue
            stop_str = f"{stop}" if stop is not None else "none"
            report.append(
                f"  {max_h:>5} {stop_str:>6} {result['n']:>6} {result['mean']:>+7.1f} "
                f"{result['median']:>+7.1f} {result['wr']:>5.1f} {result['sharpe']:>+6.2f} {result['worst']:>+7.0f}"
            )
            grid_results.append({
                "stop": stop, "tp": None, "hold": max_h,
                **{k: v for k, v in result.items() if k != "pnls"}
            })

    # ═══════════════════════════════════════════════════════════════════
    # 4. TAKE PROFIT grid
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("4. TAKE PROFIT — Simulated with daily high checks")
    report.append(f"{'═'*110}")

    tp_levels = [None, 50, 100, 150, 200, 300, 500]

    report.append(f"\n  No stop, vary TP and hold:")
    report.append(f"  {'Hold':>5} {'TP':>6} {'N':>6} {'Mean':>7} {'Median':>7} {'WR':>5} {'SR':>6} {'Worst':>7}")
    report.append(f"  {'─'*60}")

    for max_h in hold_periods:
        for tp in tp_levels:
            result = simulate_stop_tp(entries_val, None, tp, max_h)
            if result is None:
                continue
            tp_str = f"+{tp}" if tp is not None else "none"
            report.append(
                f"  {max_h:>5} {tp_str:>6} {result['n']:>6} {result['mean']:>+7.1f} "
                f"{result['median']:>+7.1f} {result['wr']:>5.1f} {result['sharpe']:>+6.2f} {result['worst']:>+7.0f}"
            )
            grid_results.append({
                "stop": None, "tp": tp, "hold": max_h,
                **{k: v for k, v in result.items() if k != "pnls"}
            })

    # ═══════════════════════════════════════════════════════════════════
    # 5. COMBINED GRID — Stop + TP + Hold
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("5. COMBINED GRID — Stop × TP × Hold (top 30 by Sharpe)")
    report.append(f"{'═'*110}")

    stops = [-75, -100, -150, -200, -300]
    tps = [100, 150, 200, 300, 500, None]
    holds = [5, 7, 10, 15, 20]

    combo_results = []
    for max_h, stop, tp in product(holds, stops, tps):
        result = simulate_stop_tp(entries_val, stop, tp, max_h)
        if result is None:
            continue
        combo_results.append({
            "stop": stop, "tp": tp, "hold": max_h,
            **{k: v for k, v in result.items() if k != "pnls"}
        })

    combo_df = pd.DataFrame(combo_results).sort_values("sharpe", ascending=False)

    report.append(f"\n  {'Hold':>5} {'Stop':>6} {'TP':>6} {'N':>6} {'Mean':>7} {'WR':>5} {'SR':>6} {'Worst':>7} {'Total':>8}")
    report.append(f"  {'─'*65}")

    for _, row in combo_df.head(30).iterrows():
        tp_str = f"+{int(row['tp'])}" if row['tp'] is not None else "none"
        report.append(
            f"  {int(row['hold']):>5} {int(row['stop']):>6} {tp_str:>6} {int(row['n']):>6} "
            f"{row['mean']:>+7.1f} {row['wr']:>5.1f} {row['sharpe']:>+6.2f} {row['worst']:>+7.0f} {row['total']:>+8.0f}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # 6. SIZING — Does anything predict magnitude?
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("6. SIZING — Does z-score magnitude or vol predict return magnitude?")
    report.append(f"{'═'*110}")

    # Split entries by z-score into terciles
    entries_val_c = entries_val.copy()
    try:
        entries_val_c["z_terc"] = pd.qcut(entries_val_c["z_signal"].abs(), 3,
                                           labels=["low_z", "mid_z", "high_z"], duplicates="drop")
    except:
        pass

    try:
        entries_val_c["vol_terc"] = pd.qcut(entries_val_c["rvol_20d"], 3,
                                             labels=["low_vol", "mid_vol", "high_vol"], duplicates="drop")
    except:
        pass

    for split_col, split_name in [("z_terc", "Z-score magnitude"), ("vol_terc", "Realized volatility")]:
        if split_col not in entries_val_c.columns:
            continue
        report.append(f"\n  Split by {split_name}:")
        report.append(f"  {'Tercile':<12} {'N':>6} {'5d bps':>8} {'10d bps':>9} {'10d SR':>7} {'10d WR':>6}")
        report.append(f"  {'─'*55}")

        for terc in entries_val_c[split_col].dropna().unique():
            subset = entries_val_c[entries_val_c[split_col] == terc]
            fwd5 = subset["fwd_c2c_5d"].dropna()
            fwd10 = subset["fwd_c2c_10d"].dropna()
            if len(fwd10) < 30:
                continue
            sr10 = fwd10.mean() / fwd10.std() * np.sqrt(252 / 10) if fwd10.std() > 0 else 0
            wr10 = 100 * (fwd10 > 0).mean()
            report.append(f"  {str(terc):<12} {len(fwd10):>6} {fwd5.mean():>+8.1f} {fwd10.mean():>+9.1f} {sr10:>+7.2f} {wr10:>6.1f}")

    # ═══════════════════════════════════════════════════════════════════
    # 7. HOLDOUT TEST — Best combo from validate on holdout
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("7. HOLDOUT TEST — Top 5 validate combos on clean holdout (2023-2026)")
    report.append(f"{'═'*110}")

    report.append(f"\n  {'Hold':>5} {'Stop':>6} {'TP':>6} {'Val SR':>7} {'Hold SR':>8} {'HoldMean':>9} {'HoldWR':>7} {'HoldN':>6} {'Worst':>7}")
    report.append(f"  {'─'*75}")

    for _, row in combo_df.head(10).iterrows():
        result = simulate_stop_tp(entries_hold, int(row["stop"]),
                                  int(row["tp"]) if row["tp"] is not None else None,
                                  int(row["hold"]))
        if result is None:
            continue
        tp_str = f"+{int(row['tp'])}" if row['tp'] is not None else "none"
        report.append(
            f"  {int(row['hold']):>5} {int(row['stop']):>6} {tp_str:>6} {row['sharpe']:>+7.2f} "
            f"{result['sharpe']:>+8.2f} {result['mean']:>+9.1f} {result['wr']:>7.1f} "
            f"{result['n']:>6} {result['worst']:>+7.0f}"
        )

        # Year by year for the best
        if _ == combo_df.index[0]:
            report.append(f"\n  Year-by-year for best combo (stop={int(row['stop'])}, tp={tp_str}, hold={int(row['hold'])}):")
            for yr in sorted(entries_hold["date"].dt.year.unique()):
                yr_entries = entries_hold[entries_hold["date"].dt.year == yr]
                yr_result = simulate_stop_tp(yr_entries, int(row["stop"]),
                                             int(row["tp"]) if row["tp"] is not None else None,
                                             int(row["hold"]))
                if yr_result:
                    report.append(f"    {yr}: N={yr_result['n']:>4}, mean={yr_result['mean']:>+7.1f} bps, "
                                f"WR={yr_result['wr']:.0f}%, SR={yr_result['sharpe']:+.2f}, worst={yr_result['worst']:+.0f}")

    # Also test the no-stop/no-TP baseline on holdout
    report.append(f"\n  BASELINE (no stop, no TP, hold 10d) on HOLDOUT:")
    baseline = simulate_stop_tp(entries_hold, None, None, 10)
    if baseline:
        report.append(f"    N={baseline['n']}, mean={baseline['mean']:+.1f}, WR={baseline['wr']}%, SR={baseline['sharpe']}, worst={baseline['worst']}")

    report.append(f"\n{'═'*110}")
    report.append("END")
    report.append(f"{'═'*110}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "trade1_deep_report.txt"), "w") as f:
        f.write(report_text)

    combo_df.to_csv(os.path.join(OUT_DIR, "trade1_grid.csv"), index=False)
    print(f"\nReport: {os.path.join(OUT_DIR, 'trade1_deep_report.txt')}")
    print(f"Grid: {os.path.join(OUT_DIR, 'trade1_grid.csv')}")


if __name__ == "__main__":
    main()
