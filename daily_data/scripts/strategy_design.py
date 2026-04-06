#!/usr/bin/env python3
"""
STRATEGY DESIGN: Turn the statistical finding into a real trading strategy.

Questions to answer from data:
1. Does z-score MAGNITUDE predict return magnitude? (conviction sizing)
2. Is there a better entry than "buy at close on signal day"?
   - Wait for N consecutive days in extreme quintile?
   - Wait for a pullback within the first day?
3. Per-position stop loss: does cutting losers at X bps help or hurt?
4. Per-position take profit: does trimming winners help?
5. How long to hold? Fixed period or signal-based exit?
6. What predicts WHICH entries fail?

All analysis on VALIDATE (2020-2022) only. HOLDOUT (2023-2026) untouched.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

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

        # Forward returns at various horizons
        for n in [1, 2, 3, 5, 10, 20]:
            df[f"fwd_{n}d_bps"] = 10000 * (c.shift(-n) / c - 1)

        # Close-to-open (overnight) return
        df["overnight_bps"] = 10000 * (o.shift(-1) / c - 1)

        # Open-to-close (intraday) next day
        df["intraday_next_bps"] = 10000 * (c.shift(-1) / o.shift(-1) - 1)

        # Max adverse excursion over next N days (using lows)
        for n in [1, 3, 5, 10]:
            future_lows = df["low"].shift(-1)
            for d in range(2, n + 1):
                future_lows = pd.concat([future_lows, df["low"].shift(-d)], axis=1).min(axis=1)
            df[f"mae_{n}d_bps"] = 10000 * (future_lows / c - 1)  # negative = adverse

        # Max favorable excursion (using highs)
        for n in [1, 3, 5, 10]:
            future_highs = df["high"].shift(-1)
            for d in range(2, n + 1):
                future_highs = pd.concat([future_highs, df["high"].shift(-d)], axis=1).max(axis=1)
            df[f"mfe_{n}d_bps"] = 10000 * (future_highs / c - 1)  # positive = favorable

        # Signal
        sma50 = c.rolling(50).mean()
        raw_slope = 100 * (sma50 / sma50.shift(10) - 1)
        roll_m = raw_slope.rolling(252, min_periods=60).mean()
        roll_s = raw_slope.rolling(252, min_periods=60).std()
        df["z_signal"] = (raw_slope - roll_m) / roll_s.replace(0, np.nan)

        # Days in extreme (how many consecutive days has this been in Q1/Q5?)
        # We'll compute this per-symbol after pooling
        df["rvol_20d"] = (np.log(c / c.shift(1))).rolling(20).std() * np.sqrt(252) * 100

        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def assign_quintiles_daily(pooled):
    """Assign quintile ranks daily across symbols."""
    pooled["quintile"] = np.nan
    for date, ddf in pooled.groupby("date"):
        idx = ddf.index
        valid = ddf["z_signal"].dropna()
        if len(valid) < 20:
            continue
        try:
            q = pd.qcut(valid, 5, labels=[1,2,3,4,5], duplicates="drop")
            pooled.loc[q.index, "quintile"] = q.values
        except ValueError:
            continue
    return pooled


def count_consecutive_extreme(pooled):
    """For each symbol, count how many consecutive days it's been in Q1 or Q5."""
    pooled["days_in_q1"] = 0
    pooled["days_in_q5"] = 0

    for sym, sdf in pooled.groupby("symbol"):
        idx = sdf.index
        q = sdf["quintile"].values
        days_q1 = np.zeros(len(q))
        days_q5 = np.zeros(len(q))

        for i in range(len(q)):
            if q[i] == 1:
                days_q1[i] = (days_q1[i-1] + 1) if i > 0 else 1
            if q[i] == 5:
                days_q5[i] = (days_q5[i-1] + 1) if i > 0 else 1

        pooled.loc[idx, "days_in_q1"] = days_q1
        pooled.loc[idx, "days_in_q5"] = days_q5

    return pooled


def main():
    print("Loading data...")
    pooled = load_data()
    print(f"Total rows: {len(pooled):,}")

    print("Assigning daily quintiles...")
    pooled = assign_quintiles_daily(pooled)

    print("Counting consecutive extreme days...")
    pooled = count_consecutive_extreme(pooled)

    # VALIDATE period only
    val = pooled[(pooled["date"].dt.year >= 2020) & (pooled["date"].dt.year <= 2022)].copy()
    print(f"Validate period: {val['date'].min().date()} to {val['date'].max().date()}")
    print(f"Validate rows: {len(val):,}\n")

    # Separate long candidates (Q1) and short candidates (Q5)
    q1 = val[val["quintile"] == 1].copy()
    q5 = val[val["quintile"] == 5].copy()

    report = []
    report.append("=" * 110)
    report.append("STRATEGY DESIGN — Building a real trade from the statistical finding")
    report.append("All analysis on VALIDATE (2020-2022)")
    report.append("=" * 110)

    # ═══════════════════════════════════════════════════════════════════
    # Q1: Does z-score magnitude predict return magnitude?
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n{'═'*110}")
    report.append("Q1: CONVICTION SIZING — Does a more extreme z-score predict bigger returns?")
    report.append(f"{'═'*110}")

    for side, data, label in [("LONG", q1, "Q1 (declining slope)"), ("SHORT", q5, "Q5 (rising slope)")]:
        report.append(f"\n  {side} side — {label}:")

        # Split by z-score magnitude within the quintile
        z_col = "z_signal"
        abs_z = data[z_col].abs()

        # Tercile within quintile by |z|
        try:
            data = data.copy()
            data["z_tercile"] = pd.qcut(abs_z, 3, labels=["mild", "moderate", "extreme"], duplicates="drop")
        except:
            continue

        for hold in [1, 3, 5, 10]:
            fwd_col = f"fwd_{hold}d_bps"
            report.append(f"\n    Hold = {hold}d:")
            report.append(f"    {'Tercile':<12} {'N':>6} {'Mean bps':>9} {'Median':>8} {'WR':>5} {'t':>6}")
            report.append(f"    {'─'*50}")

            for terc in ["mild", "moderate", "extreme"]:
                subset = data[data["z_tercile"] == terc][fwd_col].dropna()
                if len(subset) < 30:
                    continue
                mean = subset.mean()
                if side == "SHORT":
                    mean = -mean  # flip for short
                    median = -subset.median()
                    wr = 100 * (subset < 0).mean()
                else:
                    median = subset.median()
                    wr = 100 * (subset > 0).mean()
                t = mean / (subset.std() / np.sqrt(len(subset))) if subset.std() > 0 else 0

                report.append(f"    {terc:<12} {len(subset):>6} {mean:>+9.1f} {median:>+8.1f} {wr:>5.1f} {t:>+6.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # Q2: Does waiting for persistence improve entry?
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("Q2: ENTRY TIMING — Does waiting for N consecutive days in quintile improve returns?")
    report.append(f"{'═'*110}")

    for side, data, days_col, label in [
        ("LONG", q1, "days_in_q1", "Q1"),
        ("SHORT", q5, "days_in_q5", "Q5"),
    ]:
        report.append(f"\n  {side} side:")
        report.append(f"  {'Days in Q':>10} {'N':>7} {'1d bps':>8} {'3d bps':>8} {'5d bps':>8} {'10d bps':>8} {'1d WR':>6}")
        report.append(f"  {'─'*65}")

        for min_days in [1, 2, 3, 5, 7, 10, 15, 20]:
            subset = data[data[days_col] >= min_days]
            if len(subset) < 50:
                continue

            row_parts = [f"  {'>=' + str(min_days):>10} {len(subset):>7}"]
            for hold in [1, 3, 5, 10]:
                fwd = subset[f"fwd_{hold}d_bps"].dropna()
                mean = -fwd.mean() if side == "SHORT" else fwd.mean()
                row_parts.append(f"{mean:>+8.1f}")

            fwd1 = subset["fwd_1d_bps"].dropna()
            wr = 100 * (fwd1 < 0).mean() if side == "SHORT" else 100 * (fwd1 > 0).mean()
            row_parts.append(f"{wr:>6.1f}")
            report.append(" ".join(row_parts))

        # FIRST day in quintile only (the entry signal)
        first_day = data[data[days_col] == 1]
        if len(first_day) > 50:
            row_parts = [f"  {'==1 (NEW)':>10} {len(first_day):>7}"]
            for hold in [1, 3, 5, 10]:
                fwd = first_day[f"fwd_{hold}d_bps"].dropna()
                mean = -fwd.mean() if side == "SHORT" else fwd.mean()
                row_parts.append(f"{mean:>+8.1f}")
            fwd1 = first_day["fwd_1d_bps"].dropna()
            wr = 100 * (fwd1 < 0).mean() if side == "SHORT" else 100 * (fwd1 > 0).mean()
            row_parts.append(f"{wr:>6.1f}")
            report.append(" ".join(row_parts))

    # ═══════════════════════════════════════════════════════════════════
    # Q3: Stop loss analysis using MAE
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("Q3: STOP LOSS — Using max adverse excursion (MAE) to set stops")
    report.append(f"{'═'*110}")

    for side, data, label in [("LONG", q1, "Q1"), ("SHORT", q5, "Q5")]:
        report.append(f"\n  {side} side — MAE distribution (how bad does it get before recovering?):")

        for hold in [5, 10]:
            mae = data[f"mae_{hold}d_bps"].dropna()
            mfe = data[f"mfe_{hold}d_bps"].dropna()
            fwd = data[f"fwd_{hold}d_bps"].dropna()

            if side == "SHORT":
                mae = -data[f"mfe_{hold}d_bps"].dropna()  # for shorts, MFE becomes MAE
                mfe = -data[f"mae_{hold}d_bps"].dropna()
                fwd = -fwd

            report.append(f"\n    Hold = {hold}d (N={len(mae):,}):")
            report.append(f"    MAE percentiles (worst drawdown before hold end):")
            for pct in [10, 25, 50, 75, 90, 95, 99]:
                report.append(f"      P{pct}: {mae.quantile(pct/100):>+8.1f} bps")

            # What if we stopped out at various levels?
            report.append(f"\n    Stop level simulation (exit at stop if hit, else hold to {hold}d):")
            report.append(f"    {'Stop bps':>10} {'Stopped%':>9} {'AvgPnL':>8} {'vs NoStop':>10} {'WR':>5}")
            report.append(f"    {'─'*50}")

            no_stop_mean = fwd.mean()

            for stop_bps in [-50, -100, -150, -200, -300, -500]:
                stopped = mae < stop_bps
                pct_stopped = 100 * stopped.mean()
                # P&L: if stopped, lose stop_bps. If not, get actual return.
                pnl = fwd.copy()
                pnl[stopped] = stop_bps
                avg_pnl = pnl.mean()
                diff = avg_pnl - no_stop_mean
                wr = 100 * (pnl > 0).mean()
                report.append(f"    {stop_bps:>+10} {pct_stopped:>8.1f}% {avg_pnl:>+8.1f} {diff:>+10.1f} {wr:>5.1f}")

    # ═══════════════════════════════════════════════════════════════════
    # Q4: Take profit using MFE
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("Q4: TAKE PROFIT — Using max favorable excursion (MFE)")
    report.append(f"{'═'*110}")

    for side, data, label in [("LONG", q1, "Q1"), ("SHORT", q5, "Q5")]:
        report.append(f"\n  {side} side:")

        for hold in [5, 10]:
            mfe = data[f"mfe_{hold}d_bps"].dropna()
            fwd = data[f"fwd_{hold}d_bps"].dropna()

            if side == "SHORT":
                mfe = -data[f"mae_{hold}d_bps"].dropna()
                fwd = -fwd

            report.append(f"\n    Hold = {hold}d — MFE percentiles (best point before hold end):")
            for pct in [10, 25, 50, 75, 90]:
                report.append(f"      P{pct}: {mfe.quantile(pct/100):>+8.1f} bps")

            report.append(f"\n    Take profit simulation:")
            report.append(f"    {'TP bps':>10} {'Hit%':>6} {'AvgPnL':>8} {'vs Hold':>8}")
            report.append(f"    {'─'*40}")

            for tp_bps in [50, 100, 150, 200, 300, 500]:
                hit = mfe >= tp_bps
                pct_hit = 100 * hit.mean()
                pnl = fwd.copy()
                pnl[hit] = tp_bps
                avg_pnl = pnl.mean()
                diff = avg_pnl - fwd.mean()
                report.append(f"    {tp_bps:>+10} {pct_hit:>5.1f}% {avg_pnl:>+8.1f} {diff:>+8.1f}")

    # ═══════════════════════════════════════════════════════════════════
    # Q5: Overnight vs intraday — when does the P&L accrue?
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("Q5: OVERNIGHT vs INTRADAY — Where does the return come from?")
    report.append(f"{'═'*110}")

    for side, data, label in [("LONG", q1, "Q1"), ("SHORT", q5, "Q5")]:
        overnight = data["overnight_bps"].dropna()
        intraday = data["intraday_next_bps"].dropna()

        if side == "SHORT":
            overnight = -overnight
            intraday = -intraday

        report.append(f"\n  {side} side:")
        report.append(f"    Overnight (close→open):  mean={overnight.mean():>+7.1f} bps, WR={100*(overnight>0).mean():.1f}%")
        report.append(f"    Intraday  (open→close):  mean={intraday.mean():>+7.1f} bps, WR={100*(intraday>0).mean():.1f}%")
        report.append(f"    Total 1d  (close→close): {(overnight.mean() + intraday.mean()):>+7.1f} bps")
        report.append(f"    Split: {100*overnight.mean()/(overnight.mean()+intraday.mean()):.0f}% overnight / {100*intraday.mean()/(overnight.mean()+intraday.mean()):.0f}% intraday")

    # ═══════════════════════════════════════════════════════════════════
    # Q6: What predicts failures? Feature comparison: winners vs losers
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("Q6: FAILURE ANALYSIS — What's different about losing entries?")
    report.append(f"{'═'*110}")

    features_to_test = ["rvol_20d", "z_signal", "days_in_q1", "gap_bps", "range_bps",
                        "daily_return_pct", "close_location", "rvol", "adx_14", "bb_bandwidth",
                        "rsi_14", "consec_days", "dist_sma50_pct", "dist_sma200_pct"]

    for side, data, label in [("LONG", q1, "Q1"), ("SHORT", q5, "Q5")]:
        report.append(f"\n  {side} side — 5d hold, winners vs losers:")

        fwd = data["fwd_5d_bps"].copy()
        if side == "SHORT":
            fwd = -fwd

        winners = fwd > 0
        losers = fwd <= 0

        report.append(f"  Winners: {winners.sum():,} ({100*winners.mean():.1f}%)")
        report.append(f"  Losers:  {losers.sum():,} ({100*losers.mean():.1f}%)")

        report.append(f"\n  {'Feature':<25} {'Win mean':>10} {'Lose mean':>10} {'Diff':>8} {'Cohen d':>8} {'t':>7}")
        report.append(f"  {'─'*75}")

        for feat in features_to_test:
            if feat not in data.columns:
                continue
            w = data.loc[winners.index[winners], feat].dropna()
            l = data.loc[losers.index[losers], feat].dropna()
            if len(w) < 30 or len(l) < 30:
                continue
            diff = w.mean() - l.mean()
            pooled_std = np.sqrt((w.std()**2 + l.std()**2) / 2)
            cohen_d = diff / pooled_std if pooled_std > 0 else 0
            t, p = stats.ttest_ind(w, l)
            marker = "***" if abs(cohen_d) > 0.3 else "**" if abs(cohen_d) > 0.2 else "*" if abs(cohen_d) > 0.1 else ""
            report.append(f"  {feat:<25} {w.mean():>+10.2f} {l.mean():>+10.2f} {diff:>+8.2f} {cohen_d:>+8.3f} {t:>+7.2f} {marker}")

    report.append(f"\n{'═'*110}")
    report.append("END")
    report.append(f"{'═'*110}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "strategy_design_report.txt"), "w") as f:
        f.write(report_text)
    print(f"\nReport: {os.path.join(OUT_DIR, 'strategy_design_report.txt')}")


if __name__ == "__main__":
    main()
