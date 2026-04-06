#!/usr/bin/env python3
"""
TRANSITION MOMENTUM: When a stock FIRST enters Q1 or Q5, does riding the
trend (not fading it) produce edge?

If day-1 Q1 entries return -3.3 bps (stock keeps falling), then:
- SHORTING day-1 Q1 entries = momentum play on the downside
- BUYING day-1 Q5 entries = momentum play on the upside

This is the OPPOSITE of the mean reversion strategy. Let's see if both edges
coexist — momentum in the transition, reversion after persistence.

Also test: what does the ENTRY into Q1/Q5 look like? Is it a gap? A slow drift?
Does the entry character predict which direction wins?
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

        for n in [1, 2, 3, 5, 10, 20]:
            df[f"fwd_{n}d_bps"] = 10000 * (c.shift(-n) / c - 1)

        df["overnight_bps"] = 10000 * (o.shift(-1) / c - 1)
        df["intraday_next_bps"] = 10000 * (c.shift(-1) / o.shift(-1) - 1)

        # What happened BEFORE entry (trailing momentum)
        for n in [1, 3, 5, 10, 20]:
            df[f"past_{n}d_bps"] = 10000 * (c / c.shift(n) - 1)

        # Signal
        sma50 = c.rolling(50).mean()
        raw_slope = 100 * (sma50 / sma50.shift(10) - 1)
        roll_m = raw_slope.rolling(252, min_periods=60).mean()
        roll_s = raw_slope.rolling(252, min_periods=60).std()
        df["z_signal"] = (raw_slope - roll_m) / roll_s.replace(0, np.nan)

        # Entry character
        df["gap_bps"] = 10000 * (o - c.shift(1)) / c.shift(1)
        df["range_bps"] = 10000 * (df["high"] - df["low"]) / o

        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def assign_quintiles_and_transitions(pooled):
    """Assign quintiles daily, then mark transition days."""
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

    # Sort by symbol + date to track transitions
    pooled = pooled.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Previous day's quintile
    pooled["prev_q"] = pooled.groupby("symbol")["quintile"].shift(1)

    # Transition flags
    pooled["entered_q1"] = (pooled["quintile"] == 1) & (pooled["prev_q"] != 1)
    pooled["entered_q5"] = (pooled["quintile"] == 5) & (pooled["prev_q"] != 5)
    pooled["exited_q1"] = (pooled["quintile"] != 1) & (pooled["prev_q"] == 1)
    pooled["exited_q5"] = (pooled["quintile"] != 5) & (pooled["prev_q"] == 5)

    # Where did it come from?
    pooled["q1_from_q2"] = pooled["entered_q1"] & (pooled["prev_q"] == 2)
    pooled["q1_from_q3plus"] = pooled["entered_q1"] & (pooled["prev_q"] >= 3)
    pooled["q5_from_q4"] = pooled["entered_q5"] & (pooled["prev_q"] == 4)
    pooled["q5_from_q3minus"] = pooled["entered_q5"] & (pooled["prev_q"] <= 3)

    # Days in quintile
    pooled["days_in_q1"] = 0
    pooled["days_in_q5"] = 0
    for sym, sdf in pooled.groupby("symbol"):
        idx = sdf.index
        q = sdf["quintile"].values
        d1 = np.zeros(len(q))
        d5 = np.zeros(len(q))
        for i in range(len(q)):
            if q[i] == 1:
                d1[i] = (d1[i-1] + 1) if i > 0 else 1
            if q[i] == 5:
                d5[i] = (d5[i-1] + 1) if i > 0 else 1
        pooled.loc[idx, "days_in_q1"] = d1
        pooled.loc[idx, "days_in_q5"] = d5

    return pooled


def main():
    print("Loading data...")
    pooled = load_data()
    print("Assigning quintiles and transitions...")
    pooled = assign_quintiles_and_transitions(pooled)

    val = pooled[(pooled["date"].dt.year >= 2020) & (pooled["date"].dt.year <= 2022)].copy()
    hold = pooled[(pooled["date"].dt.year >= 2023)].copy()

    report = []
    report.append("=" * 110)
    report.append("TRANSITION MOMENTUM — Is there edge in JOINING the trend on quintile entry?")
    report.append("=" * 110)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: What happens on the day a stock ENTERS Q1 or Q5?
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n{'═'*110}")
    report.append("SECTION 1: Forward returns on TRANSITION days (first day entering extreme quintile)")
    report.append("MOMENTUM TRADE: Short day-1 Q1 entries, Long day-1 Q5 entries")
    report.append(f"{'═'*110}")

    for period_name, data in [("VALIDATE 2020-2022", val), ("HOLDOUT 2023-2026", hold)]:
        report.append(f"\n  ── {period_name} ──")

        for transition, direction, label in [
            ("entered_q1", "short", "Entered Q1 (slope just started declining) → SHORT momentum"),
            ("entered_q5", "long", "Entered Q5 (slope just started rising) → LONG momentum"),
        ]:
            entries = data[data[transition]]
            n = len(entries)
            report.append(f"\n  {label}")
            report.append(f"  Events: {n:,}")

            if n < 30:
                report.append("  Too few events")
                continue

            report.append(f"  {'Hold':>6} {'Mean bps':>9} {'Median':>8} {'WR%':>5} {'t':>7} {'Sharpe':>7}")
            report.append(f"  {'─'*50}")

            for h in [1, 2, 3, 5, 10, 20]:
                fwd = entries[f"fwd_{h}d_bps"].dropna()
                if len(fwd) < 20:
                    continue
                if direction == "short":
                    fwd = -fwd
                mean = fwd.mean()
                med = fwd.median()
                wr = 100 * (fwd > 0).mean()
                t = mean / (fwd.std() / np.sqrt(len(fwd))) if fwd.std() > 0 else 0
                sr = mean / fwd.std() * np.sqrt(252 / h) if fwd.std() > 0 else 0

                report.append(f"  {h:>4}d {mean:>+9.1f} {med:>+8.1f} {wr:>5.1f} {t:>+7.2f} {sr:>+7.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: Where did the stock come from? (Q2→Q1 vs Q3+→Q1)
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 2: Does the ORIGIN of the transition matter?")
    report.append("Q2→Q1 = gradual deterioration.  Q3+→Q1 = sudden collapse.")
    report.append(f"{'═'*110}")

    for period_name, data in [("VALIDATE", val), ("HOLDOUT", hold)]:
        report.append(f"\n  ── {period_name} ──")

        for label, mask, direction in [
            ("Q1 from Q2 (gradual) → SHORT", data["q1_from_q2"], "short"),
            ("Q1 from Q3+ (sudden) → SHORT", data["q1_from_q3plus"], "short"),
            ("Q5 from Q4 (gradual) → LONG", data["q5_from_q4"], "long"),
            ("Q5 from Q3- (sudden) → LONG", data["q5_from_q3minus"], "long"),
        ]:
            entries = data[mask]
            n = len(entries)
            if n < 20:
                continue

            results = []
            for h in [1, 3, 5, 10]:
                fwd = entries[f"fwd_{h}d_bps"].dropna()
                if len(fwd) < 10:
                    continue
                if direction == "short":
                    fwd = -fwd
                results.append(f"{fwd.mean():>+7.1f}")

            report.append(f"  {label:<45} N={n:>5}  1d/3d/5d/10d: {' '.join(results)}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: Momentum lifecycle — returns by day number in quintile
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 3: LIFECYCLE — Returns by day number in quintile")
    report.append("Day 1 = just entered. Day 10+ = been there a while.")
    report.append("MOMENTUM TRADES would be on early days. REVERSION on later days.")
    report.append(f"{'═'*110}")

    for period_name, data in [("VALIDATE", val)]:
        report.append(f"\n  ── {period_name} ──")

        for q_num, days_col, direction, label in [
            (1, "days_in_q1", "long", "Q1 (declining slope) — LONG side (mean reversion)"),
            (1, "days_in_q1", "short", "Q1 (declining slope) — SHORT side (momentum)"),
            (5, "days_in_q5", "short", "Q5 (rising slope) — SHORT side (mean reversion)"),
            (5, "days_in_q5", "long", "Q5 (rising slope) — LONG side (momentum)"),
        ]:
            q_data = data[data["quintile"] == q_num]
            report.append(f"\n  {label}:")
            report.append(f"  {'Day#':>6} {'N':>7} {'1d bps':>8} {'3d bps':>8} {'5d bps':>8} {'1d WR':>6} {'1d t':>6}")
            report.append(f"  {'─'*55}")

            for day in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
                subset = q_data[q_data[days_col] == day]
                if len(subset) < 30:
                    continue

                row_parts = [f"  {day:>6} {len(subset):>7}"]
                for h in [1, 3, 5]:
                    fwd = subset[f"fwd_{h}d_bps"].dropna()
                    mean = -fwd.mean() if direction == "short" else fwd.mean()
                    row_parts.append(f"{mean:>+8.1f}")

                fwd1 = subset["fwd_1d_bps"].dropna()
                mean1 = -fwd1.mean() if direction == "short" else fwd1.mean()
                wr = 100 * (fwd1 < 0).mean() if direction == "short" else 100 * (fwd1 > 0).mean()
                t = mean1 / (fwd1.std() / np.sqrt(len(fwd1))) if fwd1.std() > 0 else 0

                row_parts.append(f"{wr:>6.1f}")
                row_parts.append(f"{t:>+6.2f}")
                report.append(" ".join(row_parts))

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: Combined strategy — momentum early, reversion late
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 4: COMBINED — Momentum first N days, then switch to reversion")
    report.append("Test: short Q1 for first K days, then flip to long Q1")
    report.append(f"{'═'*110}")

    for period_name, data in [("VALIDATE", val), ("HOLDOUT", hold)]:
        report.append(f"\n  ── {period_name} ──")
        report.append(f"  {'Switch@':>8} {'MomN':>6} {'MomBps':>8} {'RevN':>6} {'RevBps':>8} {'CombBps':>9} {'CombSR':>7}")
        report.append(f"  {'─'*60}")

        q1_data = data[data["quintile"] == 1].copy()

        for switch_day in [1, 2, 3, 5, 7, 10]:
            # Momentum phase: days 1 to switch_day, SHORT q1
            mom_mask = (q1_data["days_in_q1"] >= 1) & (q1_data["days_in_q1"] <= switch_day)
            mom = q1_data[mom_mask]["fwd_1d_bps"].dropna()
            mom_ret = -mom  # short

            # Reversion phase: days > switch_day, LONG q1
            rev_mask = q1_data["days_in_q1"] > switch_day
            rev = q1_data[rev_mask]["fwd_1d_bps"].dropna()
            rev_ret = rev  # long

            if len(mom_ret) < 30 or len(rev_ret) < 30:
                continue

            # Combined (weighted average by N)
            total_n = len(mom_ret) + len(rev_ret)
            comb_mean = (mom_ret.sum() + rev_ret.sum()) / total_n
            all_rets = pd.concat([mom_ret, rev_ret])
            comb_sr = comb_mean / all_rets.std() * np.sqrt(252) if all_rets.std() > 0 else 0

            report.append(
                f"  day {switch_day:>3}   {len(mom_ret):>6} {mom_ret.mean():>+8.1f} "
                f"{len(rev_ret):>6} {rev_ret.mean():>+8.1f} {comb_mean:>+9.1f} {comb_sr:>+7.2f}"
            )

        # Same for Q5
        report.append(f"\n  Q5 side (momentum=LONG early, reversion=SHORT late):")
        report.append(f"  {'Switch@':>8} {'MomN':>6} {'MomBps':>8} {'RevN':>6} {'RevBps':>8} {'CombBps':>9} {'CombSR':>7}")
        report.append(f"  {'─'*60}")

        q5_data = data[data["quintile"] == 5].copy()

        for switch_day in [1, 2, 3, 5, 7, 10]:
            mom_mask = (q5_data["days_in_q5"] >= 1) & (q5_data["days_in_q5"] <= switch_day)
            mom = q5_data[mom_mask]["fwd_1d_bps"].dropna()
            mom_ret = mom  # long (momentum with the rise)

            rev_mask = q5_data["days_in_q5"] > switch_day
            rev = q5_data[rev_mask]["fwd_1d_bps"].dropna()
            rev_ret = -rev  # short (fade the extension)

            if len(mom_ret) < 30 or len(rev_ret) < 30:
                continue

            total_n = len(mom_ret) + len(rev_ret)
            comb_mean = (mom_ret.sum() + rev_ret.sum()) / total_n
            all_rets = pd.concat([mom_ret, rev_ret])
            comb_sr = comb_mean / all_rets.std() * np.sqrt(252) if all_rets.std() > 0 else 0

            report.append(
                f"  day {switch_day:>3}   {len(mom_ret):>6} {mom_ret.mean():>+8.1f} "
                f"{len(rev_ret):>6} {rev_ret.mean():>+8.1f} {comb_mean:>+9.1f} {comb_sr:>+7.2f}"
            )

    report.append(f"\n{'═'*110}")
    report.append("END")
    report.append(f"{'═'*110}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "transition_momentum_report.txt"), "w") as f:
        f.write(report_text)
    print(f"\nReport: {os.path.join(OUT_DIR, 'transition_momentum_report.txt')}")


if __name__ == "__main__":
    main()
