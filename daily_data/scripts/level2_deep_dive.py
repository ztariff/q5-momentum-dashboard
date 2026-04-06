#!/usr/bin/env python3
"""
LEVEL 2: Deep dive on Level 1 survivors.

For each surviving hypothesis:
  1. Walk-forward: train 2015-2019, test 2020-2022, test2 2023-2026
  2. Per-symbol consistency (does it work in most symbols or just a few?)
  3. Year-by-year stability
  4. Slippage breakeven (how much cost can it absorb?)
  5. Null model (shuffle signal dates, 100 perms)

Trade structure defined:
  - Entry: close of signal day + slippage (10 bps for market order)
  - Exit: close of day+N + slippage (10 bps)
  - Total round-trip cost: 20 bps baseline
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"

SLIPPAGE_BPS = 10  # each way

# ── Define the Level 1 survivors to test ──
# Focus on highest-consistency, highest-t conditions

def define_candidates(df):
    """Return dict of {name: (condition_mask, direction, hold_days)}"""
    c = df["close"]
    gap = df["gap_bps"]
    candidates = {}

    # MEAN REVERSION — oversold bounces
    candidates["rsi_below_30_long_3d"] = (df["rsi_14"] < 30, "long", 3)
    candidates["rsi_below_30_long_5d"] = (df["rsi_14"] < 30, "long", 5)
    candidates["rsi_below_20_long_3d"] = (df["rsi_14"] < 20, "long", 3)
    candidates["oversold_combo_long_1d"] = ((df["rsi_14"] < 30) & (df["bb_pctb"] < 0) & (df["stoch_k"] < 20), "long", 1)
    candidates["oversold_combo_long_3d"] = ((df["rsi_14"] < 30) & (df["bb_pctb"] < 0) & (df["stoch_k"] < 20), "long", 3)
    candidates["oversold_combo_long_5d"] = ((df["rsi_14"] < 30) & (df["bb_pctb"] < 0) & (df["stoch_k"] < 20), "long", 5)

    # MEAN REVERSION — MA distance
    candidates["below_sma50_10pct_long_5d"] = (df["dist_sma50_pct"] < -10, "long", 5)
    candidates["below_sma50_10pct_long_3d"] = (df["dist_sma50_pct"] < -10, "long", 3)
    candidates["below_sma20_5pct_long_3d"] = (df["dist_sma20_pct"] < -5, "long", 3)
    candidates["below_sma200_20pct_long_5d"] = (df["dist_sma200_pct"] < -20, "long", 5)

    # STREAK EXHAUSTION
    candidates["5_down_days_long_1d"] = (df["consec_days"] <= -5, "long", 1)
    candidates["5_down_days_long_2d"] = (df["consec_days"] <= -5, "long", 2)
    candidates["5_down_days_long_3d"] = (df["consec_days"] <= -5, "long", 3)
    candidates["4_down_days_long_3d"] = (df["consec_days"] <= -4, "long", 3)
    candidates["3_down_days_long_2d"] = (df["consec_days"] <= -3, "long", 2)

    # OVERBOUGHT FADE
    candidates["stoch_overbought_short_3d"] = ((df["stoch_k"] > 80) & (df["stoch_d"] > 80), "short", 3)
    candidates["stoch_overbought_short_5d"] = ((df["stoch_k"] > 80) & (df["stoch_d"] > 80), "short", 5)
    candidates["rsi_above_80_short_3d"] = (df["rsi_14"] > 80, "short", 3)

    # GAP PATTERNS
    candidates["gap_up_100_weak_close_long_1d"] = ((gap > 100) & (df["close_location"] < 0.3), "long", 1)
    candidates["gap_up_100_weak_close_long_3d"] = ((gap > 100) & (df["close_location"] < 0.3), "long", 3)
    candidates["gap_down_100_strong_close_long_3d"] = ((gap < -100) & (df["close_location"] > 0.7), "long", 3)
    candidates["gap_down_100_vol_spike_long_5d"] = ((gap < -100) & (df["rvol"] > 2.0), "long", 5)

    # BIG RED CANDLE
    candidates["big_red_5pct_long_5d"] = (df["daily_return_pct"] < -5, "long", 5)
    candidates["big_red_3pct_long_5d"] = (df["daily_return_pct"] < -3, "long", 5)
    candidates["big_red_5pct_long_3d"] = (df["daily_return_pct"] < -5, "long", 3)

    # 3-DOWN + OVERSOLD + VOL
    candidates["3down_oversold_vol_long_5d"] = (
        (df["consec_days"] <= -3) & (df["rsi_14"] < 35) & (df["rvol"] > 1.5), "long", 5)

    # MEAN REVERSION SETUP (multi-factor)
    candidates["mean_revert_setup_long_3d"] = (
        (df["dist_sma20_pct"] < -3) & (df["rsi_14"] < 35) & (df["close_location"] > 0.5), "long", 3)

    # VOLATILITY
    candidates["high_rvol_long_5d"] = (df["rvol_20d"] > 40, "long", 5)

    return candidates


def compute_pnl(fwd_bps, direction, slippage_bps=SLIPPAGE_BPS):
    """Compute net PnL after slippage."""
    cost = 2 * slippage_bps  # round trip
    if direction == "long":
        return fwd_bps - cost
    else:
        return -fwd_bps - cost


def walk_forward_test(df, mask, direction, hold_days):
    """Split into train/test1/test2 and evaluate."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    fwd_col = f"fwd_{hold_days}d_bps"
    if fwd_col not in df.columns:
        df[fwd_col] = 10000 * (df["close"].shift(-hold_days) / df["close"] - 1)

    results = {}
    periods = {
        "train_2015_2019": (df["date"].dt.year >= 2015) & (df["date"].dt.year <= 2019),
        "test1_2020_2022": (df["date"].dt.year >= 2020) & (df["date"].dt.year <= 2022),
        "test2_2023_2026": (df["date"].dt.year >= 2023) & (df["date"].dt.year <= 2026),
        "full": pd.Series(True, index=df.index),
    }

    for pname, period_mask in periods.items():
        combined = mask & period_mask
        fwd = df.loc[combined, fwd_col].dropna()
        if len(fwd) < 10:
            results[pname] = None
            continue

        net_pnl = compute_pnl(fwd, direction)
        results[pname] = {
            "n": len(fwd),
            "gross_mean_bps": round(fwd.mean() if direction == "long" else -fwd.mean(), 2),
            "net_mean_bps": round(net_pnl.mean(), 2),
            "net_median_bps": round(net_pnl.median(), 2),
            "std_bps": round(net_pnl.std(), 2),
            "t_stat": round(net_pnl.mean() / (net_pnl.std() / np.sqrt(len(net_pnl))), 3) if net_pnl.std() > 0 else 0,
            "win_rate": round(100 * (net_pnl > 0).mean(), 1),
            "worst": round(net_pnl.min(), 1),
            "best": round(net_pnl.max(), 1),
        }

    return results


def year_by_year(df, mask, direction, hold_days):
    """Test each year independently."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    fwd_col = f"fwd_{hold_days}d_bps"
    if fwd_col not in df.columns:
        df[fwd_col] = 10000 * (df["close"].shift(-hold_days) / df["close"] - 1)

    years = sorted(df["date"].dt.year.unique())
    results = {}
    for yr in years:
        yr_mask = mask & (df["date"].dt.year == yr)
        fwd = df.loc[yr_mask, fwd_col].dropna()
        if len(fwd) < 5:
            continue
        net = compute_pnl(fwd, direction)
        results[yr] = {
            "n": len(fwd),
            "net_mean": round(net.mean(), 1),
            "wr": round(100 * (net > 0).mean(), 1),
        }
    return results


def null_model_test(df, mask, direction, hold_days, n_perms=200):
    """Permutation test: shuffle which days get the signal."""
    fwd_col = f"fwd_{hold_days}d_bps"
    if fwd_col not in df.columns:
        df[fwd_col] = 10000 * (df["close"].shift(-hold_days) / df["close"] - 1)

    fwd_all = df[fwd_col].dropna()
    real_fwd = df.loc[mask, fwd_col].dropna()
    if len(real_fwd) < 30:
        return None

    real_net = compute_pnl(real_fwd, direction).mean()
    n_sig = mask.sum()

    null_means = []
    valid_idx = fwd_all.index
    for _ in range(n_perms):
        perm_idx = np.random.choice(valid_idx, size=min(n_sig, len(valid_idx)), replace=False)
        perm_fwd = fwd_all.loc[perm_idx]
        null_means.append(compute_pnl(perm_fwd, direction).mean())

    null_means = np.array(null_means)
    p_value = (null_means >= real_net).mean() if direction == "long" else (null_means <= real_net).mean()

    return {
        "real_net_bps": round(real_net, 2),
        "null_mean_bps": round(null_means.mean(), 2),
        "null_std_bps": round(null_means.std(), 2),
        "p_value": round(p_value, 4),
        "real_vs_null_ratio": round(real_net / null_means.mean(), 2) if null_means.mean() != 0 else float("inf"),
    }


def slippage_breakeven(df, mask, direction, hold_days, max_slip=50):
    """Find the slippage at which mean PnL goes to zero."""
    fwd_col = f"fwd_{hold_days}d_bps"
    if fwd_col not in df.columns:
        df[fwd_col] = 10000 * (df["close"].shift(-hold_days) / df["close"] - 1)

    fwd = df.loc[mask, fwd_col].dropna()
    if len(fwd) < 30:
        return None

    gross = fwd.mean() if direction == "long" else -fwd.mean()
    # Breakeven: gross_mean = 2 * slippage => slippage = gross_mean / 2
    be = gross / 2
    return round(be, 1)


def main():
    print("Loading all enriched data...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    data = {}
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        for n in [1, 2, 3, 5]:
            df[f"fwd_{n}d_bps"] = 10000 * (df["close"].shift(-n) / df["close"] - 1)
        data[sym] = df
    print(f"Loaded {len(data)} symbols\n")

    # Stack for pooled analysis
    all_dfs = []
    for sym, df in data.items():
        df2 = df.copy()
        df2["symbol"] = sym
        all_dfs.append(df2)
    pooled = pd.concat(all_dfs, ignore_index=True)

    candidates = define_candidates(pooled)
    print(f"Testing {len(candidates)} candidate strategies\n")

    report = []
    report.append("=" * 100)
    report.append("LEVEL 2 DEEP DIVE — Walk-Forward, Consistency, Null Model, Slippage")
    report.append(f"Slippage assumption: {SLIPPAGE_BPS} bps each way ({2*SLIPPAGE_BPS} bps round trip)")
    report.append("=" * 100)

    summary_rows = []

    for cname, (mask, direction, hold_days) in sorted(candidates.items()):
        mask = mask.fillna(False)
        report.append(f"\n{'─'*100}")
        report.append(f"STRATEGY: {cname}")
        report.append(f"Direction: {direction.upper()}, Hold: {hold_days}d")
        report.append(f"{'─'*100}")

        # 1. Walk-forward
        wf = walk_forward_test(pooled, mask, direction, hold_days)
        report.append("\n  WALK-FORWARD (pooled across all symbols):")
        for period, res in wf.items():
            if res is None:
                report.append(f"    {period}: insufficient data")
            else:
                report.append(f"    {period}: N={res['n']:,}, net={res['net_mean_bps']:+.1f}bps, "
                            f"t={res['t_stat']:.2f}, WR={res['win_rate']:.1f}%, "
                            f"median={res['net_median_bps']:+.1f}")

        # Check if OOS is positive
        t1 = wf.get("test1_2020_2022")
        t2 = wf.get("test2_2023_2026")
        oos_pass = True
        if t1 and t1["net_mean_bps"] <= 0:
            oos_pass = False
        if t2 and t2["net_mean_bps"] <= 0:
            oos_pass = False

        if not oos_pass:
            report.append("  ⚠ FAILS OOS — net negative in at least one test period")

        # 2. Year-by-year
        yby = year_by_year(pooled, mask, direction, hold_days)
        pos_years = sum(1 for v in yby.values() if v["net_mean"] > 0)
        total_years = len(yby)
        report.append(f"\n  YEAR-BY-YEAR ({pos_years}/{total_years} positive):")
        for yr, res in sorted(yby.items()):
            status = "+" if res["net_mean"] > 0 else "-"
            report.append(f"    {yr}: N={res['n']:>5}, net={res['net_mean']:>+8.1f}bps, WR={res['wr']:.0f}% [{status}]")

        # 3. Per-symbol consistency
        sym_pos = 0
        sym_neg = 0
        sym_tested = 0
        for sym, sdf in data.items():
            scands = define_candidates(sdf)
            if cname not in scands:
                continue
            smask, sdir, shd = scands[cname]
            smask = smask.fillna(False)
            fwd_col = f"fwd_{shd}d_bps"
            fwd = sdf.loc[smask, fwd_col].dropna()
            if len(fwd) < 10:
                continue
            net = compute_pnl(fwd, sdir)
            sym_tested += 1
            if net.mean() > 0:
                sym_pos += 1
            else:
                sym_neg += 1

        consist_pct = 100 * sym_pos / max(sym_tested, 1)
        report.append(f"\n  PER-SYMBOL CONSISTENCY: {sym_pos}/{sym_tested} positive ({consist_pct:.0f}%)")

        # 4. Null model
        np.random.seed(42)
        null = null_model_test(pooled, mask, direction, hold_days)
        if null:
            report.append(f"\n  NULL MODEL (200 permutations):")
            report.append(f"    Real net: {null['real_net_bps']:+.2f} bps")
            report.append(f"    Null mean: {null['null_mean_bps']:+.2f} bps (std={null['null_std_bps']:.2f})")
            report.append(f"    p-value: {null['p_value']:.4f}")
            report.append(f"    Real / Null ratio: {null['real_vs_null_ratio']:.2f}x")

        # 5. Slippage breakeven
        be = slippage_breakeven(pooled, mask, direction, hold_days)
        if be is not None:
            report.append(f"\n  SLIPPAGE BREAKEVEN: {be:.1f} bps each way ({2*be:.1f} round trip)")
            if be < 10:
                report.append(f"  ⚠ THIN EDGE — breakeven below 10 bps/side")

        # Verdict
        full = wf.get("full")
        verdict_parts = []
        if full and full["t_stat"] > 3:
            verdict_parts.append("strong pooled t-stat")
        if pos_years >= total_years * 0.7:
            verdict_parts.append(f"stable ({pos_years}/{total_years} yrs)")
        if consist_pct > 55:
            verdict_parts.append(f"consistent ({consist_pct:.0f}% syms)")
        if null and null["p_value"] < 0.01:
            verdict_parts.append("beats null")
        if be and be >= 15:
            verdict_parts.append(f"robust to costs ({be:.0f}bp BE)")

        if oos_pass and len(verdict_parts) >= 3:
            verdict = "PASS → Level 3"
        elif oos_pass and len(verdict_parts) >= 2:
            verdict = "MARGINAL — worth investigating"
        else:
            verdict = "KILL"

        report.append(f"\n  VERDICT: {verdict}")
        if verdict_parts:
            report.append(f"    Evidence: {', '.join(verdict_parts)}")

        summary_rows.append({
            "strategy": cname,
            "direction": direction,
            "hold_days": hold_days,
            "full_n": full["n"] if full else 0,
            "full_net_bps": full["net_mean_bps"] if full else 0,
            "full_t": full["t_stat"] if full else 0,
            "full_wr": full["win_rate"] if full else 0,
            "oos1_net": t1["net_mean_bps"] if t1 else None,
            "oos1_t": t1["t_stat"] if t1 else None,
            "oos2_net": t2["net_mean_bps"] if t2 else None,
            "oos2_t": t2["t_stat"] if t2 else None,
            "pos_years": pos_years,
            "total_years": total_years,
            "sym_consist_pct": consist_pct,
            "null_p": null["p_value"] if null else None,
            "slippage_be": be,
            "verdict": verdict,
        })

    # Final summary table
    report.append(f"\n\n{'='*100}")
    report.append("FINAL SUMMARY — All candidates ranked")
    report.append(f"{'='*100}")
    report.append(f"{'Strategy':<45} {'Net':>6} {'t':>6} {'WR':>5} {'Yrs':>5} {'Sym%':>5} {'NullP':>6} {'BE':>5} {'Verdict'}")
    report.append("─" * 100)

    summary_df = pd.DataFrame(summary_rows).sort_values("full_t", ascending=False)
    for _, row in summary_df.iterrows():
        null_p = f"{row['null_p']:.3f}" if row["null_p"] is not None else "n/a"
        be = f"{row['slippage_be']:.0f}" if row["slippage_be"] is not None else "n/a"
        report.append(
            f"{row['strategy']:<45} {row['full_net_bps']:>+6.1f} {row['full_t']:>6.2f} "
            f"{row['full_wr']:>5.1f} {row['pos_years']:>2}/{row['total_years']:<2} "
            f"{row['sym_consist_pct']:>5.0f} {null_p:>6} {be:>5} {row['verdict']}"
        )

    report.append("")

    report_text = "\n".join(report)
    print(report_text)

    report_path = os.path.join(OUT_DIR, "level2_deep_dive_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    summary_df.to_csv(os.path.join(OUT_DIR, "level2_summary.csv"), index=False)
    print(f"\nReport: {report_path}")
    print(f"Summary: {os.path.join(OUT_DIR, 'level2_summary.csv')}")


if __name__ == "__main__":
    main()
