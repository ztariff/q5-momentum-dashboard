#!/usr/bin/env python3
"""
LEVEL 4: Lead quant's three questions.

1. SYMMETRY: Does rising MA slope predict underperformance (short side)?
   Test all MA slopes, all periods. Decompose L/S into long-leg and short-leg P&L.

2. FACTOR OVERLAP: Is this just vanilla short-term reversal?
   Compare our signal to simple "sort by last N-day return."

3. TURNOVER: How much does the portfolio churn?
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"

HOLD_PERIODS = [1, 3, 5]


def load_and_prepare():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    all_dfs = []
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = sym
        for n in HOLD_PERIODS:
            raw_bps = 10000 * (df["close"].shift(-n) / df["close"] - 1)
            rolling_mean = raw_bps.rolling(60, min_periods=30).mean()
            df[f"fwd_{n}d_bps"] = raw_bps
            df[f"exc_{n}d_bps"] = raw_bps - rolling_mean

        # Build additional MA slopes we didn't have before
        c = df["close"]
        for w in [8, 12, 20, 50, 100, 200]:
            sma = c.rolling(w).mean()
            for lookback in [3, 5, 10]:
                col = f"sma{w}_slope_{lookback}d"
                df[col] = 100 * (sma / sma.shift(lookback) - 1)

        # Z-score all slope features within symbol
        slope_cols = [c for c in df.columns if "slope" in c]
        for sc in slope_cols:
            roll_mean = df[sc].rolling(252, min_periods=60).mean()
            roll_std = df[sc].rolling(252, min_periods=60).std()
            df[f"z_{sc}"] = (df[sc] - roll_mean) / roll_std.replace(0, np.nan)

        # Also add vanilla reversal features (raw prior returns)
        for lb in [1, 3, 5, 10, 20]:
            df[f"past_{lb}d_ret"] = 100 * (c / c.shift(lb) - 1)
            roll_mean = df[f"past_{lb}d_ret"].rolling(252, min_periods=60).mean()
            roll_std = df[f"past_{lb}d_ret"].rolling(252, min_periods=60).std()
            df[f"z_past_{lb}d_ret"] = (df[f"past_{lb}d_ret"] - roll_mean) / roll_std.replace(0, np.nan)

        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def long_short_decomposed(pooled, feature_col, return_col, hold):
    """
    Cross-sectional quintile sort. Return SEPARATE long-leg and short-leg P&L.
    Long leg = buy Q1 (lowest feature), Short leg = sell Q5 (highest feature).
    """
    results = []

    for date, ddf in pooled.groupby("date"):
        valid = ddf[[feature_col, return_col, "symbol"]].dropna()
        if len(valid) < 20:
            continue
        try:
            valid["q"] = pd.qcut(valid[feature_col], 5, labels=[1,2,3,4,5], duplicates="drop")
        except ValueError:
            continue
        if valid["q"].nunique() < 5:
            continue

        q1 = valid[valid["q"] == 1][return_col]
        q5 = valid[valid["q"] == 5][return_col]

        results.append({
            "date": date,
            "long_q1": q1.mean(),     # Long the lowest (most beaten down)
            "short_q5": -q5.mean(),   # Short the highest (most extended up)
            "ls": q1.mean() - q5.mean(),
            "n_long": len(q1),
            "n_short": len(q5),
            "q1_symbols": ",".join(sorted(valid[valid["q"] == 1]["symbol"].tolist())),
            "q5_symbols": ",".join(sorted(valid[valid["q"] == 5]["symbol"].tolist())),
        })

    rdf = pd.DataFrame(results)
    rdf["date"] = pd.to_datetime(rdf["date"])
    # Space by hold period
    rdf = rdf.iloc[::hold].reset_index(drop=True)
    return rdf


def compute_turnover(rdf, leg="q1"):
    """Compute average turnover per rebalance for a given leg."""
    sym_col = f"{leg}_symbols"
    turnovers = []
    prev_set = None
    for _, row in rdf.iterrows():
        curr_set = set(row[sym_col].split(",")) if row[sym_col] else set()
        if prev_set is not None and len(curr_set) > 0:
            # Turnover = fraction of names that changed
            overlap = len(curr_set & prev_set)
            total = max(len(curr_set), 1)
            turnover = 1 - overlap / total
            turnovers.append(turnover)
        prev_set = curr_set
    return np.mean(turnovers) if turnovers else 1.0


def year_stats(rdf, col):
    """Year-by-year stats for a column."""
    rdf = rdf.copy()
    rdf["year"] = rdf["date"].dt.year
    results = {}
    for yr, ydf in rdf.groupby("year"):
        vals = ydf[col]
        if len(vals) < 5:
            continue
        results[yr] = {
            "n": len(vals),
            "mean": round(vals.mean(), 2),
            "sharpe": round(vals.mean() / vals.std() * np.sqrt(252 / max(1, len(vals) / (vals.index[-1] - vals.index[0] + 1))), 2) if vals.std() > 0 else 0,
            "positive": vals.mean() > 0,
        }
    return results


def main():
    print("Loading and preparing data (all MA slopes + vanilla reversal)...")
    pooled = load_and_prepare()
    print(f"Total rows: {len(pooled):,}, Symbols: {pooled['symbol'].nunique()}\n")

    # Test period only
    test_mask = pooled["date"].dt.year >= 2020
    test = pooled[test_mask].copy()

    report = []
    report.append("=" * 110)
    report.append("LEVEL 4: SYMMETRY, FACTOR OVERLAP, TURNOVER")
    report.append("=" * 110)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: SYMMETRY — Does it work on both sides?
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n{'═'*110}")
    report.append("SECTION 1: SYMMETRY — Long leg vs Short leg decomposition")
    report.append("Long Q1 = stocks with most DECLINING slope (buy beaten down)")
    report.append("Short Q5 = stocks with most RISING slope (sell extended up)")
    report.append("If both legs contribute, the effect is symmetric.")
    report.append(f"{'═'*110}")

    # Test all MA slopes
    slope_features = [f"z_sma{w}_slope_{lb}d" for w in [8, 12, 20, 50, 100, 200] for lb in [3, 5, 10]]

    decomp_results = []

    for hold in [1, 5]:
        exc_col = f"exc_{hold}d_bps"
        report.append(f"\n  Hold = {hold}d:")
        report.append(f"  {'Feature':<30} {'L/S':>8} {'LongLeg':>8} {'ShortLeg':>8} {'%Long':>6} {'%Short':>6} {'LongSR':>7} {'ShortSR':>7} {'L/S SR':>7}")
        report.append(f"  {'─'*100}")

        for feat in slope_features:
            if feat not in test.columns:
                continue

            rdf = long_short_decomposed(test, feat, exc_col, hold)
            if len(rdf) < 50:
                continue

            ls_mean = rdf["ls"].mean()
            long_mean = rdf["long_q1"].mean()
            short_mean = rdf["short_q5"].mean()

            # What % of L/S comes from each leg?
            total_abs = abs(long_mean) + abs(short_mean)
            pct_long = 100 * abs(long_mean) / total_abs if total_abs > 0 else 50
            pct_short = 100 * abs(short_mean) / total_abs if total_abs > 0 else 50

            # Per-leg Sharpe
            scale = np.sqrt(252 / hold)
            long_sr = long_mean / rdf["long_q1"].std() * scale if rdf["long_q1"].std() > 0 else 0
            short_sr = short_mean / rdf["short_q5"].std() * scale if rdf["short_q5"].std() > 0 else 0
            ls_sr = ls_mean / rdf["ls"].std() * scale if rdf["ls"].std() > 0 else 0

            # Is the short leg positive (profitable)?
            short_works = short_mean > 0

            report.append(
                f"  {feat:<30} {ls_mean:>+8.1f} {long_mean:>+8.1f} {short_mean:>+8.1f} "
                f"{pct_long:>5.0f}% {pct_short:>5.0f}% {long_sr:>+7.2f} {short_sr:>+7.2f} {ls_sr:>+7.2f}"
                f"  {'✓ BOTH' if short_works and long_mean > 0 else '✗ ONE-SIDED' if long_mean > 0 else '✗ BROKEN'}"
            )

            decomp_results.append({
                "feature": feat, "hold": hold,
                "ls_mean": round(ls_mean, 2),
                "long_mean": round(long_mean, 2),
                "short_mean": round(short_mean, 2),
                "pct_long": round(pct_long, 1),
                "pct_short": round(pct_short, 1),
                "long_sr": round(long_sr, 2),
                "short_sr": round(short_sr, 2),
                "ls_sr": round(ls_sr, 2),
                "symmetric": short_works and long_mean > 0,
            })

    pd.DataFrame(decomp_results).to_csv(os.path.join(OUT_DIR, "level4_symmetry.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: Best MA — which lookback and which MA works best?
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 2: BEST MA SLOPE — Heatmap of MA period × slope lookback")
    report.append(f"{'═'*110}")

    for hold in [1, 5]:
        exc_col = f"exc_{hold}d_bps"
        report.append(f"\n  Hold = {hold}d — L/S Sharpe by (MA period, slope lookback):")
        report.append(f"  {'':>12} {'3d':>8} {'5d':>8} {'10d':>8}")
        report.append(f"  {'─'*40}")

        for w in [8, 12, 20, 50, 100, 200]:
            line = f"  SMA{w:>4}    "
            for lb in [3, 5, 10]:
                feat = f"z_sma{w}_slope_{lb}d"
                row = [r for r in decomp_results if r["feature"] == feat and r["hold"] == hold]
                if row:
                    line += f"  {row[0]['ls_sr']:>+6.2f}"
                else:
                    line += f"  {'n/a':>6}"
            report.append(line)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: FACTOR OVERLAP — Vanilla reversal comparison
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 3: FACTOR OVERLAP — Is this just short-term reversal?")
    report.append("Compare MA slope signal to vanilla 'sort by prior N-day return'")
    report.append(f"{'═'*110}")

    # Test vanilla reversal signals
    vanilla_features = [f"z_past_{lb}d_ret" for lb in [1, 3, 5, 10, 20]]
    best_slope = "z_sma50_slope_5d"  # our best signal from Level 3

    for hold in [1, 5]:
        exc_col = f"exc_{hold}d_bps"
        report.append(f"\n  Hold = {hold}d:")
        report.append(f"  {'Signal':<30} {'L/S bps':>8} {'L/S SR':>7} {'LongSR':>7} {'ShortSR':>7}")
        report.append(f"  {'─'*70}")

        # Our best signal
        rdf_best = long_short_decomposed(test, best_slope, exc_col, hold)
        if len(rdf_best) > 50:
            scale = np.sqrt(252 / hold)
            sr = rdf_best["ls"].mean() / rdf_best["ls"].std() * scale
            report.append(
                f"  {'OUR: sma50_slope_5d':<30} {rdf_best['ls'].mean():>+8.1f} {sr:>+7.2f} "
                f"{rdf_best['long_q1'].mean() / rdf_best['long_q1'].std() * scale:>+7.2f} "
                f"{rdf_best['short_q5'].mean() / rdf_best['short_q5'].std() * scale:>+7.2f}"
            )

        for vf in vanilla_features:
            if vf not in test.columns:
                continue
            rdf_v = long_short_decomposed(test, vf, exc_col, hold)
            if len(rdf_v) < 50:
                continue
            scale = np.sqrt(252 / hold)
            sr = rdf_v["ls"].mean() / rdf_v["ls"].std() * scale
            report.append(
                f"  {vf:<30} {rdf_v['ls'].mean():>+8.1f} {sr:>+7.2f} "
                f"{rdf_v['long_q1'].mean() / rdf_v['long_q1'].std() * scale:>+7.2f} "
                f"{rdf_v['short_q5'].mean() / rdf_v['short_q5'].std() * scale:>+7.2f}"
            )

        # Correlation between our signal's L/S returns and vanilla's L/S returns
        report.append(f"\n  Correlation of L/S daily returns (our signal vs vanilla):")
        if len(rdf_best) > 50:
            for vf in vanilla_features:
                if vf not in test.columns:
                    continue
                rdf_v = long_short_decomposed(test, vf, exc_col, hold)
                if len(rdf_v) < 50:
                    continue
                # Align by date
                merged = rdf_best[["date", "ls"]].merge(rdf_v[["date", "ls"]], on="date", suffixes=("_ours", "_vanilla"))
                if len(merged) > 30:
                    corr = merged["ls_ours"].corr(merged["ls_vanilla"])
                    report.append(f"    vs {vf:<28}: r = {corr:+.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: TURNOVER ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 4: TURNOVER — How much does the portfolio churn?")
    report.append(f"{'═'*110}")

    for hold in [1, 5]:
        exc_col = f"exc_{hold}d_bps"

        for feat in [best_slope, "z_past_5d_ret", "z_sma200_slope_10d", "z_sma8_slope_3d"]:
            if feat not in test.columns:
                continue
            rdf = long_short_decomposed(test, feat, exc_col, hold)
            if len(rdf) < 50:
                continue

            long_to = compute_turnover(rdf, "q1")
            short_to = compute_turnover(rdf, "q5")

            report.append(f"  {feat:<30} hold={hold}d: Long turnover={100*long_to:.1f}%, Short turnover={100*short_to:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: YEAR-BY-YEAR for best signal, decomposed
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 5: YEAR-BY-YEAR DECOMPOSITION — Best signal (sma50_slope_5d)")
    report.append(f"{'═'*110}")

    for hold in [1, 5]:
        exc_col = f"exc_{hold}d_bps"
        rdf = long_short_decomposed(test, best_slope, exc_col, hold)
        if len(rdf) < 50:
            continue

        rdf["year"] = rdf["date"].dt.year

        report.append(f"\n  Hold = {hold}d:")
        report.append(f"  {'Year':>6} {'N':>5} {'L/S':>8} {'LongLeg':>8} {'ShortLeg':>8} {'L/S WR':>7}")
        report.append(f"  {'─'*50}")

        for yr in sorted(rdf["year"].unique()):
            ydf = rdf[rdf["year"] == yr]
            report.append(
                f"  {yr:>6} {len(ydf):>5} {ydf['ls'].mean():>+8.1f} "
                f"{ydf['long_q1'].mean():>+8.1f} {ydf['short_q5'].mean():>+8.1f} "
                f"{100*(ydf['ls']>0).mean():>6.1f}%"
            )

    report.append(f"\n{'═'*110}")
    report.append("END")
    report.append(f"{'═'*110}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "level4_symmetry_report.txt"), "w") as f:
        f.write(report_text)
    print(f"\nReport: {os.path.join(OUT_DIR, 'level4_symmetry_report.txt')}")


if __name__ == "__main__":
    main()
