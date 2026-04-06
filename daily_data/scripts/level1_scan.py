#!/usr/bin/env python3
"""
LEVEL 1: Broad hypothesis scan across 230 symbols × ~50 conditions × multiple holding periods.

No preconceptions. Test everything. Apply multiple-testing correction.
Surface what survives. Kill what doesn't.

Trade structure:
  - Entry at CLOSE of signal day (you see the signal, you act EOD)
  - Exit at CLOSE of day+N (N = 1, 2, 3, 5)
  - Returns in bps
  - No slippage yet (Level 1 = does the raw effect exist?)

Null model: unconditional mean return for each symbol over same period.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

HOLD_PERIODS = [1, 2, 3, 5]


def load_all():
    """Load all enriched CSVs into a dict."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    data = {}
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        # Forward returns (close-to-close, in bps)
        for n in HOLD_PERIODS:
            df[f"fwd_{n}d_bps"] = 10000 * (df["close"].shift(-n) / df["close"] - 1)
        data[sym] = df
    return data


def define_conditions(df):
    """
    Define all binary conditions to test. Returns dict of {name: boolean_series}.
    Each condition = "if this is true today, what happens over the next N days?"
    """
    conds = {}
    c = df["close"]

    # ── GAP CONDITIONS ──
    gap = df["gap_bps"]
    conds["gap_up_50bps"] = gap > 50
    conds["gap_up_100bps"] = gap > 100
    conds["gap_up_200bps"] = gap > 200
    conds["gap_down_50bps"] = gap < -50
    conds["gap_down_100bps"] = gap < -100
    conds["gap_down_200bps"] = gap < -200
    # Gap + close location (did it fill or extend?)
    conds["gap_up_100_closed_weak"] = (gap > 100) & (df["close_location"] < 0.3)
    conds["gap_up_100_closed_strong"] = (gap > 100) & (df["close_location"] > 0.7)
    conds["gap_down_100_closed_weak"] = (gap < -100) & (df["close_location"] < 0.3)
    conds["gap_down_100_closed_strong"] = (gap < -100) & (df["close_location"] > 0.7)

    # ── MEAN REVERSION / OVEREXTENSION ──
    conds["rsi_below_30"] = df["rsi_14"] < 30
    conds["rsi_below_20"] = df["rsi_14"] < 20
    conds["rsi_above_70"] = df["rsi_14"] > 70
    conds["rsi_above_80"] = df["rsi_14"] > 80
    conds["bb_below_lower"] = df["bb_pctb"] < 0
    conds["bb_above_upper"] = df["bb_pctb"] > 1
    conds["dist_sma20_below_neg3pct"] = df["dist_sma20_pct"] < -3
    conds["dist_sma20_below_neg5pct"] = df["dist_sma20_pct"] < -5
    conds["dist_sma20_above_3pct"] = df["dist_sma20_pct"] > 3
    conds["dist_sma20_above_5pct"] = df["dist_sma20_pct"] > 5
    conds["dist_sma50_below_neg5pct"] = df["dist_sma50_pct"] < -5
    conds["dist_sma50_below_neg10pct"] = df["dist_sma50_pct"] < -10
    conds["dist_sma50_above_5pct"] = df["dist_sma50_pct"] > 5
    conds["dist_sma50_above_10pct"] = df["dist_sma50_pct"] > 10
    conds["dist_sma200_below_neg10pct"] = df["dist_sma200_pct"] < -10
    conds["dist_sma200_below_neg20pct"] = df["dist_sma200_pct"] < -20
    conds["dist_sma200_above_10pct"] = df["dist_sma200_pct"] > 10
    conds["dist_sma200_above_20pct"] = df["dist_sma200_pct"] > 20

    # Stochastic extremes
    conds["stoch_oversold"] = (df["stoch_k"] < 20) & (df["stoch_d"] < 20)
    conds["stoch_overbought"] = (df["stoch_k"] > 80) & (df["stoch_d"] > 80)

    # ── CONSECUTIVE DAYS ──
    conds["3_down_days"] = df["consec_days"] <= -3
    conds["4_down_days"] = df["consec_days"] <= -4
    conds["5_down_days"] = df["consec_days"] <= -5
    conds["3_up_days"] = df["consec_days"] >= 3
    conds["4_up_days"] = df["consec_days"] >= 4
    conds["5_up_days"] = df["consec_days"] >= 5

    # ── VOLUME CONDITIONS ──
    conds["volume_spike_2x"] = df["rvol"] > 2.0
    conds["volume_spike_3x"] = df["rvol"] > 3.0
    conds["volume_dry_below_0.5x"] = df["rvol"] < 0.5
    # Volume spike + direction
    conds["vol_spike_2x_up_day"] = (df["rvol"] > 2.0) & (df["daily_return_pct"] > 0)
    conds["vol_spike_2x_down_day"] = (df["rvol"] > 2.0) & (df["daily_return_pct"] < 0)
    conds["vol_spike_3x_big_down"] = (df["rvol"] > 3.0) & (df["daily_return_pct"] < -2)

    # ── TREND / MOMENTUM ──
    conds["above_all_mas"] = (c > df["sma_8"]) & (c > df["sma_20"]) & (c > df["sma_50"]) & (c > df["sma_200"])
    conds["below_all_mas"] = (c < df["sma_8"]) & (c < df["sma_20"]) & (c < df["sma_50"]) & (c < df["sma_200"])
    conds["golden_cross_today"] = (df["sma_50"] > df["sma_200"]) & (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
    conds["death_cross_today"] = (df["sma_50"] < df["sma_200"]) & (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
    conds["sma8_cross_above_sma20"] = (df["sma_8"] > df["sma_20"]) & (df["sma_8"].shift(1) <= df["sma_20"].shift(1))
    conds["sma8_cross_below_sma20"] = (df["sma_8"] < df["sma_20"]) & (df["sma_8"].shift(1) >= df["sma_20"].shift(1))
    conds["macd_cross_above"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    conds["macd_cross_below"] = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    # ADX trend strength
    conds["strong_trend_adx_above_30"] = df["adx_14"] > 30
    conds["weak_trend_adx_below_15"] = df["adx_14"] < 15
    # Momentum
    conds["roc10_above_10pct"] = df["roc_10"] > 10
    conds["roc10_below_neg10pct"] = df["roc_10"] < -10
    conds["roc20_above_15pct"] = df["roc_20"] > 15
    conds["roc20_below_neg15pct"] = df["roc_20"] < -15

    # ── VOLATILITY ──
    conds["squeeze_on"] = df["squeeze"] == 1
    conds["squeeze_release_today"] = (df["squeeze"] == 0) & (df["squeeze"].shift(1) == 1)
    conds["range_expansion_2x_atr"] = df["range_bps"] > (df["atr_14"] / c * 10000) * 2
    conds["narrow_range_below_0.5x_atr"] = df["range_bps"] < (df["atr_14"] / c * 10000) * 0.5
    conds["rvol_20d_above_40"] = df["rvol_20d"] > 40
    conds["rvol_20d_below_15"] = df["rvol_20d"] < 15

    # ── PROXIMITY TO HIGHS/LOWS ──
    conds["within_2pct_of_52w_high"] = df["pct_from_52w_high"] > -2
    conds["within_5pct_of_52w_high"] = df["pct_from_52w_high"] > -5
    conds["down_20pct_from_52w_high"] = df["pct_from_52w_high"] < -20
    conds["down_30pct_from_52w_high"] = df["pct_from_52w_high"] < -30
    conds["at_20d_low"] = df["pct_from_20d_low"] < 1
    conds["at_20d_high"] = df["pct_from_20d_high"] > -1

    # ── CANDLE PATTERNS ──
    conds["big_red_candle_3pct"] = df["daily_return_pct"] < -3
    conds["big_green_candle_3pct"] = df["daily_return_pct"] > 3
    conds["big_red_candle_5pct"] = df["daily_return_pct"] < -5
    conds["big_green_candle_5pct"] = df["daily_return_pct"] > 5
    conds["doji_body_lt_10pct"] = df["body_pct"].abs() < 0.10
    conds["hammer_low_wick"] = (df["close_location"] > 0.7) & ((df["low"] - df[["open","close"]].min(axis=1)) > 2 * (df["close"] - df["open"]).abs())

    # ── COMBINATION CONDITIONS ──
    conds["oversold_combo"] = (df["rsi_14"] < 30) & (df["bb_pctb"] < 0) & (df["stoch_k"] < 20)
    conds["overbought_combo"] = (df["rsi_14"] > 70) & (df["bb_pctb"] > 1) & (df["stoch_k"] > 80)
    conds["vol_spike_at_52w_low"] = (df["rvol"] > 2) & (df["pct_from_52w_high"] < -20)
    conds["vol_spike_squeeze_break"] = (df["rvol"] > 1.5) & (df["squeeze"].shift(1) == 1) & (df["squeeze"] == 0)
    conds["3down_oversold_vol"] = (df["consec_days"] <= -3) & (df["rsi_14"] < 35) & (df["rvol"] > 1.5)
    conds["gap_down_100_vol_spike"] = (gap < -100) & (df["rvol"] > 2.0)
    conds["rsi_divergence_bullish"] = (df["rsi_14"] < 35) & (df["rsi_14"] > df["rsi_14"].shift(5)) & (c < c.shift(5))
    conds["mean_revert_setup"] = (df["dist_sma20_pct"] < -3) & (df["rsi_14"] < 35) & (df["close_location"] > 0.5)

    return conds


def test_condition(fwd_returns, mask, uncond_mean, uncond_std, uncond_n):
    """Test one condition: is the conditional mean different from unconditional?"""
    subset = fwd_returns[mask].dropna()
    n = len(subset)
    if n < 30:
        return None

    mean_ret = subset.mean()
    std_ret = subset.std()

    # t-test vs unconditional mean
    excess = mean_ret - uncond_mean
    se = std_ret / np.sqrt(n)
    t_stat = excess / se if se > 0 else 0

    # Win rate
    wr = (subset > 0).mean() * 100

    # Median (robust)
    median_ret = subset.median()

    return {
        "n": n,
        "mean_bps": round(mean_ret, 2),
        "median_bps": round(median_ret, 2),
        "std_bps": round(std_ret, 2),
        "excess_bps": round(excess, 2),
        "t_stat": round(t_stat, 3),
        "win_rate": round(wr, 1),
        "uncond_mean": round(uncond_mean, 2),
    }


def main():
    print("Loading all enriched data...")
    data = load_all()
    print(f"Loaded {len(data)} symbols\n")

    # ── PHASE A: Per-symbol scan (find effects that work across many symbols) ──
    # For each condition × hold period, count how many symbols show t > 2

    # First pass: get all condition names from one symbol
    sample_df = list(data.values())[0]
    sample_conds = define_conditions(sample_df)
    cond_names = sorted(sample_conds.keys())
    print(f"Testing {len(cond_names)} conditions × {len(HOLD_PERIODS)} hold periods × {len(data)} symbols")
    print(f"= {len(cond_names) * len(HOLD_PERIODS) * len(data):,} hypothesis tests\n")

    # Track: for each (condition, hold_period), how many symbols significant?
    results_pooled = []  # pooled across all symbols
    results_consistency = defaultdict(lambda: {"pos_t2": 0, "neg_t2": 0, "tested": 0, "total_n": 0})

    # Also collect per-symbol results for the detail file
    per_symbol_rows = []

    for si, (sym, df) in enumerate(data.items()):
        if si % 50 == 0:
            print(f"  Processing symbol {si+1}/{len(data)} ({sym})...")

        conds = define_conditions(df)

        for hp in HOLD_PERIODS:
            fwd_col = f"fwd_{hp}d_bps"
            fwd = df[fwd_col]
            valid = fwd.dropna()
            if len(valid) < 100:
                continue
            uncond_mean = valid.mean()
            uncond_std = valid.std()
            uncond_n = len(valid)

            for cname in cond_names:
                mask = conds[cname].fillna(False)
                result = test_condition(fwd, mask, uncond_mean, uncond_std, uncond_n)
                if result is None:
                    continue

                key = (cname, hp)
                results_consistency[key]["tested"] += 1
                results_consistency[key]["total_n"] += result["n"]
                if result["t_stat"] > 2:
                    results_consistency[key]["pos_t2"] += 1
                elif result["t_stat"] < -2:
                    results_consistency[key]["neg_t2"] += 1

                per_symbol_rows.append({
                    "symbol": sym,
                    "condition": cname,
                    "hold_days": hp,
                    **result,
                })

    print(f"\nPer-symbol tests done. {len(per_symbol_rows):,} results.\n")

    # ── PHASE B: Pooled test (stack all symbols, test condition pooled) ──
    print("Running pooled tests...")

    # Stack all data with a symbol column
    all_dfs = []
    for sym, df in data.items():
        df2 = df.copy()
        df2["symbol"] = sym
        all_dfs.append(df2)
    pooled = pd.concat(all_dfs, ignore_index=True)

    pooled_conds = define_conditions(pooled)

    pooled_rows = []
    for hp in HOLD_PERIODS:
        fwd_col = f"fwd_{hp}d_bps"
        fwd = pooled[fwd_col]
        valid = fwd.dropna()
        uncond_mean = valid.mean()
        uncond_std = valid.std()
        uncond_n = len(valid)

        for cname in cond_names:
            mask = pooled_conds[cname].fillna(False)
            result = test_condition(fwd, mask, uncond_mean, uncond_std, uncond_n)
            if result is None:
                continue

            key = (cname, hp)
            cons = results_consistency[key]

            pooled_rows.append({
                "condition": cname,
                "hold_days": hp,
                "pooled_n": result["n"],
                "pooled_mean_bps": result["mean_bps"],
                "pooled_excess_bps": result["excess_bps"],
                "pooled_t_stat": result["t_stat"],
                "pooled_win_rate": result["win_rate"],
                "uncond_mean_bps": result["uncond_mean"],
                "symbols_tested": cons["tested"],
                "symbols_pos_t2": cons["pos_t2"],
                "symbols_neg_t2": cons["neg_t2"],
                "consistency_pct": round(100 * cons["pos_t2"] / max(cons["tested"], 1), 1),
                "total_obs_across_symbols": cons["total_n"],
            })

    pooled_df = pd.DataFrame(pooled_rows)

    # ── Multiple testing correction (Benjamini-Hochberg) ──
    from scipy.stats import norm
    pooled_df["p_value"] = 2 * (1 - norm.cdf(pooled_df["pooled_t_stat"].abs()))
    pooled_df = pooled_df.sort_values("p_value")
    m = len(pooled_df)
    pooled_df["rank"] = range(1, m + 1)
    pooled_df["bh_threshold"] = 0.05 * pooled_df["rank"] / m
    pooled_df["bh_significant"] = pooled_df["p_value"] <= pooled_df["bh_threshold"]

    # Sort by absolute t-stat for display
    pooled_df = pooled_df.sort_values("pooled_t_stat", key=abs, ascending=False)

    # Save everything
    pooled_df.to_csv(os.path.join(OUT_DIR, "level1_pooled_results.csv"), index=False)

    per_sym_df = pd.DataFrame(per_symbol_rows)
    per_sym_df.to_csv(os.path.join(OUT_DIR, "level1_per_symbol_results.csv"), index=False)

    # ── REPORT ──
    sig = pooled_df[pooled_df["bh_significant"]]

    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("LEVEL 1 SCAN — BROAD HYPOTHESIS TEST")
    report_lines.append("=" * 90)
    report_lines.append(f"Symbols: {len(data)}")
    report_lines.append(f"Conditions tested: {len(cond_names)}")
    report_lines.append(f"Hold periods: {HOLD_PERIODS}")
    report_lines.append(f"Total pooled tests: {len(pooled_df)}")
    report_lines.append(f"BH-significant (α=0.05): {len(sig)}")
    report_lines.append(f"Per-symbol tests: {len(per_symbol_rows):,}")
    report_lines.append("")

    # Unconditional baseline
    report_lines.append("─" * 90)
    report_lines.append("UNCONDITIONAL BASELINE (all symbols pooled)")
    report_lines.append("─" * 90)
    for hp in HOLD_PERIODS:
        fwd = pooled[f"fwd_{hp}d_bps"].dropna()
        report_lines.append(f"  {hp}d: mean={fwd.mean():.1f} bps, median={fwd.median():.1f}, "
                          f"std={fwd.std():.1f}, WR={100*(fwd>0).mean():.1f}%, N={len(fwd):,}")
    report_lines.append("")

    # Top findings (BH significant, sorted by |t|)
    report_lines.append("─" * 90)
    report_lines.append("TOP FINDINGS — BH-significant, sorted by |t-stat|")
    report_lines.append("─" * 90)
    report_lines.append(f"{'Condition':<45} {'Hold':>4} {'N':>8} {'Excess':>7} {'t':>7} {'WR%':>5} {'Cons%':>5} {'#Sym':>4}")
    report_lines.append("─" * 90)

    for _, row in sig.head(80).iterrows():
        report_lines.append(
            f"{row['condition']:<45} {row['hold_days']:>4}d "
            f"{row['pooled_n']:>7,} {row['pooled_excess_bps']:>+7.1f} "
            f"{row['pooled_t_stat']:>7.2f} {row['pooled_win_rate']:>5.1f} "
            f"{row['consistency_pct']:>5.1f} {row['symbols_pos_t2']:>4}"
        )

    report_lines.append("")

    # Key patterns — group by condition theme
    report_lines.append("─" * 90)
    report_lines.append("THEME SUMMARY — What does the data say?")
    report_lines.append("─" * 90)

    themes = {
        "GAP": [c for c in cond_names if "gap" in c],
        "MEAN_REVERSION": [c for c in cond_names if any(x in c for x in ["rsi_below", "rsi_above", "bb_below", "bb_above", "oversold", "overbought", "overext", "dist_sma", "mean_revert", "down_days"])],
        "MOMENTUM": [c for c in cond_names if any(x in c for x in ["roc", "cross", "above_all", "below_all", "up_days", "52w_high"])],
        "VOLUME": [c for c in cond_names if any(x in c for x in ["vol_spike", "volume", "rvol"])],
        "VOLATILITY": [c for c in cond_names if any(x in c for x in ["squeeze", "range_exp", "narrow", "rvol_20d"])],
        "CANDLE": [c for c in cond_names if any(x in c for x in ["big_red", "big_green", "doji", "hammer"])],
    }

    for theme, theme_conds in themes.items():
        theme_sig = sig[sig["condition"].isin(theme_conds)]
        if len(theme_sig) == 0:
            report_lines.append(f"\n  {theme}: NO significant findings")
        else:
            report_lines.append(f"\n  {theme}: {len(theme_sig)} significant findings")
            for _, row in theme_sig.head(10).iterrows():
                direction = "LONG" if row["pooled_excess_bps"] > 0 else "SHORT"
                report_lines.append(
                    f"    {direction} | {row['condition']:<40} {row['hold_days']}d "
                    f"excess={row['pooled_excess_bps']:+.1f}bps t={row['pooled_t_stat']:.2f} "
                    f"N={row['pooled_n']:,} consist={row['consistency_pct']:.0f}%"
                )

    report_lines.append("")

    # NON-significant notable — things people commonly believe that DON'T work
    report_lines.append("─" * 90)
    report_lines.append("NOTABLE NON-FINDINGS — Popular beliefs that FAIL statistical significance")
    report_lines.append("─" * 90)

    popular = ["golden_cross_today", "death_cross_today", "hammer_low_wick", "doji_body_lt_10pct",
               "sma8_cross_above_sma20", "sma8_cross_below_sma20", "macd_cross_above", "macd_cross_below",
               "squeeze_release_today"]

    for cname in popular:
        rows = pooled_df[pooled_df["condition"] == cname]
        if len(rows) == 0:
            continue
        best = rows.loc[rows["pooled_t_stat"].abs().idxmax()]
        status = "SIGNIFICANT" if best["bh_significant"] else "NOT SIGNIFICANT"
        report_lines.append(
            f"  {cname:<40} best t={best['pooled_t_stat']:.2f} at {best['hold_days']}d "
            f"excess={best['pooled_excess_bps']:+.1f}bps [{status}]"
        )

    report_lines.append("")
    report_lines.append("=" * 90)

    report = "\n".join(report_lines)
    print(report)

    report_path = os.path.join(OUT_DIR, "level1_scan_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
    print(f"Pooled results: {os.path.join(OUT_DIR, 'level1_pooled_results.csv')}")
    print(f"Per-symbol results: {os.path.join(OUT_DIR, 'level1_per_symbol_results.csv')}")


if __name__ == "__main__":
    main()
