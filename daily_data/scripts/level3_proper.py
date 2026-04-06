#!/usr/bin/env python3
"""
LEVEL 3: Do it properly.

Lead quant's requirements:
1. EXCESS returns — subtract each symbol's own unconditional mean
2. NORMALIZE returns by each symbol's own volatility (z-score)
3. TRAIN/TEST split — rank features on 2015-2019, validate on 2020-2026
4. PER-SYMBOL feature ranking before pooling
5. Define actual trades, not just ICs

Features are also standardized per-symbol (z-score) before pooling.
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

HOLD_PERIODS = [1, 3, 5]
TRAIN_YEARS = range(2015, 2020)  # 2015-2019
TEST_YEARS = range(2020, 2027)   # 2020-2026


def get_feature_cols(df):
    """Features that are actual indicators, not price levels or raw MAs."""
    # EXCLUDE raw price-level features (MAs, BBands in dollar terms, KC in dollar terms)
    # These just measure "how big is the stock" or "where is price" — not signals
    exclude = {
        "timestamp", "date", "symbol", "open", "high", "low", "close",
        "volume", "vwap", "transactions", "obv",
        # Raw dollar-denominated — these just measure price level
        "sma_8", "sma_12", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_8", "ema_12", "ema_20", "ema_50",
        "bb_upper", "bb_lower", "bb_mid",
        "kc_upper", "kc_lower", "kc_mid",
        "true_range", "atr_14",
        "vol_sma20", "vol_sma50",
        "macd", "macd_signal",  # dollar-denominated
    }
    exclude.update(c for c in df.columns if c.startswith("fwd_") or c.startswith("exc_"))
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in exclude]


def load_and_prepare():
    """
    Load all data. Compute per-symbol:
    - Excess returns (subtract symbol's rolling mean)
    - Volatility-normalized returns (z-score)
    - Feature z-scores (within symbol)
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    all_dfs = []

    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = sym

        for n in HOLD_PERIODS:
            raw_ret = df["close"].shift(-n) / df["close"] - 1
            raw_bps = raw_ret * 10000

            # Rolling unconditional mean (60-day trailing) — what the stock "normally" does
            rolling_mean = raw_bps.rolling(60, min_periods=30).mean()
            # Rolling vol (60-day trailing)
            rolling_vol = raw_bps.rolling(60, min_periods=30).std()

            # Excess return: actual - expected
            df[f"fwd_{n}d_bps"] = raw_bps
            df[f"exc_{n}d_bps"] = raw_bps - rolling_mean
            # Vol-normalized excess return
            df[f"znorm_{n}d"] = (raw_bps - rolling_mean) / rolling_vol.replace(0, np.nan)

        # Feature z-scores within symbol (rolling 252-day)
        feat_cols = get_feature_cols(df)
        for fc in feat_cols:
            roll_mean = df[fc].rolling(252, min_periods=60).mean()
            roll_std = df[fc].rolling(252, min_periods=60).std()
            df[f"z_{fc}"] = (df[fc] - roll_mean) / roll_std.replace(0, np.nan)

        all_dfs.append(df)

    pooled = pd.concat(all_dfs, ignore_index=True)
    return pooled


def ic_scan(pooled, feature_cols, return_col, period_mask=None):
    """Compute rank IC for each feature against returns, optionally filtered."""
    if period_mask is not None:
        df = pooled[period_mask].copy()
    else:
        df = pooled.copy()

    results = []
    for fc in feature_cols:
        valid = df[[fc, return_col]].dropna()
        if len(valid) < 500:
            continue
        ic, p = stats.spearmanr(valid[fc], valid[return_col])
        results.append({
            "feature": fc,
            "ic": round(ic, 5),
            "abs_ic": round(abs(ic), 5),
            "p_value": round(p, 8),
            "n": len(valid),
        })

    return pd.DataFrame(results).sort_values("abs_ic", ascending=False)


def per_symbol_ic_check(pooled, feature_col, return_col, period_mask=None):
    """Check IC consistency across symbols."""
    if period_mask is not None:
        df = pooled[period_mask].copy()
    else:
        df = pooled.copy()

    ics = []
    for sym, sdf in df.groupby("symbol"):
        valid = sdf[[feature_col, return_col]].dropna()
        if len(valid) < 50:
            continue
        ic, p = stats.spearmanr(valid[feature_col], valid[return_col])
        ics.append({"symbol": sym, "ic": ic})

    if not ics:
        return None

    ic_arr = np.array([x["ic"] for x in ics])
    return {
        "median_ic": round(np.median(ic_arr), 5),
        "mean_ic": round(np.mean(ic_arr), 5),
        "pct_positive": round(100 * np.mean(ic_arr > 0), 1) if np.median(ic_arr) > 0 else round(100 * np.mean(ic_arr < 0), 1),
        "pct_same_sign": round(100 * np.mean(np.sign(ic_arr) == np.sign(np.median(ic_arr))), 1),
        "n_symbols": len(ics),
        "std_ic": round(np.std(ic_arr), 5),
        "t_stat": round(np.mean(ic_arr) / (np.std(ic_arr) / np.sqrt(len(ic_arr))), 2) if np.std(ic_arr) > 0 else 0,
    }


def quintile_excess(pooled, feature_col, return_col, period_mask=None):
    """Quintile spread on excess returns."""
    if period_mask is not None:
        df = pooled[period_mask].copy()
    else:
        df = pooled.copy()

    valid = df[[feature_col, return_col]].dropna()
    if len(valid) < 2000:
        return None

    try:
        valid["q"] = pd.qcut(valid[feature_col], 5, labels=[1,2,3,4,5], duplicates="drop")
    except ValueError:
        return None

    qstats = valid.groupby("q")[return_col].agg(["mean", "std", "count"])
    if len(qstats) < 5:
        return None

    means = [round(qstats.loc[i, "mean"], 2) for i in range(1, 6)]
    q1 = valid[valid["q"] == 1][return_col]
    q5 = valid[valid["q"] == 5][return_col]
    t, p = stats.ttest_ind(q5, q1)

    mono_up = all(means[i] <= means[i+1] for i in range(4))
    mono_down = all(means[i] >= means[i+1] for i in range(4))

    return {
        "q_means": means,
        "spread": round(means[4] - means[0], 2),
        "t_stat": round(t, 3),
        "monotonic": mono_up or mono_down,
        "q1_n": int(qstats.loc[1, "count"]),
    }


def backtest_long_short(pooled, feature_col, return_col, hold, period_mask=None, n_quantiles=5):
    """
    Actual trade backtest: go long Q1 (lowest feature), short Q5 (highest).
    Or vice versa based on IC sign.
    Compute day-by-day P&L with no overlap.
    """
    if period_mask is not None:
        df = pooled[period_mask].copy()
    else:
        df = pooled.copy()

    # Need date-level grouping to avoid overlapping trades
    results_by_date = []

    for date, ddf in df.groupby("date"):
        valid = ddf[[feature_col, return_col, "symbol"]].dropna()
        if len(valid) < 20:  # need enough symbols for quintiles
            continue

        try:
            valid["q"] = pd.qcut(valid[feature_col], n_quantiles, labels=range(1, n_quantiles+1), duplicates="drop")
        except ValueError:
            continue

        if valid["q"].nunique() < n_quantiles:
            continue

        q1_ret = valid[valid["q"] == 1][return_col].mean()
        q5_ret = valid[valid["q"] == n_quantiles][return_col].mean()

        # Long Q1 (lowest feature value), short Q5
        ls_ret = q1_ret - q5_ret

        results_by_date.append({
            "date": date,
            "long_q1": q1_ret,
            "short_q5": -q5_ret,
            "ls_return": ls_ret,
            "n_long": len(valid[valid["q"] == 1]),
            "n_short": len(valid[valid["q"] == n_quantiles]),
        })

    if not results_by_date:
        return None

    rdf = pd.DataFrame(results_by_date)
    rdf["date"] = pd.to_datetime(rdf["date"])
    rdf["year"] = rdf["date"].dt.year

    # Adjust for holding period — only take trades every N days
    rdf = rdf.iloc[::hold]

    daily_ls = rdf["ls_return"]
    sharpe = daily_ls.mean() / daily_ls.std() * np.sqrt(252 / hold) if daily_ls.std() > 0 else 0

    # Year by year
    yearly = {}
    for yr, ydf in rdf.groupby("year"):
        yd = ydf["ls_return"]
        yr_sharpe = yd.mean() / yd.std() * np.sqrt(252 / hold) if yd.std() > 0 and len(yd) > 5 else 0
        yearly[yr] = {
            "n": len(yd),
            "mean_bps": round(yd.mean(), 2),
            "sharpe": round(yr_sharpe, 2),
            "positive": yd.mean() > 0,
        }

    # Cumulative P&L
    rdf["cum_pnl"] = rdf["ls_return"].cumsum()
    max_dd = 0
    peak = 0
    for pnl in rdf["cum_pnl"]:
        peak = max(peak, pnl)
        dd = peak - pnl
        max_dd = max(max_dd, dd)

    return {
        "n_trades": len(rdf),
        "mean_ls_bps": round(daily_ls.mean(), 2),
        "std_ls_bps": round(daily_ls.std(), 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(100 * (daily_ls > 0).mean(), 1),
        "total_pnl_bps": round(daily_ls.sum(), 1),
        "max_drawdown_bps": round(max_dd, 1),
        "yearly": yearly,
        "pos_years": sum(1 for v in yearly.values() if v["positive"]),
        "total_years": len(yearly),
    }


def main():
    print("Loading and preparing data (excess returns, z-scores)...")
    pooled = load_and_prepare()
    print(f"Total rows: {len(pooled):,}, Symbols: {pooled['symbol'].nunique()}\n")

    raw_features = get_feature_cols(pooled)
    z_features = [f"z_{fc}" for fc in raw_features if f"z_{fc}" in pooled.columns]

    print(f"Raw features: {len(raw_features)}")
    print(f"Z-scored features: {len(z_features)}")
    print(f"Features: {raw_features}\n")

    train_mask = pooled["date"].dt.year.isin(TRAIN_YEARS)
    test_mask = pooled["date"].dt.year.isin(TEST_YEARS)

    report = []
    report.append("=" * 110)
    report.append("LEVEL 3: PROPER ANALYSIS — Excess returns, per-symbol z-scores, train/test split")
    report.append("=" * 110)
    report.append(f"Train: 2015-2019, Test: 2020-2026")
    report.append(f"Returns: EXCESS (actual - 60d rolling mean), VOL-NORMALIZED")
    report.append(f"Features: Z-scored within each symbol (252d rolling)")
    report.append(f"Total obs: {len(pooled):,}, Train: {train_mask.sum():,}, Test: {test_mask.sum():,}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: IC SCAN ON EXCESS RETURNS — TRAIN PERIOD ONLY
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 1: FEATURE IC ON EXCESS RETURNS — TRAIN (2015-2019)")
    report.append("Using z-scored features and excess returns. If it's just beta, IC should be ~0.")
    report.append(f"{'═'*110}")

    train_ic_results = {}

    for hold in HOLD_PERIODS:
        exc_col = f"exc_{hold}d_bps"
        znorm_col = f"znorm_{hold}d"

        # IC on excess returns using z-scored features
        ic_exc = ic_scan(pooled, z_features, exc_col, train_mask)
        # IC on vol-normalized excess returns
        ic_znorm = ic_scan(pooled, z_features, znorm_col, train_mask)

        report.append(f"\n  Hold = {hold}d — EXCESS returns, z-scored features (TRAIN 2015-2019)")
        report.append(f"  {'Feature':<35} {'IC(exc)':>9} {'IC(znorm)':>9} {'p(exc)':>10} {'N':>9}")
        report.append(f"  {'─'*80}")

        # Merge the two
        merged = ic_exc.merge(ic_znorm, on="feature", suffixes=("_exc", "_znorm"))
        merged = merged.sort_values("abs_ic_exc", ascending=False)

        train_ic_results[hold] = merged

        for _, row in merged.head(25).iterrows():
            marker = "***" if row["abs_ic_exc"] > 0.015 else "**" if row["abs_ic_exc"] > 0.008 else "*" if row["abs_ic_exc"] > 0.004 else ""
            report.append(
                f"  {row['feature']:<35} {row['ic_exc']:>+9.5f} {row['ic_znorm']:>+9.5f} "
                f"{row['p_value_exc']:>10.6f} {row['n_exc']:>9,} {marker}"
            )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: VALIDATE TOP TRAIN FEATURES ON TEST PERIOD
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 2: TRAIN → TEST VALIDATION")
    report.append("Features ranked by IC in train (2015-2019), then checked in test (2020-2026)")
    report.append(f"{'═'*110}")

    survivors = []

    for hold in HOLD_PERIODS:
        exc_col = f"exc_{hold}d_bps"
        train_top = train_ic_results[hold].head(15)

        report.append(f"\n  Hold = {hold}d:")
        report.append(f"  {'Feature':<35} {'TrainIC':>9} {'TestIC':>9} {'Same?':>6} {'SymConsist':>10} {'SymT':>6}")
        report.append(f"  {'─'*85}")

        for _, row in train_top.iterrows():
            feat = row["feature"]
            train_ic = row["ic_exc"]

            # Test IC
            test_ic_result = ic_scan(pooled, [feat], exc_col, test_mask)
            if test_ic_result.empty:
                continue
            test_ic = test_ic_result.iloc[0]["ic"]

            # Same sign?
            same_sign = np.sign(train_ic) == np.sign(test_ic)

            # Per-symbol consistency in test
            sym_check = per_symbol_ic_check(pooled, feat, exc_col, test_mask)
            if sym_check is None:
                continue

            survived = same_sign and abs(test_ic) > 0.003 and sym_check["pct_same_sign"] > 55

            if survived:
                survivors.append({
                    "feature": feat,
                    "hold": hold,
                    "train_ic": train_ic,
                    "test_ic": test_ic,
                    "sym_consistency": sym_check["pct_same_sign"],
                    "sym_t": sym_check["t_stat"],
                    "n_symbols": sym_check["n_symbols"],
                })

            marker = "✓ SURVIVES" if survived else "✗ DIES"
            report.append(
                f"  {feat:<35} {train_ic:>+9.5f} {test_ic:>+9.5f} {'YES' if same_sign else 'NO':>6} "
                f"{sym_check['pct_same_sign']:>9.1f}% {sym_check['t_stat']:>6.1f}  {marker}"
            )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: QUINTILE SPREADS ON EXCESS RETURNS — TEST PERIOD
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 3: QUINTILE EXCESS RETURNS — TEST (2020-2026) for survivors")
    report.append(f"{'═'*110}")

    for s in survivors:
        feat = s["feature"]
        hold = s["hold"]
        exc_col = f"exc_{hold}d_bps"

        qs = quintile_excess(pooled, feat, exc_col, test_mask)
        if qs is None:
            continue

        mono = "MONO" if qs["monotonic"] else "not mono"
        report.append(
            f"  {feat:<35} {hold}d: Q1={qs['q_means'][0]:>+7.1f} Q2={qs['q_means'][1]:>+7.1f} "
            f"Q3={qs['q_means'][2]:>+7.1f} Q4={qs['q_means'][3]:>+7.1f} Q5={qs['q_means'][4]:>+7.1f} "
            f"spread={qs['spread']:>+7.1f} t={qs['t_stat']:>6.2f} {mono}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: ACTUAL TRADE BACKTESTS — L/S PORTFOLIO
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 4: LONG/SHORT PORTFOLIO BACKTEST — TEST PERIOD (2020-2026)")
    report.append("Daily cross-sectional sort: long bottom quintile, short top quintile")
    report.append("Using EXCESS returns. No overlap (trade every N days).")
    report.append(f"{'═'*110}")

    final_results = []

    for s in survivors:
        feat = s["feature"]
        hold = s["hold"]
        exc_col = f"exc_{hold}d_bps"

        # Determine sort direction from IC sign
        # Negative IC = low feature → high return → long Q1, short Q5
        # Positive IC = high feature → high return → long Q5, short Q1
        bt = backtest_long_short(pooled, feat, exc_col, hold, test_mask)
        if bt is None:
            continue

        # If IC was positive, we need to flip (long Q5, short Q1)
        if s["train_ic"] > 0:
            bt["mean_ls_bps"] = -bt["mean_ls_bps"]
            bt["total_pnl_bps"] = -bt["total_pnl_bps"]
            bt["win_rate"] = 100 - bt["win_rate"]
            for yr in bt["yearly"]:
                bt["yearly"][yr]["mean_bps"] = -bt["yearly"][yr]["mean_bps"]
                bt["yearly"][yr]["positive"] = bt["yearly"][yr]["mean_bps"] > 0
            bt["pos_years"] = sum(1 for v in bt["yearly"].values() if v["positive"])
            # Recalc sharpe
            bt["sharpe"] = -bt["sharpe"]

        report.append(f"\n  ── {feat} ({hold}d hold) ──")
        report.append(f"  Trades: {bt['n_trades']}, Mean L/S: {bt['mean_ls_bps']:+.2f} bps/trade")
        report.append(f"  Sharpe: {bt['sharpe']:.2f}, WR: {bt['win_rate']:.1f}%, MaxDD: {bt['max_drawdown_bps']:.0f} bps")
        report.append(f"  Total PnL: {bt['total_pnl_bps']:+.0f} bps, Years positive: {bt['pos_years']}/{bt['total_years']}")

        report.append(f"  Year-by-year:")
        for yr in sorted(bt["yearly"].keys()):
            ydata = bt["yearly"][yr]
            status = "+" if ydata["positive"] else "-"
            report.append(f"    {yr}: N={ydata['n']:>4}, mean={ydata['mean_bps']:>+8.2f} bps, SR={ydata['sharpe']:>+6.2f} [{status}]")

        final_results.append({
            "feature": feat,
            "hold": hold,
            "train_ic": s["train_ic"],
            "test_ic": s["test_ic"],
            "sym_consist": s["sym_consistency"],
            "sharpe": bt["sharpe"],
            "mean_ls_bps": bt["mean_ls_bps"],
            "win_rate": bt["win_rate"],
            "pos_years": bt["pos_years"],
            "total_years": bt["total_years"],
            "n_trades": bt["n_trades"],
            "max_dd": bt["max_drawdown_bps"],
        })

    # ═══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("FINAL SUMMARY — What survives after removing beta, normalizing, and splitting?")
    report.append(f"{'═'*110}")

    if final_results:
        final_df = pd.DataFrame(final_results).sort_values("sharpe", ascending=False)
        report.append(f"\n{'Feature':<35} {'Hold':>4} {'TrainIC':>8} {'TestIC':>8} {'Consist':>7} {'SR':>6} {'LS bps':>7} {'WR':>5} {'Yrs':>5}")
        report.append("─" * 110)

        for _, row in final_df.iterrows():
            report.append(
                f"{row['feature']:<35} {row['hold']:>4}d {row['train_ic']:>+8.5f} {row['test_ic']:>+8.5f} "
                f"{row['sym_consist']:>6.1f}% {row['sharpe']:>+6.2f} {row['mean_ls_bps']:>+7.2f} "
                f"{row['win_rate']:>5.1f} {row['pos_years']:>2}/{row['total_years']}"
            )

        final_df.to_csv(os.path.join(OUT_DIR, "level3_final_results.csv"), index=False)
    else:
        report.append("\n  NO FEATURES SURVIVED the train/test split on excess returns.")
        report.append("  The lead quant was right — it was all beta.")

    report.append("")
    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "level3_proper_report.txt"), "w") as f:
        f.write(report_text)

    print(f"\nReport: {os.path.join(OUT_DIR, 'level3_proper_report.txt')}")


if __name__ == "__main__":
    main()
