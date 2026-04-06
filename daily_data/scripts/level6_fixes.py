#!/usr/bin/env python3
"""
LEVEL 6: Fix all three problems the lead quant identified.

FIX 1: Use real Fama-French 5 factors + momentum (not proxy factors)
FIX 2: Lag the rolling mean by hold period (no look-ahead)
FIX 3: Vol-equalize the legs (risk parity across long and short)
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"

HOLD_PERIODS = [1, 5]
SIGNAL = "z_sma50_slope_5d"


def load_ff_factors():
    """Load real Fama-French factors."""
    ff5 = pd.read_csv(os.path.join(DATA_DIR, "ff5_daily.csv"))
    mom = pd.read_csv(os.path.join(DATA_DIR, "mom_daily.csv"))
    ff5["date"] = pd.to_datetime(ff5["date"])
    mom["date"] = pd.to_datetime(mom["date"])
    factors = ff5.merge(mom, on="date", how="inner")
    return factors


def load_and_prepare():
    """Load with LAGGED rolling mean (fix #2)."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    all_dfs = []
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = sym

        c = df["close"]
        for n in HOLD_PERIODS:
            raw_bps = 10000 * (c.shift(-n) / c - 1)

            # FIX 2: Lag the rolling mean by hold period
            # rolling_mean is computed on returns ENDING hold_period days ago
            # This means: on day T, the "expected" return uses data up to day T-n only
            raw_bps_shifted = raw_bps.shift(n)  # shift forward = use older data
            rolling_mean = raw_bps_shifted.rolling(60, min_periods=30).mean()
            rolling_vol = raw_bps_shifted.rolling(60, min_periods=30).std()

            df[f"fwd_{n}d_bps"] = raw_bps
            df[f"exc_{n}d_bps"] = raw_bps - rolling_mean
            df[f"fwd_{n}d_vol"] = rolling_vol  # for vol-equalizing

        # SMA50 slope
        sma50 = c.rolling(50).mean()
        df["sma50_slope_5d"] = 100 * (sma50 / sma50.shift(5) - 1)
        roll_mean = df["sma50_slope_5d"].rolling(252, min_periods=60).mean()
        roll_std = df["sma50_slope_5d"].rolling(252, min_periods=60).std()
        df[SIGNAL] = (df["sma50_slope_5d"] - roll_mean) / roll_std.replace(0, np.nan)

        # Realized vol for vol-equalization
        log_ret = np.log(c / c.shift(1))
        df["rvol_20d"] = log_ret.rolling(20).std() * np.sqrt(252) * 100

        df["avg_dollar_vol_20d"] = (c * df["volume"]).rolling(20).mean()

        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def build_portfolio_vol_equalized(pooled, hold):
    """
    FIX 3: Vol-equalize the legs.
    Instead of equal-weighting names, weight inversely by realized vol.
    This makes each name contribute roughly equal risk.
    Report both equal-weight and vol-equalized returns.
    """
    exc_col = f"exc_{hold}d_bps"
    raw_col = f"fwd_{hold}d_bps"  # raw (not excess) for factor regression
    results = []

    for date, ddf in pooled.groupby("date"):
        valid = ddf[[SIGNAL, exc_col, raw_col, "symbol", "close", "rvol_20d",
                      "avg_dollar_vol_20d"]].dropna()
        if len(valid) < 20:
            continue
        try:
            valid["q"] = pd.qcut(valid[SIGNAL], 5, labels=[1,2,3,4,5], duplicates="drop")
        except ValueError:
            continue
        if valid["q"].nunique() < 5:
            continue

        q1 = valid[valid["q"] == 1].copy()  # Long
        q5 = valid[valid["q"] == 5].copy()  # Short

        if len(q1) < 3 or len(q5) < 3:
            continue

        # Equal-weight returns
        ew_long_exc = q1[exc_col].mean()
        ew_short_exc = -q5[exc_col].mean()
        ew_ls_exc = ew_long_exc + ew_short_exc

        ew_long_raw = q1[raw_col].mean()
        ew_short_raw = -q5[raw_col].mean()
        ew_ls_raw = ew_long_raw + ew_short_raw

        # Vol-equalized returns (inverse vol weighting)
        q1["inv_vol"] = 1 / q1["rvol_20d"].clip(lower=5)  # floor at 5% vol
        q5["inv_vol"] = 1 / q5["rvol_20d"].clip(lower=5)
        q1["weight"] = q1["inv_vol"] / q1["inv_vol"].sum()
        q5["weight"] = q5["inv_vol"] / q5["inv_vol"].sum()

        ve_long_exc = (q1[exc_col] * q1["weight"]).sum()
        ve_short_exc = -(q5[exc_col] * q5["weight"]).sum()
        ve_ls_exc = ve_long_exc + ve_short_exc

        ve_long_raw = (q1[raw_col] * q1["weight"]).sum()
        ve_short_raw = -(q5[raw_col] * q5["weight"]).sum()
        ve_ls_raw = ve_long_raw + ve_short_raw

        # Vol diagnostics
        long_avg_vol = q1["rvol_20d"].mean()
        short_avg_vol = q5["rvol_20d"].mean()

        results.append({
            "date": date,
            # Equal weight
            "ew_long_exc": ew_long_exc,
            "ew_short_exc": ew_short_exc,
            "ew_ls_exc": ew_ls_exc,
            "ew_ls_raw": ew_ls_raw,
            "ew_long_raw": ew_long_raw,
            "ew_short_raw": ew_short_raw,
            # Vol equalized
            "ve_long_exc": ve_long_exc,
            "ve_short_exc": ve_short_exc,
            "ve_ls_exc": ve_ls_exc,
            "ve_ls_raw": ve_ls_raw,
            "ve_long_raw": (q1[raw_col] * q1["weight"]).sum(),
            "ve_short_raw": -(q5[raw_col] * q5["weight"]).sum(),
            # Diagnostics
            "long_avg_vol": long_avg_vol,
            "short_avg_vol": short_avg_vol,
            "vol_ratio": long_avg_vol / short_avg_vol if short_avg_vol > 0 else 0,
            "n_long": len(q1),
            "n_short": len(q5),
            "long_symbols": set(q1["symbol"].tolist()),
        })

    rdf = pd.DataFrame(results)
    rdf["date"] = pd.to_datetime(rdf["date"])
    return rdf


def run_factor_regression(returns_series, dates, factors_df, label, hold):
    """Run regression on real FF5 + Mom factors."""
    # Merge by date
    ret_df = pd.DataFrame({"date": dates, "ret": returns_series.values})
    merged = ret_df.merge(factors_df, on="date", how="inner")

    if len(merged) < 50:
        return f"  {label}: insufficient data ({len(merged)} obs)"

    y = merged["ret"].values

    # Convert factor returns from % to bps for comparability
    # Actually keep in native units — our returns are in bps, factors in %
    # So betas will have units bps/% which is fine for interpretation

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    available = [c for c in factor_cols if c in merged.columns and merged[c].notna().sum() > 50]
    X = merged[available].values
    X = np.column_stack([np.ones(len(X)), X])

    betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ betas
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    n_obs = len(y)
    n_params = len(betas)
    mse = ss_res / max(n_obs - n_params, 1)
    try:
        var_betas = mse * np.linalg.inv(X.T @ X).diagonal()
        se_betas = np.sqrt(np.abs(var_betas))
        t_stats = betas / se_betas
    except:
        t_stats = np.zeros_like(betas)
        se_betas = np.ones_like(betas)

    alpha_bps = betas[0]
    alpha_t = t_stats[0]
    alpha_annual = alpha_bps * 252 / hold

    lines = []
    lines.append(f"\n  {label} (N={n_obs}, R²={r_squared:.4f})")
    lines.append(f"  Alpha = {alpha_bps:+.2f} bps/rebalance (t={alpha_t:.2f}), annualized = {alpha_annual:+.0f} bps/yr")
    lines.append(f"  {'Factor':<8} {'Beta':>10} {'t-stat':>8} {'p-value':>10}")
    lines.append(f"  {'─'*42}")

    labels_list = ["Alpha"] + available
    for lbl, beta, t in zip(labels_list, betas, t_stats):
        p = 2 * (1 - stats.t.cdf(abs(t), max(n_obs - n_params, 1)))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(f"  {lbl:<8} {beta:>+10.4f} {t:>8.2f} {p:>10.4f} {sig}")

    lines.append(f"  Explained by factors: {r_squared*100:.1f}%, Pure alpha: {(1-r_squared)*100:.1f}%")

    return "\n".join(lines), r_squared, alpha_bps, alpha_t, alpha_annual


def main():
    print("Loading data with lagged rolling mean...")
    pooled = load_and_prepare()
    test = pooled[pooled["date"].dt.year >= 2020].copy()
    print(f"Test: {test['date'].min().date()} to {test['date'].max().date()}, {test['symbol'].nunique()} symbols\n")

    print("Loading Fama-French factors...")
    factors = load_ff_factors()
    print(f"FF factors: {factors['date'].min().date()} to {factors['date'].max().date()}, {len(factors)} days\n")

    report = []
    report.append("=" * 110)
    report.append("LEVEL 6: ALL THREE FIXES APPLIED")
    report.append("  FIX 1: Real Fama-French 5 factors + momentum")
    report.append("  FIX 2: Lagged rolling mean (no look-ahead in excess return computation)")
    report.append("  FIX 3: Vol-equalized legs (inverse realized vol weighting)")
    report.append("=" * 110)

    for hold in HOLD_PERIODS:
        report.append(f"\n\n{'═'*110}")
        report.append(f"HOLD = {hold}d")
        report.append(f"{'═'*110}")

        # Build portfolio
        rdf = build_portfolio_vol_equalized(test, hold)
        rdf_spaced = rdf.iloc[::hold].reset_index(drop=True)

        # ── VOL ASYMMETRY DIAGNOSTIC ──
        report.append(f"\n  ── VOL ASYMMETRY (Fix #3 diagnostic) ──")
        report.append(f"  Average 20d realized vol:")
        report.append(f"    Long leg (Q1, declining slope): {rdf['long_avg_vol'].mean():.1f}%")
        report.append(f"    Short leg (Q5, rising slope):   {rdf['short_avg_vol'].mean():.1f}%")
        report.append(f"    Ratio (long/short):             {rdf['vol_ratio'].mean():.2f}x")

        # ── COMPARE EW vs VE ──
        report.append(f"\n  ── EQUAL-WEIGHT vs VOL-EQUALIZED ──")
        scale = np.sqrt(252 / hold)

        for method, prefix, label in [("ew", "ew_", "Equal-Weight"), ("ve", "ve_", "Vol-Equalized")]:
            ls_col = f"{prefix}ls_exc"
            long_col = f"{prefix}long_exc"
            short_col = f"{prefix}short_exc"

            ls = rdf_spaced[ls_col]
            lg = rdf_spaced[long_col]
            sh = rdf_spaced[short_col]

            ls_sr = ls.mean() / ls.std() * scale if ls.std() > 0 else 0
            lg_sr = lg.mean() / lg.std() * scale if lg.std() > 0 else 0
            sh_sr = sh.mean() / sh.std() * scale if sh.std() > 0 else 0

            report.append(f"\n  {label}:")
            report.append(f"    L/S: {ls.mean():>+8.1f} bps/trade, Sharpe {ls_sr:>+.2f}, WR {100*(ls>0).mean():.1f}%")
            report.append(f"    Long:  {lg.mean():>+8.1f} bps, Sharpe {lg_sr:>+.2f}")
            report.append(f"    Short: {sh.mean():>+8.1f} bps, Sharpe {sh_sr:>+.2f}")

            # Year by year
            rdf_tmp = rdf_spaced.copy()
            rdf_tmp["year"] = rdf_tmp["date"].dt.year
            pos_yrs = 0
            total_yrs = 0
            yr_lines = []
            for yr, ydf in rdf_tmp.groupby("year"):
                yd = ydf[ls_col]
                yr_sr = yd.mean() / yd.std() * scale if yd.std() > 0 and len(yd) > 5 else 0
                is_pos = yd.mean() > 0
                pos_yrs += int(is_pos)
                total_yrs += 1
                yr_lines.append(f"      {yr}: N={len(yd):>4}, L/S={yd.mean():>+8.1f} bps, "
                              f"Long={ydf[long_col].mean():>+8.1f}, Short={ydf[short_col].mean():>+8.1f}, "
                              f"SR={yr_sr:>+.2f} [{'+'if is_pos else '-'}]")

            report.append(f"    Years positive: {pos_yrs}/{total_yrs}")
            for yl in yr_lines:
                report.append(yl)

        # ── FACTOR REGRESSION WITH REAL FF5+MOM ──
        report.append(f"\n  ── FACTOR REGRESSION (Real FF5 + Momentum) ──")

        # EW L/S on raw returns (not excess — factor model handles that via Mkt-RF)
        for method, prefix, label in [("ew", "ew_", "Equal-Weight"), ("ve", "ve_", "Vol-Equalized")]:
            ls_raw = f"{prefix}ls_raw"

            result = run_factor_regression(
                rdf_spaced[ls_raw], rdf_spaced["date"],
                factors, f"{label} L/S (raw returns)", hold
            )

            if isinstance(result, str):
                report.append(result)
            else:
                text, r2, alpha, alpha_t, alpha_ann = result
                report.append(text)

        # Also regress each leg separately
        report.append(f"\n  ── PER-LEG FACTOR REGRESSION ──")
        for method, prefix, label in [("ew", "ew_", "EW"), ("ve", "ve_", "VE")]:
            for leg, leg_col in [("Long", f"{prefix}long_raw"), ("Short", f"{prefix}short_raw")]:
                result = run_factor_regression(
                    rdf_spaced[leg_col], rdf_spaced["date"],
                    factors, f"{label} {leg} leg", hold
                )
                if isinstance(result, str):
                    report.append(result)
                else:
                    text, _, _, _, _ = result
                    report.append(text)

        # ── DOLLAR P&L AFTER ALL FIXES ──
        report.append(f"\n  ── NET P&L AFTER ALL FIXES (Vol-Equalized, Excess, 10bp RT) ──")

        # Compute turnover
        turnovers = []
        prev = None
        for _, row in rdf_spaced.iterrows():
            if prev is not None:
                overlap = len(row["long_symbols"] & prev)
                total = max(len(row["long_symbols"]), 1)
                turnovers.append(1 - overlap / total)
            prev = row["long_symbols"]
        avg_to = np.mean(turnovers) if turnovers else 0

        cost_drag = avg_to * 10  # 10 bps RT

        ve_ls = rdf_spaced["ve_ls_exc"]
        net_ls = ve_ls - cost_drag

        net_sr = net_ls.mean() / net_ls.std() * scale if net_ls.std() > 0 else 0

        report.append(f"    Turnover: {100*avg_to:.1f}%/rebalance")
        report.append(f"    Cost drag: {cost_drag:.1f} bps/rebalance")
        report.append(f"    Net L/S: {net_ls.mean():+.1f} bps/trade")
        report.append(f"    Net Sharpe: {net_sr:.2f}")
        report.append(f"    Net annual (at $10M/leg): ${net_ls.mean() * 252/hold / 10000 * 10e6 / 1e6:+.2f}M")

    # ── FINAL HONEST ASSESSMENT ──
    report.append(f"\n\n{'═'*110}")
    report.append("HONEST ASSESSMENT — What's left after all three fixes?")
    report.append(f"{'═'*110}")

    report.append("""
  Three things changed:

  FIX 1 (Real FF factors): How much of our 'alpha' was just known factors?
  FIX 2 (Lagged rolling mean): Was the excess return computation leaking future info?
  FIX 3 (Vol equalization): Was the long leg's higher vol inflating the L/S spread?

  Compare the Sharpes, R², and alpha t-stats above to Level 5's numbers:
  - Level 5 proxy-factor R² was 12.8%
  - Level 5 alpha t-stat was 14.12
  - Level 5 Sharpe was 5.58

  If the numbers above are dramatically lower, the lead quant was right and
  the edge was (partially) an artifact.
  """)

    report.append(f"{'═'*110}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "level6_fixes_report.txt"), "w") as f:
        f.write(report_text)
    print(f"\nReport: {os.path.join(OUT_DIR, 'level6_fixes_report.txt')}")


if __name__ == "__main__":
    main()
