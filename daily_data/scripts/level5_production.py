#!/usr/bin/env python3
"""
LEVEL 5: Production reality check.

1. DOLLAR P&L WITH COSTS — realistic transaction cost model
2. FACTOR REGRESSION — FF5 + momentum, what's the residual alpha?
3. CAPACITY — volume analysis of the stocks in each quintile
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

HOLD_PERIODS = [1, 5]
SIGNAL = "z_sma50_slope_5d"


def load_and_prepare():
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
            rolling_mean = raw_bps.rolling(60, min_periods=30).mean()
            df[f"fwd_{n}d_bps"] = raw_bps
            df[f"exc_{n}d_bps"] = raw_bps - rolling_mean

        # SMA50 slope
        sma50 = c.rolling(50).mean()
        df["sma50_slope_5d"] = 100 * (sma50 / sma50.shift(5) - 1)
        roll_mean = df["sma50_slope_5d"].rolling(252, min_periods=60).mean()
        roll_std = df["sma50_slope_5d"].rolling(252, min_periods=60).std()
        df[SIGNAL] = (df["sma50_slope_5d"] - roll_mean) / roll_std.replace(0, np.nan)

        # Keep volume and price for capacity analysis
        df["avg_dollar_vol_20d"] = (c * df["volume"]).rolling(20).mean()

        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def build_portfolio(pooled, hold):
    """
    Build daily L/S portfolio with full tracking of:
    - Names in each leg
    - Entry/exit prices implied
    - Turnover
    - Dollar volume of each leg
    """
    exc_col = f"exc_{hold}d_bps"
    results = []

    for date, ddf in pooled.groupby("date"):
        valid = ddf[[SIGNAL, exc_col, "symbol", "close", "volume", "avg_dollar_vol_20d"]].dropna()
        if len(valid) < 20:
            continue
        try:
            valid["q"] = pd.qcut(valid[SIGNAL], 5, labels=[1,2,3,4,5], duplicates="drop")
        except ValueError:
            continue
        if valid["q"].nunique() < 5:
            continue

        q1 = valid[valid["q"] == 1]  # Most declining slope → long
        q5 = valid[valid["q"] == 5]  # Most rising slope → short

        results.append({
            "date": date,
            "long_ret": q1[exc_col].mean(),
            "short_ret": -q5[exc_col].mean(),
            "ls_ret": q1[exc_col].mean() - q5[exc_col].mean(),
            "n_long": len(q1),
            "n_short": len(q5),
            "long_symbols": set(q1["symbol"].tolist()),
            "short_symbols": set(q5["symbol"].tolist()),
            # Capacity metrics
            "long_min_adv": q1["avg_dollar_vol_20d"].min(),
            "long_median_adv": q1["avg_dollar_vol_20d"].median(),
            "long_mean_adv": q1["avg_dollar_vol_20d"].mean(),
            "short_min_adv": q5["avg_dollar_vol_20d"].min(),
            "short_median_adv": q5["avg_dollar_vol_20d"].median(),
            "short_mean_adv": q5["avg_dollar_vol_20d"].mean(),
            # Price range
            "long_median_price": q1["close"].median(),
            "short_median_price": q5["close"].median(),
        })

    rdf = pd.DataFrame(results)
    rdf["date"] = pd.to_datetime(rdf["date"])
    return rdf


def compute_costs_and_pnl(rdf, hold, capital_per_leg, cost_bps):
    """
    Compute dollar P&L after transaction costs.

    capital_per_leg: $ allocated to each side (e.g. $5M long, $5M short)
    cost_bps: round-trip cost per trade in bps
    """
    rdf = rdf.copy()

    # Space trades by hold period
    rdf = rdf.iloc[::hold].reset_index(drop=True)

    # Turnover
    turnovers_long = []
    turnovers_short = []
    prev_long = None
    prev_short = None

    for i, row in rdf.iterrows():
        if prev_long is not None:
            overlap = len(row["long_symbols"] & prev_long)
            total = max(len(row["long_symbols"]), 1)
            turnovers_long.append(1 - overlap / total)

            overlap_s = len(row["short_symbols"] & prev_short)
            total_s = max(len(row["short_symbols"]), 1)
            turnovers_short.append(1 - overlap_s / total_s)

        prev_long = row["long_symbols"]
        prev_short = row["short_symbols"]

    avg_turnover = np.mean(turnovers_long) if turnovers_long else 0

    # Per-rebalance cost = turnover × cost_bps × 2 (both legs)
    cost_per_rebal_bps = avg_turnover * cost_bps * 2  # both sides turn over

    # Net return per rebalance
    rdf["gross_ls_bps"] = rdf["ls_ret"]
    rdf["cost_bps"] = cost_per_rebal_bps
    rdf["net_ls_bps"] = rdf["gross_ls_bps"] - cost_per_rebal_bps

    # Dollar P&L
    rdf["gross_dollar"] = rdf["gross_ls_bps"] / 10000 * capital_per_leg
    rdf["cost_dollar"] = cost_per_rebal_bps / 10000 * capital_per_leg
    rdf["net_dollar"] = rdf["net_ls_bps"] / 10000 * capital_per_leg

    return rdf, avg_turnover


def factor_regression(rdf, hold):
    """
    Since we don't have FF5 factor data downloaded, we'll build our own
    proxy factors from our universe:
    - MKT: equal-weighted universe return (market)
    - SMB: small vs large (by median price as proxy)
    - HML: high vs low book (use dist_sma200 as value proxy)
    - MOM: past 20d return winners vs losers
    - STR: past 5d return (short-term reversal)

    Then regress our L/S returns on these.
    """
    # We need to build factor returns from the same universe
    # This is a simplified version — real FF would use proper size/value sorts
    return None  # Will build from scratch below


def build_factor_returns(pooled, hold):
    """Build proxy factor returns from our universe."""
    exc_col = f"exc_{hold}d_bps"
    factor_data = []

    for date, ddf in pooled.groupby("date"):
        valid = ddf[["symbol", exc_col, "close", "volume", "avg_dollar_vol_20d", SIGNAL]].dropna()
        if len(valid) < 20:
            continue

        # Need past returns for momentum/reversal factors
        # We'll use the signal itself and price as proxies

        mkt_ret = valid[exc_col].mean()

        # SIZE: split at median market cap proxy (price × volume)
        valid["mcap_proxy"] = valid["close"] * valid["avg_dollar_vol_20d"]
        med_mcap = valid["mcap_proxy"].median()
        small = valid[valid["mcap_proxy"] < med_mcap][exc_col].mean()
        big = valid[valid["mcap_proxy"] >= med_mcap][exc_col].mean()
        smb = small - big

        # VALUE proxy: use close relative to its level (low price = "value")
        med_price = valid["close"].median()
        low_p = valid[valid["close"] < med_price][exc_col].mean()
        high_p = valid[valid["close"] >= med_price][exc_col].mean()
        hml = low_p - high_p  # "value" minus "growth"

        factor_data.append({
            "date": date,
            "MKT": mkt_ret,
            "SMB": smb,
            "HML": hml,
        })

    return pd.DataFrame(factor_data)


def build_reversal_factor(pooled, hold):
    """Build short-term reversal factor returns."""
    exc_col = f"exc_{hold}d_bps"
    results = []

    for date, ddf in pooled.groupby("date"):
        # Need past 5d return — use daily_return_pct as proxy for recent performance
        valid = ddf[["symbol", exc_col, "daily_return_pct"]].dropna()
        if len(valid) < 20:
            continue

        # Sort by recent return
        try:
            valid["q"] = pd.qcut(valid["daily_return_pct"], 5, labels=[1,2,3,4,5], duplicates="drop")
        except ValueError:
            continue
        if valid["q"].nunique() < 5:
            continue

        # Reversal = long losers, short winners
        q1 = valid[valid["q"] == 1][exc_col].mean()
        q5 = valid[valid["q"] == 5][exc_col].mean()

        results.append({"date": date, "STR": q1 - q5})

    return pd.DataFrame(results)


def main():
    print("Loading data...")
    pooled = load_and_prepare()
    test = pooled[pooled["date"].dt.year >= 2020].copy()
    print(f"Test period: {test['date'].min().date()} to {test['date'].max().date()}")
    print(f"Symbols: {test['symbol'].nunique()}, Rows: {len(test):,}\n")

    report = []
    report.append("=" * 110)
    report.append("LEVEL 5: PRODUCTION REALITY — Dollar P&L, Factor Regression, Capacity")
    report.append("=" * 110)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: DOLLAR P&L WITH COSTS
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n{'═'*110}")
    report.append("SECTION 1: DOLLAR P&L WITH TRANSACTION COSTS")
    report.append(f"{'═'*110}")

    capital_scenarios = [5_000_000, 10_000_000, 25_000_000]  # per leg
    cost_scenarios = [5, 10, 15, 20, 30]  # round-trip bps

    for hold in HOLD_PERIODS:
        rdf = build_portfolio(test, hold)
        report.append(f"\n  Hold = {hold}d, Signal = {SIGNAL}")
        report.append(f"  Rebalance dates: {len(rdf.iloc[::hold])}")

        rdf_spaced = rdf.iloc[::hold].reset_index(drop=True)

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

        report.append(f"  Average turnover per rebalance: {100*avg_to:.1f}%")
        report.append(f"  Names turned per rebalance: {avg_to * rdf_spaced['n_long'].mean():.1f}")
        report.append(f"  Gross L/S per rebalance: {rdf_spaced['ls_ret'].mean():+.1f} bps")
        report.append(f"  Gross annual L/S: {rdf_spaced['ls_ret'].mean() * 252/hold:+.0f} bps")

        report.append(f"\n  {'Capital/leg':>12} | ", )
        header = f"  {'Capital/leg':>12} |"
        for cost in cost_scenarios:
            header += f" {cost}bp RT |"
        report.append(header)
        report.append(f"  {'─'*75}")

        for cap in capital_scenarios:
            line = f"  ${cap/1e6:.0f}M/leg      |"
            for cost in cost_scenarios:
                cost_per_rebal = avg_to * cost  # bps eaten by costs per rebalance
                net_per_rebal = rdf_spaced["ls_ret"].mean() - cost_per_rebal
                annual_net_bps = net_per_rebal * 252 / hold
                annual_net_dollar = annual_net_bps / 10000 * cap
                annual_gross_dollar = rdf_spaced["ls_ret"].mean() * 252 / hold / 10000 * cap

                # Sharpe after costs
                net_series = rdf_spaced["ls_ret"] - cost_per_rebal
                net_sr = net_series.mean() / net_series.std() * np.sqrt(252/hold) if net_series.std() > 0 else 0

                line += f" ${annual_net_dollar/1e6:+.2f}M |"
            report.append(line)

        # Detailed cost analysis for best scenario
        report.append(f"\n  Detailed (10 bps RT, $10M/leg):")
        cost_per_rebal = avg_to * 10
        net_series = rdf_spaced["ls_ret"] - cost_per_rebal
        scale = np.sqrt(252/hold)
        gross_sr = rdf_spaced["ls_ret"].mean() / rdf_spaced["ls_ret"].std() * scale
        net_sr = net_series.mean() / net_series.std() * scale

        report.append(f"    Gross Sharpe: {gross_sr:.2f}")
        report.append(f"    Net Sharpe:   {net_sr:.2f}")
        report.append(f"    Cost drag:    {cost_per_rebal:.1f} bps/rebalance ({100*cost_per_rebal/rdf_spaced['ls_ret'].mean():.0f}% of gross)")
        report.append(f"    Gross annual: ${rdf_spaced['ls_ret'].mean() * 252/hold/10000 * 10e6/1e6:.2f}M")
        report.append(f"    Net annual:   ${net_series.mean() * 252/hold/10000 * 10e6/1e6:.2f}M")
        report.append(f"    Win rate:     {100*(net_series>0).mean():.1f}%")

        # Year by year net
        rdf_spaced_copy = rdf_spaced.copy()
        rdf_spaced_copy["year"] = rdf_spaced_copy["date"].dt.year
        rdf_spaced_copy["net"] = rdf_spaced_copy["ls_ret"] - cost_per_rebal
        report.append(f"\n    Year-by-year (10 bps RT, $10M/leg):")
        for yr, ydf in rdf_spaced_copy.groupby("year"):
            ann_dollar = ydf["net"].mean() * 252/hold / 10000 * 10e6
            report.append(f"      {yr}: N={len(ydf):>4}, net={ydf['net'].mean():>+6.1f} bps/trade, ~${ann_dollar/1e6:+.2f}M/yr")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: FACTOR REGRESSION
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 2: FACTOR REGRESSION — Is there residual alpha after known factors?")
    report.append("Factors: MKT (equal-weight universe), SMB (size), HML (value proxy), STR (reversal)")
    report.append(f"{'═'*110}")

    for hold in HOLD_PERIODS:
        rdf = build_portfolio(test, hold)
        rdf_spaced = rdf.iloc[::hold].reset_index(drop=True)

        # Build factors
        factor_df = build_factor_returns(test, hold)
        reversal_df = build_reversal_factor(test, hold)

        # Merge
        merged = rdf_spaced[["date", "ls_ret"]].merge(factor_df, on="date", how="inner")
        if not reversal_df.empty:
            merged = merged.merge(reversal_df, on="date", how="inner")

        if len(merged) < 50:
            report.append(f"\n  Hold = {hold}d: insufficient data for regression")
            continue

        # Run regression
        y = merged["ls_ret"].values
        factor_cols = [c for c in ["MKT", "SMB", "HML", "STR"] if c in merged.columns]
        X = merged[factor_cols].values
        X = np.column_stack([np.ones(len(X)), X])  # add intercept

        try:
            betas, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except:
            report.append(f"\n  Hold = {hold}d: regression failed")
            continue

        y_pred = X @ betas
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r_squared = 1 - ss_res / ss_tot

        # Standard errors
        n_obs = len(y)
        n_params = len(betas)
        mse = ss_res / (n_obs - n_params)
        var_betas = mse * np.linalg.inv(X.T @ X).diagonal()
        se_betas = np.sqrt(var_betas)
        t_stats = betas / se_betas

        # Alpha (intercept) in bps per rebalance
        alpha_bps = betas[0]
        alpha_t = t_stats[0]
        alpha_annual = alpha_bps * 252 / hold

        report.append(f"\n  Hold = {hold}d (N={n_obs}):")
        report.append(f"  R² = {r_squared:.4f}")
        report.append(f"  Alpha (intercept) = {alpha_bps:+.2f} bps/rebalance (t={alpha_t:.2f})")
        report.append(f"  Alpha annualized  = {alpha_annual:+.0f} bps/year")
        report.append(f"\n  {'Factor':<8} {'Beta':>8} {'t-stat':>8} {'p-value':>10}")
        report.append(f"  {'─'*40}")

        labels = ["Alpha"] + factor_cols
        for i, (label, beta, t, se) in enumerate(zip(labels, betas, t_stats, se_betas)):
            p = 2 * (1 - stats.t.cdf(abs(t), n_obs - n_params))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            report.append(f"  {label:<8} {beta:>+8.3f} {t:>8.2f} {p:>10.4f} {sig}")

        # Residual analysis
        resid = y - y_pred
        resid_sr = resid.mean() / resid.std() * np.sqrt(252/hold) if resid.std() > 0 else 0
        report.append(f"\n  Residual Sharpe (unexplained by factors): {resid_sr:.2f}")
        report.append(f"  Fraction of signal explained by factors: {r_squared*100:.1f}%")
        report.append(f"  Fraction that is PURE ALPHA: {(1-r_squared)*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: CAPACITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("SECTION 3: CAPACITY — Can you actually trade this?")
    report.append("Rule of thumb: don't exceed 1% of ADV per name per day")
    report.append(f"{'═'*110}")

    for hold in HOLD_PERIODS:
        rdf = build_portfolio(test, hold)
        rdf_recent = rdf[rdf["date"].dt.year >= 2024]  # use recent data for volume

        if len(rdf_recent) < 20:
            continue

        report.append(f"\n  Hold = {hold}d (2024-2026 data):")

        # Average dollar volume stats
        long_min_adv = rdf_recent["long_min_adv"].mean()
        long_med_adv = rdf_recent["long_median_adv"].mean()
        long_mean_adv = rdf_recent["long_mean_adv"].mean()
        short_min_adv = rdf_recent["short_min_adv"].mean()
        short_med_adv = rdf_recent["short_median_adv"].mean()
        short_mean_adv = rdf_recent["short_mean_adv"].mean()

        report.append(f"  Long leg (Q1 — declining slope):")
        report.append(f"    Avg names per leg: {rdf_recent['n_long'].mean():.0f}")
        report.append(f"    Min ADV in leg:    ${long_min_adv/1e6:.1f}M")
        report.append(f"    Median ADV in leg: ${long_med_adv/1e6:.1f}M")
        report.append(f"    Mean ADV in leg:   ${long_mean_adv/1e6:.1f}M")

        report.append(f"  Short leg (Q5 — rising slope):")
        report.append(f"    Avg names per leg: {rdf_recent['n_short'].mean():.0f}")
        report.append(f"    Min ADV in leg:    ${short_min_adv/1e6:.1f}M")
        report.append(f"    Median ADV in leg: ${short_med_adv/1e6:.1f}M")
        report.append(f"    Mean ADV in leg:   ${short_mean_adv/1e6:.1f}M")

        # Max capacity at 1% ADV constraint
        # Position size per name = capital / n_names
        # Must be < 1% of that name's ADV
        # So: capital / n_names < 0.01 * min_ADV
        # => capital < n_names * 0.01 * min_ADV
        n_names = rdf_recent["n_long"].mean()
        max_cap_long = n_names * 0.01 * long_min_adv
        max_cap_short = n_names * 0.01 * short_min_adv
        max_cap = min(max_cap_long, max_cap_short)

        # At 0.5% ADV (more conservative)
        max_cap_conservative = min(
            n_names * 0.005 * long_min_adv,
            n_names * 0.005 * short_min_adv
        )

        # If we use median instead of min (exclude worst name)
        max_cap_median = min(
            n_names * 0.01 * long_med_adv,
            n_names * 0.01 * short_med_adv
        )

        report.append(f"\n  CAPACITY ESTIMATES:")
        report.append(f"    At 1% ADV (min name):         ${max_cap/1e6:.0f}M per leg (${2*max_cap/1e6:.0f}M total)")
        report.append(f"    At 0.5% ADV (conservative):   ${max_cap_conservative/1e6:.0f}M per leg")
        report.append(f"    At 1% ADV (median name):      ${max_cap_median/1e6:.0f}M per leg (${2*max_cap_median/1e6:.0f}M total)")

        # What about excluding the smallest names?
        report.append(f"\n  If we filter to names with ADV > $50M:")
        rdf_filtered = rdf_recent.copy()
        # Count how many names would survive
        # We need to look at actual quintile composition
        # For now, estimate based on median
        pct_above = (rdf_recent["long_min_adv"] > 50e6).mean()
        report.append(f"    {100*pct_above:.0f}% of days have all long-leg names above $50M ADV")

        # Expected P&L at various capital levels
        report.append(f"\n  EXPECTED ANNUAL P&L (10 bps RT cost, hold={hold}d):")
        rdf_spaced = rdf.iloc[::hold].reset_index(drop=True)

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

        cost_drag = avg_to * 10  # bps per rebalance
        net_per_rebal = rdf_spaced["ls_ret"].mean() - cost_drag

        for cap in [1e6, 5e6, 10e6, 25e6, 50e6]:
            annual_net = net_per_rebal * 252 / hold / 10000 * cap
            feasible = "✓" if cap <= max_cap else "⚠ OVER CAPACITY" if cap <= max_cap_median else "✗ WAY OVER"
            report.append(f"    ${cap/1e6:.0f}M/leg: ${annual_net/1e6:+.2f}M/yr  {feasible}")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("FINAL VERDICT")
    report.append(f"{'═'*110}")

    report.append("""
  SIGNAL: z_sma50_slope_5d (z-scored 5-day change in 50-day SMA, within each symbol)
  TRADE: Cross-sectional L/S quintile sort across 230 mid/large-cap names
  LONG: Bottom quintile (most declining SMA50 slope)
  SHORT: Top quintile (most rising SMA50 slope)
  REBALANCE: Daily (1d hold) or weekly (5d hold)
  PERIOD: OOS 2020-2026, trained on 2015-2019
    """)

    report.append(f"{'═'*110}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "level5_production_report.txt"), "w") as f:
        f.write(report_text)
    print(f"\nReport: {os.path.join(OUT_DIR, 'level5_production_report.txt')}")


if __name__ == "__main__":
    main()
