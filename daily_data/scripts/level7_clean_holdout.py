#!/usr/bin/env python3
"""
LEVEL 7: Proper three-way split.

TRAIN:    2015-2019 — feature selection only
VALIDATE: 2020-2022 — all design choices frozen here
HOLDOUT:  2023-2026 — touched ONCE, reported, never revisited

Strategy is FROZEN before seeing holdout:
  Signal: z_sma50_slope_5d
  Weighting: vol-equalized (inverse 20d realized vol)
  Rebalance: daily (1d hold)
  Universe: all 230 names
  Legs: quintile sort, long Q1, short Q5
  Excess returns: lagged 60d rolling mean
  Costs: 10 bps round-trip

We also test robustness variants WITHOUT looking at holdout first:
  - Different MA periods (20, 50, 100)
  - Different slope lookbacks (3, 5, 10)
  - Different holding periods (1, 3, 5)
  - Tercile sort instead of quintile
  - Top/bottom decile
All variants are designed and evaluated on VALIDATE only, then run on HOLDOUT.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"

TRAIN = range(2015, 2020)
VALIDATE = range(2020, 2023)
HOLDOUT = range(2023, 2027)


def load_all():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    all_dfs = []
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = sym

        c = df["close"]
        for n in [1, 3, 5]:
            raw_bps = 10000 * (c.shift(-n) / c - 1)
            lagged = raw_bps.shift(n)
            rolling_mean = lagged.rolling(60, min_periods=30).mean()
            df[f"fwd_{n}d_bps"] = raw_bps
            df[f"exc_{n}d_bps"] = raw_bps - rolling_mean

        # Build all MA slopes
        for w in [8, 20, 50, 100]:
            sma = c.rolling(w).mean()
            for lb in [3, 5, 10]:
                raw_slope = 100 * (sma / sma.shift(lb) - 1)
                roll_m = raw_slope.rolling(252, min_periods=60).mean()
                roll_s = raw_slope.rolling(252, min_periods=60).std()
                df[f"z_sma{w}_slope_{lb}d"] = (raw_slope - roll_m) / roll_s.replace(0, np.nan)

        # Realized vol for weighting
        log_ret = np.log(c / c.shift(1))
        df["rvol_20d"] = log_ret.rolling(20).std() * np.sqrt(252) * 100

        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def run_strategy(pooled, signal, hold, n_quantiles, period_mask, vol_equalize=True):
    """
    Run the L/S strategy on a specific period.
    Returns per-rebalance results.
    """
    exc_col = f"exc_{hold}d_bps"
    df = pooled[period_mask].copy()
    results = []

    for date, ddf in df.groupby("date"):
        valid = ddf[[signal, exc_col, "symbol", "rvol_20d"]].dropna()
        if len(valid) < 20:
            continue
        try:
            valid["q"] = pd.qcut(valid[signal], n_quantiles,
                                 labels=range(1, n_quantiles + 1), duplicates="drop")
        except ValueError:
            continue
        if valid["q"].nunique() < n_quantiles:
            continue

        q_long = valid[valid["q"] == 1].copy()
        q_short = valid[valid["q"] == n_quantiles].copy()

        if len(q_long) < 3 or len(q_short) < 3:
            continue

        if vol_equalize:
            q_long["inv_vol"] = 1 / q_long["rvol_20d"].clip(lower=5)
            q_short["inv_vol"] = 1 / q_short["rvol_20d"].clip(lower=5)
            q_long["w"] = q_long["inv_vol"] / q_long["inv_vol"].sum()
            q_short["w"] = q_short["inv_vol"] / q_short["inv_vol"].sum()
            long_ret = (q_long[exc_col] * q_long["w"]).sum()
            short_ret = -(q_short[exc_col] * q_short["w"]).sum()
        else:
            long_ret = q_long[exc_col].mean()
            short_ret = -q_short[exc_col].mean()

        results.append({
            "date": date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": long_ret + short_ret,
            "long_symbols": set(q_long["symbol"].tolist()),
            "n": len(q_long),
        })

    rdf = pd.DataFrame(results)
    if rdf.empty:
        return None

    rdf["date"] = pd.to_datetime(rdf["date"])
    rdf = rdf.iloc[::hold].reset_index(drop=True)

    # Turnover
    turnovers = []
    prev = None
    for _, row in rdf.iterrows():
        if prev is not None:
            overlap = len(row["long_symbols"] & prev)
            total = max(len(row["long_symbols"]), 1)
            turnovers.append(1 - overlap / total)
        prev = row["long_symbols"]
    avg_to = np.mean(turnovers) if turnovers else 0

    cost_drag = avg_to * 10  # 10 bps RT
    net = rdf["ls_ret"] - cost_drag
    scale = np.sqrt(252 / hold)

    gross_sr = rdf["ls_ret"].mean() / rdf["ls_ret"].std() * scale if rdf["ls_ret"].std() > 0 else 0
    net_sr = net.mean() / net.std() * scale if net.std() > 0 else 0

    # Year by year
    rdf["year"] = rdf["date"].dt.year
    yearly = {}
    for yr, ydf in rdf.groupby("year"):
        yd = ydf["ls_ret"] - cost_drag
        yearly[yr] = {
            "n": len(yd),
            "net_mean": round(yd.mean(), 1),
            "wr": round(100 * (yd > 0).mean(), 1),
            "positive": yd.mean() > 0,
        }

    # Max drawdown
    cum = net.cumsum()
    peak = cum.cummax()
    dd = peak - cum
    max_dd = dd.max()

    return {
        "n_trades": len(rdf),
        "gross_mean": round(rdf["ls_ret"].mean(), 1),
        "net_mean": round(net.mean(), 1),
        "long_mean": round(rdf["long_ret"].mean(), 1),
        "short_mean": round(rdf["short_ret"].mean(), 1),
        "gross_sr": round(gross_sr, 2),
        "net_sr": round(net_sr, 2),
        "wr": round(100 * (net > 0).mean(), 1),
        "turnover": round(100 * avg_to, 1),
        "cost_drag": round(cost_drag, 1),
        "max_dd": round(max_dd, 0),
        "yearly": yearly,
        "pos_years": sum(1 for v in yearly.values() if v["positive"]),
        "total_years": len(yearly),
    }


def main():
    print("Loading all data...")
    pooled = load_all()
    print(f"Total: {len(pooled):,} rows, {pooled['symbol'].nunique()} symbols\n")

    train_mask = pooled["date"].dt.year.isin(TRAIN)
    val_mask = pooled["date"].dt.year.isin(VALIDATE)
    hold_mask = pooled["date"].dt.year.isin(HOLDOUT)

    report = []
    report.append("=" * 110)
    report.append("LEVEL 7: CLEAN THREE-WAY SPLIT")
    report.append("  TRAIN:    2015-2019 (feature selection)")
    report.append("  VALIDATE: 2020-2022 (design choices, parameter selection)")
    report.append("  HOLDOUT:  2023-2026 (single clean test, never revisited)")
    report.append("=" * 110)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Confirm feature ranking on TRAIN only
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n{'═'*110}")
    report.append("STEP 1: TRAIN (2015-2019) — Feature IC ranking")
    report.append(f"{'═'*110}")

    train_data = pooled[train_mask]
    signals = [f"z_sma{w}_slope_{lb}d" for w in [8, 20, 50, 100] for lb in [3, 5, 10]]

    train_ics = []
    for sig in signals:
        for hold in [1, 5]:
            exc_col = f"exc_{hold}d_bps"
            valid = train_data[[sig, exc_col]].dropna()
            if len(valid) < 500:
                continue
            ic, p = stats.spearmanr(valid[sig], valid[exc_col])
            train_ics.append({"signal": sig, "hold": hold, "ic": round(ic, 5), "p": round(p, 6)})

    train_ic_df = pd.DataFrame(train_ics).sort_values("ic", ascending=False)
    report.append(f"\n  Top signals by IC (on TRAIN excess returns):")
    report.append(f"  {'Signal':<30} {'Hold':>4} {'IC':>9}")
    report.append(f"  {'─'*50}")
    for _, row in train_ic_df.head(15).iterrows():
        report.append(f"  {row['signal']:<30} {row['hold']:>4}d {row['ic']:>+9.5f}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Run all variants on VALIDATE (2020-2022)
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("STEP 2: VALIDATE (2020-2022) — Test all variants, pick the best")
    report.append("All design choices are made here. Holdout is NOT touched.")
    report.append(f"{'═'*110}")

    variants = []
    for sig in signals:
        for hold in [1, 3, 5]:
            for nq in [3, 5, 10]:  # tercile, quintile, decile
                for ve in [True, False]:
                    result = run_strategy(pooled, sig, hold, nq, val_mask, vol_equalize=ve)
                    if result is None:
                        continue
                    variants.append({
                        "signal": sig,
                        "hold": hold,
                        "n_quantiles": nq,
                        "vol_eq": ve,
                        **result,
                    })

    var_df = pd.DataFrame(variants)
    var_df = var_df.sort_values("net_sr", ascending=False)

    report.append(f"\n  {len(var_df)} variants tested on VALIDATE")
    report.append(f"\n  Top 20 by net Sharpe (VALIDATE 2020-2022):")
    report.append(f"  {'Signal':<25} {'Hold':>4} {'Nq':>3} {'VE':>3} {'NetSR':>6} {'Net':>7} {'Long':>7} {'Short':>7} {'WR':>5} {'TO':>5} {'Yrs':>4}")
    report.append(f"  {'─'*95}")

    for _, row in var_df.head(20).iterrows():
        ve_str = "Y" if row["vol_eq"] else "N"
        report.append(
            f"  {row['signal']:<25} {row['hold']:>4}d {row['n_quantiles']:>3} {ve_str:>3} "
            f"{row['net_sr']:>+6.2f} {row['net_mean']:>+7.1f} {row['long_mean']:>+7.1f} "
            f"{row['short_mean']:>+7.1f} {row['wr']:>5.1f} {row['turnover']:>5.1f} {row['pos_years']:>1}/{row['total_years']}"
        )

    # Pick best strategy from VALIDATE
    best = var_df.iloc[0]
    report.append(f"\n  SELECTED STRATEGY (best net Sharpe on VALIDATE):")
    report.append(f"    Signal:     {best['signal']}")
    report.append(f"    Hold:       {best['hold']}d")
    report.append(f"    Quantiles:  {best['n_quantiles']}")
    report.append(f"    Vol-eq:     {best['vol_eq']}")
    report.append(f"    Val Net SR: {best['net_sr']}")

    # Also pick the PRIMARY hypothesis (sma50_slope_5d, quintile, vol-eq, 1d)
    primary_mask = ((var_df["signal"] == "z_sma50_slope_5d") &
                    (var_df["hold"] == 1) &
                    (var_df["n_quantiles"] == 5) &
                    (var_df["vol_eq"] == True))
    primary = var_df[primary_mask].iloc[0] if primary_mask.sum() > 0 else None

    if primary is not None:
        report.append(f"\n  PRIMARY HYPOTHESIS (pre-specified sma50_slope_5d, Q5, VE, 1d):")
        report.append(f"    Val Net SR: {primary['net_sr']}")
        report.append(f"    Val Net:    {primary['net_mean']:+.1f} bps/trade")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: HOLDOUT (2023-2026) — Single clean test
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("STEP 3: HOLDOUT (2023-2026) — CLEAN OUT-OF-SAMPLE TEST")
    report.append("Strategy frozen from VALIDATE. These numbers are reported ONCE.")
    report.append(f"{'═'*110}")

    # Run best strategy on holdout
    best_result = run_strategy(
        pooled, best["signal"], int(best["hold"]), int(best["n_quantiles"]),
        hold_mask, vol_equalize=best["vol_eq"]
    )

    report.append(f"\n  ── BEST (optimized on validate): {best['signal']}, {int(best['hold'])}d, Q{int(best['n_quantiles'])}, VE={best['vol_eq']} ──")
    if best_result:
        report.append(f"    Trades:      {best_result['n_trades']}")
        report.append(f"    Gross L/S:   {best_result['gross_mean']:+.1f} bps/trade")
        report.append(f"    Net L/S:     {best_result['net_mean']:+.1f} bps/trade")
        report.append(f"    Long leg:    {best_result['long_mean']:+.1f} bps")
        report.append(f"    Short leg:   {best_result['short_mean']:+.1f} bps")
        report.append(f"    Gross SR:    {best_result['gross_sr']}")
        report.append(f"    Net SR:      {best_result['net_sr']}")
        report.append(f"    Win rate:    {best_result['wr']}%")
        report.append(f"    Turnover:    {best_result['turnover']}%/rebalance")
        report.append(f"    Max DD:      {best_result['max_dd']:.0f} bps")
        report.append(f"    Years pos:   {best_result['pos_years']}/{best_result['total_years']}")
        report.append(f"\n    Year-by-year:")
        for yr in sorted(best_result["yearly"].keys()):
            yd = best_result["yearly"][yr]
            report.append(f"      {yr}: N={yd['n']:>4}, net={yd['net_mean']:>+8.1f} bps, WR={yd['wr']:.0f}% [{'+'if yd['positive'] else '-'}]")

        # Compare to VALIDATE
        report.append(f"\n    VALIDATE vs HOLDOUT:")
        report.append(f"      Validate Net SR: {best['net_sr']}")
        report.append(f"      Holdout  Net SR: {best_result['net_sr']}")
        degradation = (best_result['net_sr'] - best['net_sr']) / abs(best['net_sr']) * 100
        report.append(f"      Change:          {degradation:+.1f}%")

    # Run primary hypothesis on holdout
    if primary is not None:
        primary_result = run_strategy(
            pooled, "z_sma50_slope_5d", 1, 5, hold_mask, vol_equalize=True
        )

        report.append(f"\n  ── PRIMARY (pre-specified): z_sma50_slope_5d, 1d, Q5, VE ──")
        if primary_result:
            report.append(f"    Trades:      {primary_result['n_trades']}")
            report.append(f"    Net L/S:     {primary_result['net_mean']:+.1f} bps/trade")
            report.append(f"    Long leg:    {primary_result['long_mean']:+.1f} bps")
            report.append(f"    Short leg:   {primary_result['short_mean']:+.1f} bps")
            report.append(f"    Net SR:      {primary_result['net_sr']}")
            report.append(f"    Win rate:    {primary_result['wr']}%")
            report.append(f"    Max DD:      {primary_result['max_dd']:.0f} bps")
            report.append(f"    Years pos:   {primary_result['pos_years']}/{primary_result['total_years']}")
            report.append(f"\n    Year-by-year:")
            for yr in sorted(primary_result["yearly"].keys()):
                yd = primary_result["yearly"][yr]
                report.append(f"      {yr}: N={yd['n']:>4}, net={yd['net_mean']:>+8.1f} bps, WR={yd['wr']:.0f}% [{'+'if yd['positive'] else '-'}]")

            report.append(f"\n    VALIDATE vs HOLDOUT:")
            report.append(f"      Validate Net SR: {primary['net_sr']}")
            report.append(f"      Holdout  Net SR: {primary_result['net_sr']}")
            deg2 = (primary_result['net_sr'] - primary['net_sr']) / abs(primary['net_sr']) * 100
            report.append(f"      Change:          {deg2:+.1f}%")

    # Run a few more frozen variants on holdout to check robustness
    report.append(f"\n  ── ROBUSTNESS: Top 5 validate strategies on HOLDOUT ──")
    report.append(f"  {'Signal':<25} {'Hold':>4} {'Nq':>3} {'VE':>3} {'ValSR':>6} {'HoldSR':>7} {'HoldNet':>8} {'Long':>7} {'Short':>7} {'Yrs':>4}")
    report.append(f"  {'─'*95}")

    for _, row in var_df.head(10).iterrows():
        hr = run_strategy(
            pooled, row["signal"], int(row["hold"]), int(row["n_quantiles"]),
            hold_mask, vol_equalize=row["vol_eq"]
        )
        if hr is None:
            continue
        ve_str = "Y" if row["vol_eq"] else "N"
        report.append(
            f"  {row['signal']:<25} {int(row['hold']):>4}d {int(row['n_quantiles']):>3} {ve_str:>3} "
            f"{row['net_sr']:>+6.2f} {hr['net_sr']:>+7.2f} {hr['net_mean']:>+8.1f} "
            f"{hr['long_mean']:>+7.1f} {hr['short_mean']:>+7.1f} {hr['pos_years']:>1}/{hr['total_years']}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # FINAL HONEST ASSESSMENT
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*110}")
    report.append("FINAL ASSESSMENT")
    report.append(f"{'═'*110}")
    report.append("""
  This is the first time 2023-2026 has been tested with a frozen strategy.

  Compare:
  - VALIDATE Sharpe vs HOLDOUT Sharpe: Did it degrade?
  - Both legs positive in HOLDOUT?
  - Positive in EVERY holdout year?
  - Does the optimized variant (best from validate) beat the pre-specified primary?
    If so, optimization helped. If not, the primary was already good enough and
    optimizing on validate was just fitting noise in that 3-year window.
  """)

    report.append(f"{'═'*110}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "level7_holdout_report.txt"), "w") as f:
        f.write(report_text)
    print(f"\nReport: {os.path.join(OUT_DIR, 'level7_holdout_report.txt')}")


if __name__ == "__main__":
    main()
