#!/usr/bin/env python3
"""
DATA-DRIVEN EDGE DISCOVERY

No pre-selected indicators. No textbook conditions. Let the data surface
non-obvious structure.

Approach:
1. Feature importance: which of the 55 numeric features actually predict
   forward returns? Rank by information coefficient, not by whether I
   "believe" in them.
2. Interaction discovery: which PAIRS of features create non-linear edge
   that individual features don't?
3. Regime detection: are there natural clusters in the feature space where
   forward returns are significantly different?
4. Residual analysis: after removing the unconditional drift (equity premium),
   what predicts the EXCESS return?
5. Asymmetry: which features predict differently for UP moves vs DOWN moves?

No RSI, no "oversold", no textbook. Just numbers.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"

HOLD_PERIODS = [1, 3, 5]


def load_pooled():
    """Load all enriched, add forward returns, stack."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    dfs = []
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["symbol"] = sym
        df["date"] = pd.to_datetime(df["date"])
        for n in HOLD_PERIODS:
            df[f"fwd_{n}d_bps"] = 10000 * (df["close"].shift(-n) / df["close"] - 1)
        dfs.append(df)
    pooled = pd.concat(dfs, ignore_index=True)
    return pooled


def get_feature_cols(df):
    """Get all numeric feature columns (exclude meta, targets, raw price)."""
    exclude = {"timestamp", "date", "symbol", "open", "high", "low", "close",
               "volume", "vwap", "transactions", "obv"}
    exclude.update(c for c in df.columns if c.startswith("fwd_"))
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in exclude]


def information_coefficient(feature, forward_ret):
    """Rank IC: Spearman correlation between feature rank and forward return rank."""
    valid = pd.DataFrame({"f": feature, "r": forward_ret}).dropna()
    if len(valid) < 100:
        return None, None, None
    ic, p = stats.spearmanr(valid["f"], valid["r"])
    return round(ic, 5), round(p, 6), len(valid)


def quintile_spread(feature, forward_ret):
    """
    Sort into quintiles by feature value.
    Return spread (Q5 mean - Q1 mean), and per-quintile stats.
    This makes NO assumption about linearity.
    """
    valid = pd.DataFrame({"f": feature, "r": forward_ret}).dropna()
    if len(valid) < 500:
        return None

    try:
        valid["q"] = pd.qcut(valid["f"], 5, labels=[1,2,3,4,5], duplicates="drop")
    except ValueError:
        return None

    qstats = valid.groupby("q")["r"].agg(["mean", "std", "count"])
    if len(qstats) < 5:
        return None

    q1_mean = qstats.loc[1, "mean"]
    q5_mean = qstats.loc[5, "mean"]
    spread = q5_mean - q1_mean

    # t-stat on the spread
    q1 = valid[valid["q"] == 1]["r"]
    q5 = valid[valid["q"] == 5]["r"]
    t, p = stats.ttest_ind(q5, q1)

    # Monotonicity: do quintile means increase monotonically?
    means = qstats["mean"].values
    mono_up = all(means[i] <= means[i+1] for i in range(len(means)-1))
    mono_down = all(means[i] >= means[i+1] for i in range(len(means)-1))
    monotonic = mono_up or mono_down

    return {
        "spread_bps": round(spread, 2),
        "t_stat": round(t, 3),
        "p_value": round(p, 6),
        "monotonic": monotonic,
        "q1_mean": round(q1_mean, 2),
        "q2_mean": round(qstats.loc[2, "mean"], 2),
        "q3_mean": round(qstats.loc[3, "mean"], 2),
        "q4_mean": round(qstats.loc[4, "mean"], 2),
        "q5_mean": round(q5_mean, 2),
        "q1_n": int(qstats.loc[1, "count"]),
        "q5_n": int(qstats.loc[5, "count"]),
    }


def per_symbol_ic(data_dict, feature_col, hold):
    """Compute IC per symbol, return distribution of ICs."""
    ics = []
    for sym, df in data_dict.items():
        if feature_col not in df.columns:
            continue
        fwd_col = f"fwd_{hold}d_bps"
        valid = df[[feature_col, fwd_col]].dropna()
        if len(valid) < 100:
            continue
        ic, p, n = information_coefficient(valid[feature_col], valid[fwd_col])
        if ic is not None:
            ics.append(ic)
    return ics


def interaction_scan(pooled, features, fwd_col, top_n=20):
    """
    For top features, check if PAIRS produce non-linear edge.
    Split each feature at median, create 4 quadrants, check if
    any quadrant has significantly different returns from the others.
    """
    results = []
    n_features = len(features)

    for i in range(min(n_features, top_n)):
        for j in range(i+1, min(n_features, top_n)):
            f1, f2 = features[i], features[j]
            valid = pooled[[f1, f2, fwd_col]].dropna()
            if len(valid) < 1000:
                continue

            m1 = valid[f1].median()
            m2 = valid[f2].median()

            # 4 quadrants
            q_ll = valid[(valid[f1] < m1) & (valid[f2] < m2)][fwd_col]
            q_lh = valid[(valid[f1] < m1) & (valid[f2] >= m2)][fwd_col]
            q_hl = valid[(valid[f1] >= m1) & (valid[f2] < m2)][fwd_col]
            q_hh = valid[(valid[f1] >= m1) & (valid[f2] >= m2)][fwd_col]

            if min(len(q_ll), len(q_lh), len(q_hl), len(q_hh)) < 100:
                continue

            means = [q_ll.mean(), q_lh.mean(), q_hl.mean(), q_hh.mean()]
            best_q = np.argmax(np.abs(np.array(means) - np.mean(means)))
            worst_q = np.argmin(np.abs(np.array(means) - np.mean(means)))

            # Is the spread between best and worst quadrant significant?
            quads = [q_ll, q_lh, q_hl, q_hh]
            quad_names = ["low_low", "low_high", "high_low", "high_high"]

            best_data = quads[np.argmax(means)]
            worst_data = quads[np.argmin(means)]
            spread = max(means) - min(means)

            t, p = stats.ttest_ind(best_data, worst_data)

            # Check for non-linearity: is interaction > sum of individual effects?
            # Individual effects
            low_f1 = valid[valid[f1] < m1][fwd_col].mean()
            high_f1 = valid[valid[f1] >= m1][fwd_col].mean()
            low_f2 = valid[valid[f2] < m2][fwd_col].mean()
            high_f2 = valid[valid[f2] >= m2][fwd_col].mean()

            individual_spread = abs(high_f1 - low_f1) + abs(high_f2 - low_f2)
            interaction_ratio = spread / max(individual_spread, 0.01)

            results.append({
                "f1": f1,
                "f2": f2,
                "spread_bps": round(spread, 2),
                "t_stat": round(t, 3),
                "p_value": round(p, 6),
                "interaction_ratio": round(interaction_ratio, 2),
                "best_quadrant": quad_names[np.argmax(means)],
                "best_mean": round(max(means), 2),
                "worst_quadrant": quad_names[np.argmin(means)],
                "worst_mean": round(min(means), 2),
                "ll_mean": round(means[0], 2),
                "lh_mean": round(means[1], 2),
                "hl_mean": round(means[2], 2),
                "hh_mean": round(means[3], 2),
            })

    return pd.DataFrame(results).sort_values("spread_bps", ascending=False)


def tail_analysis(pooled, features, fwd_col, percentile=5):
    """
    For each feature, look at the extreme tails (top/bottom 5%).
    Sometimes the edge is ONLY in the tails, not linear.
    """
    results = []
    for feat in features:
        valid = pooled[[feat, fwd_col]].dropna()
        if len(valid) < 1000:
            continue

        low_thresh = valid[feat].quantile(percentile / 100)
        high_thresh = valid[feat].quantile(1 - percentile / 100)

        tail_low = valid[valid[feat] <= low_thresh][fwd_col]
        tail_high = valid[valid[feat] >= high_thresh][fwd_col]
        middle = valid[(valid[feat] > low_thresh) & (valid[feat] < high_thresh)][fwd_col]

        if min(len(tail_low), len(tail_high), len(middle)) < 50:
            continue

        # Does the low tail differ from middle?
        t_low, p_low = stats.ttest_ind(tail_low, middle)
        t_high, p_high = stats.ttest_ind(tail_high, middle)

        results.append({
            "feature": feat,
            "low_tail_mean": round(tail_low.mean(), 2),
            "middle_mean": round(middle.mean(), 2),
            "high_tail_mean": round(tail_high.mean(), 2),
            "low_tail_excess": round(tail_low.mean() - middle.mean(), 2),
            "high_tail_excess": round(tail_high.mean() - middle.mean(), 2),
            "low_tail_t": round(t_low, 3),
            "high_tail_t": round(t_high, 3),
            "low_tail_n": len(tail_low),
            "high_tail_n": len(tail_high),
            "tail_asymmetry": round(abs(tail_low.mean() - middle.mean()) -
                                    abs(tail_high.mean() - middle.mean()), 2),
        })

    return pd.DataFrame(results)


def year_stability_of_ic(data_dict, feature_col, hold):
    """Check if IC is stable across years or time-varying."""
    by_year = defaultdict(list)
    for sym, df in data_dict.items():
        if feature_col not in df.columns:
            continue
        fwd_col = f"fwd_{hold}d_bps"
        for yr in df["date"].dt.year.unique():
            ydf = df[df["date"].dt.year == yr]
            valid = ydf[[feature_col, fwd_col]].dropna()
            if len(valid) < 30:
                continue
            ic, p, n = information_coefficient(valid[feature_col], valid[fwd_col])
            if ic is not None:
                by_year[yr].append(ic)

    result = {}
    for yr in sorted(by_year.keys()):
        ics = by_year[yr]
        result[yr] = {
            "median_ic": round(np.median(ics), 4),
            "mean_ic": round(np.mean(ics), 4),
            "pct_positive": round(100 * np.mean(np.array(ics) > 0), 1),
            "n_symbols": len(ics),
        }
    return result


def main():
    print("Loading all enriched data...")
    pooled = load_pooled()
    print(f"Loaded {pooled['symbol'].nunique()} symbols, {len(pooled):,} total rows\n")

    # Also load per-symbol for consistency checks
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    data = {}
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        for n in HOLD_PERIODS:
            df[f"fwd_{n}d_bps"] = 10000 * (df["close"].shift(-n) / df["close"] - 1)
        data[sym] = df

    features = get_feature_cols(pooled)
    print(f"Feature columns: {len(features)}")
    print(f"Features: {features}\n")

    report = []
    report.append("=" * 100)
    report.append("DATA-DRIVEN EDGE DISCOVERY — No priors, let the data speak")
    report.append("=" * 100)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: INFORMATION COEFFICIENT RANKING
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n{'═'*100}")
    report.append("SECTION 1: INFORMATION COEFFICIENT (Rank IC) — All features vs forward returns")
    report.append(f"{'═'*100}")

    ic_rows = []
    for hold in HOLD_PERIODS:
        fwd_col = f"fwd_{hold}d_bps"
        report.append(f"\n  Hold = {hold}d:")
        report.append(f"  {'Feature':<30} {'IC':>8} {'|IC|':>8} {'p-value':>10} {'N':>10}")
        report.append(f"  {'─'*70}")

        feat_ics = []
        for feat in features:
            ic, p, n = information_coefficient(pooled[feat], pooled[fwd_col])
            if ic is not None:
                feat_ics.append((feat, ic, p, n))
                ic_rows.append({"feature": feat, "hold": hold, "ic": ic, "p": p, "n": n})

        # Sort by |IC|
        feat_ics.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, ic, p, n in feat_ics[:25]:
            marker = "***" if abs(ic) > 0.02 else "**" if abs(ic) > 0.01 else "*" if abs(ic) > 0.005 else ""
            report.append(f"  {feat:<30} {ic:>+8.4f} {abs(ic):>8.4f} {p:>10.6f} {n:>10,} {marker}")

    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(os.path.join(OUT_DIR, "data_driven_ic_all.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: QUINTILE SPREAD ANALYSIS (non-linear check)
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*100}")
    report.append("SECTION 2: QUINTILE SPREAD — Does the edge live in tails or is it linear?")
    report.append(f"{'═'*100}")

    # Use top IC features
    top_features = ic_df.groupby("feature")["ic"].apply(lambda x: x.abs().max()).sort_values(ascending=False).head(20).index.tolist()

    quint_rows = []
    for hold in HOLD_PERIODS:
        fwd_col = f"fwd_{hold}d_bps"
        report.append(f"\n  Hold = {hold}d:")
        report.append(f"  {'Feature':<25} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'Q5':>8} {'Spread':>8} {'t':>7} {'Mono':>5}")
        report.append(f"  {'─'*90}")

        for feat in top_features:
            qs = quintile_spread(pooled[feat], pooled[fwd_col])
            if qs is None:
                continue
            mono = "YES" if qs["monotonic"] else "no"
            report.append(
                f"  {feat:<25} {qs['q1_mean']:>+8.1f} {qs['q2_mean']:>+8.1f} {qs['q3_mean']:>+8.1f} "
                f"{qs['q4_mean']:>+8.1f} {qs['q5_mean']:>+8.1f} {qs['spread_bps']:>+8.1f} "
                f"{qs['t_stat']:>7.2f} {mono:>5}"
            )
            quint_rows.append({"feature": feat, "hold": hold, **qs})

    quint_df = pd.DataFrame(quint_rows)
    quint_df.to_csv(os.path.join(OUT_DIR, "data_driven_quintiles.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: TAIL ANALYSIS (5th percentile extremes)
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*100}")
    report.append("SECTION 3: TAIL ANALYSIS — Edge in extreme 5% tails vs middle 90%")
    report.append(f"{'═'*100}")

    for hold in HOLD_PERIODS:
        fwd_col = f"fwd_{hold}d_bps"
        tail_df = tail_analysis(pooled, features, fwd_col, percentile=5)
        if tail_df.empty:
            continue

        # Sort by max absolute tail excess
        tail_df["max_tail_excess"] = tail_df[["low_tail_excess", "high_tail_excess"]].abs().max(axis=1)
        tail_df = tail_df.sort_values("max_tail_excess", ascending=False)

        report.append(f"\n  Hold = {hold}d (top 20 by tail excess):")
        report.append(f"  {'Feature':<25} {'LowTail':>8} {'Middle':>8} {'HiTail':>8} {'LowExc':>8} {'HiExc':>8} {'LowT':>7} {'HiT':>7}")
        report.append(f"  {'─'*90}")

        for _, row in tail_df.head(20).iterrows():
            report.append(
                f"  {row['feature']:<25} {row['low_tail_mean']:>+8.1f} {row['middle_mean']:>+8.1f} "
                f"{row['high_tail_mean']:>+8.1f} {row['low_tail_excess']:>+8.1f} "
                f"{row['high_tail_excess']:>+8.1f} {row['low_tail_t']:>7.2f} {row['high_tail_t']:>7.2f}"
            )

        tail_df.to_csv(os.path.join(OUT_DIR, f"data_driven_tails_{hold}d.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: INTERACTION DISCOVERY
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*100}")
    report.append("SECTION 4: INTERACTION DISCOVERY — Pairs of features with non-linear edge")
    report.append(f"{'═'*100}")

    for hold in HOLD_PERIODS:
        fwd_col = f"fwd_{hold}d_bps"
        interact_df = interaction_scan(pooled, top_features, fwd_col, top_n=15)

        if interact_df.empty:
            continue

        # Focus on interactions where ratio > 1.3 (interaction > sum of parts)
        nonlinear = interact_df[interact_df["interaction_ratio"] > 1.3].head(15)

        report.append(f"\n  Hold = {hold}d — Non-linear interactions (ratio > 1.3):")
        report.append(f"  {'F1':<22} {'F2':<22} {'Spread':>7} {'t':>7} {'Ratio':>6} {'BestQ':<10} {'WorstQ':<10}")
        report.append(f"  {'─'*90}")

        for _, row in nonlinear.iterrows():
            report.append(
                f"  {row['f1']:<22} {row['f2']:<22} {row['spread_bps']:>+7.1f} "
                f"{row['t_stat']:>7.2f} {row['interaction_ratio']:>6.2f} "
                f"{row['best_quadrant']:<10} {row['worst_quadrant']:<10}"
            )

        interact_df.to_csv(os.path.join(OUT_DIR, f"data_driven_interactions_{hold}d.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: PER-SYMBOL IC STABILITY (is it consistent or concentrated?)
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*100}")
    report.append("SECTION 5: PER-SYMBOL IC CONSISTENCY — Does the feature work everywhere?")
    report.append(f"{'═'*100}")

    for hold in HOLD_PERIODS:
        report.append(f"\n  Hold = {hold}d:")
        report.append(f"  {'Feature':<30} {'MedianIC':>9} {'%Pos':>6} {'#Sym':>5}")
        report.append(f"  {'─'*55}")

        consistency = []
        for feat in top_features[:15]:
            sym_ics = per_symbol_ic(data, feat, hold)
            if len(sym_ics) < 50:
                continue
            med_ic = np.median(sym_ics)
            pct_pos = 100 * np.mean(np.array(sym_ics) > 0)
            consistency.append((feat, med_ic, pct_pos, len(sym_ics)))

        consistency.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, mic, pp, ns in consistency:
            report.append(f"  {feat:<30} {mic:>+9.4f} {pp:>6.1f} {ns:>5}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 6: YEAR-OVER-YEAR STABILITY OF TOP FEATURES
    # ═══════════════════════════════════════════════════════════════════
    report.append(f"\n\n{'═'*100}")
    report.append("SECTION 6: YEAR-OVER-YEAR IC STABILITY — Does the edge persist or decay?")
    report.append(f"{'═'*100}")

    for feat in top_features[:8]:
        report.append(f"\n  Feature: {feat}")
        for hold in [3, 5]:
            yr_stab = year_stability_of_ic(data, feat, hold)
            report.append(f"    Hold={hold}d: ", )
            line = "    "
            for yr, info in sorted(yr_stab.items()):
                line += f"{yr}:{info['median_ic']:+.3f}({info['pct_positive']:.0f}%) "
            report.append(line)

    report.append(f"\n{'═'*100}")
    report.append("END OF DATA-DRIVEN SCAN")
    report.append(f"{'═'*100}")

    report_text = "\n".join(report)
    print(report_text)

    report_path = os.path.join(OUT_DIR, "data_driven_scan_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
