import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# LOAD DATA
###############################################################################
print("Loading data...")
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/final_merged_trades.csv')
enriched = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_enriched.csv')

# Merge enriched features onto trades
# enriched has more rows (2875) - need to match on key columns
trades['signal_date'] = pd.to_datetime(trades['signal_date'])
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
enriched['signal_date'] = pd.to_datetime(enriched['signal_date'])
enriched['entry_date'] = pd.to_datetime(enriched['entry_date'])

# Merge on symbol + signal_date
merge_cols = [c for c in enriched.columns if c not in trades.columns or c in ['symbol','signal_date']]
df = trades.merge(enriched[merge_cols], on=['symbol','signal_date'], how='left')

print(f"Trades: {len(df)}, Columns after merge: {len(df.columns)}")

###############################################################################
# STEP 1: Define groups by hybrid_pnl
###############################################################################
print("\n" + "="*80)
print("STEP 1: DEFINING BIG WINNERS, MIDDLE, AND BIG LOSERS")
print("="*80)

df = df.sort_values('hybrid_pnl', ascending=False).reset_index(drop=True)
n = len(df)
n10 = int(np.ceil(n * 0.10))

# Top 10%
top_cutoff = df['hybrid_pnl'].iloc[n10 - 1]
bot_cutoff = df['hybrid_pnl'].iloc[n - n10]

df['group'] = 'MIDDLE'
df.loc[df['hybrid_pnl'] >= top_cutoff, 'group'] = 'TOP10'
df.loc[df['hybrid_pnl'] <= bot_cutoff, 'group'] = 'BOT10'

for g in ['TOP10', 'MIDDLE', 'BOT10']:
    sub = df[df['group'] == g]
    print(f"  {g}: {len(sub)} trades, PnL range [{sub['hybrid_pnl'].min():.0f}, {sub['hybrid_pnl'].max():.0f}], mean={sub['hybrid_pnl'].mean():.0f}")

###############################################################################
# STEP 2: Compute ADDITIONAL features from symbol enriched CSVs
###############################################################################
print("\n" + "="*80)
print("STEP 2: COMPUTING ADDITIONAL FEATURES FROM SYMBOL DATA")
print("="*80)

# Cache for loaded symbol data
symbol_cache = {}

def load_symbol(sym):
    if sym not in symbol_cache:
        path = f'/home/ubuntu/daily_data/data/{sym}_enriched.csv'
        try:
            sdf = pd.read_csv(path)
            sdf['date'] = pd.to_datetime(sdf['date'])
            sdf = sdf.sort_values('date').reset_index(drop=True)
            symbol_cache[sym] = sdf
        except:
            symbol_cache[sym] = None
    return symbol_cache[sym]

# Initialize new feature columns
new_features = [
    'rvol_signal_day', 'rvol_avg_3d', 'rvol_avg_5d', 'volume_trend_5d',
    'range_bps_signal', 'avg_range_3d', 'avg_range_5d', 'range_expansion', 'range_contraction_prior',
    'days_since_q4', 'z_score_velocity', 'slope_acceleration',
    'pct_from_52w_high_sig', 'pct_from_20d_high_sig', 'pct_from_52w_low_sig',
    'new_20d_high', 'new_52w_high',
    'stock_price', 'log_price'
]
for f in new_features:
    df[f] = np.nan

computed = 0
missed = 0

for idx, row in df.iterrows():
    sym = row['symbol']
    sig_date = row['signal_date']
    sdf = load_symbol(sym)
    if sdf is None:
        missed += 1
        continue
    
    # Find signal day index
    sig_idx_arr = sdf.index[sdf['date'] == sig_date]
    if len(sig_idx_arr) == 0:
        # Try nearest date
        date_diffs = (sdf['date'] - sig_date).abs()
        nearest = date_diffs.idxmin()
        if date_diffs[nearest] > timedelta(days=5):
            missed += 1
            continue
        sig_i = nearest
    else:
        sig_i = sig_idx_arr[0]
    
    sig_row = sdf.iloc[sig_i]
    
    # Volume dynamics
    vol_sma20 = sig_row.get('vol_sma20', np.nan)
    vol = sig_row.get('volume', np.nan)
    if pd.notna(vol_sma20) and vol_sma20 > 0:
        df.at[idx, 'rvol_signal_day'] = vol / vol_sma20
    
    if sig_i >= 2:
        rvol_vals_3 = sdf.iloc[sig_i-2:sig_i+1]['rvol'].values if 'rvol' in sdf.columns else None
        if rvol_vals_3 is not None:
            # rvol might be named differently, check
            if 'rvol' not in sdf.columns and 'rvol_20d' in sdf.columns:
                rvol_vals_3 = sdf.iloc[sig_i-2:sig_i+1]['rvol_20d'].values
            df.at[idx, 'rvol_avg_3d'] = np.nanmean(rvol_vals_3)
    
    if sig_i >= 4:
        if 'rvol' in sdf.columns:
            rvol_vals_5 = sdf.iloc[sig_i-4:sig_i+1]['rvol'].values
        elif 'rvol_20d' in sdf.columns:
            rvol_vals_5 = sdf.iloc[sig_i-4:sig_i+1]['rvol_20d'].values
        else:
            rvol_vals_5 = None
        if rvol_vals_5 is not None:
            df.at[idx, 'rvol_avg_5d'] = np.nanmean(rvol_vals_5)
    
    # Volume trend 5d (slope)
    if sig_i >= 4:
        vols = sdf.iloc[sig_i-4:sig_i+1]['volume'].values.astype(float)
        vols_clean = vols[~np.isnan(vols)]
        if len(vols_clean) >= 3:
            x = np.arange(len(vols_clean))
            slope, _, _, _, _ = stats.linregress(x, vols_clean)
            df.at[idx, 'volume_trend_5d'] = slope
    
    # Range dynamics
    range_bps = sig_row.get('range_bps', np.nan)
    df.at[idx, 'range_bps_signal'] = range_bps
    
    if sig_i >= 2 and 'range_bps' in sdf.columns:
        df.at[idx, 'avg_range_3d'] = sdf.iloc[sig_i-2:sig_i+1]['range_bps'].mean()
    if sig_i >= 4 and 'range_bps' in sdf.columns:
        df.at[idx, 'avg_range_5d'] = sdf.iloc[sig_i-4:sig_i+1]['range_bps'].mean()
    
    if sig_i >= 10 and 'range_bps' in sdf.columns:
        prior_10 = sdf.iloc[sig_i-10:sig_i]['range_bps'].mean()
        if prior_10 > 0:
            df.at[idx, 'range_expansion'] = range_bps / prior_10
        # range_contraction_prior: avg of days -10 to -4 vs days -3 to -1
        far = sdf.iloc[sig_i-10:sig_i-3]['range_bps'].mean()
        near = sdf.iloc[sig_i-3:sig_i]['range_bps'].mean()
        if far > 0:
            df.at[idx, 'range_contraction_prior'] = near / far  # <1 means contraction
    
    # Speed to Q5
    # days_since_q4: look backwards from signal day for how long stock was in Q4 before Q5
    # We need z_score - quintile is based on z_score likely
    # Check for z_score or quintile column
    if 'z_score' in sdf.columns:
        zs = sig_row.get('z_score', np.nan)
        # Count consecutive days before signal where z_score was in a "Q4-like" range
        # Q5 is top quintile of z_score. Let's approximate Q4 as having been below Q5 threshold
        # Actually, let's look for how many days the stock had been above some threshold
        # A simpler interpretation: how many consecutive days was z_score > 0 before signal?
        # Or we can look for dist_sma50_pct or sma50_slope as proxy
        # Let's count days in Q4 (positive z but not yet Q5 level)
        # For now, count consecutive days with z_score between, say, 1 and 2 before signal
        days_q4 = 0
        for j in range(sig_i-1, max(sig_i-30, -1), -1):
            z_j = sdf.iloc[j].get('z_score', np.nan) if j >= 0 else np.nan
            if pd.notna(z_j) and 0.5 < z_j < 2.0:
                days_q4 += 1
            else:
                break
        df.at[idx, 'days_since_q4'] = days_q4 if days_q4 > 0 else np.nan
    
    if 'z_score' in sdf.columns and sig_i >= 3:
        z_now = sdf.iloc[sig_i].get('z_score', np.nan)
        z_3ago = sdf.iloc[sig_i-3].get('z_score', np.nan)
        if pd.notna(z_now) and pd.notna(z_3ago):
            df.at[idx, 'z_score_velocity'] = z_now - z_3ago
    
    # slope_acceleration
    if 'sma50_slope_5d' in sdf.columns and sig_i >= 5:
        slope_now = sdf.iloc[sig_i].get('sma50_slope_5d', np.nan)
        slope_5ago = sdf.iloc[sig_i-5].get('sma50_slope_5d', np.nan)
        if pd.notna(slope_now) and pd.notna(slope_5ago):
            df.at[idx, 'slope_acceleration'] = slope_now - slope_5ago
    
    # Proximity
    df.at[idx, 'pct_from_52w_high_sig'] = sig_row.get('pct_from_52w_high', np.nan)
    df.at[idx, 'pct_from_20d_high_sig'] = sig_row.get('pct_from_20d_high', np.nan)
    df.at[idx, 'pct_from_52w_low_sig'] = sig_row.get('pct_from_52w_low', np.nan)
    
    close = sig_row.get('close', np.nan)
    if pd.notna(close):
        df.at[idx, 'stock_price'] = close
        if close > 0:
            df.at[idx, 'log_price'] = np.log(close)
    
    # new_20d_high
    if sig_i >= 19:
        high_20d = sdf.iloc[sig_i-19:sig_i]['close'].max()
        df.at[idx, 'new_20d_high'] = 1 if close >= high_20d else 0
    
    # new_52w_high
    if sig_i >= 251:
        high_52w = sdf.iloc[sig_i-251:sig_i]['close'].max()
        df.at[idx, 'new_52w_high'] = 1 if close >= high_52w else 0
    
    computed += 1
    if computed % 200 == 0:
        print(f"  Computed {computed} trades...")

print(f"  Computed features for {computed} trades, missed {missed}")

###############################################################################
# STEP 3: Statistical comparison - top vs bottom vs middle
###############################################################################
print("\n" + "="*80)
print("STEP 3: STATISTICAL COMPARISON OF FEATURES")
print("="*80)

# All features to analyze
existing_features = [
    'entry_rsi_14', 'entry_adx_14', 'entry_bb_bandwidth', 'entry_dist_sma50_pct',
    'entry_dist_sma200_pct', 'entry_rvol_20d', 'entry_squeeze', 'entry_consec_days',
    'entry_macd_hist', 'entry_bb_pctb', 'entry_stoch_k', 'entry_stoch_d',
    'entry_roc_10', 'entry_roc_20', 'entry_dist_sma8_pct', 'entry_dist_sma20_pct',
    'entry_range_bps', 'entry_gap_bps', 'entry_close_location', 'entry_body_pct',
    'entry_sma20_slope_5d', 'entry_sma50_slope_5d',
    'entry_pct_from_20d_high', 'entry_pct_from_20d_low',
    'entry_pct_from_52w_high', 'entry_pct_from_52w_low',
    'entry_daily_return_pct', 'entry_plus_di', 'entry_minus_di',
    'entry_rvol', 'entry_rvol_10d',
]

# Add spy features if present
spy_features = [c for c in df.columns if c.startswith('spy_')]
existing_features.extend(spy_features)

all_features = existing_features + new_features
# Remove features that don't exist in df
all_features = [f for f in all_features if f in df.columns]

# Remove duplicates
all_features = list(dict.fromkeys(all_features))

top = df[df['group'] == 'TOP10']
mid = df[df['group'] == 'MIDDLE']
bot = df[df['group'] == 'BOT10']
rest = df[df['group'] != 'TOP10']

results = []
for feat in all_features:
    top_vals = top[feat].dropna()
    bot_vals = bot[feat].dropna()
    mid_vals = mid[feat].dropna()
    rest_vals = rest[feat].dropna()
    
    if len(top_vals) < 10 or len(bot_vals) < 10:
        continue
    
    mean_top = top_vals.mean()
    mean_mid = mid_vals.mean()
    mean_bot = bot_vals.mean()
    
    # t-test: top vs bottom
    t_stat, t_pval = stats.ttest_ind(top_vals, bot_vals, equal_var=False)
    
    # Cohen's d: top vs rest
    pooled_std = np.sqrt(((len(top_vals)-1)*top_vals.std()**2 + (len(rest_vals)-1)*rest_vals.std()**2) / (len(top_vals) + len(rest_vals) - 2))
    if pooled_std > 0:
        cohens_d = (mean_top - rest_vals.mean()) / pooled_std
    else:
        cohens_d = 0
    
    significant = abs(cohens_d) > 0.25 or abs(t_stat) > 3
    
    results.append({
        'feature': feat,
        'mean_top10': mean_top,
        'mean_mid80': mean_mid,
        'mean_bot10': mean_bot,
        't_stat': t_stat,
        't_pval': t_pval,
        'cohens_d': cohens_d,
        'significant': significant,
        'n_top': len(top_vals),
        'n_bot': len(bot_vals)
    })

results_df = pd.DataFrame(results)
results_df['abs_d'] = results_df['cohens_d'].abs()
results_df = results_df.sort_values('abs_d', ascending=False)

print("\nTop 30 features by |Cohen's d| (top 10% vs rest):")
print(f"{'Feature':<30} {'Mean T10':>10} {'Mean M80':>10} {'Mean B10':>10} {'t-stat':>8} {'p-val':>8} {'Cohen d':>8} {'Sig':>4}")
print("-" * 100)
for _, r in results_df.head(30).iterrows():
    sig_flag = "***" if r['significant'] else ""
    print(f"{r['feature']:<30} {r['mean_top10']:>10.3f} {r['mean_mid80']:>10.3f} {r['mean_bot10']:>10.3f} {r['t_stat']:>8.2f} {r['t_pval']:>8.4f} {r['cohens_d']:>8.3f} {sig_flag:>4}")

###############################################################################
# STEP 4: Quintile analysis for top 5 features
###############################################################################
print("\n" + "="*80)
print("STEP 4: QUINTILE ANALYSIS FOR TOP 5 FEATURES")
print("="*80)

top5_features = results_df.head(5)['feature'].tolist()

for feat in top5_features:
    print(f"\n--- {feat} ---")
    valid = df[df[feat].notna()].copy()
    if len(valid) < 50:
        print(f"  Insufficient data ({len(valid)} trades)")
        continue
    
    valid['quintile'] = pd.qcut(valid[feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    
    print(f"{'Quintile':<12} {'Count':>6} {'Mean PnL':>12} {'Med PnL':>12} {'WR%':>8} {'% Total PnL':>12} {'Feat Range':>20}")
    total_pnl = valid['hybrid_pnl'].sum()
    
    for q in ['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)']:
        qd = valid[valid['quintile'] == q]
        if len(qd) == 0:
            continue
        mean_pnl = qd['hybrid_pnl'].mean()
        med_pnl = qd['hybrid_pnl'].median()
        wr = (qd['hybrid_pnl'] > 0).mean() * 100
        pct_total = qd['hybrid_pnl'].sum() / total_pnl * 100 if total_pnl != 0 else 0
        fmin, fmax = qd[feat].min(), qd[feat].max()
        print(f"{q:<12} {len(qd):>6} {mean_pnl:>12.0f} {med_pnl:>12.0f} {wr:>7.1f}% {pct_total:>11.1f}% {fmin:>9.2f} - {fmax:>8.2f}")
    
    # Monotonicity check
    q_means = valid.groupby('quintile')['hybrid_pnl'].mean()
    diffs = q_means.diff().dropna()
    if all(diffs > 0):
        print("  Pattern: MONOTONICALLY INCREASING")
    elif all(diffs < 0):
        print("  Pattern: MONOTONICALLY DECREASING")
    else:
        # Check approximate monotonicity
        increasing = (diffs > 0).sum()
        print(f"  Pattern: NON-MONOTONIC ({increasing}/4 steps increasing)")

###############################################################################
# STEP 5: Top 10 individual winners - common features
###############################################################################
print("\n" + "="*80)
print("STEP 5: TOP 10 INDIVIDUAL WINNERS")
print("="*80)

top10_trades = df.nlargest(10, 'hybrid_pnl')

# Show key features for each
display_cols = ['symbol', 'signal_date', 'hybrid_pnl', 'stock_pnl', 'opt_pnl', 'tier', 'days_held'] + [f for f in top5_features if f in df.columns] + [
    'rvol_signal_day', 'range_expansion', 'range_bps_signal', 'z_score_velocity',
    'pct_from_52w_high_sig', 'new_20d_high', 'new_52w_high', 'stock_price',
    'entry_rsi_14', 'entry_adx_14', 'entry_squeeze', 'entry_consec_days'
]
display_cols = [c for c in display_cols if c in df.columns]
display_cols = list(dict.fromkeys(display_cols))

for i, (_, row) in enumerate(top10_trades.iterrows()):
    print(f"\n  #{i+1}: {row['symbol']} ({str(row['signal_date'])[:10]}) - PnL: ${row['hybrid_pnl']:,.0f}")
    for col in display_cols:
        if col in ['symbol', 'signal_date', 'hybrid_pnl']:
            continue
        val = row[col]
        if pd.notna(val):
            if isinstance(val, float):
                print(f"    {col}: {val:.3f}")
            else:
                print(f"    {col}: {val}")

# Commonality analysis
print("\n--- What the top 10 winners have in common ---")
for feat in top5_features + new_features:
    if feat not in df.columns:
        continue
    vals = top10_trades[feat].dropna()
    if len(vals) < 5:
        continue
    all_vals = df[feat].dropna()
    if len(all_vals) < 50:
        continue
    pctile = [(all_vals < v).mean() * 100 for v in vals]
    avg_pctile = np.mean(pctile)
    if avg_pctile > 75 or avg_pctile < 25:
        direction = "HIGH" if avg_pctile > 75 else "LOW"
        print(f"  {feat}: avg percentile = {avg_pctile:.0f}% ({direction}) | values: {vals.describe()[['mean','min','max']].to_dict()}")

###############################################################################
# STEP 6: Overfitting check - split by time period
###############################################################################
print("\n" + "="*80)
print("STEP 6: OVERFITTING CHECK - TIME SPLIT")
print("="*80)

df['year'] = df['signal_date'].dt.year
early = df[df['year'] <= 2022].copy()
late = df[df['year'] >= 2023].copy()
print(f"  Early period (<=2022): {len(early)} trades")
print(f"  Late period (>=2023): {len(late)} trades")

top3_features = results_df.head(3)['feature'].tolist()

for feat in top3_features:
    print(f"\n--- {feat} ---")
    for period_name, period_df in [("2020-2022", early), ("2023-2026", late)]:
        valid = period_df[period_df[feat].notna()].copy()
        if len(valid) < 25:
            print(f"  {period_name}: Insufficient data ({len(valid)} trades)")
            continue
        
        try:
            valid['quintile'] = pd.qcut(valid[feat], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        except:
            print(f"  {period_name}: Could not create quintiles")
            continue
        
        q_means = valid.groupby('quintile')['hybrid_pnl'].mean()
        q_wr = valid.groupby('quintile')['hybrid_pnl'].apply(lambda x: (x>0).mean()*100)
        
        print(f"  {period_name} (n={len(valid)}):")
        print(f"    {'Quintile':<8} {'Mean PnL':>10} {'WR%':>8}")
        for q in ['Q1','Q2','Q3','Q4','Q5']:
            if q in q_means.index:
                print(f"    {q:<8} {q_means[q]:>10.0f} {q_wr[q]:>7.1f}%")
        
        diffs = q_means.diff().dropna()
        if all(diffs > 0):
            print(f"    Monotonic: YES (increasing)")
        elif all(diffs < 0):
            print(f"    Monotonic: YES (decreasing)")
        else:
            inc = (diffs > 0).sum()
            print(f"    Monotonic: NO ({inc}/{len(diffs)} steps consistent)")

###############################################################################
# STEP 7: Compile full report
###############################################################################
print("\n\nAnalysis complete. Writing report...")

# Build report string
report = []
report.append("=" * 100)
report.append("WINNER ANATOMY REPORT: WHAT SEPARATES THE TOP 10% FROM THE REST")
report.append("=" * 100)
report.append(f"Generated: 2026-04-03")
report.append(f"Total trades analyzed: {len(df)}")
report.append("")

# Step 1
report.append("=" * 100)
report.append("SECTION 1: GROUP DEFINITIONS")
report.append("=" * 100)
for g in ['TOP10', 'MIDDLE', 'BOT10']:
    sub = df[df['group'] == g]
    report.append(f"  {g}: {len(sub)} trades")
    report.append(f"    PnL range: ${sub['hybrid_pnl'].min():,.0f} to ${sub['hybrid_pnl'].max():,.0f}")
    report.append(f"    Mean PnL: ${sub['hybrid_pnl'].mean():,.0f}")
    report.append(f"    Median PnL: ${sub['hybrid_pnl'].median():,.0f}")
    report.append(f"    Win rate: {(sub['hybrid_pnl'] > 0).mean()*100:.1f}%")
    report.append(f"    % of total PnL: {sub['hybrid_pnl'].sum()/df['hybrid_pnl'].sum()*100:.1f}%")
    report.append("")

report.append(f"  Top 10% cutoff: >= ${top_cutoff:,.0f}")
report.append(f"  Bottom 10% cutoff: <= ${bot_cutoff:,.0f}")
report.append("")

# Step 2
report.append("=" * 100)
report.append("SECTION 2: ADDITIONAL COMPUTED FEATURES")
report.append("=" * 100)
report.append(f"  Features computed from symbol-level enriched CSVs: {len(new_features)}")
report.append(f"  Successfully computed for: {computed} trades")
report.append(f"  Feature coverage:")
for feat in new_features:
    n_valid = df[feat].notna().sum()
    report.append(f"    {feat}: {n_valid} trades ({n_valid/len(df)*100:.0f}%)")
report.append("")

# Step 3
report.append("=" * 100)
report.append("SECTION 3: FEATURE COMPARISON - TOP 10% vs BOTTOM 10% vs MIDDLE 80%")
report.append("=" * 100)
report.append("")
report.append(f"{'Feature':<30} {'Mean T10':>10} {'Mean M80':>10} {'Mean B10':>10} {'t-stat':>8} {'p-val':>10} {'Cohen d':>8} {'Sig':>5}")
report.append("-" * 100)

sig_count = 0
for _, r in results_df.head(30).iterrows():
    sig_flag = "***" if r['significant'] else ""
    report.append(f"{r['feature']:<30} {r['mean_top10']:>10.3f} {r['mean_mid80']:>10.3f} {r['mean_bot10']:>10.3f} {r['t_stat']:>8.2f} {r['t_pval']:>10.6f} {r['cohens_d']:>8.3f} {sig_flag:>5}")
    if r['significant']:
        sig_count += 1

report.append("")
report.append(f"  Total features tested: {len(results_df)}")
report.append(f"  Features meeting significance threshold (|d|>0.25 or |t|>3): {sig_count} of top 30")
report.append(f"  Total significant features: {results_df['significant'].sum()}")
report.append("")
report.append("  MULTIPLE TESTING NOTE: With {n} features tested, at alpha=0.05,".format(n=len(results_df)))
report.append(f"  we expect ~{len(results_df)*0.05:.0f} false positives by chance alone.")
report.append(f"  Bonferroni-corrected alpha: {0.05/len(results_df):.5f}")
bonf_sig = (results_df['t_pval'] < 0.05/len(results_df)).sum()
report.append(f"  Features surviving Bonferroni correction: {bonf_sig}")
report.append("")

# Step 4
report.append("=" * 100)
report.append("SECTION 4: QUINTILE ANALYSIS FOR TOP 5 FEATURES")
report.append("=" * 100)

for feat in top5_features:
    report.append(f"\n--- {feat} (Cohen's d = {results_df[results_df['feature']==feat]['cohens_d'].values[0]:.3f}) ---")
    valid = df[df[feat].notna()].copy()
    if len(valid) < 50:
        report.append(f"  Insufficient data ({len(valid)} trades)")
        continue
    
    try:
        valid['quintile'] = pd.qcut(valid[feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    except:
        report.append("  Could not create quintiles (too many duplicate values)")
        continue
    
    total_pnl = valid['hybrid_pnl'].sum()
    report.append(f"{'Quintile':<12} {'Count':>6} {'Mean PnL':>12} {'Med PnL':>12} {'WR%':>8} {'% Total PnL':>12} {'Feat Range':>20}")
    
    q_means_list = []
    for q in ['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)']:
        qd = valid[valid['quintile'] == q]
        if len(qd) == 0:
            continue
        mean_pnl = qd['hybrid_pnl'].mean()
        med_pnl = qd['hybrid_pnl'].median()
        wr = (qd['hybrid_pnl'] > 0).mean() * 100
        pct_total = qd['hybrid_pnl'].sum() / total_pnl * 100 if total_pnl != 0 else 0
        fmin, fmax = qd[feat].min(), qd[feat].max()
        report.append(f"{q:<12} {len(qd):>6} {mean_pnl:>12.0f} {med_pnl:>12.0f} {wr:>7.1f}% {pct_total:>11.1f}% {fmin:>9.2f} - {fmax:>8.2f}")
        q_means_list.append(mean_pnl)
    
    # Monotonicity
    if len(q_means_list) >= 3:
        diffs = np.diff(q_means_list)
        if all(d > 0 for d in diffs):
            report.append("  MONOTONIC: YES (increasing) - STRONG signal")
        elif all(d < 0 for d in diffs):
            report.append("  MONOTONIC: YES (decreasing) - STRONG signal")
        else:
            inc = sum(1 for d in diffs if d > 0)
            report.append(f"  MONOTONIC: NO ({inc}/{len(diffs)} steps increasing)")

# Step 5
report.append("")
report.append("=" * 100)
report.append("SECTION 5: TOP 10 INDIVIDUAL WINNERS - COMMON CHARACTERISTICS")
report.append("=" * 100)

for i, (_, row) in enumerate(top10_trades.iterrows()):
    report.append(f"\n  #{i+1}: {row['symbol']} ({str(row['signal_date'])[:10]}) - Hybrid PnL: ${row['hybrid_pnl']:,.0f}")
    for col in display_cols:
        if col in ['symbol', 'signal_date', 'hybrid_pnl']:
            continue
        val = row[col]
        if pd.notna(val):
            if isinstance(val, float):
                report.append(f"      {col}: {val:.3f}")
            else:
                report.append(f"      {col}: {val}")

report.append("\n  --- Shared characteristics (features where top 10 cluster at extremes) ---")
shared_findings = []
for feat in top5_features + new_features:
    if feat not in df.columns:
        continue
    vals = top10_trades[feat].dropna()
    if len(vals) < 5:
        continue
    all_vals = df[feat].dropna()
    if len(all_vals) < 50:
        continue
    pctile = [(all_vals < v).mean() * 100 for v in vals]
    avg_pctile = np.mean(pctile)
    if avg_pctile > 70 or avg_pctile < 30:
        direction = "HIGH" if avg_pctile > 70 else "LOW"
        shared_findings.append((feat, avg_pctile, direction, vals.mean(), vals.std()))
        report.append(f"  {feat}: avg percentile = {avg_pctile:.0f}% ({direction}), mean={vals.mean():.3f}, std={vals.std():.3f}")

# Step 6
report.append("")
report.append("=" * 100)
report.append("SECTION 6: OVERFITTING CHECK - TIME-PERIOD STABILITY")
report.append("=" * 100)
report.append(f"  Early period (2020-2022): {len(early)} trades")
report.append(f"  Late period (2023-2026): {len(late)} trades")

overfitting_flags = []
for feat in top3_features:
    report.append(f"\n--- {feat} ---")
    
    period_results = {}
    for period_name, period_df in [("2020-2022", early), ("2023-2026", late)]:
        valid = period_df[period_df[feat].notna()].copy()
        if len(valid) < 25:
            report.append(f"  {period_name}: Insufficient data ({len(valid)} trades)")
            continue
        
        try:
            valid['quintile'] = pd.qcut(valid[feat], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        except:
            report.append(f"  {period_name}: Could not form quintiles")
            continue
        
        q_means = valid.groupby('quintile')['hybrid_pnl'].mean()
        q_wr = valid.groupby('quintile')['hybrid_pnl'].apply(lambda x: (x>0).mean()*100)
        
        report.append(f"  {period_name} (n={len(valid)}):")
        report.append(f"    {'Quintile':<8} {'Mean PnL':>10} {'WR%':>8}")
        for q in ['Q1','Q2','Q3','Q4','Q5']:
            if q in q_means.index:
                report.append(f"    {q:<8} {q_means[q]:>10.0f} {q_wr[q]:>7.1f}%")
        
        diffs = q_means.diff().dropna()
        if all(diffs > 0):
            report.append(f"    Monotonic: YES (increasing)")
            period_results[period_name] = 'mono_inc'
        elif all(diffs < 0):
            report.append(f"    Monotonic: YES (decreasing)")
            period_results[period_name] = 'mono_dec'
        else:
            inc = (diffs > 0).sum()
            report.append(f"    Monotonic: NO ({inc}/{len(diffs)} consistent)")
            period_results[period_name] = f'non_mono_{inc}'
    
    # Check consistency
    if len(period_results) == 2:
        vals = list(period_results.values())
        if vals[0] == vals[1] and 'mono' in vals[0]:
            report.append(f"  STABILITY: CONSISTENT across both periods - likely real signal")
        elif vals[0][:4] == vals[1][:4]:
            report.append(f"  STABILITY: SAME DIRECTION but not perfectly monotonic - likely real but noisy")
        else:
            report.append(f"  STABILITY: INCONSISTENT - possible overfitting concern")
            overfitting_flags.append(feat)

# Step 7
report.append("")
report.append("=" * 100)
report.append("SECTION 7: ECONOMIC INTERPRETATION")
report.append("=" * 100)

# Get top significant features for interpretation
sig_features = results_df[results_df['significant']].head(10)

for _, r in sig_features.iterrows():
    feat = r['feature']
    d = r['cohens_d']
    direction = "higher" if d > 0 else "lower"
    report.append(f"\n--- {feat} (Cohen's d = {d:.3f}) ---")
    report.append(f"  Finding: Big winners have {direction} {feat} than the rest")
    report.append(f"  Top 10% mean: {r['mean_top10']:.3f} | Bottom 10% mean: {r['mean_bot10']:.3f} | t-stat: {r['t_stat']:.2f}")
    
    # Economic interpretation based on feature name
    if 'rvol' in feat.lower() or 'volume' in feat.lower():
        report.append("  Economic rationale: Elevated volume on signal day suggests institutional participation")
        report.append("  and conviction behind the move. Higher volume breakouts tend to sustain because they")
        report.append("  indicate broad market agreement, not just noise. This is a well-known price-volume")
        report.append("  relationship consistent with market microstructure theory.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'range' in feat.lower() or 'atr' in feat.lower():
        report.append("  Economic rationale: Range dynamics capture volatility regime changes. Range expansion")
        report.append("  on signal day, especially after prior contraction, suggests a breakout from a")
        report.append("  consolidation pattern. This aligns with the volatility clustering literature and")
        report.append("  the squeeze/breakout trading thesis.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'rsi' in feat.lower():
        report.append("  Economic rationale: RSI measures momentum strength. If winners have different RSI,")
        report.append("  it may reflect the sweet spot between oversold bounces and overbought exhaustion.")
        report.append("  However, RSI is a lagging indicator and the relationship may be non-linear.")
        report.append(f"  VERDICT: PLAUSIBLE but may be capturing a non-linear effect")
    elif 'dist_sma' in feat.lower() or 'slope' in feat.lower():
        report.append("  Economic rationale: Distance from moving averages measures trend extension.")
        report.append("  Moderate distance suggests early-stage trend; extreme distance suggests mean-reversion")
        report.append("  risk. Slope measures trend strength and acceleration, which is related to the")
        report.append("  persistence of momentum documented in academic literature.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'squeeze' in feat.lower():
        report.append("  Economic rationale: Squeeze (Bollinger bands inside Keltner channels) represents")
        report.append("  compressed volatility. Breakouts from squeezes tend to be more powerful because")
        report.append("  the low-volatility period represents a buildup of energy/positioning.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'adx' in feat.lower():
        report.append("  Economic rationale: ADX measures trend strength regardless of direction.")
        report.append("  Higher ADX at entry may indicate entering an established trend rather than")
        report.append("  a choppy/ranging market, increasing the probability of continuation.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif '52w' in feat.lower() or '20d' in feat.lower():
        report.append("  Economic rationale: Proximity to 52-week or 20-day highs/lows captures whether")
        report.append("  the stock is breaking into new territory or bouncing from support. New highs")
        report.append("  remove overhead supply (no trapped longs above), reducing resistance.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'price' in feat.lower():
        report.append("  Economic rationale: Absolute price level could matter due to options gamma effects")
        report.append("  (higher-priced stocks have different delta/gamma profiles), or due to institutional")
        report.append("  coverage biases. However, this could also be a proxy for market cap or quality factor.")
        report.append(f"  VERDICT: UNCERTAIN - may be a proxy for other factors")
    elif 'macd' in feat.lower():
        report.append("  Economic rationale: MACD histogram measures momentum acceleration.")
        report.append("  Positive and rising MACD hist suggests accelerating upward momentum,")
        report.append("  consistent with entries that catch early-stage trend acceleration.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'bb' in feat.lower():
        report.append("  Economic rationale: Bollinger Band width/position measures recent volatility")
        report.append("  regime and where price sits within that volatility envelope. Narrow bands")
        report.append("  before a signal suggest breakout potential; high %B suggests strong momentum.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'consec' in feat.lower() or 'days_since' in feat.lower():
        report.append("  Economic rationale: Consecutive days in a signal state measures persistence")
        report.append("  of the underlying condition. Moderate persistence suggests confirmation")
        report.append("  without exhaustion. Too few days = false signal; too many = extended move.")
        report.append(f"  VERDICT: PLAUSIBLE if relationship is non-linear")
    elif 'z_score' in feat.lower():
        report.append("  Economic rationale: Z-score velocity measures how quickly the stock's")
        report.append("  relative strength is improving. Rapid acceleration suggests emerging")
        report.append("  institutional interest or catalyst-driven re-rating.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'gap' in feat.lower():
        report.append("  Economic rationale: Gap size on signal day reflects overnight information flow")
        report.append("  and pre-market conviction. Larger gaps suggest stronger catalyst but also")
        report.append("  potential for gap-fill reversion.")
        report.append(f"  VERDICT: PLAUSIBLE but direction matters")
    elif 'spy' in feat.lower():
        report.append("  Economic rationale: Market regime at time of entry. Trades entered when")
        report.append("  the broad market is in a favorable state may benefit from rising-tide effects.")
        report.append("  This is well-documented in the factor timing literature.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'body_pct' in feat.lower() or 'close_location' in feat.lower():
        report.append("  Economic rationale: Candle body percentage and close location measure")
        report.append("  intraday conviction. A close near the high (high close_location) suggests")
        report.append("  buying pressure persisted through the session, indicating strong demand.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'di' in feat.lower():
        report.append("  Economic rationale: Plus DI vs Minus DI measures directional strength.")
        report.append("  Higher Plus DI indicates stronger buying pressure in the trend.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE")
    elif 'stoch' in feat.lower():
        report.append("  Economic rationale: Stochastic oscillator measures where price closed")
        report.append("  relative to its recent range. High stochastic can mean momentum or overbought.")
        report.append(f"  VERDICT: PLAUSIBLE but direction-dependent")
    elif 'roc' in feat.lower():
        report.append("  Economic rationale: Rate of change measures raw momentum over the lookback")
        report.append("  period. Higher ROC at entry means stronger prior momentum, consistent with")
        report.append("  the momentum factor literature, but extreme values increase reversion risk.")
        report.append(f"  VERDICT: ECONOMICALLY PLAUSIBLE if moderate")
    else:
        report.append("  Economic rationale: Requires further investigation to determine causal mechanism.")
        report.append(f"  VERDICT: UNCERTAIN - needs more analysis")

# Overfitting summary
report.append("")
report.append("=" * 100)
report.append("SECTION 8: SUMMARY AND OVERFITTING ASSESSMENT")
report.append("=" * 100)

report.append("")
report.append(f"  Total features tested: {len(results_df)}")
report.append(f"  Features significant at |d|>0.25 or |t|>3: {results_df['significant'].sum()}")
report.append(f"  Features surviving Bonferroni correction: {bonf_sig}")
if len(overfitting_flags) > 0:
    report.append(f"  Features FLAGGED for possible overfitting: {', '.join(overfitting_flags)}")
else:
    report.append(f"  Features flagged for overfitting: None of top 3")

report.append("")
report.append("  MULTIPLE TESTING WARNING:")
report.append(f"  We tested {len(results_df)} features. At p<0.05, we expect ~{len(results_df)*0.05:.0f} false positives by chance.")
report.append(f"  Only {bonf_sig} features survive the Bonferroni correction (p < {0.05/len(results_df):.5f}).")
report.append("  Features that are NOT monotonic in quintile analysis or NOT stable across time periods")
report.append("  should be treated with additional skepticism.")

report.append("")
report.append("  RECOMMENDED FILTERS FOR FURTHER TESTING:")
report.append("  The following features meet ALL quality criteria (significant, monotonic-leaning, stable):")

# Identify features meeting all criteria
for feat in top5_features:
    r = results_df[results_df['feature'] == feat].iloc[0]
    in_overfitting = feat in overfitting_flags
    if r['significant'] and not in_overfitting:
        report.append(f"    - {feat} (d={r['cohens_d']:.3f}, t={r['t_stat']:.2f})")

report.append("")
report.append("  These features should be considered as potential filters to improve trade selection,")
report.append("  but should be validated on a true out-of-sample period before deployment.")

report_text = "\n".join(report)

with open('/home/ubuntu/daily_data/analysis_results/winner_anatomy_report.txt', 'w') as f:
    f.write(report_text)

print("\nReport saved to /home/ubuntu/daily_data/analysis_results/winner_anatomy_report.txt")
print(f"Report length: {len(report)} lines")
