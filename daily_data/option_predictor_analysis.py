import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# LOAD AND MERGE DATA
###############################################################################
print("Loading data...")
opts = pd.read_csv('/home/ubuntu/daily_data/analysis_results/options_full_trades.csv')
opts_30d = opts[opts['option_type'] == '30d_1m'].copy()
print(f"  30d_1m option trades: {len(opts_30d)}")

enr = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_enriched.csv')
print(f"  Enriched trades: {len(enr)}")

# Merge on symbol + entry_date
merged = opts_30d.merge(enr, on=['symbol', 'entry_date'], how='inner', suffixes=('_opt', '_enr'))
print(f"  Merged trades: {len(merged)}")

###############################################################################
# STEP 1: Build additional pre-entry features from symbol daily data
###############################################################################
print("\nStep 1: Building pre-entry features from daily data...")

new_features = []
missing_symbols = set()

for idx, row in merged.iterrows():
    sym = row['symbol']
    signal_date = row['signal_date']
    
    fpath = f'/home/ubuntu/daily_data/data/{sym}_enriched.csv'
    try:
        df = pd.read_csv(fpath)
    except:
        missing_symbols.add(sym)
        new_features.append({})
        continue
    
    df['date'] = pd.to_datetime(df['date'])
    sig_dt = pd.to_datetime(signal_date)
    
    # Find signal day index
    mask = df['date'] == sig_dt
    if mask.sum() == 0:
        # Try closest date before
        prior = df[df['date'] <= sig_dt]
        if len(prior) == 0:
            new_features.append({})
            continue
        sig_idx = prior.index[-1]
    else:
        sig_idx = df[mask].index[0]
    
    loc = df.index.get_loc(sig_idx)
    if loc < 60:
        new_features.append({})
        continue
    
    feats = {}
    
    # ATR dynamics
    atr_now = df.iloc[loc]['atr_14']
    close_now = df.iloc[loc]['close']
    
    if loc >= 1 and df.iloc[loc-1]['atr_14'] > 0:
        feats['atr_change_1d'] = (atr_now / df.iloc[loc-1]['atr_14'] - 1) * 100
    if loc >= 3 and df.iloc[loc-3]['atr_14'] > 0:
        feats['atr_change_3d'] = (atr_now / df.iloc[loc-3]['atr_14'] - 1) * 100
    if loc >= 5 and df.iloc[loc-5]['atr_14'] > 0:
        feats['atr_change_5d'] = (atr_now / df.iloc[loc-5]['atr_14'] - 1) * 100
    
    # ATR percentile
    atr_20 = df.iloc[loc-19:loc+1]['atr_14'].values
    atr_60 = df.iloc[loc-59:loc+1]['atr_14'].values
    feats['atr_percentile_20d'] = (atr_20 < atr_now).sum() / len(atr_20) * 100
    feats['atr_percentile_60d'] = (atr_60 < atr_now).sum() / len(atr_60) * 100
    
    atr_20_mean = atr_20.mean()
    feats['atr_vs_avg'] = atr_now / atr_20_mean if atr_20_mean > 0 else np.nan
    
    # Volatility dynamics
    rvol20_now = df.iloc[loc]['rvol_20d']
    if loc >= 5:
        rvol20_5ago = df.iloc[loc-5]['rvol_20d']
        if rvol20_5ago > 0:
            feats['rvol_change_5d'] = (rvol20_now / rvol20_5ago - 1) * 100
    
    feats['range_vs_atr'] = df.iloc[loc]['range_bps'] / (atr_now / close_now * 10000) if atr_now > 0 else np.nan
    
    # avg range 3d
    ranges_3d = df.iloc[loc-2:loc+1]['range_bps'].values
    feats['avg_range_3d'] = ranges_3d.mean()
    
    # Price action leading in
    feats['ret_1d'] = df.iloc[loc]['daily_return_pct']
    if loc >= 3:
        feats['ret_3d'] = (df.iloc[loc]['close'] / df.iloc[loc-3]['close'] - 1) * 100
    if loc >= 5:
        feats['ret_5d'] = (df.iloc[loc]['close'] / df.iloc[loc-5]['close'] - 1) * 100
    
    feats['signal_gap_bps'] = df.iloc[loc]['gap_bps']
    
    # close vs 3-day high
    high_3d = df.iloc[loc-2:loc+1]['high'].max()
    feats['close_vs_high_3d'] = (close_now / high_3d - 1) * 100 if high_3d > 0 else np.nan
    
    # volume trend 5d
    if loc >= 5:
        vols = df.iloc[loc-4:loc+1]['volume'].values.astype(float)
        x = np.arange(5)
        try:
            slope = np.polyfit(x, vols, 1)[0]
            feats['vol_trend_5d'] = slope / vols.mean() * 100 if vols.mean() > 0 else np.nan
        except:
            feats['vol_trend_5d'] = np.nan
    
    # Option-relevant
    feats['implied_move_proxy'] = atr_now / close_now * 100 if close_now > 0 else np.nan
    feats['days_to_expiry_estimate'] = 30
    
    new_features.append(feats)

if missing_symbols:
    print(f"  Missing symbol files: {missing_symbols}")

new_df = pd.DataFrame(new_features, index=merged.index)
merged = pd.concat([merged, new_df], axis=1)

# Drop rows where we couldn't compute features
valid_mask = merged['atr_change_1d'].notna()
merged = merged[valid_mask].copy()
print(f"  Trades with valid features: {len(merged)}")

###############################################################################
# Identify entry features (pre-entry only)
###############################################################################
entry_cols = [c for c in enr.columns if c.startswith('entry_')]
spy_entry_cols = [c for c in enr.columns if c.startswith('spy_entry_')]
new_feat_cols = ['atr_change_1d', 'atr_change_3d', 'atr_change_5d',
                 'atr_percentile_20d', 'atr_percentile_60d', 'atr_vs_avg',
                 'rvol_change_5d', 'range_vs_atr', 'avg_range_3d',
                 'ret_1d', 'ret_3d', 'ret_5d', 'signal_gap_bps',
                 'close_vs_high_3d', 'vol_trend_5d', 'implied_move_proxy']

# Also include z_score from enriched
extra_cols = ['z_score']

all_features = entry_cols + spy_entry_cols + new_feat_cols + extra_cols
# Remove any that don't exist
all_features = [f for f in all_features if f in merged.columns]
# Remove duplicates
all_features = list(dict.fromkeys(all_features))
# Keep only numeric columns
all_features = [f for f in all_features if pd.api.types.is_numeric_dtype(merged[f])]

print(f"  Total features to test: {len(all_features)}")

###############################################################################
# STEP 2: Test every feature against option_return_pct
###############################################################################
print("\nStep 2: Testing all features...")

target = 'option_return_pct'
stock_target = 'stock_return_pct'

results = []
quintile_data = {}

for feat in all_features:
    vals = merged[[feat, target, stock_target]].dropna()
    if len(vals) < 30:
        continue
    
    # Spearman correlation
    sp_corr, sp_p = stats.spearmanr(vals[feat], vals[target])
    sp_corr_stock, sp_p_stock = stats.spearmanr(vals[feat], vals[stock_target])
    
    # Cohen's d: winners vs losers
    winners = vals[vals[target] > 0][feat]
    losers = vals[vals[target] <= 0][feat]
    if len(winners) > 5 and len(losers) > 5:
        pooled_std = np.sqrt((winners.var() * (len(winners)-1) + losers.var() * (len(losers)-1)) / 
                             (len(winners) + len(losers) - 2))
        cohens_d = (winners.mean() - losers.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0
    
    # Quintile analysis
    try:
        vals['quintile'] = pd.qcut(vals[feat], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
    except:
        continue
    
    q_stats = vals.groupby('quintile').agg(
        N=(target, 'count'),
        mean_opt_ret=(target, 'mean'),
        win_rate=(target, lambda x: (x > 0).mean() * 100),
        total_pnl=(target, 'sum'),
        mean_stock_ret=(stock_target, 'mean')
    )
    
    if len(q_stats) < 4:
        continue
    
    # Quintile spread
    q_spread = q_stats['mean_opt_ret'].iloc[-1] - q_stats['mean_opt_ret'].iloc[0]
    wr_spread = q_stats['win_rate'].iloc[-1] - q_stats['win_rate'].iloc[0]
    
    flagged = abs(sp_corr) > 0.08 or abs(cohens_d) > 0.15 or abs(q_spread) > 50
    
    results.append({
        'feature': feat,
        'spearman_r': sp_corr,
        'spearman_p': sp_p,
        'spearman_r_stock': sp_corr_stock,
        'cohens_d': cohens_d,
        'q_spread_opt_ret': q_spread,
        'q_spread_win_rate': wr_spread,
        'n': len(vals),
        'flagged': flagged
    })
    quintile_data[feat] = q_stats

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('spearman_r', key=abs, ascending=False)

print(f"  Features tested: {len(results_df)}")
print(f"  Features flagged: {results_df['flagged'].sum()}")

###############################################################################
# BUILD REPORT
###############################################################################
print("\nBuilding report...")

report = []
report.append("=" * 100)
report.append("OPTION RETURN PREDICTORS REPORT")
report.append("30-Delta 1-Month Call, Held ~13 Days")
report.append(f"Total trades analyzed: {len(merged)}")
report.append(f"Total features tested: {len(results_df)}")
report.append(f"Date range: {merged['entry_date'].min()} to {merged['entry_date'].max()}")
report.append("=" * 100)

# Summary stats
report.append("")
report.append("BASELINE STATISTICS")
report.append("-" * 60)
report.append(f"  Mean option return:  {merged[target].mean():.2f}%")
report.append(f"  Median option return: {merged[target].median():.2f}%")
report.append(f"  Std dev:             {merged[target].std():.2f}%")
report.append(f"  Win rate:            {(merged[target] > 0).mean()*100:.1f}%")
report.append(f"  Mean stock return:   {merged[stock_target].mean():.2f}%")
report.append(f"  Stock win rate:      {(merged[stock_target] > 0).mean()*100:.1f}%")
report.append(f"  Skewness:            {merged[target].skew():.2f}")

###############################################################################
# STEP 2 OUTPUT: All features ranked
###############################################################################
report.append("")
report.append("=" * 100)
report.append("STEP 2: ALL FEATURES RANKED BY |SPEARMAN CORRELATION| WITH OPTION RETURN")
report.append("=" * 100)
report.append(f"{'Feature':<40} {'Spear_r':>8} {'p-val':>8} {'Cohen_d':>8} {'Q_spread':>9} {'WR_spread':>10} {'Flag':>5}")
report.append("-" * 100)

for _, r in results_df.iterrows():
    flag = "***" if r['flagged'] else ""
    report.append(f"{r['feature']:<40} {r['spearman_r']:>8.4f} {r['spearman_p']:>8.4f} {r['cohens_d']:>8.3f} {r['q_spread_opt_ret']:>8.1f}pp {r['q_spread_win_rate']:>9.1f}pp {flag:>5}")

# Show quintile detail for flagged features
flagged_feats = results_df[results_df['flagged']].sort_values('spearman_r', key=abs, ascending=False)
report.append("")
report.append("=" * 100)
report.append("QUINTILE DETAIL FOR FLAGGED FEATURES")
report.append("=" * 100)

for _, r in flagged_feats.iterrows():
    feat = r['feature']
    if feat not in quintile_data:
        continue
    qs = quintile_data[feat]
    report.append("")
    report.append(f"--- {feat} (r={r['spearman_r']:.4f}, d={r['cohens_d']:.3f}) ---")
    report.append(f"  {'Quintile':<8} {'N':>5} {'MeanOptRet%':>12} {'WinRate%':>10} {'TotalPnL':>10} {'MeanStkRet%':>12}")
    for q_name, q_row in qs.iterrows():
        report.append(f"  {str(q_name):<8} {q_row['N']:>5.0f} {q_row['mean_opt_ret']:>11.2f}% {q_row['win_rate']:>9.1f}% {q_row['total_pnl']:>10.1f} {q_row['mean_stock_ret']:>11.2f}%")

###############################################################################
# STEP 3: Option vs Stock correlation comparison
###############################################################################
report.append("")
report.append("=" * 100)
report.append("STEP 3: TOP 20 FEATURES - OPTION vs STOCK CORRELATION")
report.append("Identifies features that predict OPTIONS differently than STOCKS")
report.append("=" * 100)

top20 = results_df.head(20).copy()
top20['opt_minus_stock'] = top20['spearman_r'].abs() - top20['spearman_r_stock'].abs()
top20 = top20.sort_values('opt_minus_stock', ascending=False)

report.append(f"{'Feature':<40} {'Opt_corr':>9} {'Stk_corr':>9} {'Diff':>9} {'OptSpecific':>12}")
report.append("-" * 85)
for _, r in top20.iterrows():
    opt_spec = "OPTION-SPEC" if r['opt_minus_stock'] > 0.03 else ("STOCK-SPEC" if r['opt_minus_stock'] < -0.03 else "SIMILAR")
    report.append(f"{r['feature']:<40} {r['spearman_r']:>9.4f} {r['spearman_r_stock']:>9.4f} {r['opt_minus_stock']:>9.4f} {opt_spec:>12}")

###############################################################################
# STEP 4: Interaction effects
###############################################################################
report.append("")
report.append("=" * 100)
report.append("STEP 4: INTERACTION EFFECTS - TOP 5 FEATURES IN PAIRS")
report.append("=" * 100)

top5_feats = results_df.head(5)['feature'].tolist()
report.append(f"Top 5 features: {top5_feats}")
report.append("")

for i in range(len(top5_feats)):
    for j in range(i+1, len(top5_feats)):
        f1, f2 = top5_feats[i], top5_feats[j]
        sub = merged[[f1, f2, target, stock_target]].dropna()
        if len(sub) < 50:
            continue
        
        # Split each feature at median
        m1 = sub[f1].median()
        m2 = sub[f2].median()
        
        combos = {
            f'{f1[:18]}_HI + {f2[:18]}_HI': sub[(sub[f1] >= m1) & (sub[f2] >= m2)],
            f'{f1[:18]}_HI + {f2[:18]}_LO': sub[(sub[f1] >= m1) & (sub[f2] < m2)],
            f'{f1[:18]}_LO + {f2[:18]}_HI': sub[(sub[f1] < m1) & (sub[f2] >= m2)],
            f'{f1[:18]}_LO + {f2[:18]}_LO': sub[(sub[f1] < m1) & (sub[f2] < m2)],
        }
        
        report.append(f"--- {f1} x {f2} ---")
        report.append(f"  {'Combo':<45} {'N':>5} {'MeanOptRet%':>12} {'WinRate%':>10} {'MeanStkRet%':>12}")
        
        any_strong = False
        for label, grp in combos.items():
            if len(grp) == 0:
                continue
            wr = (grp[target] > 0).mean() * 100
            mr = grp[target].mean()
            sr = grp[stock_target].mean()
            flag = " <<<" if wr > 60 or mr > 100 else ""
            if wr > 60 or mr > 100:
                any_strong = True
            report.append(f"  {label:<45} {len(grp):>5} {mr:>11.2f}% {wr:>9.1f}% {sr:>11.2f}%{flag}")
        
        if any_strong:
            report.append("  *** STRONG INTERACTION FOUND ***")
        report.append("")

###############################################################################
# STEP 5: ATR DEEP DIVE
###############################################################################
report.append("")
report.append("=" * 100)
report.append("STEP 5: ATR DEEP DIVE")
report.append("=" * 100)

atr_feats = ['entry_atr_14', 'entry_bb_bandwidth', 'atr_change_1d', 'atr_change_3d', 'atr_change_5d',
             'atr_percentile_20d', 'atr_percentile_60d', 'atr_vs_avg',
             'range_vs_atr', 'avg_range_3d', 'implied_move_proxy',
             'entry_rvol_20d', 'entry_rvol_10d', 'rvol_change_5d']
atr_feats = [f for f in atr_feats if f in merged.columns]

for feat in atr_feats:
    vals = merged[[feat, target, stock_target]].dropna()
    if len(vals) < 30:
        continue
    
    try:
        vals['quintile'] = pd.qcut(vals[feat], 5, labels=['Q1(low)','Q2','Q3','Q4','Q5(high)'], duplicates='drop')
    except:
        continue
    
    qs = vals.groupby('quintile').agg(
        N=(target, 'count'),
        mean_opt_ret=(target, 'mean'),
        median_opt_ret=(target, 'median'),
        win_rate=(target, lambda x: (x > 0).mean() * 100),
        std_opt_ret=(target, 'std'),
        mean_stock_ret=(stock_target, 'mean'),
        feat_mean=(feat, 'mean'),
        feat_min=(feat, 'min'),
        feat_max=(feat, 'max')
    )
    
    sp_corr, _ = stats.spearmanr(vals[feat], vals[target])
    
    report.append("")
    report.append(f"--- {feat} (Spearman r = {sp_corr:.4f}) ---")
    report.append(f"  {'Quintile':<12} {'N':>4} {'FeatRange':>25} {'MeanOptRet%':>12} {'MedOptRet%':>11} {'WinRate%':>9} {'StdOptRet':>10} {'MeanStkRet%':>12}")
    for q_name, q_row in qs.iterrows():
        frange = f"[{q_row['feat_min']:.3f}, {q_row['feat_max']:.3f}]"
        report.append(f"  {str(q_name):<12} {q_row['N']:>4.0f} {frange:>25} {q_row['mean_opt_ret']:>11.2f}% {q_row['median_opt_ret']:>10.2f}% {q_row['win_rate']:>8.1f}% {q_row['std_opt_ret']:>10.2f} {q_row['mean_stock_ret']:>11.2f}%")

# ATR direction analysis
report.append("")
report.append("--- ATR Direction Summary: Compression vs Expansion ---")
report.append(f"  {'Feature':<25} {'Declining(<0) OptRet%':>22} {'Rising(>0) OptRet%':>20} {'Declining WR%':>14} {'Rising WR%':>11}")
for feat in ['atr_change_1d', 'atr_change_3d', 'atr_change_5d']:
    if feat not in merged.columns:
        continue
    vals = merged[[feat, target]].dropna()
    dec = vals[vals[feat] < 0]
    ris = vals[vals[feat] >= 0]
    if len(dec) > 5 and len(ris) > 5:
        report.append(f"  {feat:<25} {dec[target].mean():>21.2f}% {ris[target].mean():>19.2f}% {(dec[target]>0).mean()*100:>13.1f}% {(ris[target]>0).mean()*100:>10.1f}%")

# Speed of ATR change
report.append("")
report.append("--- Speed of ATR Change: Does Timeframe Matter? ---")
for feat in ['atr_change_1d', 'atr_change_3d', 'atr_change_5d']:
    if feat not in merged.columns:
        continue
    sp, p = stats.spearmanr(merged[feat].dropna(), merged.loc[merged[feat].notna(), target])
    report.append(f"  {feat}: Spearman r = {sp:.4f} (p = {p:.4f})")

###############################################################################
# STEP 6: Regime Interaction - Year-by-year stability
###############################################################################
report.append("")
report.append("=" * 100)
report.append("STEP 6: REGIME INTERACTION - TOP 3 FEATURES YEAR-BY-YEAR")
report.append("=" * 100)

merged['year'] = pd.to_datetime(merged['entry_date']).dt.year
years = sorted(merged['year'].unique())
top3_feats = results_df.head(3)['feature'].tolist()

for feat in top3_feats:
    report.append("")
    report.append(f"--- {feat} ---")
    report.append(f"  {'Year':>6} {'N':>5} {'Spearman_r':>11} {'p-val':>8} {'Q1_OptRet%':>12} {'Q5_OptRet%':>12} {'Spread':>10}")
    
    for yr in years:
        sub = merged[merged['year'] == yr][[feat, target]].dropna()
        if len(sub) < 15:
            report.append(f"  {yr:>6} {len(sub):>5} {'(too few)':>11}")
            continue
        
        sp, p = stats.spearmanr(sub[feat], sub[target])
        
        try:
            sub['q'] = pd.qcut(sub[feat], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
            q1_ret = sub[sub['q'] == 'Q1'][target].mean()
            q5_ret = sub[sub['q'] == 'Q5'][target].mean()
            spread = q5_ret - q1_ret
        except:
            q1_ret = q5_ret = spread = np.nan
        
        report.append(f"  {yr:>6} {len(sub):>5} {sp:>11.4f} {p:>8.4f} {q1_ret:>11.2f}% {q5_ret:>11.2f}% {spread:>9.1f}pp")
    
    # Overall
    sub = merged[[feat, target]].dropna()
    sp, p = stats.spearmanr(sub[feat], sub[target])
    report.append(f"  {'ALL':>6} {len(sub):>5} {sp:>11.4f} {p:>8.4f}")

###############################################################################
# FINAL SUMMARY
###############################################################################
report.append("")
report.append("=" * 100)
report.append("EXECUTIVE SUMMARY")
report.append("=" * 100)

# Top positive predictors
report.append("")
report.append("TOP 10 POSITIVE PREDICTORS (higher feature -> higher option return):")
pos = results_df[results_df['spearman_r'] > 0].head(10)
for i, (_, r) in enumerate(pos.iterrows(), 1):
    report.append(f"  {i:>2}. {r['feature']:<40} r={r['spearman_r']:.4f}  d={r['cohens_d']:.3f}  Q-spread={r['q_spread_opt_ret']:.1f}pp")

report.append("")
report.append("TOP 10 NEGATIVE PREDICTORS (higher feature -> lower option return):")
neg = results_df[results_df['spearman_r'] < 0].sort_values('spearman_r').head(10)
for i, (_, r) in enumerate(neg.iterrows(), 1):
    report.append(f"  {i:>2}. {r['feature']:<40} r={r['spearman_r']:.4f}  d={r['cohens_d']:.3f}  Q-spread={r['q_spread_opt_ret']:.1f}pp")

# Option-specific predictors
report.append("")
report.append("OPTION-SPECIFIC PREDICTORS (predict options better than stocks):")
top20_check = results_df.head(20).copy()
top20_check['opt_edge'] = top20_check['spearman_r'].abs() - top20_check['spearman_r_stock'].abs()
opt_spec = top20_check[top20_check['opt_edge'] > 0.02].sort_values('opt_edge', ascending=False)
for _, r in opt_spec.iterrows():
    report.append(f"  {r['feature']:<40} opt_r={r['spearman_r']:.4f}  stk_r={r['spearman_r_stock']:.4f}  edge={r['opt_edge']:.4f}")

# ATR summary
report.append("")
report.append("ATR FINDINGS:")
for feat in ['atr_change_1d', 'atr_change_3d', 'atr_change_5d']:
    if feat in results_df['feature'].values:
        row = results_df[results_df['feature'] == feat].iloc[0]
        report.append(f"  {feat}: r={row['spearman_r']:.4f}, Q-spread={row['q_spread_opt_ret']:.1f}pp")

# Best interaction
report.append("")
report.append("KEY INTERACTION EFFECTS: See Step 4 for combinations yielding >60% win rate or >100% avg return")

# Write report
report_text = '\n'.join(report)
with open('/home/ubuntu/daily_data/analysis_results/option_predictors_report.txt', 'w') as f:
    f.write(report_text)

print("\nReport saved.")
print(report_text)
