import pandas as pd
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD AND JOIN DATA
# ============================================================
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_atr_stop_trades.csv')
enriched = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_enriched.csv')

# Join on symbol + entry_date
merged = trades.merge(enriched, on=['symbol', 'entry_date'], suffixes=('', '_enr'), how='inner')
print(f"ATR trades: {len(trades)}, Enriched: {len(enriched)}, Merged: {len(merged)}")

# Use ATR-stop outcomes
merged['win'] = (merged['net_pnl'] > 0).astype(int)
merged['year'] = pd.to_datetime(merged['entry_date']).dt.year

# Identify entry-time features (only entry_* and spy_entry_* from enriched, plus z_score)
entry_features = [c for c in enriched.columns if c.startswith('entry_') or c.startswith('spy_entry_')]
entry_features.append('z_score')
# Remove non-numeric
entry_features = [f for f in entry_features if merged[f].dtype in ['float64', 'int64', 'float32', 'int32']]
# Remove price-level features (SMAs, prices) - keep ratios/pcts/indicators
price_level_cols = [c for c in entry_features if any(x in c for x in ['_sma_8', '_sma_12', '_sma_20', '_sma_50', '_sma_100', '_sma_200',
                                                                        '_ema_8', '_ema_12', '_ema_20', '_ema_50',
                                                                        'spy_entry_close', 'spy_entry_sma_50', 'spy_entry_sma_200',
                                                                        'entry_atr_14'])]
# Actually keep all - the quintile analysis will still work on level features
# Just note that price levels are less useful as filters
print(f"Entry features to analyze: {len(entry_features)}")

# ============================================================
# STEP 2: QUINTILE ANALYSIS FOR EVERY FEATURE
# ============================================================
report = []
report.append("=" * 90)
report.append("TRADE 1 FILTER & SIZING ANALYSIS")
report.append("=" * 90)
report.append(f"\nBASELINE: {len(merged)} trades, {merged['win'].mean()*100:.1f}% win rate, "
              f"${merged['net_pnl'].sum():,.0f} total P&L, "
              f"${merged['net_pnl'].mean():,.0f} avg P&L/trade")
report.append(f"Profit Factor: {merged[merged['net_pnl']>0]['net_pnl'].sum() / abs(merged[merged['net_pnl']<=0]['net_pnl'].sum()):.2f}")
report.append("")

quintile_rows = []
flagged_features = []

for feat in entry_features:
    col = merged[feat].dropna()
    if len(col) < 100:
        continue
    try:
        merged['_q'] = pd.qcut(merged[feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    except:
        continue

    if merged['_q'].nunique() < 4:
        continue

    for q in ['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)']:
        subset = merged[merged['_q'] == q]
        if len(subset) < 20:
            continue
        quintile_rows.append({
            'feature': feat,
            'quintile': q,
            'n_trades': len(subset),
            'win_rate': subset['win'].mean() * 100,
            'mean_return_pct': subset['return_pct'].mean() if 'return_pct' in subset.columns else subset['net_pnl'].mean() / 1e6 * 100,
            'total_pnl': subset['net_pnl'].sum(),
            'mean_pnl': subset['net_pnl'].mean()
        })

    # Check flag conditions
    q_stats = merged.groupby('_q').agg(
        n=('win', 'count'),
        wr=('win', 'mean'),
        total_pnl=('net_pnl', 'sum'),
        mean_pnl=('net_pnl', 'mean'),
        mean_ret=('return_pct', 'mean')
    ).reindex(['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])

    top_wr = q_stats.loc['Q5(high)', 'wr'] if 'Q5(high)' in q_stats.index else 0
    bot_wr = q_stats.loc['Q1(low)', 'wr'] if 'Q1(low)' in q_stats.index else 1
    top2_wr = q_stats.loc['Q4', 'wr'] if 'Q4' in q_stats.index else 0
    bot2_wr = q_stats.loc['Q2', 'wr'] if 'Q2' in q_stats.index else 1

    spread = top_wr - bot_wr
    monotonic_score = 0
    vals = q_stats['wr'].values
    for i in range(1, len(vals)):
        if vals[i] > vals[i-1]:
            monotonic_score += 1

    flagged = False
    flag_reason = []
    if top_wr > 0.58:
        flag_reason.append(f"Top Q WR={top_wr*100:.1f}%")
        flagged = True
    if bot_wr < 0.44:
        flag_reason.append(f"Bot Q WR={bot_wr*100:.1f}%")
        flagged = True
    if spread > 0.12:
        flag_reason.append(f"Spread={spread*100:.1f}pp")
        flagged = True

    if flagged:
        flagged_features.append({
            'feature': feat,
            'bot_q_wr': bot_wr,
            'q2_wr': bot2_wr,
            'q4_wr': top2_wr,
            'top_q_wr': top_wr,
            'spread': spread,
            'monotonic': monotonic_score,
            'reason': ', '.join(flag_reason)
        })

merged.drop(columns=['_q'], errors='ignore', inplace=True)

# Save quintile CSV
qdf = pd.DataFrame(quintile_rows)
qdf.to_csv('/home/ubuntu/daily_data/analysis_results/filter_quintiles.csv', index=False)

# Print flagged features
flagged_df = pd.DataFrame(flagged_features).sort_values('spread', ascending=False)
report.append("\n" + "=" * 90)
report.append("STEP 2: FLAGGED FEATURES (top Q WR > 58% or bottom Q WR < 44% or spread > 12pp)")
report.append("=" * 90)
report.append(f"\n{'Feature':<35} {'BotQ WR':>8} {'Q2 WR':>8} {'Q4 WR':>8} {'TopQ WR':>8} {'Spread':>8} {'Mono':>5}  Reason")
report.append("-" * 120)
for _, row in flagged_df.iterrows():
    report.append(f"{row['feature']:<35} {row['bot_q_wr']*100:>7.1f}% {row['q2_wr']*100:>7.1f}% {row['q4_wr']*100:>7.1f}% {row['top_q_wr']*100:>7.1f}% {row['spread']*100:>7.1f}pp {row['monotonic']:>5}  {row['reason']}")

# Print quintile detail for flagged features
report.append("\n\nDETAILED QUINTILE TABLES FOR FLAGGED FEATURES:")
report.append("-" * 90)
for _, frow in flagged_df.iterrows():
    feat = frow['feature']
    sub = qdf[qdf['feature'] == feat]
    report.append(f"\n  {feat}  (spread={frow['spread']*100:.1f}pp, monotonic={frow['monotonic']}/4)")
    report.append(f"  {'Quintile':<12} {'N':>6} {'WinRate':>8} {'MeanRet%':>10} {'TotalPnL':>14} {'MeanPnL':>12}")
    for _, r in sub.iterrows():
        report.append(f"  {r['quintile']:<12} {r['n_trades']:>6.0f} {r['win_rate']:>7.1f}% {r['mean_return_pct']:>9.2f}% {r['total_pnl']:>13,.0f} {r['mean_pnl']:>11,.0f}")

print(f"\nFlagged {len(flagged_df)} features")

# ============================================================
# STEP 3: TEST FILTERS - skip worst quintile(s)
# ============================================================
report.append("\n\n" + "=" * 90)
report.append("STEP 3: FILTER SIMULATIONS — SKIP WORST QUINTILE(S)")
report.append("=" * 90)

def calc_metrics(df, label=""):
    """Calculate standard metrics for a trade set."""
    n = len(df)
    if n == 0:
        return None
    wr = df['win'].mean() * 100
    total_pnl = df['net_pnl'].sum()
    mean_pnl = df['net_pnl'].mean()
    winners = df[df['net_pnl'] > 0]['net_pnl'].sum()
    losers = abs(df[df['net_pnl'] <= 0]['net_pnl'].sum())
    pf = winners / losers if losers > 0 else float('inf')

    # Max drawdown (cumulative P&L)
    cum = df.sort_values('entry_date')['net_pnl'].cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()

    return {'label': label, 'n': n, 'wr': wr, 'total_pnl': total_pnl, 'mean_pnl': mean_pnl,
            'profit_factor': pf, 'max_dd': dd}

baseline = calc_metrics(merged, "BASELINE")

filter_results = []

# For each flagged feature, test skipping Q1 and Q1+Q2
for _, frow in flagged_df.iterrows():
    feat = frow['feature']
    try:
        merged['_q'] = pd.qcut(merged[feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    except:
        continue

    # Determine direction: if bot WR < top WR, skip low quintiles; else skip high
    if frow['bot_q_wr'] < frow['top_q_wr']:
        # Skip lowest
        skip1 = merged[merged['_q'] != 'Q1(low)']
        skip2 = merged[~merged['_q'].isin(['Q1(low)', 'Q2'])]
        dir_label = "skip LOW"
    else:
        # Skip highest
        skip1 = merged[merged['_q'] != 'Q5(high)']
        skip2 = merged[~merged['_q'].isin(['Q5(high)', 'Q4'])]
        dir_label = "skip HIGH"

    m1 = calc_metrics(skip1, f"{feat} — skip worst Q")
    m2 = calc_metrics(skip2, f"{feat} — skip worst 2Q")

    if m1:
        m1['feature'] = feat
        m1['direction'] = dir_label
        m1['filter_level'] = 'skip_1Q'
        filter_results.append(m1)
    if m2:
        m2['feature'] = feat
        m2['direction'] = dir_label
        m2['filter_level'] = 'skip_2Q'
        filter_results.append(m2)

merged.drop(columns=['_q'], errors='ignore', inplace=True)

# Sort by mean_pnl improvement
filter_df = pd.DataFrame(filter_results)
filter_df['pnl_per_trade_lift'] = filter_df['mean_pnl'] - baseline['mean_pnl']
filter_df = filter_df.sort_values('pnl_per_trade_lift', ascending=False)

report.append(f"\nBaseline: N={baseline['n']}, WR={baseline['wr']:.1f}%, PnL=${baseline['total_pnl']:,.0f}, "
              f"PnL/trade=${baseline['mean_pnl']:,.0f}, PF={baseline['profit_factor']:.2f}, MaxDD=${baseline['max_dd']:,.0f}")
report.append(f"\n{'Filter':<45} {'N':>6} {'WR':>7} {'TotalPnL':>14} {'PnL/trade':>11} {'PF':>6} {'MaxDD':>14} {'Lift/trade':>11}")
report.append("-" * 130)
for _, r in filter_df.iterrows():
    report.append(f"{r['label']:<45} {r['n']:>6} {r['wr']:>6.1f}% ${r['total_pnl']:>12,.0f} ${r['mean_pnl']:>9,.0f} {r['profit_factor']:>5.2f} ${r['max_dd']:>12,.0f} ${r['pnl_per_trade_lift']:>9,.0f}")

# ============================================================
# STEP 4: COMBINATION FILTERS
# ============================================================
report.append("\n\n" + "=" * 90)
report.append("STEP 4: COMBINATION FILTERS — TOP 2-3 FEATURES TOGETHER")
report.append("=" * 90)

# Pick the top features by PnL/trade lift (skip_1Q variants)
top_single_filters = filter_df[filter_df['filter_level'] == 'skip_1Q'].head(10)
top_features = top_single_filters['feature'].tolist()[:6]  # top 6

# For each feature, create the boolean mask for "keep" (skip worst Q)
masks = {}
for feat in top_features:
    frow = flagged_df[flagged_df['feature'] == feat].iloc[0]
    try:
        merged['_q'] = pd.qcut(merged[feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    except:
        continue
    if frow['bot_q_wr'] < frow['top_q_wr']:
        masks[feat] = merged['_q'] != 'Q1(low)'
    else:
        masks[feat] = merged['_q'] != 'Q5(high)'

merged.drop(columns=['_q'], errors='ignore', inplace=True)

# Test all pairs and triples
from itertools import combinations

combo_results = []
for r in [2, 3]:
    for combo in combinations(masks.keys(), r):
        combined_mask = pd.Series(True, index=merged.index)
        for f in combo:
            combined_mask &= masks[f]
        sub = merged[combined_mask]
        m = calc_metrics(sub, ' + '.join([c.replace('entry_','').replace('spy_entry_','spy_') for c in combo]))
        if m:
            m['features'] = combo
            m['n_filters'] = r
            m['pnl_per_trade_lift'] = m['mean_pnl'] - baseline['mean_pnl']
            combo_results.append(m)

combo_df = pd.DataFrame(combo_results).sort_values('pnl_per_trade_lift', ascending=False)

report.append(f"\n{'Combination':<55} {'N':>6} {'WR':>7} {'TotalPnL':>14} {'PnL/trade':>11} {'PF':>6} {'MaxDD':>14} {'Lift':>11}")
report.append("-" * 140)
for _, r in combo_df.head(20).iterrows():
    report.append(f"{r['label']:<55} {r['n']:>6} {r['wr']:>6.1f}% ${r['total_pnl']:>12,.0f} ${r['mean_pnl']:>9,.0f} {r['profit_factor']:>5.2f} ${r['max_dd']:>12,.0f} ${r['pnl_per_trade_lift']:>9,.0f}")

# Also show worst combos
report.append("\n  (Bottom 5 combos by lift):")
for _, r in combo_df.tail(5).iterrows():
    report.append(f"  {r['label']:<55} {r['n']:>6} {r['wr']:>6.1f}% ${r['total_pnl']:>12,.0f} ${r['mean_pnl']:>9,.0f} {r['profit_factor']:>5.2f} ${r['max_dd']:>12,.0f} ${r['pnl_per_trade_lift']:>9,.0f}")


# ============================================================
# STEP 5: SIZING TESTS
# ============================================================
report.append("\n\n" + "=" * 90)
report.append("STEP 5: FEATURE-BASED SIZING (vs flat $1M per trade)")
report.append("=" * 90)

sizing_features = ['z_score', 'entry_body_pct', 'entry_close_location', 'entry_daily_return_pct', 'entry_stoch_k']
# Also add top flagged features
for f in top_features[:3]:
    if f not in sizing_features:
        sizing_features.append(f)

sizing_results = []

# Baseline Sharpe
merged_sorted = merged.sort_values('entry_date')
baseline_daily_pnl = merged_sorted.groupby('entry_date')['net_pnl'].sum()
baseline_sharpe = baseline_daily_pnl.mean() / baseline_daily_pnl.std() * np.sqrt(252) if baseline_daily_pnl.std() > 0 else 0

for feat in sizing_features:
    if feat not in merged.columns:
        continue
    col = merged[feat].dropna()
    if len(col) < 100:
        continue

    try:
        merged['_q'] = pd.qcut(merged[feat], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    except:
        continue

    # Determine direction: check which quintile has better WR
    q_wr = merged.groupby('_q')['win'].mean()
    if q_wr.get('Q5', 0) > q_wr.get('Q1', 0):
        # Higher = better -> size up on high
        size_map = {'Q1': 0.5, 'Q2': 0.75, 'Q3': 1.0, 'Q4': 1.25, 'Q5': 1.5}
        direction = "higher=bigger"
    else:
        # Lower = better -> size up on low
        size_map = {'Q1': 1.5, 'Q2': 1.25, 'Q3': 1.0, 'Q4': 0.75, 'Q5': 0.5}
        direction = "lower=bigger"

    # Also test more aggressive: 0.25x to 2x
    if q_wr.get('Q5', 0) > q_wr.get('Q1', 0):
        agg_map = {'Q1': 0.25, 'Q2': 0.5, 'Q3': 1.0, 'Q4': 1.5, 'Q5': 2.0}
    else:
        agg_map = {'Q1': 2.0, 'Q2': 1.5, 'Q3': 1.0, 'Q4': 0.5, 'Q5': 0.25}

    for label, smap in [("moderate", size_map), ("aggressive", agg_map)]:
        merged['_size'] = merged['_q'].map(smap).astype(float).fillna(1.0)
        # Adjusted PnL = return_pct * size_factor * $1M (base notional)
        # Since net_pnl is based on $1M notional, scaled_pnl = net_pnl * size_factor
        merged['_scaled_pnl'] = merged['net_pnl'] * merged['_size']

        total_scaled = merged['_scaled_pnl'].sum()
        mean_scaled = merged['_scaled_pnl'].mean()
        winners_s = merged[merged['_scaled_pnl'] > 0]['_scaled_pnl'].sum()
        losers_s = abs(merged[merged['_scaled_pnl'] <= 0]['_scaled_pnl'].sum())
        pf_s = winners_s / losers_s if losers_s > 0 else float('inf')

        # Sharpe on daily aggregated scaled PnL
        daily_s = merged.sort_values('entry_date').groupby('entry_date')['_scaled_pnl'].sum()
        sharpe_s = daily_s.mean() / daily_s.std() * np.sqrt(252) if daily_s.std() > 0 else 0

        # Max DD
        cum_s = merged.sort_values('entry_date')['_scaled_pnl'].cumsum()
        dd_s = (cum_s - cum_s.cummax()).min()

        sizing_results.append({
            'feature': feat, 'direction': direction, 'aggressiveness': label,
            'total_pnl': total_scaled, 'mean_pnl': mean_scaled, 'profit_factor': pf_s,
            'sharpe': sharpe_s, 'max_dd': dd_s,
            'pnl_lift': total_scaled - baseline['total_pnl'],
            'pnl_lift_pct': (total_scaled - baseline['total_pnl']) / abs(baseline['total_pnl']) * 100
        })

    merged.drop(columns=['_q', '_size', '_scaled_pnl'], errors='ignore', inplace=True)

sizing_df = pd.DataFrame(sizing_results).sort_values('pnl_lift', ascending=False)

report.append(f"\nBaseline: TotalPnL=${baseline['total_pnl']:,.0f}, PnL/trade=${baseline['mean_pnl']:,.0f}, "
              f"PF={baseline['profit_factor']:.2f}, Sharpe={baseline_sharpe:.2f}")
report.append(f"\n{'Feature':<30} {'Dir':<15} {'Agg':<10} {'TotalPnL':>14} {'PnL/trade':>11} {'PF':>6} {'Sharpe':>7} {'MaxDD':>14} {'Lift%':>7}")
report.append("-" * 130)
for _, r in sizing_df.iterrows():
    report.append(f"{r['feature']:<30} {r['direction']:<15} {r['aggressiveness']:<10} ${r['total_pnl']:>12,.0f} ${r['mean_pnl']:>9,.0f} {r['profit_factor']:>5.2f} {r['sharpe']:>6.2f} ${r['max_dd']:>12,.0f} {r['pnl_lift_pct']:>6.1f}%")


# ============================================================
# STEP 6: BEST FILTER & BEST SIZING — YEAR-BY-YEAR
# ============================================================
report.append("\n\n" + "=" * 90)
report.append("STEP 6: YEAR-BY-YEAR PERFORMANCE — BEST FILTER vs BASELINE")
report.append("=" * 90)

# Pick best single filter (highest PnL/trade lift among skip_1Q)
best_filter_row = filter_df[filter_df['filter_level'] == 'skip_1Q'].iloc[0]
best_filter_feat = best_filter_row['feature']

# Also pick best combo
best_combo_row = combo_df.iloc[0]
best_combo_feats = best_combo_row['features']

# Also pick best sizing
best_sizing_row = sizing_df.iloc[0]
best_sizing_feat = best_sizing_row['feature']

report.append(f"\nBest single filter: {best_filter_feat} ({best_filter_row['direction']})")
report.append(f"Best combo filter: {' + '.join(best_combo_feats)}")
report.append(f"Best sizing feature: {best_sizing_feat} ({best_sizing_row['direction']}, {best_sizing_row['aggressiveness']})")

# Create filtered datasets
# Best single filter
frow = flagged_df[flagged_df['feature'] == best_filter_feat].iloc[0]
try:
    merged['_q_best'] = pd.qcut(merged[best_filter_feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
except:
    merged['_q_best'] = pd.cut(merged[best_filter_feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])

if frow['bot_q_wr'] < frow['top_q_wr']:
    filtered = merged[merged['_q_best'] != 'Q1(low)']
else:
    filtered = merged[merged['_q_best'] != 'Q5(high)']

# Best combo
combo_mask = pd.Series(True, index=merged.index)
for f in best_combo_feats:
    combo_mask &= masks[f]
combo_filtered = merged[combo_mask]

# Best sizing
try:
    merged['_q_sz'] = pd.qcut(merged[best_sizing_feat], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
except:
    merged['_q_sz'] = pd.cut(merged[best_sizing_feat], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

q_wr_sz = merged.groupby('_q_sz')['win'].mean()
if q_wr_sz.get('Q5', 0) > q_wr_sz.get('Q1', 0):
    if best_sizing_row['aggressiveness'] == 'aggressive':
        sz_map = {'Q1': 0.25, 'Q2': 0.5, 'Q3': 1.0, 'Q4': 1.5, 'Q5': 2.0}
    else:
        sz_map = {'Q1': 0.5, 'Q2': 0.75, 'Q3': 1.0, 'Q4': 1.25, 'Q5': 1.5}
else:
    if best_sizing_row['aggressiveness'] == 'aggressive':
        sz_map = {'Q1': 2.0, 'Q2': 1.5, 'Q3': 1.0, 'Q4': 0.5, 'Q5': 0.25}
    else:
        sz_map = {'Q1': 1.5, 'Q2': 1.25, 'Q3': 1.0, 'Q4': 0.75, 'Q5': 0.5}

merged['_sz_factor'] = merged['_q_sz'].map(sz_map).astype(float).fillna(1.0)
merged['_scaled_pnl'] = merged['net_pnl'] * merged['_sz_factor']

# Year-by-year
years = sorted(merged['year'].unique())

report.append(f"\n--- Year-by-Year: BASELINE ---")
report.append(f"{'Year':>6} {'N':>6} {'WR':>7} {'TotalPnL':>14} {'PnL/trade':>11} {'PF':>6}")
report.append("-" * 60)
for y in years:
    sub = merged[merged['year'] == y]
    m = calc_metrics(sub)
    report.append(f"{y:>6} {m['n']:>6} {m['wr']:>6.1f}% ${m['total_pnl']:>12,.0f} ${m['mean_pnl']:>9,.0f} {m['profit_factor']:>5.2f}")
m = calc_metrics(merged)
report.append(f"{'ALL':>6} {m['n']:>6} {m['wr']:>6.1f}% ${m['total_pnl']:>12,.0f} ${m['mean_pnl']:>9,.0f} {m['profit_factor']:>5.2f}")

report.append(f"\n--- Year-by-Year: BEST SINGLE FILTER ({best_filter_feat}) ---")
report.append(f"{'Year':>6} {'N':>6} {'WR':>7} {'TotalPnL':>14} {'PnL/trade':>11} {'PF':>6}")
report.append("-" * 60)
for y in years:
    sub = filtered[filtered['year'] == y]
    if len(sub) == 0:
        continue
    m = calc_metrics(sub)
    report.append(f"{y:>6} {m['n']:>6} {m['wr']:>6.1f}% ${m['total_pnl']:>12,.0f} ${m['mean_pnl']:>9,.0f} {m['profit_factor']:>5.2f}")
m = calc_metrics(filtered)
report.append(f"{'ALL':>6} {m['n']:>6} {m['wr']:>6.1f}% ${m['total_pnl']:>12,.0f} ${m['mean_pnl']:>9,.0f} {m['profit_factor']:>5.2f}")

report.append(f"\n--- Year-by-Year: BEST COMBO FILTER ({' + '.join([c.replace('entry_','') for c in best_combo_feats])}) ---")
report.append(f"{'Year':>6} {'N':>6} {'WR':>7} {'TotalPnL':>14} {'PnL/trade':>11} {'PF':>6}")
report.append("-" * 60)
for y in years:
    sub = combo_filtered[combo_filtered['year'] == y]
    if len(sub) == 0:
        continue
    m = calc_metrics(sub)
    report.append(f"{y:>6} {m['n']:>6} {m['wr']:>6.1f}% ${m['total_pnl']:>12,.0f} ${m['mean_pnl']:>9,.0f} {m['profit_factor']:>5.2f}")
m = calc_metrics(combo_filtered)
report.append(f"{'ALL':>6} {m['n']:>6} {m['wr']:>6.1f}% ${m['total_pnl']:>12,.0f} ${m['mean_pnl']:>9,.0f} {m['profit_factor']:>5.2f}")

report.append(f"\n--- Year-by-Year: BEST SIZING ({best_sizing_feat}, {best_sizing_row['aggressiveness']}) ---")
report.append(f"{'Year':>6} {'N':>6} {'Flat PnL':>14} {'Sized PnL':>14} {'Lift':>14} {'Lift%':>7}")
report.append("-" * 70)
for y in years:
    sub = merged[merged['year'] == y]
    flat = sub['net_pnl'].sum()
    sized = sub['_scaled_pnl'].sum()
    lift = sized - flat
    lift_pct = lift / abs(flat) * 100 if flat != 0 else 0
    report.append(f"{y:>6} {len(sub):>6} ${flat:>12,.0f} ${sized:>12,.0f} ${lift:>12,.0f} {lift_pct:>6.1f}%")
flat_all = merged['net_pnl'].sum()
sized_all = merged['_scaled_pnl'].sum()
report.append(f"{'ALL':>6} {len(merged):>6} ${flat_all:>12,.0f} ${sized_all:>12,.0f} ${sized_all-flat_all:>12,.0f} {(sized_all-flat_all)/abs(flat_all)*100:>6.1f}%")


# ============================================================
# STEP 7: OVERFITTING CHECK
# ============================================================
report.append("\n\n" + "=" * 90)
report.append("STEP 7: OVERFITTING CHECKS")
report.append("=" * 90)

# Split: 2020-2022 vs 2023-2026
early = merged[merged['year'].between(2020, 2022)]
late = merged[merged['year'].between(2023, 2026)]

early_filtered = filtered[filtered['year'].between(2020, 2022)]
late_filtered = filtered[filtered['year'].between(2023, 2026)]

report.append(f"\n--- BEST SINGLE FILTER: {best_filter_feat} ---")
report.append(f"\n  Period 1 (2020-2022):")
m_e_base = calc_metrics(early)
m_e_filt = calc_metrics(early_filtered)
report.append(f"    Baseline: N={m_e_base['n']}, WR={m_e_base['wr']:.1f}%, PnL/trade=${m_e_base['mean_pnl']:,.0f}")
if m_e_filt:
    report.append(f"    Filtered: N={m_e_filt['n']}, WR={m_e_filt['wr']:.1f}%, PnL/trade=${m_e_filt['mean_pnl']:,.0f}")
    report.append(f"    Lift: {m_e_filt['wr'] - m_e_base['wr']:+.1f}pp WR, ${m_e_filt['mean_pnl'] - m_e_base['mean_pnl']:+,.0f}/trade")

report.append(f"\n  Period 2 (2023-2026):")
m_l_base = calc_metrics(late)
m_l_filt = calc_metrics(late_filtered)
report.append(f"    Baseline: N={m_l_base['n']}, WR={m_l_base['wr']:.1f}%, PnL/trade=${m_l_base['mean_pnl']:,.0f}")
if m_l_filt:
    report.append(f"    Filtered: N={m_l_filt['n']}, WR={m_l_filt['wr']:.1f}%, PnL/trade=${m_l_filt['mean_pnl']:,.0f}")
    report.append(f"    Lift: {m_l_filt['wr'] - m_l_base['wr']:+.1f}pp WR, ${m_l_filt['mean_pnl'] - m_l_base['mean_pnl']:+,.0f}/trade")

report.append(f"\n  VERDICT: {'PASSES — effect present in both periods' if (m_e_filt and m_l_filt and m_e_filt['wr'] > m_e_base['wr'] and m_l_filt['wr'] > m_l_base['wr']) else 'FAILS — effect not stable across periods'}")

# Monotonicity check for best filter
report.append(f"\n--- MONOTONICITY CHECK: {best_filter_feat} ---")
try:
    merged['_q_mono'] = pd.qcut(merged[best_filter_feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
except:
    merged['_q_mono'] = pd.cut(merged[best_filter_feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])

q_mono = merged.groupby('_q_mono').agg(
    n=('win', 'count'),
    wr=('win', 'mean'),
    mean_pnl=('net_pnl', 'mean'),
    mean_ret=('return_pct', 'mean')
).reindex(['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])

report.append(f"  {'Quintile':<12} {'N':>6} {'WR':>8} {'MeanPnL':>12} {'MeanRet%':>10}")
mono_count = 0
prev_wr = None
for q, row in q_mono.iterrows():
    report.append(f"  {q:<12} {row['n']:>6.0f} {row['wr']*100:>7.1f}% ${row['mean_pnl']:>10,.0f} {row['mean_ret']:>9.2f}%")
    if prev_wr is not None:
        if (frow['top_q_wr'] > frow['bot_q_wr'] and row['wr'] >= prev_wr) or \
           (frow['top_q_wr'] <= frow['bot_q_wr'] and row['wr'] <= prev_wr):
            mono_count += 1
    prev_wr = row['wr']

report.append(f"\n  Monotonic steps: {mono_count}/4 — {'GOOD: mostly monotonic' if mono_count >= 3 else 'CAUTION: not fully monotonic (possible tail effect)'}")

# Check best combo in both periods
report.append(f"\n--- BEST COMBO FILTER STABILITY ---")
early_combo = combo_filtered[combo_filtered['year'].between(2020, 2022)]
late_combo = combo_filtered[combo_filtered['year'].between(2023, 2026)]
m_ec = calc_metrics(early_combo)
m_lc = calc_metrics(late_combo)
report.append(f"  2020-2022: N={m_ec['n']}, WR={m_ec['wr']:.1f}%, PnL/trade=${m_ec['mean_pnl']:,.0f}")
report.append(f"  2023-2026: N={m_lc['n']}, WR={m_lc['wr']:.1f}%, PnL/trade=${m_lc['mean_pnl']:,.0f}")
report.append(f"  VERDICT: {'PASSES' if m_ec['wr'] > m_e_base['wr'] and m_lc['wr'] > m_l_base['wr'] else 'MIXED/FAILS'}")

# Monotonicity for all flagged features (top 10)
report.append(f"\n--- MONOTONICITY SUMMARY FOR ALL FLAGGED FEATURES ---")
report.append(f"  {'Feature':<35} {'Q1 WR':>7} {'Q2 WR':>7} {'Q3 WR':>7} {'Q4 WR':>7} {'Q5 WR':>7} {'Mono':>5} {'Direction':>10}")
report.append("  " + "-" * 100)

for _, frow2 in flagged_df.iterrows():
    feat = frow2['feature']
    try:
        merged['_qt'] = pd.qcut(merged[feat], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    except:
        continue
    q_wrs = merged.groupby('_qt')['win'].mean()
    vals = [q_wrs.get(q, np.nan)*100 for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    # Count monotonic steps
    mc = 0
    direction = "higher" if frow2['top_q_wr'] > frow2['bot_q_wr'] else "lower"
    for i in range(1, len(vals)):
        if not np.isnan(vals[i]) and not np.isnan(vals[i-1]):
            if (direction == "higher" and vals[i] >= vals[i-1]) or (direction == "lower" and vals[i] <= vals[i-1]):
                mc += 1
    report.append(f"  {feat:<35} {vals[0]:>6.1f}% {vals[1]:>6.1f}% {vals[2]:>6.1f}% {vals[3]:>6.1f}% {vals[4]:>6.1f}% {mc:>5}/4 {direction:>10}")

merged.drop(columns=['_qt', '_q_mono', '_q_best', '_q_sz', '_sz_factor', '_scaled_pnl'], errors='ignore', inplace=True)


# ============================================================
# FINAL SUMMARY
# ============================================================
report.append("\n\n" + "=" * 90)
report.append("EXECUTIVE SUMMARY")
report.append("=" * 90)

report.append(f"""
BASELINE: {baseline['n']} trades, {baseline['wr']:.1f}% WR, ${baseline['total_pnl']:,.0f} total PnL,
          ${baseline['mean_pnl']:,.0f}/trade, PF={baseline['profit_factor']:.2f}

BEST SINGLE FILTER: {best_filter_feat} ({best_filter_row['direction']})
  Result: {best_filter_row['n']:.0f} trades, {best_filter_row['wr']:.1f}% WR, ${best_filter_row['total_pnl']:,.0f} total PnL,
          ${best_filter_row['mean_pnl']:,.0f}/trade, PF={best_filter_row['profit_factor']:.2f}
  Lift: {best_filter_row['wr'] - baseline['wr']:+.1f}pp WR, ${best_filter_row['pnl_per_trade_lift']:+,.0f}/trade

BEST COMBO FILTER: {' + '.join([c.replace('entry_','') for c in best_combo_feats])}
  Result: {best_combo_row['n']:.0f} trades, {best_combo_row['wr']:.1f}% WR, ${best_combo_row['total_pnl']:,.0f} total PnL,
          ${best_combo_row['mean_pnl']:,.0f}/trade, PF={best_combo_row['profit_factor']:.2f}
  Lift: {best_combo_row['wr'] - baseline['wr']:+.1f}pp WR, ${best_combo_row['pnl_per_trade_lift']:+,.0f}/trade

BEST SIZING: {best_sizing_feat} ({best_sizing_row['direction']}, {best_sizing_row['aggressiveness']})
  Result: ${best_sizing_row['total_pnl']:,.0f} total PnL (vs ${baseline['total_pnl']:,.0f} baseline)
  Lift: ${best_sizing_row['pnl_lift']:+,.0f} ({best_sizing_row['pnl_lift_pct']:+.1f}%)
  Sharpe: {best_sizing_row['sharpe']:.2f}
""")

# Write report
with open('/home/ubuntu/daily_data/analysis_results/filter_sizing_report.txt', 'w') as f:
    f.write('\n'.join(report))

print("\n\nDONE. Report saved.")
print(f"Flagged features: {len(flagged_df)}")
print(f"Quintile rows: {len(qdf)}")
print(f"\nTop 5 single filters by PnL/trade lift:")
for _, r in filter_df[filter_df['filter_level']=='skip_1Q'].head(5).iterrows():
    print(f"  {r['feature']:<35} WR={r['wr']:.1f}% PnL/trade=${r['mean_pnl']:,.0f} lift=${r['pnl_per_trade_lift']:,.0f}")
print(f"\nTop 3 combos:")
for _, r in combo_df.head(3).iterrows():
    print(f"  {r['label']:<55} WR={r['wr']:.1f}% PnL/trade=${r['mean_pnl']:,.0f} lift=${r['pnl_per_trade_lift']:,.0f}")
print(f"\nTop 3 sizing:")
for _, r in sizing_df.head(3).iterrows():
    print(f"  {r['feature']:<25} {r['aggressiveness']:<10} PnL=${r['total_pnl']:,.0f} lift={r['pnl_lift_pct']:+.1f}%")
