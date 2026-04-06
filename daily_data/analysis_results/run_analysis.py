"""
Comprehensive Winner vs Loser Pattern Analysis
No prior assumptions - let data speak.
"""
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Load ──
df = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_enriched.csv')
print(f"Loaded {len(df)} trades, {len(df.columns)} columns")

df['winner'] = (df['net_pnl'] > 0).astype(int)
winners = df[df['winner'] == 1]
losers = df[df['winner'] == 0]
overall_wr = df['winner'].mean()
print(f"Overall WR: {overall_wr:.4f}  Winners: {len(winners)}  Losers: {len(losers)}")
print(f"Mean return: {df['return_pct'].mean():.4f}%  Median: {df['return_pct'].median():.4f}%")

# ── Identify numeric columns (exclude outcome cols) ──
exclude = {'net_pnl', 'gross_pnl', 'return_pct', 'winner', 'shares'}
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
print(f"\nTesting {len(num_cols)} numeric features...")

# ── Feature-level stats ──
results = []
for col in num_cols:
    s = df[col].dropna()
    if len(s) < 50:
        continue

    w_vals = df.loc[df['winner'] == 1, col].dropna()
    l_vals = df.loc[df['winner'] == 0, col].dropna()

    if len(w_vals) < 20 or len(l_vals) < 20:
        continue

    # Cohen's d
    pooled_std = np.sqrt(((len(w_vals)-1)*w_vals.std()**2 + (len(l_vals)-1)*l_vals.std()**2) / (len(w_vals)+len(l_vals)-2))
    if pooled_std == 0:
        continue
    cohens_d = (w_vals.mean() - l_vals.mean()) / pooled_std

    # t-test
    t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False)

    # Spearman correlation with return_pct
    valid = df[[col, 'return_pct']].dropna()
    if len(valid) > 30:
        spear_r, spear_p = stats.spearmanr(valid[col], valid['return_pct'])
    else:
        spear_r, spear_p = 0, 1

    # Quintile analysis
    try:
        df['_q'] = pd.qcut(df[col].rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'])
        q_stats = df.groupby('_q').agg(
            mean_ret=('return_pct', 'mean'),
            wr=('winner', 'mean'),
            count=('winner', 'count')
        )
        q1_ret = q_stats.loc['Q1', 'mean_ret']
        q5_ret = q_stats.loc['Q5', 'mean_ret']
        q1_wr = q_stats.loc['Q1', 'wr']
        q5_wr = q_stats.loc['Q5', 'wr']
        q_spread = q5_ret - q1_ret
        wr_spread = q5_wr - q1_wr
        # Monotonicity check
        q_rets = [q_stats.loc[q, 'mean_ret'] for q in ['Q1','Q2','Q3','Q4','Q5']]
        diffs = [q_rets[i+1] - q_rets[i] for i in range(4)]
        pos_diffs = sum(1 for d in diffs if d > 0)
        monotonic_score = pos_diffs / 4  # 1.0 = perfectly monotonic increasing
        if sum(1 for d in diffs if d < 0) > pos_diffs:
            monotonic_score = -sum(1 for d in diffs if d < 0) / 4  # negative = decreasing
        df.drop('_q', axis=1, inplace=True)
    except:
        q1_ret = q5_ret = q1_wr = q5_wr = q_spread = wr_spread = monotonic_score = 0
        q_rets = [0]*5
        if '_q' in df.columns:
            df.drop('_q', axis=1, inplace=True)

    results.append({
        'feature': col,
        'winner_mean': w_vals.mean(),
        'loser_mean': l_vals.mean(),
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_val,
        'spearman_r': spear_r,
        'spearman_p': spear_p,
        'q1_mean_ret': q1_ret,
        'q2_mean_ret': q_rets[1] if len(q_rets)>1 else 0,
        'q3_mean_ret': q_rets[2] if len(q_rets)>2 else 0,
        'q4_mean_ret': q_rets[3] if len(q_rets)>3 else 0,
        'q5_mean_ret': q5_ret,
        'q1_wr': q1_wr,
        'q5_wr': q5_wr,
        'q_return_spread': q_spread,
        'q_wr_spread': wr_spread,
        'monotonic_score': monotonic_score,
        'n_valid': len(s)
    })

feat_df = pd.DataFrame(results)
feat_df['abs_d'] = feat_df['cohens_d'].abs()
feat_df['abs_t'] = feat_df['t_stat'].abs()
feat_df = feat_df.sort_values('abs_d', ascending=False)

# Save feature ranking
feat_df.to_csv('/home/ubuntu/daily_data/analysis_results/winner_loser_features.csv', index=False)
print(f"\nSaved {len(feat_df)} features to winner_loser_features.csv")

# ── Significant features ──
sig = feat_df[(feat_df['abs_d'] > 0.1) | (feat_df['abs_t'] > 3)].copy()
print(f"Significant features (|d|>0.1 or |t|>3): {len(sig)}")

# ── INTERACTION ANALYSIS ──
print("\n--- Interaction Analysis ---")
# Take top features for interaction testing
# For interactions, get top entry-time features sorted by abs effect size
entry_feats_df = feat_df[feat_df['feature'].str.startswith('entry_') | feat_df['feature'].str.startswith('spy_entry_') | feat_df['feature'].isin(['z_score'])]
entry_feats_df = entry_feats_df.sort_values('abs_d', ascending=False)
actionable = entry_feats_df.head(15)['feature'].tolist()
print(f"  Actionable features for interaction: {len(actionable)} -> {actionable[:5]}...")

interaction_results = []
for f1, f2 in combinations(actionable, 2):
    valid = df[[f1, f2, 'return_pct', 'winner']].dropna()
    if len(valid) < 100:
        continue
    med1 = valid[f1].median()
    med2 = valid[f2].median()

    # 4 quadrants
    hh = valid[(valid[f1] >= med1) & (valid[f2] >= med2)]
    hl = valid[(valid[f1] >= med1) & (valid[f2] < med2)]
    lh = valid[(valid[f1] < med1) & (valid[f2] >= med2)]
    ll = valid[(valid[f1] < med1) & (valid[f2] < med2)]

    if min(len(hh), len(hl), len(lh), len(ll)) < 20:
        continue

    rets = {
        'HH': hh['return_pct'].mean(),
        'HL': hl['return_pct'].mean(),
        'LH': lh['return_pct'].mean(),
        'LL': ll['return_pct'].mean()
    }
    wrs = {
        'HH': hh['winner'].mean(),
        'HL': hl['winner'].mean(),
        'LH': lh['winner'].mean(),
        'LL': ll['winner'].mean()
    }

    best_q = max(rets, key=rets.get)
    worst_q = min(rets, key=rets.get)
    spread = rets[best_q] - rets[worst_q]
    wr_spread = wrs[best_q] - wrs[worst_q]

    interaction_results.append({
        'feat1': f1, 'feat2': f2,
        'best_quadrant': best_q, 'worst_quadrant': worst_q,
        'best_ret': rets[best_q], 'worst_ret': rets[worst_q],
        'return_spread': spread,
        'best_wr': wrs[best_q], 'worst_wr': wrs[worst_q],
        'wr_spread': wr_spread,
        'HH_ret': rets['HH'], 'HL_ret': rets['HL'],
        'LH_ret': rets['LH'], 'LL_ret': rets['LL'],
        'HH_wr': wrs['HH'], 'HL_wr': wrs['HL'],
        'LH_wr': wrs['LH'], 'LL_wr': wrs['LL'],
        'HH_n': len(hh), 'HL_n': len(hl), 'LH_n': len(lh), 'LL_n': len(ll)
    })

if interaction_results:
    int_df = pd.DataFrame(interaction_results).sort_values('return_spread', ascending=False)
else:
    int_df = pd.DataFrame()
print(f"Tested {len(int_df)} interactions")

# ── SPY REGIME ANALYSIS ──
print("\n--- SPY Regime Analysis ---")
spy_regime_results = {}
if 'spy_entry_pct_from_52w_high' in df.columns:
    col = 'spy_entry_pct_from_52w_high'
    valid = df[col].dropna()
    # Near highs: within 5% of 52w high; Drawdown: more than 10% from high
    near_high = df[df[col] > -5]
    moderate = df[(df[col] <= -5) & (df[col] > -10)]
    drawdown = df[df[col] <= -10]

    for label, subset in [('Near 52w High (>-5%)', near_high),
                           ('Moderate (-5% to -10%)', moderate),
                           ('Drawdown (<-10%)', drawdown)]:
        if len(subset) > 10:
            spy_regime_results[label] = {
                'n': len(subset),
                'mean_ret': subset['return_pct'].mean(),
                'median_ret': subset['return_pct'].median(),
                'wr': subset['winner'].mean(),
                'avg_pnl': subset['net_pnl'].mean()
            }

# Also by SPY trend (dist from sma200)
spy_trend_results = {}
if 'spy_entry_dist_sma200_pct' in df.columns:
    col = 'spy_entry_dist_sma200_pct'
    above = df[df[col] > 0]
    below = df[df[col] <= 0]
    for label, subset in [('SPY Above 200 SMA', above), ('SPY Below 200 SMA', below)]:
        if len(subset) > 10:
            spy_trend_results[label] = {
                'n': len(subset),
                'mean_ret': subset['return_pct'].mean(),
                'median_ret': subset['return_pct'].median(),
                'wr': subset['winner'].mean(),
                'avg_pnl': subset['net_pnl'].mean()
            }

# ── SECTOR / SYMBOL ANALYSIS ──
print("\n--- Symbol Analysis ---")
sym_stats = df.groupby('symbol').agg(
    n_trades=('winner', 'count'),
    wr=('winner', 'mean'),
    mean_ret=('return_pct', 'mean'),
    median_ret=('return_pct', 'median'),
    total_pnl=('net_pnl', 'sum'),
    mean_pnl=('net_pnl', 'mean')
).sort_values('mean_ret', ascending=False)
sym_stats_min = sym_stats[sym_stats['n_trades'] >= 5]
print(f"Symbols with >= 5 trades: {len(sym_stats_min)}")

# ── TIME PATTERNS ──
print("\n--- Time Pattern Analysis ---")
df['entry_date_parsed'] = pd.to_datetime(df['entry_date'], errors='coerce')
df['entry_month'] = df['entry_date_parsed'].dt.month
df['entry_quarter'] = df['entry_date_parsed'].dt.quarter
df['entry_year'] = df['entry_date_parsed'].dt.year
df['entry_dow'] = df['entry_date_parsed'].dt.dayofweek  # 0=Mon

time_results = {}
for label, col in [('Month', 'entry_month'), ('Quarter', 'entry_quarter'),
                     ('Year', 'entry_year'), ('Day of Week', 'entry_dow')]:
    grp = df.groupby(col).agg(
        n=('winner', 'count'),
        wr=('winner', 'mean'),
        mean_ret=('return_pct', 'mean'),
        median_ret=('return_pct', 'median'),
        total_pnl=('net_pnl', 'sum')
    )
    time_results[label] = grp

# ── DAYS HELD analysis ──
days_held_results = {}
if 'days_held' in df.columns:
    for label, lo, hi in [('1-3 days', 1, 3), ('4-7 days', 4, 7), ('8-14 days', 8, 14), ('15-20 days', 15, 20), ('20+ days', 21, 999)]:
        subset = df[(df['days_held'] >= lo) & (df['days_held'] <= hi)]
        if len(subset) > 5:
            days_held_results[label] = {
                'n': len(subset),
                'wr': subset['winner'].mean(),
                'mean_ret': subset['return_pct'].mean(),
                'median_ret': subset['return_pct'].median()
            }

# ═════════════════════════════════════════════
# GENERATE REPORT
# ═════════════════════════════════════════════
lines = []
def w(s=''):
    lines.append(s)

w("=" * 90)
w("WINNER vs LOSER PATTERN ANALYSIS")
w("=" * 90)
w(f"Data: trade1_20d_enriched.csv")
w(f"Total trades: {len(df)}")
w(f"Winners: {len(winners)} ({overall_wr:.1%})   Losers: {len(losers)} ({1-overall_wr:.1%})")
w(f"Mean return: {df['return_pct'].mean():.4f}%   Median: {df['return_pct'].median():.4f}%")
w(f"Mean net PnL: ${df['net_pnl'].mean():.2f}   Total: ${df['net_pnl'].sum():,.2f}")
w(f"Features tested: {len(feat_df)}")
w(f"Significant features (|Cohen's d|>0.1 or |t|>3): {len(sig)}")
w()

# ── SECTION 1: Top Features ──
w("=" * 90)
w("SECTION 1: TOP FEATURES BY EFFECT SIZE (Cohen's d)")
w("=" * 90)
w()
w("Positive d = winners have HIGHER values; Negative d = winners have LOWER values")
w()
w(f"{'Feature':<35} {'d':>7} {'t-stat':>8} {'p-value':>10} {'Spear r':>8} {'Q1→Q5 ret':>11} {'Q1→Q5 WR':>10} {'Mono':>5}")
w("-" * 96)

for _, r in sig.head(40).iterrows():
    mono_str = f"{r['monotonic_score']:+.2f}"
    w(f"{r['feature']:<35} {r['cohens_d']:>+7.3f} {r['t_stat']:>8.2f} {r['p_value']:>10.2e} {r['spearman_r']:>+8.3f} "
      f"{r['q1_mean_ret']:>+5.2f}→{r['q5_mean_ret']:>+5.2f} "
      f"{r['q1_wr']:>4.1%}→{r['q5_wr']:>4.1%} {mono_str:>5}")

w()
w("Interpretation: 'Mono' = monotonicity score from -1 (decreasing Q1→Q5) to +1 (increasing).")
w("A feature is most useful if it has |d|>0.2, significant p-value, AND monotonic quintile pattern.")

# ── SECTION 2: Detailed Quintile Breakdowns for top features ──
w()
w("=" * 90)
w("SECTION 2: QUINTILE BREAKDOWNS — TOP 15 FEATURES")
w("=" * 90)

for _, r in sig.head(15).iterrows():
    col = r['feature']
    try:
        df['_q'] = pd.qcut(df[col].rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'])
        q_stats = df.groupby('_q').agg(
            mean_ret=('return_pct', 'mean'),
            median_ret=('return_pct', 'median'),
            wr=('winner', 'mean'),
            count=('winner', 'count'),
            mean_pnl=('net_pnl', 'mean')
        )
        w()
        w(f"  {col}  (d={r['cohens_d']:+.3f}, p={r['p_value']:.2e})")
        w(f"  Winner mean: {r['winner_mean']:.4f}   Loser mean: {r['loser_mean']:.4f}")
        w(f"  {'Quintile':<8} {'N':>5} {'Mean Ret%':>10} {'Med Ret%':>10} {'WR':>7} {'Avg PnL':>10}")
        w(f"  {'-'*52}")
        for q in ['Q1','Q2','Q3','Q4','Q5']:
            row = q_stats.loc[q]
            w(f"  {q:<8} {int(row['count']):>5} {row['mean_ret']:>+10.3f} {row['median_ret']:>+10.3f} {row['wr']:>6.1%} {row['mean_pnl']:>+10.2f}")
        df.drop('_q', axis=1, inplace=True)
    except:
        if '_q' in df.columns:
            df.drop('_q', axis=1, inplace=True)

# ── SECTION 3: Entry-only features (actionable) ──
w()
w("=" * 90)
w("SECTION 3: ACTIONABLE ENTRY-TIME FEATURES ONLY")
w("=" * 90)
w()
w("These are features known at entry time (entry_ and spy_entry_ prefixes, plus z_score).")
w("Exit features are post-hoc and cannot be used for trade selection.")
w()

entry_sig = sig[sig['feature'].str.startswith('entry_') | sig['feature'].str.startswith('spy_entry_') | sig['feature'].isin(['z_score', 'days_held'])]
w(f"{'Feature':<35} {'d':>7} {'t-stat':>8} {'p-value':>10} {'Spear r':>8} {'Q spread':>9} {'Mono':>5}")
w("-" * 84)
for _, r in entry_sig.iterrows():
    w(f"{r['feature']:<35} {r['cohens_d']:>+7.3f} {r['t_stat']:>8.2f} {r['p_value']:>10.2e} {r['spearman_r']:>+8.3f} {r['q_return_spread']:>+9.3f} {r['monotonic_score']:>+5.2f}")

# ── SECTION 4: Interactions ──
w()
w("=" * 90)
w("SECTION 4: FEATURE INTERACTIONS (Top 20 by return spread)")
w("=" * 90)
w()
w("Each pair is split into 4 quadrants by median. H=above median, L=below.")
w(f"{'Feat1':<25} {'Feat2':<25} {'Best':>4} {'Worst':>5} {'Spread':>7} {'BestWR':>7} {'WorstWR':>8} {'WR Sp':>6}")
w("-" * 90)
for _, r in int_df.head(20).iterrows():
    w(f"{r['feat1']:<25} {r['feat2']:<25} {r['best_quadrant']:>4} {r['worst_quadrant']:>5} "
      f"{r['return_spread']:>+7.3f} {r['best_wr']:>6.1%} {r['worst_wr']:>7.1%} {r['wr_spread']:>+6.1%}")

# Show full detail for top 5 interactions
w()
w("--- Top 5 Interaction Details ---")
for i, (_, r) in enumerate(int_df.head(5).iterrows()):
    w()
    w(f"  Interaction #{i+1}: {r['feat1']} x {r['feat2']}")
    w(f"  {'Quadrant':<6} {'N':>5} {'Mean Ret%':>10} {'WR':>7}")
    w(f"  {'-'*30}")
    for q in ['HH', 'HL', 'LH', 'LL']:
        w(f"  {q:<6} {int(r[f'{q}_n']):>5} {r[f'{q}_ret']:>+10.3f} {r[f'{q}_wr']:>6.1%}")

# ── SECTION 5: SPY Regime ──
w()
w("=" * 90)
w("SECTION 5: SPY MARKET REGIME ANALYSIS")
w("=" * 90)
w()
w("A) SPY Distance from 52-Week High at Entry:")
w(f"  {'Regime':<30} {'N':>6} {'Mean Ret%':>10} {'Med Ret%':>10} {'WR':>7} {'Avg PnL':>10}")
w(f"  {'-'*75}")
for label, d in spy_regime_results.items():
    w(f"  {label:<30} {d['n']:>6} {d['mean_ret']:>+10.3f} {d['median_ret']:>+10.3f} {d['wr']:>6.1%} {d['avg_pnl']:>+10.2f}")

w()
w("B) SPY vs 200-day SMA at Entry:")
w(f"  {'Regime':<30} {'N':>6} {'Mean Ret%':>10} {'Med Ret%':>10} {'WR':>7} {'Avg PnL':>10}")
w(f"  {'-'*75}")
for label, d in spy_trend_results.items():
    w(f"  {label:<30} {d['n']:>6} {d['mean_ret']:>+10.3f} {d['median_ret']:>+10.3f} {d['wr']:>6.1%} {d['avg_pnl']:>+10.2f}")

# Additional SPY quintile analysis
w()
w("C) SPY Regime Quintile Detail (spy_entry_pct_from_52w_high):")
if 'spy_entry_pct_from_52w_high' in df.columns:
    try:
        df['_q'] = pd.qcut(df['spy_entry_pct_from_52w_high'].rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'])
        q_stats = df.groupby('_q').agg(
            mean_ret=('return_pct', 'mean'),
            wr=('winner', 'mean'),
            count=('winner', 'count'),
            spy_range=('spy_entry_pct_from_52w_high', lambda x: f"{x.min():.1f}% to {x.max():.1f}%")
        )
        w(f"  {'Quintile':<8} {'N':>5} {'SPY Range':>20} {'Mean Ret%':>10} {'WR':>7}")
        w(f"  {'-'*55}")
        for q in ['Q1','Q2','Q3','Q4','Q5']:
            row = q_stats.loc[q]
            w(f"  {q:<8} {int(row['count']):>5} {row['spy_range']:>20} {row['mean_ret']:>+10.3f} {row['wr']:>6.1%}")
        df.drop('_q', axis=1, inplace=True)
    except:
        if '_q' in df.columns:
            df.drop('_q', axis=1, inplace=True)

# ── SECTION 6: Symbol Analysis ──
w()
w("=" * 90)
w("SECTION 6: SYMBOL / SECTOR ANALYSIS")
w("=" * 90)
w()

# Top and bottom symbols
w(f"Symbols with >= 5 trades: {len(sym_stats_min)}")
w()
w("Top 15 symbols by mean return:")
w(f"  {'Symbol':<8} {'Trades':>6} {'WR':>7} {'Mean Ret%':>10} {'Med Ret%':>10} {'Total PnL':>12}")
w(f"  {'-'*56}")
for sym, r in sym_stats_min.head(15).iterrows():
    w(f"  {sym:<8} {int(r['n_trades']):>6} {r['wr']:>6.1%} {r['mean_ret']:>+10.3f} {r['median_ret']:>+10.3f} {r['total_pnl']:>+12.2f}")

w()
w("Bottom 15 symbols by mean return:")
w(f"  {'Symbol':<8} {'Trades':>6} {'WR':>7} {'Mean Ret%':>10} {'Med Ret%':>10} {'Total PnL':>12}")
w(f"  {'-'*56}")
for sym, r in sym_stats_min.tail(15).iterrows():
    w(f"  {sym:<8} {int(r['n_trades']):>6} {r['wr']:>6.1%} {r['mean_ret']:>+10.3f} {r['median_ret']:>+10.3f} {r['total_pnl']:>+12.2f}")

# ── SECTION 7: Time Patterns ──
w()
w("=" * 90)
w("SECTION 7: TIME PATTERNS")
w("=" * 90)

for label, grp in time_results.items():
    w()
    w(f"  By {label}:")
    w(f"  {'Period':>8} {'N':>6} {'WR':>7} {'Mean Ret%':>10} {'Med Ret%':>10} {'Total PnL':>12}")
    w(f"  {'-'*58}")
    for idx, r in grp.iterrows():
        period_str = str(int(idx)) if not pd.isna(idx) else 'N/A'
        if label == 'Day of Week':
            dow_names = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri'}
            period_str = dow_names.get(int(idx), str(int(idx)))
        w(f"  {period_str:>8} {int(r['n']):>6} {r['wr']:>6.1%} {r['mean_ret']:>+10.3f} {r['median_ret']:>+10.3f} {r['total_pnl']:>+12.2f}")

# ANOVA test for month effect
w()
w("  Statistical test for month effect:")
month_groups = [g['return_pct'].dropna().values for _, g in df.groupby('entry_month') if len(g) > 10]
if len(month_groups) > 2:
    f_stat, f_p = stats.f_oneway(*month_groups)
    w(f"  One-way ANOVA: F={f_stat:.3f}, p={f_p:.4f}")
    w(f"  {'Significant' if f_p < 0.05 else 'Not significant'} at 5% level")

# ── SECTION 8: Days Held ──
w()
w("=" * 90)
w("SECTION 8: HOLDING PERIOD ANALYSIS")
w("=" * 90)
w()
w(f"  {'Period':<15} {'N':>6} {'WR':>7} {'Mean Ret%':>10} {'Med Ret%':>10}")
w(f"  {'-'*50}")
for label, d in days_held_results.items():
    w(f"  {label:<15} {d['n']:>6} {d['wr']:>6.1%} {d['mean_ret']:>+10.3f} {d['median_ret']:>+10.3f}")

# ── SECTION 9: Key Takeaways ──
w()
w("=" * 90)
w("SECTION 9: KEY OBSERVATIONS (data-driven, no assumptions)")
w("=" * 90)
w()

# Auto-identify strongest patterns
w("A) STRONGEST SINGLE FEATURES (|d| > 0.15 and monotonic):")
strong = sig[(sig['abs_d'] > 0.15) & (sig['monotonic_score'].abs() >= 0.5)]
entry_strong = strong[strong['feature'].str.startswith('entry_') | strong['feature'].str.startswith('spy_entry_') | strong['feature'].isin(['z_score'])]
for _, r in entry_strong.iterrows():
    direction = "HIGHER" if r['cohens_d'] > 0 else "LOWER"
    w(f"   - {r['feature']}: Winners have {direction} values (d={r['cohens_d']:+.3f})")
    w(f"     Q1 ret={r['q1_mean_ret']:+.3f}%, Q5 ret={r['q5_mean_ret']:+.3f}%, monotonic={r['monotonic_score']:+.2f}")

w()
w("B) STRONGEST INTERACTIONS:")
for _, r in int_df.head(3).iterrows():
    w(f"   - {r['feat1']} x {r['feat2']}")
    w(f"     Best: {r['best_quadrant']} ({r['best_ret']:+.3f}%, WR={r['best_wr']:.1%})")
    w(f"     Worst: {r['worst_quadrant']} ({r['worst_ret']:+.3f}%, WR={r['worst_wr']:.1%})")
    w(f"     Spread: {r['return_spread']:+.3f}% return, {r['wr_spread']:+.1%} WR")

w()
w("C) SPY REGIME IMPACT:")
for label, d in spy_regime_results.items():
    w(f"   - {label}: WR={d['wr']:.1%}, Mean ret={d['mean_ret']:+.3f}%, n={d['n']}")

w()
w("D) POTENTIAL FILTERS (data suggests these could improve strategy):")
# Identify concrete filter candidates
entry_actionable = entry_sig[entry_sig['feature'].str.startswith('entry_') | entry_sig['feature'].str.startswith('spy_entry_')]
for _, r in entry_actionable.head(5).iterrows():
    if r['cohens_d'] > 0:
        w(f"   - Favor trades where {r['feature']} is HIGH (top quintile WR={r['q5_wr']:.1%} vs bottom={r['q1_wr']:.1%})")
    else:
        w(f"   - Favor trades where {r['feature']} is LOW (bottom quintile WR={r['q1_wr']:.1%} vs top={r['q5_wr']:.1%})")

w()
w("=" * 90)
w("END OF REPORT")
w("=" * 90)

report = '\n'.join(lines)
with open('/home/ubuntu/daily_data/analysis_results/winner_loser_patterns_report.txt', 'w') as f:
    f.write(report)

print("\n✓ Report saved to winner_loser_patterns_report.txt")
print("✓ Features saved to winner_loser_features.csv")
