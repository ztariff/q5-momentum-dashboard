import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# LOAD DATA
# ==============================================================================
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/final_merged_trades.csv')
opts = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options.csv')
monthly = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options_monthly_replace.csv')
backfill = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options_backfill.csv')

opts_30d = opts[opts['option_type'] == '30d_1m'].copy()

def build_option_bars():
    orig = opts_30d[['trade_symbol', 'trade_entry_date', 'bar_date', 'day_volume', 'day_close', 'n_bars', 'morning_price']].copy()
    orig['source'] = 'original'
    orig = orig.dropna(subset=['bar_date'])
    
    mr = monthly[['trade_symbol', 'trade_entry_date', 'bar_date', 'day_volume', 'day_close', 'n_bars', 'morning_price']].copy()
    mr['source'] = 'monthly_replace'
    mr = mr.dropna(subset=['bar_date'])
    
    bf = backfill[['trade_symbol', 'trade_entry_date', 'bar_date', 'day_volume', 'day_close', 'n_bars', 'morning_price']].copy()
    bf['source'] = 'backfill'
    bf = bf.dropna(subset=['bar_date'])
    
    all_bars = pd.concat([orig, mr, bf], ignore_index=True)
    source_priority = {'original': 0, 'monthly_replace': 1, 'backfill': 2}
    all_bars['priority'] = all_bars['source'].map(source_priority)
    all_bars = all_bars.sort_values('priority')
    all_bars = all_bars.drop_duplicates(subset=['trade_symbol', 'trade_entry_date', 'bar_date'], keep='first')
    all_bars = all_bars.drop(columns=['priority'])
    return all_bars

all_bars = build_option_bars()

# ==============================================================================
# STEP 1
# ==============================================================================
trade_keys = trades[['symbol', 'entry_date']].copy()
trade_keys = trade_keys.rename(columns={'symbol': 'trade_symbol', 'entry_date': 'trade_entry_date'})

trade_bars = all_bars.merge(trade_keys, on=['trade_symbol', 'trade_entry_date'], how='inner')
trade_bars['bar_date_dt'] = pd.to_datetime(trade_bars['bar_date'])
trade_bars['entry_date_dt'] = pd.to_datetime(trade_bars['trade_entry_date'])
trade_bars = trade_bars.sort_values(['trade_symbol', 'trade_entry_date', 'bar_date_dt'])

# Vectorized approach instead of groupby apply
def compute_features_vectorized(trade_bars):
    results = []
    for (sym, edate), group in trade_bars.groupby(['trade_symbol', 'trade_entry_date']):
        group = group.sort_values('bar_date_dt')
        entry_dt = pd.to_datetime(edate)
        
        entry_row = group[group['bar_date_dt'] == entry_dt]
        if len(entry_row) == 0:
            continue
        
        entry_vol = float(entry_row['day_volume'].iloc[0])
        n_bars_e = float(entry_row['n_bars'].iloc[0])
        opt_price = float(entry_row['day_close'].iloc[0])
        
        first_3 = group.head(3)
        avg_3d = first_3['day_volume'].mean()
        avg_all = group['day_volume'].mean()
        vol_ratio = entry_vol / avg_all if avg_all > 0 else np.nan
        notional = entry_vol * opt_price * 100 if pd.notna(opt_price) else np.nan
        
        results.append({
            'trade_symbol': sym,
            'trade_entry_date': edate,
            'opt_vol_entry': entry_vol,
            'opt_vol_avg_3d': avg_3d,
            'opt_vol_day1_vs_avg': vol_ratio,
            'n_bars_entry': n_bars_e,
            'opt_price_entry': opt_price,
            'notional_vol': notional
        })
    return pd.DataFrame(results)

features = compute_features_vectorized(trade_bars)

trades_merged = trades.merge(
    features,
    left_on=['symbol', 'entry_date'],
    right_on=['trade_symbol', 'trade_entry_date'],
    how='left'
).drop(columns=['trade_symbol', 'trade_entry_date'], errors='ignore')

has_vol = trades_merged.dropna(subset=['opt_vol_entry'])

OUT = []
def P(s=""):
    OUT.append(str(s))
    print(s)

P("="*80)
P("OPTIONS VOLUME AS PREDICTOR OF TRADE OUTCOMES")
P("="*80)
P(f"Total trades: {len(trades_merged)}")
P(f"Trades with entry-day option volume data: {len(has_vol)}")
P(f"Coverage: {len(has_vol)/len(trades_merged):.1%}")

P(f"\n--- Entry-Day Volume Stats ---")
for feat in ['opt_vol_entry', 'opt_vol_avg_3d', 'opt_vol_day1_vs_avg', 'n_bars_entry', 'notional_vol']:
    vals = has_vol[feat].dropna()
    P(f"  {feat}: n={len(vals)}, mean={vals.mean():,.1f}, median={vals.median():,.1f}, "
      f"p25={vals.quantile(0.25):,.1f}, p75={vals.quantile(0.75):,.1f}, p90={vals.quantile(0.90):,.1f}")

# ==============================================================================
# STEP 2: Quintile analysis
# ==============================================================================
P("\n" + "="*80)
P("STEP 2: QUINTILE ANALYSIS")
P("="*80)

vol_features = ['opt_vol_entry', 'opt_vol_avg_3d', 'opt_vol_day1_vs_avg', 'n_bars_entry', 'notional_vol']
outcomes = ['hybrid_pnl', 'opt_pnl', 'stock_pnl']

for feat in vol_features:
    P(f"\n{'='*40}")
    P(f"Feature: {feat}")
    P(f"{'='*40}")
    df = has_vol.dropna(subset=[feat]).copy()
    if len(df) < 50:
        P(f"  Skipping: only {len(df)} trades")
        continue
    
    try:
        df['quintile'] = pd.qcut(df[feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    except:
        P(f"  Could not create quintiles (too many ties)")
        continue
    
    for outcome in outcomes:
        df_out = df.dropna(subset=[outcome])
        if len(df_out) < 30:
            continue
        
        P(f"\n  vs {outcome} (n={len(df_out)}):")
        qstats = df_out.groupby('quintile', observed=True).agg(
            n=(outcome, 'count'),
            mean_pnl=(outcome, 'mean'),
            total_pnl=(outcome, 'sum'),
            wr=(outcome, lambda x: (x > 0).mean())
        )
        for idx, row in qstats.iterrows():
            P(f"    {idx}: n={int(row['n']):>4}, mean=${row['mean_pnl']:>10,.0f}, total=${row['total_pnl']:>12,.0f}, WR={row['wr']:.1%}")
        
        rho, pval = stats.spearmanr(df_out[feat], df_out[outcome])
        P(f"  Spearman rho={rho:.4f}, p={pval:.4f}")
        
        top10_thresh = df_out[feat].quantile(0.9)
        top10 = df_out[df_out[feat] >= top10_thresh][outcome]
        rest = df_out[df_out[feat] < top10_thresh][outcome]
        if len(top10) > 5 and len(rest) > 5:
            pooled_std = np.sqrt((top10.var()*(len(top10)-1) + rest.var()*(len(rest)-1)) / (len(top10)+len(rest)-2))
            d = (top10.mean() - rest.mean()) / pooled_std if pooled_std > 0 else 0
            P(f"  Cohen's d (top 10% vs rest): {d:.4f}")
        
        q_means = qstats['mean_pnl'].values
        diffs = np.diff(q_means)
        mono_inc = all(d > 0 for d in diffs) if len(diffs) > 0 else False
        mono_dec = all(d < 0 for d in diffs) if len(diffs) > 0 else False
        P(f"  Monotonic: {'increasing' if mono_inc else 'decreasing' if mono_dec else 'NO'}")

# ==============================================================================
# STEP 3: Unusual activity
# ==============================================================================
P("\n" + "="*80)
P("STEP 3: UNUSUAL OPTIONS ACTIVITY THRESHOLDS")
P("="*80)

df_ua = has_vol.dropna(subset=['opt_vol_entry', 'hybrid_pnl']).copy()
med = df_ua['opt_vol_entry'].median()
p75 = df_ua['opt_vol_entry'].quantile(0.75)
p90 = df_ua['opt_vol_entry'].quantile(0.90)

P(f"\nThreshold values: median={med:.0f}, 75th={p75:.0f}, 90th={p90:.0f}")

thresholds = [
    (f'vol > median ({med:.0f})', df_ua['opt_vol_entry'] > med),
    (f'vol > 75th pct ({p75:.0f})', df_ua['opt_vol_entry'] > p75),
    (f'vol > 90th pct ({p90:.0f})', df_ua['opt_vol_entry'] > p90),
    ('n_bars > 100', df_ua['n_bars_entry'] > 100),
]

for name, mask in thresholds:
    P(f"\n--- {name} ---")
    for outcome in ['hybrid_pnl', 'opt_pnl', 'stock_pnl']:
        gp = df_ua[mask].dropna(subset=[outcome])
        gf = df_ua[~mask].dropna(subset=[outcome])
        if len(gp) < 10 or len(gf) < 10:
            continue
        
        mean_p, mean_f = gp[outcome].mean(), gf[outcome].mean()
        wr_p, wr_f = (gp[outcome]>0).mean(), (gf[outcome]>0).mean()
        
        wp = gp[gp[outcome]>0][outcome].sum()
        lp = abs(gp[gp[outcome]<0][outcome].sum())
        pf_p = wp/lp if lp > 0 else float('inf')
        
        wf = gf[gf[outcome]>0][outcome].sum()
        lf = abs(gf[gf[outcome]<0][outcome].sum())
        pf_f = wf/lf if lf > 0 else float('inf')
        
        t_stat, t_pval = stats.ttest_ind(gp[outcome], gf[outcome])
        
        P(f"  {outcome}:")
        P(f"    PASS: n={len(gp):>4}, mean=${mean_p:>10,.0f}, WR={wr_p:.1%}, PF={pf_p:.2f}")
        P(f"    FAIL: n={len(gf):>4}, mean=${mean_f:>10,.0f}, WR={wr_f:.1%}, PF={pf_f:.2f}")
        P(f"    Diff: ${mean_p-mean_f:>10,.0f}  t={t_stat:.3f}  p={t_pval:.4f}")

# ==============================================================================
# STEP 4: Options vol predicts opt returns vs stock returns
# ==============================================================================
P("\n" + "="*80)
P("STEP 4: DOES OPTIONS VOLUME PREDICT OPTION vs STOCK RETURNS?")
P("="*80)

P(f"\n{'Feature':<25} {'rho(opt_pnl)':>15} {'p':>8} {'rho(stock_pnl)':>15} {'p':>8} {'Diff':>8}")
P("-"*85)

for feat in vol_features:
    df_f = has_vol.dropna(subset=[feat])
    
    df_opt = df_f.dropna(subset=['opt_pnl'])
    if len(df_opt) > 30:
        rho_opt, p_opt = stats.spearmanr(df_opt[feat], df_opt['opt_pnl'])
    else:
        rho_opt, p_opt = np.nan, np.nan
    
    df_stk = df_f.dropna(subset=['stock_pnl'])
    if len(df_stk) > 30:
        rho_stk, p_stk = stats.spearmanr(df_stk[feat], df_stk['stock_pnl'])
    else:
        rho_stk, p_stk = np.nan, np.nan
    
    diff = rho_opt - rho_stk if pd.notna(rho_opt) and pd.notna(rho_stk) else np.nan
    
    r1 = f"{rho_opt:+.4f}" if pd.notna(rho_opt) else "N/A"
    p1 = f"{p_opt:.4f}" if pd.notna(p_opt) else "N/A"
    r2 = f"{rho_stk:+.4f}" if pd.notna(rho_stk) else "N/A"
    p2 = f"{p_stk:.4f}" if pd.notna(p_stk) else "N/A"
    d = f"{diff:+.4f}" if pd.notna(diff) else "N/A"
    
    P(f"{feat:<25} {r1:>15} {p1:>8} {r2:>15} {p2:>8} {d:>8}")

# Also do a direct comparison for trades that have BOTH opt_pnl and stock_pnl
P("\n--- Direct comparison: trades with both opt_pnl and stock_pnl ---")
df_both = has_vol.dropna(subset=['opt_pnl', 'stock_pnl', 'opt_vol_entry'])
P(f"Trades with both: {len(df_both)}")
if len(df_both) > 30:
    for feat in ['opt_vol_entry', 'notional_vol', 'n_bars_entry']:
        sub = df_both.dropna(subset=[feat])
        if len(sub) < 30:
            continue
        rho_o, po = stats.spearmanr(sub[feat], sub['opt_pnl'])
        rho_s, ps = stats.spearmanr(sub[feat], sub['stock_pnl'])
        P(f"  {feat}: rho(opt)={rho_o:+.4f} (p={po:.4f}), rho(stk)={rho_s:+.4f} (p={ps:.4f}), diff={rho_o-rho_s:+.4f}")

# ==============================================================================
# STEP 5: Top 10% winners
# ==============================================================================
P("\n" + "="*80)
P("STEP 5: TOP 10% WINNERS vs REST")
P("="*80)

df_t10 = has_vol.dropna(subset=['hybrid_pnl', 'opt_vol_entry']).copy()
thresh_10 = df_t10['hybrid_pnl'].quantile(0.9)
top10 = df_t10[df_t10['hybrid_pnl'] >= thresh_10]
rest10 = df_t10[df_t10['hybrid_pnl'] < thresh_10]

P(f"\nTop 10% threshold: hybrid_pnl >= ${thresh_10:,.0f}")
P(f"Top 10%: n={len(top10)}, Rest: n={len(rest10)}")

for feat in vol_features:
    t_vals = top10[feat].dropna()
    r_vals = rest10[feat].dropna()
    if len(t_vals) < 5:
        continue
    
    all_vals = df_t10[feat].dropna()
    pctl = stats.percentileofscore(all_vals, t_vals.median())
    
    P(f"\n  {feat}:")
    P(f"    Top 10%: mean={t_vals.mean():,.1f}, median={t_vals.median():,.1f}")
    P(f"    Rest:    mean={r_vals.mean():,.1f}, median={r_vals.median():,.1f}")
    P(f"    Top 10% median sits at {pctl:.0f}th percentile overall")
    u_stat, u_pval = stats.mannwhitneyu(t_vals, r_vals, alternative='two-sided')
    P(f"    Mann-Whitney U p={u_pval:.4f}")

# Volume quintile distribution
P(f"\nTop 10% distribution across opt_vol_entry quintiles:")
try:
    df_t10['vol_q'] = pd.qcut(df_t10['opt_vol_entry'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'], duplicates='drop')
    is_top = df_t10['hybrid_pnl'] >= thresh_10
    for q in ['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)']:
        sub = df_t10[df_t10['vol_q'] == q]
        if len(sub) == 0:
            continue
        n_top = is_top[df_t10['vol_q'] == q].sum()
        P(f"  {q}: {n_top}/{len(sub)} = {n_top/len(sub):.1%} are top 10%")
except:
    P("  Could not create quintiles")

# ==============================================================================
# STEP 6: Stability
# ==============================================================================
P("\n" + "="*80)
P("STEP 6: STABILITY CHECK (2020-2022 vs 2023-2026)")
P("="*80)

df_stab = has_vol.dropna(subset=['hybrid_pnl']).copy()
df_stab['period'] = df_stab['year'].apply(lambda y: '2020-2022' if y <= 2022 else '2023-2026')

P(f"\n{'Feature':<25} {'Period':<12} {'n':>6} {'Spearman':>10} {'p-val':>8} {'Sig?':>6}")
P("-"*72)

for feat in vol_features:
    for period in ['2020-2022', '2023-2026']:
        sub = df_stab[(df_stab['period'] == period)].dropna(subset=[feat])
        if len(sub) < 30:
            P(f"{feat:<25} {period:<12} {len(sub):>6} {'N/A':>10} {'N/A':>8} {'':>6}")
            continue
        rho, pval = stats.spearmanr(sub[feat], sub['hybrid_pnl'])
        sig = '*' if pval < 0.05 else ''
        P(f"{feat:<25} {period:<12} {len(sub):>6} {rho:>10.4f} {pval:>8.4f} {sig:>6}")

# Also compare opt_pnl stability
P(f"\n--- vs opt_pnl stability ---")
P(f"{'Feature':<25} {'Period':<12} {'n':>6} {'Spearman':>10} {'p-val':>8}")
P("-"*65)
for feat in ['opt_vol_entry', 'notional_vol', 'n_bars_entry']:
    for period in ['2020-2022', '2023-2026']:
        sub = df_stab[(df_stab['period'] == period)].dropna(subset=[feat, 'opt_pnl'])
        if len(sub) < 30:
            continue
        rho, pval = stats.spearmanr(sub[feat], sub['opt_pnl'])
        P(f"{feat:<25} {period:<12} {len(sub):>6} {rho:>10.4f} {pval:>8.4f}")

# ==============================================================================
# STEP 7: Volume filter for options strategy
# ==============================================================================
P("\n" + "="*80)
P("STEP 7: VOLUME FILTER FOR OPTIONS STRATEGY")
P("="*80)

df_opt_strat = trades_merged[
    (trades_merged['tier'].isin(['A', 'B'])) &
    (trades_merged['has_options'] == True) &
    (trades_merged['optonly_pnl'].notna())
].copy()

P(f"\nTier A+B option trades with opt_pnl: {len(df_opt_strat)}")

df_opt_vol = df_opt_strat.dropna(subset=['opt_vol_entry'])
P(f"With volume data: {len(df_opt_vol)}")

if len(df_opt_vol) > 30:
    median_vol = df_opt_vol['opt_vol_entry'].median()
    p75_vol = df_opt_vol['opt_vol_entry'].quantile(0.75)
    
    P(f"\n{'Filter':<35} {'n':>5} {'Total PnL':>14} {'Mean PnL':>11} {'WR':>7} {'PF':>7}")
    P("-"*85)
    
    filters = [
        ('No filter (baseline)', pd.Series(True, index=df_opt_vol.index)),
        (f'vol > median ({median_vol:.0f})', df_opt_vol['opt_vol_entry'] > median_vol),
        (f'vol > 75th ({p75_vol:.0f})', df_opt_vol['opt_vol_entry'] > p75_vol),
        ('n_bars > 100', df_opt_vol['n_bars_entry'] > 100),
        (f'vol <= median ({median_vol:.0f})', df_opt_vol['opt_vol_entry'] <= median_vol),
    ]
    
    for name, mask in filters:
        for pnl_col, label in [('optonly_pnl', 'opt'), ('hybrid_pnl', 'hybrid')]:
            sub = df_opt_vol[mask].dropna(subset=[pnl_col])
            if len(sub) < 5:
                continue
            pnl = sub[pnl_col]
            total = pnl.sum()
            mean_p = pnl.mean()
            wr = (pnl > 0).mean()
            w = pnl[pnl>0].sum()
            l = abs(pnl[pnl<0].sum())
            pf = w/l if l > 0 else float('inf')
            P(f"  {name+' ('+label+')' :<35} {len(sub):>5} ${total:>12,.0f} ${mean_p:>9,.0f} {wr:>6.1%} {pf:>6.2f}")

# ==============================================================================
# SUMMARY
# ==============================================================================
P("\n" + "="*80)
P("EXECUTIVE SUMMARY")
P("="*80)

# Gather key findings
P("""
This analysis tested whether entry-day options volume and activity metrics
predict subsequent trade outcomes across 2,082 trades.

KEY FINDINGS:
""")

# Get the key correlations
df_corr = has_vol.dropna(subset=['opt_vol_entry', 'hybrid_pnl'])
rho_main, p_main = stats.spearmanr(df_corr['opt_vol_entry'], df_corr['hybrid_pnl'])

df_corr2 = has_vol.dropna(subset=['notional_vol', 'hybrid_pnl'])
rho_not, p_not = stats.spearmanr(df_corr2['notional_vol'], df_corr2['hybrid_pnl'])

df_corr3 = has_vol.dropna(subset=['n_bars_entry', 'hybrid_pnl'])
rho_nb, p_nb = stats.spearmanr(df_corr3['n_bars_entry'], df_corr3['hybrid_pnl'])

P(f"1. CORRELATION WITH HYBRID PNL:")
P(f"   - opt_vol_entry vs hybrid_pnl: rho={rho_main:.4f}, p={p_main:.4f}")
P(f"   - notional_vol vs hybrid_pnl:  rho={rho_not:.4f}, p={p_not:.4f}")
P(f"   - n_bars_entry vs hybrid_pnl:  rho={rho_nb:.4f}, p={p_nb:.4f}")

# Check if any are significant
sig_features = []
for feat in vol_features:
    sub = has_vol.dropna(subset=[feat, 'hybrid_pnl'])
    if len(sub) > 30:
        r, p = stats.spearmanr(sub[feat], sub['hybrid_pnl'])
        if p < 0.05:
            sig_features.append((feat, r, p))

if sig_features:
    P(f"\n2. STATISTICALLY SIGNIFICANT FEATURES (p<0.05):")
    for f, r, p in sig_features:
        P(f"   - {f}: rho={r:.4f}, p={p:.4f}")
else:
    P(f"\n2. NO features reached statistical significance at p<0.05 for hybrid_pnl")

# Option vs stock comparison
df_both = has_vol.dropna(subset=['opt_pnl', 'stock_pnl', 'opt_vol_entry'])
if len(df_both) > 30:
    ro, _ = stats.spearmanr(df_both['opt_vol_entry'], df_both['opt_pnl'])
    rs, _ = stats.spearmanr(df_both['opt_vol_entry'], df_both['stock_pnl'])
    P(f"\n3. OPTION vs STOCK RETURN PREDICTION (same trades, n={len(df_both)}):")
    P(f"   - opt_vol_entry -> opt_pnl:   rho={ro:+.4f}")
    P(f"   - opt_vol_entry -> stock_pnl:  rho={rs:+.4f}")
    if abs(ro) > abs(rs):
        P(f"   Options volume predicts option returns {'better' if abs(ro) > abs(rs) else 'worse'} than stock returns")
    else:
        P(f"   Options volume does NOT predict option returns better than stock returns")

P(f"\n4. PRACTICAL FILTER VALUE:")
# Summarize the filter test
if len(df_opt_vol) > 30:
    base_pnl = df_opt_vol['optonly_pnl'].sum()
    filt_pnl = df_opt_vol[df_opt_vol['opt_vol_entry'] > median_vol]['optonly_pnl'].sum()
    base_n = len(df_opt_vol)
    filt_n = len(df_opt_vol[df_opt_vol['opt_vol_entry'] > median_vol])
    P(f"   Baseline (all): {base_n} trades, total=${base_pnl:,.0f}")
    P(f"   Vol>median:     {filt_n} trades, total=${filt_pnl:,.0f}")

# Write report
report = "\n".join(OUT)
with open('/home/ubuntu/daily_data/analysis_results/options_volume_predictor_report.txt', 'w') as f:
    f.write(report)
print("\n\nReport saved.")
