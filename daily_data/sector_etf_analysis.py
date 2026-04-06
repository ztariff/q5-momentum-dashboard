import pandas as pd
import numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

###############################################################################
# STEP 1: Sector mapping
###############################################################################
SECTOR_MAP = {}
ETF_MAP = {}

sectors = {
    'TECH': (['AAPL','MSFT','GOOGL','META','NVDA','ADBE','CRM','NOW','INTC','CSCO','IBM','ORCL',
              'AMD','MU','AVGO','QCOM','MRVL','AMAT','LRCX','KLAC','TSM','TXN','ADI','SMCI','ARM',
              'MCHP','SWKS','ON','NET','DDOG','ZS','CRWD','PANW','SNOW','PLTR','SHOP','SQ','TWLO',
              'U','ROKU','TTD','HUBS','WDAY','DOCU','OKTA','MDB','TEAM','VEEV','BILL','PATH','AI'], 'QQQ'),
    'CONSUMER': (['WMT','COST','TGT','HD','LOW','NKE','SBUX','MCD','DIS','ABNB','UBER','LYFT',
                  'DASH','CMG','LULU','PTON','W','ETSY','BKNG','NFLX','SPOT','RBLX'], 'XLY'),
    'FINANCIALS': (['JPM','GS','MS','BAC','C','WFC','SCHW','BLK','AXP','COF','V','MA','PYPL',
                    'COIN','HOOD','SOFI','AFRM','UPST'], 'XLF'),
    'ENERGY': (['XOM','CVX','OXY','COP','SLB','DVN','MPC','VLO','HAL','PSX','EOG','PXD'], 'XLE'),
    'BIOTECH': (['MRNA','BNTX','PFE','JNJ','ABBV','LLY','NVO','AMGN','BIIB','GILD','REGN','VRTX',
                 'BMY','MRK','AZN'], 'XLV'),
    'INDUSTRIALS': (['LMT','RTX','NOC','BA','GE','CAT','DE','HON','UPS','FDX','MMM'], 'XLI'),
    'METALS': (['FCX','NEM','CLF','X','AA','VALE','NUE','MP'], 'XLE'),
    'TRAVEL': (['DAL','UAL','AAL','LUV','CCL','RCL','NCLH','MAR','HLT','EXPE'], 'XLY'),
    'CHINA': (['BABA','JD','PDD','BIDU','BEKE','XPEV','NIO','LI'], 'KWEB'),
    'CRYPTO': (['MSTR','MARA','RIOT','HUT'], 'SPY'),  # COIN in financials
    'EV_CLEAN': (['RIVN','LCID','ENPH','SEDG','FSLR','RUN','PLUG','CHPT','JOBY'], 'QQQ'),
    'MEME': (['GME','AMC','SPCE','BYND','DKNG','LAZR','QS','CVNA','CLOV','IONQ','RKLB'], 'IWM'),
}

ETF_SYMBOLS = ['SPY','QQQ','IWM','XLY','XLF','XLE','XLV','XLI','KWEB','ARKK','DIA',
               'SMH','SOXL','SOXX','TQQQ','XBI','XLC','XLK','XLP','XLU','XOP',
               'GLD','SLV','TLT','HYG','EEM','VXX']

for sname, (syms, etf) in sectors.items():
    for s in syms:
        SECTOR_MAP[s] = sname
        ETF_MAP[s] = etf

for e in ETF_SYMBOLS:
    if e not in SECTOR_MAP:
        SECTOR_MAP[e] = 'ETF'
        ETF_MAP[e] = 'SPY'

# Special: TSLA, AMZN, etc. that aren't in the mapping
EXTRA = {
    'TSLA': ('TECH', 'QQQ'), 'AMZN': ('TECH', 'QQQ'), 'ZM': ('TECH', 'QQQ'),
    'DXCM': ('BIOTECH', 'XLV'), 'ISRG': ('BIOTECH', 'XLV'),
    'HCA': ('BIOTECH', 'XLV'), 'CI': ('BIOTECH', 'XLV'), 'ELV': ('BIOTECH', 'XLV'),
    'UNH': ('BIOTECH', 'XLV'), 'SYK': ('BIOTECH', 'XLV'), 'ZTS': ('BIOTECH', 'XLV'),
    'TMO': ('BIOTECH', 'XLV'), 'ABT': ('BIOTECH', 'XLV'), 'GEHC': ('BIOTECH', 'XLV'),
    'PARA': ('CONSUMER', 'XLY'), 'WBD': ('CONSUMER', 'XLY'), 'CMCSA': ('CONSUMER', 'XLY'),
    'T': ('CONSUMER', 'XLY'), 'VZ': ('CONSUMER', 'XLY'),
    'AMT': ('FINANCIALS', 'XLF'), 'EQIX': ('FINANCIALS', 'XLF'), 'O': ('FINANCIALS', 'XLF'),
    'PLD': ('FINANCIALS', 'XLF'), 'SPG': ('FINANCIALS', 'XLF'),
}
for s, (sec, etf) in EXTRA.items():
    if s not in SECTOR_MAP:
        SECTOR_MAP[s] = sec
        ETF_MAP[s] = etf

###############################################################################
# STEP 2: Load trades and compute sector ETF conditions at entry
###############################################################################
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/final_merged_trades.csv')
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])

# Assign sector and ETF
trades['sector'] = trades['symbol'].map(SECTOR_MAP).fillna('OTHER')
trades['sector_etf'] = trades['symbol'].map(ETF_MAP).fillna('SPY')

# Outcome
trades['winner'] = (trades['hybrid_pnl'] > 0).astype(int)

print(f"Total trades: {len(trades)}")
print(f"\nSector distribution:")
print(trades['sector'].value_counts().to_string())
print(f"\nSector ETF distribution:")
print(trades['sector_etf'].value_counts().to_string())

# Check for unmapped symbols
unmapped = trades[trades['sector'] == 'OTHER']['symbol'].unique()
if len(unmapped) > 0:
    print(f"\nUnmapped symbols (using SPY): {unmapped}")

###############################################################################
# Load ETF data cache
###############################################################################
DATA_DIR = '/home/ubuntu/daily_data/data/'
etf_cache = {}

needed_etfs = trades['sector_etf'].unique()
for etf in needed_etfs:
    fp = os.path.join(DATA_DIR, f'{etf}_enriched.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # Compute ROC 5d
        df['roc_5'] = df['close'].pct_change(5) * 100
        # SMA8 
        if 'sma_8' not in df.columns:
            df['sma_8'] = df['close'].rolling(8).mean()
        etf_cache[etf] = df
        print(f"Loaded {etf}: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        print(f"WARNING: {etf} enriched data not found!")

###############################################################################
# Compute ETF features for each trade
###############################################################################
def get_etf_features(row):
    etf = row['sector_etf']
    entry = row['entry_date']
    
    if etf not in etf_cache:
        return pd.Series({k: np.nan for k in ['etf_above_sma20','etf_above_sma50','etf_above_sma200',
                                                'etf_rsi','etf_roc_5d','etf_roc_10d','etf_roc_20d',
                                                'etf_dist_sma50_pct','etf_consec_days','etf_pct_from_20d_high',
                                                'etf_trend_aligned']})
    
    edf = etf_cache[etf]
    mask = edf['date'] <= entry
    if mask.sum() == 0:
        return pd.Series({k: np.nan for k in ['etf_above_sma20','etf_above_sma50','etf_above_sma200',
                                                'etf_rsi','etf_roc_5d','etf_roc_10d','etf_roc_20d',
                                                'etf_dist_sma50_pct','etf_consec_days','etf_pct_from_20d_high',
                                                'etf_trend_aligned']})
    
    row_etf = edf[mask].iloc[-1]
    
    close = row_etf['close']
    sma20 = row_etf.get('sma_20', np.nan)
    sma50 = row_etf.get('sma_50', np.nan)
    sma200 = row_etf.get('sma_200', np.nan)
    sma8 = row_etf.get('sma_8', np.nan)
    
    return pd.Series({
        'etf_above_sma20': 1 if (pd.notna(sma20) and close > sma20) else 0,
        'etf_above_sma50': 1 if (pd.notna(sma50) and close > sma50) else 0,
        'etf_above_sma200': 1 if (pd.notna(sma200) and close > sma200) else 0,
        'etf_rsi': row_etf.get('rsi_14', np.nan),
        'etf_roc_5d': row_etf.get('roc_5', np.nan),
        'etf_roc_10d': row_etf.get('roc_10', np.nan),
        'etf_roc_20d': row_etf.get('roc_20', np.nan),
        'etf_dist_sma50_pct': row_etf.get('dist_sma50_pct', np.nan),
        'etf_consec_days': row_etf.get('consec_days', np.nan),
        'etf_pct_from_20d_high': row_etf.get('pct_from_20d_high', np.nan),
        'etf_trend_aligned': 1 if (pd.notna(sma8) and pd.notna(sma20) and pd.notna(sma50) 
                                    and sma8 > sma20 > sma50) else 0,
    })

print("\nComputing ETF features for each trade...")
etf_features = trades.apply(get_etf_features, axis=1)
trades = pd.concat([trades, etf_features], axis=1)
print("Done. Non-null counts:")
for c in etf_features.columns:
    print(f"  {c}: {trades[c].notna().sum()}")

###############################################################################
# STEP 3: Analysis functions
###############################################################################
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled == 0:
        return np.nan
    return (g1.mean() - g2.mean()) / pooled

def quintile_analysis(df, feature, outcome='hybrid_pnl', n_bins=5):
    valid = df[[feature, outcome, 'winner']].dropna()
    if len(valid) < 20:
        return None
    try:
        valid['q'] = pd.qcut(valid[feature], n_bins, labels=False, duplicates='drop')
    except:
        return None
    
    results = []
    for q in sorted(valid['q'].unique()):
        sub = valid[valid['q'] == q]
        results.append({
            'quintile': int(q) + 1,
            'n': len(sub),
            'feat_range': f"{sub[feature].min():.2f} to {sub[feature].max():.2f}",
            'mean_pnl': sub[outcome].mean(),
            'median_pnl': sub[outcome].median(),
            'win_rate': sub['winner'].mean() * 100,
            'total_pnl': sub[outcome].sum(),
        })
    return pd.DataFrame(results)

def binary_analysis(df, feature, outcome='hybrid_pnl'):
    valid = df[[feature, outcome, 'winner']].dropna()
    g0 = valid[valid[feature] == 0]
    g1 = valid[valid[feature] == 1]
    if len(g0) < 5 or len(g1) < 5:
        return None
    
    d = cohens_d(g1[outcome], g0[outcome])
    tstat, pval = stats.ttest_ind(g1[outcome], g0[outcome], equal_var=False)
    
    return {
        'n_0': len(g0), 'n_1': len(g1),
        'mean_pnl_0': g0[outcome].mean(), 'mean_pnl_1': g1[outcome].mean(),
        'median_pnl_0': g0[outcome].median(), 'median_pnl_1': g1[outcome].median(),
        'wr_0': g0['winner'].mean()*100, 'wr_1': g1['winner'].mean()*100,
        'total_pnl_0': g0[outcome].sum(), 'total_pnl_1': g1[outcome].sum(),
        'cohens_d': d, 't_stat': tstat, 'p_value': pval
    }

def profit_factor(pnl_series):
    gains = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    return gains / losses if losses > 0 else np.inf

###############################################################################
# Build output
###############################################################################
output = []
output.append("=" * 90)
output.append("SECTOR ETF CONDITIONS AT ENTRY: DO THEY PREDICT TRADE OUTCOMES?")
output.append("=" * 90)
output.append(f"Total trades analyzed: {len(trades)}")
output.append(f"Date range: {trades['entry_date'].min().date()} to {trades['entry_date'].max().date()}")
output.append(f"Sectors: {trades['sector'].nunique()}")
output.append("")

# Sector summary
output.append("-" * 90)
output.append("SECTOR DISTRIBUTION")
output.append("-" * 90)
for sec in trades['sector'].value_counts().index:
    sub = trades[trades['sector'] == sec]
    etf = sub['sector_etf'].iloc[0]
    output.append(f"  {sec:20s} ETF={etf:5s}  Trades={len(sub):4d}  "
                  f"WR={sub['winner'].mean()*100:.1f}%  Mean P&L=${sub['hybrid_pnl'].mean():,.0f}  "
                  f"Total=${sub['hybrid_pnl'].sum():,.0f}")

###############################################################################
# STEP 3a: Cross-sector analysis (ALL trades)
###############################################################################
output.append("")
output.append("=" * 90)
output.append("STEP 3: CROSS-SECTOR ANALYSIS (ALL TRADES)")
output.append("=" * 90)

binary_features = ['etf_above_sma20', 'etf_above_sma50', 'etf_above_sma200', 'etf_trend_aligned']
continuous_features = ['etf_rsi', 'etf_roc_5d', 'etf_roc_10d', 'etf_roc_20d', 
                       'etf_dist_sma50_pct', 'etf_consec_days', 'etf_pct_from_20d_high']
all_features = binary_features + continuous_features

# Spearman correlations
output.append("")
output.append("--- Spearman Correlations (ALL trades) ---")
output.append(f"{'Feature':30s} {'rho':>8s} {'p-value':>10s} {'Significant':>12s}")
for feat in all_features:
    valid = trades[[feat, 'hybrid_pnl']].dropna()
    if len(valid) > 20:
        rho, p = stats.spearmanr(valid[feat], valid['hybrid_pnl'])
        sig = "YES" if p < 0.05 else "no"
        output.append(f"  {feat:28s} {rho:8.4f} {p:10.4f} {sig:>12s}")

# Binary feature analysis
output.append("")
output.append("--- Binary Feature Analysis (ALL trades) ---")
for feat in binary_features:
    res = binary_analysis(trades, feat)
    if res is None:
        continue
    output.append(f"\n  {feat}:")
    output.append(f"    When 0: n={res['n_0']:4d}  Mean=${res['mean_pnl_0']:>10,.0f}  "
                  f"Median=${res['median_pnl_0']:>10,.0f}  WR={res['wr_0']:.1f}%  Total=${res['total_pnl_0']:>12,.0f}")
    output.append(f"    When 1: n={res['n_1']:4d}  Mean=${res['mean_pnl_1']:>10,.0f}  "
                  f"Median=${res['median_pnl_1']:>10,.0f}  WR={res['wr_1']:.1f}%  Total=${res['total_pnl_1']:>12,.0f}")
    output.append(f"    Cohen's d={res['cohens_d']:.4f}  t={res['t_stat']:.3f}  p={res['p_value']:.4f}"
                  f"  {'*** SIGNIFICANT ***' if res['p_value'] < 0.05 else ''}")

# Quintile analysis for continuous
output.append("")
output.append("--- Quintile Analysis (ALL trades, continuous features) ---")
for feat in continuous_features:
    qa = quintile_analysis(trades, feat)
    if qa is None:
        continue
    output.append(f"\n  {feat}:")
    output.append(f"  {'Q':>4s} {'N':>5s} {'Range':>25s} {'Mean PnL':>12s} {'Median PnL':>12s} {'WR%':>6s} {'Total PnL':>14s}")
    for _, r in qa.iterrows():
        output.append(f"  {r['quintile']:4.0f} {r['n']:5d} {r['feat_range']:>25s} "
                      f"${r['mean_pnl']:>10,.0f} ${r['median_pnl']:>10,.0f} {r['win_rate']:5.1f}% ${r['total_pnl']:>12,.0f}")

# Cohen's d for winners vs losers on each feature
output.append("")
output.append("--- Cohen's d: Feature values for Winners vs Losers (ALL trades) ---")
output.append(f"{'Feature':30s} {'d':>8s} {'Winners Mean':>14s} {'Losers Mean':>14s}")
for feat in all_features:
    valid = trades[[feat, 'hybrid_pnl', 'winner']].dropna()
    if len(valid) < 20:
        continue
    w = valid[valid['winner'] == 1][feat]
    l = valid[valid['winner'] == 0][feat]
    d = cohens_d(w, l)
    output.append(f"  {feat:28s} {d:8.4f} {w.mean():14.4f} {l.mean():14.4f}")

###############################################################################
# STEP 3b: Within-sector analysis
###############################################################################
output.append("")
output.append("=" * 90)
output.append("WITHIN-SECTOR ANALYSIS")
output.append("=" * 90)

sector_findings = []

for sec in sorted(trades['sector'].unique()):
    sub = trades[trades['sector'] == sec]
    if len(sub) < 20:
        continue
    etf_name = sub['sector_etf'].iloc[0]
    
    output.append(f"\n{'─' * 70}")
    output.append(f"SECTOR: {sec} (ETF: {etf_name}, n={len(sub)}, WR={sub['winner'].mean()*100:.1f}%)")
    output.append(f"{'─' * 70}")
    
    # Binary features
    for feat in binary_features:
        res = binary_analysis(sub, feat)
        if res is None:
            continue
        sig_marker = " ***" if res['p_value'] < 0.10 else ""
        output.append(f"  {feat}: 0→WR={res['wr_0']:.1f}% Mean=${res['mean_pnl_0']:,.0f} (n={res['n_0']}) | "
                      f"1→WR={res['wr_1']:.1f}% Mean=${res['mean_pnl_1']:,.0f} (n={res['n_1']}) | "
                      f"d={res['cohens_d']:.3f} p={res['p_value']:.3f}{sig_marker}")
        if abs(res['cohens_d']) > 0.2:
            sector_findings.append((sec, feat, res['cohens_d'], res['p_value'], res['n_0']+res['n_1']))
    
    # Continuous: Spearman
    for feat in continuous_features:
        valid = sub[[feat, 'hybrid_pnl']].dropna()
        if len(valid) < 15:
            continue
        rho, p = stats.spearmanr(valid[feat], valid['hybrid_pnl'])
        sig_marker = " ***" if p < 0.10 else ""
        if abs(rho) > 0.1 or p < 0.10:
            output.append(f"  {feat}: rho={rho:.3f} p={p:.3f}{sig_marker}")

###############################################################################
# STEP 4: Specific questions
###############################################################################
output.append("")
output.append("=" * 90)
output.append("STEP 4: SPECIFIC QUESTIONS")
output.append("=" * 90)

# 4a) Tech trades when QQQ above 50 SMA / 200 SMA
output.append("")
output.append("--- 4a: Tech trades (QQQ sector) vs QQQ SMA conditions ---")
tech = trades[trades['sector_etf'] == 'QQQ']
output.append(f"Tech/QQQ-mapped trades: {len(tech)}")

for feat in ['etf_above_sma50', 'etf_above_sma200']:
    res = binary_analysis(tech, feat)
    if res:
        output.append(f"\n  {feat}:")
        output.append(f"    Below: n={res['n_0']}, Mean=${res['mean_pnl_0']:,.0f}, WR={res['wr_0']:.1f}%, Total=${res['total_pnl_0']:,.0f}")
        output.append(f"    Above: n={res['n_1']}, Mean=${res['mean_pnl_1']:,.0f}, WR={res['wr_1']:.1f}%, Total=${res['total_pnl_1']:,.0f}")
        output.append(f"    Cohen's d={res['cohens_d']:.4f}, p={res['p_value']:.4f}")

# 4b) Energy trades vs XLE roc_20d
output.append("")
output.append("--- 4b: Energy trades vs XLE trending (roc_20d > 0) ---")
energy = trades[trades['sector'] == 'ENERGY']
output.append(f"Energy trades: {len(energy)}")
if len(energy) > 10:
    energy_valid = energy[energy['etf_roc_20d'].notna()].copy()
    energy_valid['xle_trending'] = (energy_valid['etf_roc_20d'] > 0).astype(int)
    res = binary_analysis(energy_valid, 'xle_trending')
    if res:
        output.append(f"  XLE not trending (roc20d<=0): n={res['n_0']}, Mean=${res['mean_pnl_0']:,.0f}, WR={res['wr_0']:.1f}%")
        output.append(f"  XLE trending (roc20d>0):      n={res['n_1']}, Mean=${res['mean_pnl_1']:,.0f}, WR={res['wr_1']:.1f}%")
        output.append(f"  Cohen's d={res['cohens_d']:.4f}, p={res['p_value']:.4f}")

# 4c) Trend aligned
output.append("")
output.append("--- 4c: Trend aligned (SMA8 > SMA20 > SMA50) across all trades ---")
res = binary_analysis(trades, 'etf_trend_aligned')
if res:
    output.append(f"  Not aligned: n={res['n_0']}, Mean=${res['mean_pnl_0']:,.0f}, WR={res['wr_0']:.1f}%")
    output.append(f"  Aligned:     n={res['n_1']}, Mean=${res['mean_pnl_1']:,.0f}, WR={res['wr_1']:.1f}%")
    output.append(f"  Cohen's d={res['cohens_d']:.4f}, p={res['p_value']:.4f}")
    output.append(f"  {'SIGNIFICANT' if res['p_value'] < 0.05 else 'Not significant'}")

# 4d) Any sector with Cohen's d > 0.3?
output.append("")
output.append("--- 4d: Sectors with Cohen's d > 0.3 on any feature ---")
big_d = [f for f in sector_findings if abs(f[2]) > 0.3]
if big_d:
    for sec, feat, d, p, n in sorted(big_d, key=lambda x: abs(x[2]), reverse=True):
        output.append(f"  {sec:20s} {feat:25s} d={d:+.3f} p={p:.3f} n={n}")
else:
    output.append("  No sector-feature combination found with |d| > 0.3")

# Also check all combos more carefully
output.append("")
output.append("  Full scan of all sector-feature Cohen's d values:")
for sec in sorted(trades['sector'].unique()):
    sub = trades[trades['sector'] == sec]
    if len(sub) < 20:
        continue
    for feat in binary_features:
        valid = sub[[feat, 'hybrid_pnl', 'winner']].dropna()
        w = valid[valid['winner'] == 1][feat]
        l = valid[valid['winner'] == 0][feat]
        if len(w) > 5 and len(l) > 5:
            d = cohens_d(w, l)
            if abs(d) > 0.2:
                output.append(f"    {sec:20s} {feat:25s} d={d:+.4f} (n={len(valid)})")

# 4e) Does "sector ETF above 50 SMA" consistently help?
output.append("")
output.append("--- 4e: 'ETF above 50 SMA' consistency across sectors ---")
output.append(f"{'Sector':20s} {'ETF':>5s} {'n_below':>8s} {'n_above':>8s} {'WR_below':>9s} {'WR_above':>9s} {'Mean_below':>12s} {'Mean_above':>12s} {'d':>8s}")
consistent_help = 0
consistent_hurt = 0
for sec in sorted(trades['sector'].unique()):
    sub = trades[trades['sector'] == sec]
    if len(sub) < 15:
        continue
    etf_name = sub['sector_etf'].iloc[0]
    res = binary_analysis(sub, 'etf_above_sma50')
    if res is None:
        continue
    direction = "HELPS" if res['mean_pnl_1'] > res['mean_pnl_0'] else "HURTS"
    if res['mean_pnl_1'] > res['mean_pnl_0']:
        consistent_help += 1
    else:
        consistent_hurt += 1
    output.append(f"  {sec:18s} {etf_name:>5s} {res['n_0']:8d} {res['n_1']:8d} "
                  f"{res['wr_0']:8.1f}% {res['wr_1']:8.1f}% "
                  f"${res['mean_pnl_0']:>10,.0f} ${res['mean_pnl_1']:>10,.0f} {res['cohens_d']:+8.3f} {direction}")
output.append(f"\n  Sectors where above-50SMA helps: {consistent_help}")
output.append(f"  Sectors where above-50SMA hurts: {consistent_hurt}")

###############################################################################
# STEP 5: Filter testing
###############################################################################
output.append("")
output.append("=" * 90)
output.append("STEP 5: FILTER TESTING")
output.append("=" * 90)

def filter_comparison(df, filter_col, filter_val=1, label=""):
    """Compare filtered vs unfiltered trading"""
    valid = df[df[filter_col].notna()]
    filtered = valid[valid[filter_col] == filter_val] if isinstance(filter_val, int) else valid[valid[filter_col] > filter_val]
    excluded = valid[~valid.index.isin(filtered.index)]
    
    if len(filtered) < 10 or len(excluded) < 10:
        return None
    
    return {
        'label': label,
        'all_n': len(valid), 'all_wr': valid['winner'].mean()*100, 
        'all_mean': valid['hybrid_pnl'].mean(), 'all_total': valid['hybrid_pnl'].sum(),
        'all_pf': profit_factor(valid['hybrid_pnl']),
        'filt_n': len(filtered), 'filt_wr': filtered['winner'].mean()*100,
        'filt_mean': filtered['hybrid_pnl'].mean(), 'filt_total': filtered['hybrid_pnl'].sum(),
        'filt_pf': profit_factor(filtered['hybrid_pnl']),
        'excl_n': len(excluded), 'excl_wr': excluded['winner'].mean()*100,
        'excl_mean': excluded['hybrid_pnl'].mean(), 'excl_total': excluded['hybrid_pnl'].sum(),
        'excl_pf': profit_factor(excluded['hybrid_pnl']),
    }

# Filter 1: Only trade when sector ETF above 50 SMA
output.append("")
output.append("--- Filter 1: Only trade when sector ETF above 50 SMA ---")
r = filter_comparison(trades, 'etf_above_sma50', 1, "ETF > SMA50")
if r:
    output.append(f"  {'':18s} {'Trades':>7s} {'WR%':>7s} {'Mean PnL':>12s} {'Total PnL':>14s} {'PF':>6s}")
    output.append(f"  {'All trades':18s} {r['all_n']:7d} {r['all_wr']:6.1f}% ${r['all_mean']:>10,.0f} ${r['all_total']:>12,.0f} {r['all_pf']:6.2f}")
    output.append(f"  {'Filtered (above)':18s} {r['filt_n']:7d} {r['filt_wr']:6.1f}% ${r['filt_mean']:>10,.0f} ${r['filt_total']:>12,.0f} {r['filt_pf']:6.2f}")
    output.append(f"  {'Excluded (below)':18s} {r['excl_n']:7d} {r['excl_wr']:6.1f}% ${r['excl_mean']:>10,.0f} ${r['excl_total']:>12,.0f} {r['excl_pf']:6.2f}")

# Filter 2: Only trade when sector ETF roc_20d > 0
output.append("")
output.append("--- Filter 2: Only trade when sector ETF roc_20d > 0 ---")
trades_valid = trades[trades['etf_roc_20d'].notna()].copy()
trades_valid['etf_roc20_pos'] = (trades_valid['etf_roc_20d'] > 0).astype(int)
r = filter_comparison(trades_valid, 'etf_roc20_pos', 1, "ETF ROC20 > 0")
if r:
    output.append(f"  {'':18s} {'Trades':>7s} {'WR%':>7s} {'Mean PnL':>12s} {'Total PnL':>14s} {'PF':>6s}")
    output.append(f"  {'All trades':18s} {r['all_n']:7d} {r['all_wr']:6.1f}% ${r['all_mean']:>10,.0f} ${r['all_total']:>12,.0f} {r['all_pf']:6.2f}")
    output.append(f"  {'Filtered (pos)':18s} {r['filt_n']:7d} {r['filt_wr']:6.1f}% ${r['filt_mean']:>10,.0f} ${r['filt_total']:>12,.0f} {r['filt_pf']:6.2f}")
    output.append(f"  {'Excluded (neg)':18s} {r['excl_n']:7d} {r['excl_wr']:6.1f}% ${r['excl_mean']:>10,.0f} ${r['excl_total']:>12,.0f} {r['excl_pf']:6.2f}")

# Filter 3: Only trade when trend aligned
output.append("")
output.append("--- Filter 3: Only trade when sector ETF trend aligned (8>20>50) ---")
r = filter_comparison(trades, 'etf_trend_aligned', 1, "Trend Aligned")
if r:
    output.append(f"  {'':18s} {'Trades':>7s} {'WR%':>7s} {'Mean PnL':>12s} {'Total PnL':>14s} {'PF':>6s}")
    output.append(f"  {'All trades':18s} {r['all_n']:7d} {r['all_wr']:6.1f}% ${r['all_mean']:>10,.0f} ${r['all_total']:>12,.0f} {r['all_pf']:6.2f}")
    output.append(f"  {'Filtered (aligned)':18s} {r['filt_n']:7d} {r['filt_wr']:6.1f}% ${r['filt_mean']:>10,.0f} ${r['filt_total']:>12,.0f} {r['filt_pf']:6.2f}")
    output.append(f"  {'Excluded':18s} {r['excl_n']:7d} {r['excl_wr']:6.1f}% ${r['excl_mean']:>10,.0f} ${r['excl_total']:>12,.0f} {r['excl_pf']:6.2f}")

# Filter per sector
output.append("")
output.append("--- Filter: ETF above 50 SMA, by sector ---")
output.append(f"{'Sector':20s} {'ETF':>5s} {'All_n':>6s} {'Filt_n':>7s} {'All_WR':>7s} {'Filt_WR':>8s} {'All_Mean':>10s} {'Filt_Mean':>11s} {'All_PF':>7s} {'Filt_PF':>8s}")
for sec in sorted(trades['sector'].unique()):
    sub = trades[trades['sector'] == sec]
    if len(sub) < 15:
        continue
    etf_name = sub['sector_etf'].iloc[0]
    r = filter_comparison(sub, 'etf_above_sma50', 1)
    if r is None:
        continue
    output.append(f"  {sec:18s} {etf_name:>5s} {r['all_n']:6d} {r['filt_n']:7d} "
                  f"{r['all_wr']:6.1f}% {r['filt_wr']:7.1f}% "
                  f"${r['all_mean']:>8,.0f} ${r['filt_mean']:>9,.0f} "
                  f"{r['all_pf']:7.2f} {r['filt_pf']:8.2f}")

###############################################################################
# STEP 6: Stability check
###############################################################################
output.append("")
output.append("=" * 90)
output.append("STEP 6: STABILITY CHECK (2020-2022 vs 2023-2026)")
output.append("=" * 90)

trades['year_int'] = trades['entry_date'].dt.year
early = trades[trades['year_int'] <= 2022]
late = trades[trades['year_int'] >= 2023]

output.append(f"\nEarly period (2020-2022): {len(early)} trades")
output.append(f"Late period (2023-2026): {len(late)} trades")

for feat in binary_features:
    output.append(f"\n  {feat}:")
    for period_name, period_df in [("2020-2022", early), ("2023-2026", late)]:
        res = binary_analysis(period_df, feat)
        if res:
            output.append(f"    {period_name}: 0→WR={res['wr_0']:.1f}% Mean=${res['mean_pnl_0']:>10,.0f} (n={res['n_0']}) | "
                          f"1→WR={res['wr_1']:.1f}% Mean=${res['mean_pnl_1']:>10,.0f} (n={res['n_1']}) | "
                          f"d={res['cohens_d']:+.3f} p={res['p_value']:.3f}")

# Stability of roc_20d > 0 filter
output.append("")
output.append("  etf_roc_20d > 0 filter:")
for period_name, period_df in [("2020-2022", early), ("2023-2026", late)]:
    pv = period_df[period_df['etf_roc_20d'].notna()].copy()
    pv['roc20_pos'] = (pv['etf_roc_20d'] > 0).astype(int)
    res = binary_analysis(pv, 'roc20_pos')
    if res:
        output.append(f"    {period_name}: neg→WR={res['wr_0']:.1f}% Mean=${res['mean_pnl_0']:>10,.0f} (n={res['n_0']}) | "
                      f"pos→WR={res['wr_1']:.1f}% Mean=${res['mean_pnl_1']:>10,.0f} (n={res['n_1']}) | "
                      f"d={res['cohens_d']:+.3f} p={res['p_value']:.3f}")

# Stability by sector for the most promising features
output.append("")
output.append("  Stability by sector for 'ETF above 50 SMA':")
for sec in sorted(trades['sector'].unique()):
    sub = trades[trades['sector'] == sec]
    if len(sub) < 30:
        continue
    etf_name = sub['sector_etf'].iloc[0]
    output.append(f"    {sec} ({etf_name}):")
    for period_name, pslice in [("2020-2022", sub[sub['year_int'] <= 2022]), ("2023-2026", sub[sub['year_int'] >= 2023])]:
        if len(pslice) < 10:
            output.append(f"      {period_name}: n={len(pslice)} (too few)")
            continue
        res = binary_analysis(pslice, 'etf_above_sma50')
        if res:
            output.append(f"      {period_name}: below→Mean=${res['mean_pnl_0']:>10,.0f} WR={res['wr_0']:.1f}% (n={res['n_0']}) | "
                          f"above→Mean=${res['mean_pnl_1']:>10,.0f} WR={res['wr_1']:.1f}% (n={res['n_1']}) | d={res['cohens_d']:+.3f}")
        else:
            output.append(f"      {period_name}: n={len(pslice)} (insufficient split)")

###############################################################################
# FINAL SUMMARY
###############################################################################
output.append("")
output.append("=" * 90)
output.append("FINAL SUMMARY: KEY FINDINGS")
output.append("=" * 90)

# Compute summary stats for the final section
output.append("")
output.append("1. CROSS-SECTOR SPEARMAN CORRELATIONS (significant at p<0.05):")
for feat in all_features:
    valid = trades[[feat, 'hybrid_pnl']].dropna()
    if len(valid) > 20:
        rho, p = stats.spearmanr(valid[feat], valid['hybrid_pnl'])
        if p < 0.05:
            output.append(f"   {feat}: rho={rho:.4f}, p={p:.4f}")

output.append("")
output.append("2. STRONGEST SECTOR-SPECIFIC EFFECTS (|d| > 0.2):")
if sector_findings:
    for sec, feat, d, p, n in sorted(sector_findings, key=lambda x: abs(x[2]), reverse=True):
        output.append(f"   {sec:20s} {feat:25s} d={d:+.3f} p={p:.3f} n={n}")
else:
    output.append("   None found.")

output.append("")
output.append("3. BOTTOM LINE:")
# Compute key summary numbers
res_sma50 = binary_analysis(trades, 'etf_above_sma50')
res_aligned = binary_analysis(trades, 'etf_trend_aligned')
res_sma200 = binary_analysis(trades, 'etf_above_sma200')

if res_sma50:
    output.append(f"   ETF above 50 SMA:  d={res_sma50['cohens_d']:+.4f} p={res_sma50['p_value']:.4f}")
if res_aligned:
    output.append(f"   Trend aligned:     d={res_aligned['cohens_d']:+.4f} p={res_aligned['p_value']:.4f}")
if res_sma200:
    output.append(f"   ETF above 200 SMA: d={res_sma200['cohens_d']:+.4f} p={res_sma200['p_value']:.4f}")

# Write output
output_text = '\n'.join(output)
with open('/home/ubuntu/daily_data/analysis_results/sector_etf_analysis.txt', 'w') as f:
    f.write(output_text)

print("\n\nAnalysis complete. Written to sector_etf_analysis.txt")
print(f"Output length: {len(output)} lines")
