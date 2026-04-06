import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# LOAD DATA
###############################################################################

# 1) Stock trade log
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/backtest_optionD_noReentry_trades.csv')
trades['signal_date'] = pd.to_datetime(trades['signal_date'])
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])
print(f"Loaded {len(trades)} stock trades")

# 2) Options data
opts1 = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options.csv')
opts2 = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options_backfill.csv')

# Normalize option_type: treat 'backfill' as '30d_1m' equivalent
opts1['bar_date'] = pd.to_datetime(opts1['bar_date'])
opts1['trade_entry_date'] = pd.to_datetime(opts1['trade_entry_date'])
opts2['bar_date'] = pd.to_datetime(opts2['bar_date'])
opts2['trade_entry_date'] = pd.to_datetime(opts2['trade_entry_date'])

# Filter opts1 to 30d_1m only
opts1_30d = opts1[opts1['option_type'] == '30d_1m'].copy()
print(f"Options main (30d_1m): {len(opts1_30d)} rows, backfill: {len(opts2)} rows")

# Combine
all_opts = pd.concat([opts1_30d, opts2], ignore_index=True)
print(f"Combined options rows: {len(all_opts)}")

###############################################################################
# STEP 1: Classify trades by tier
###############################################################################

# For each trade, load the enriched data for the symbol and get signal_date features
enriched_cache = {}

def get_enriched(symbol):
    if symbol not in enriched_cache:
        fpath = f'/home/ubuntu/daily_data/data/{symbol}_enriched.csv'
        if Path(fpath).exists():
            df = pd.read_csv(fpath, parse_dates=['date'])
            df = df.sort_values('date').reset_index(drop=True)
            enriched_cache[symbol] = df
        else:
            enriched_cache[symbol] = None
    return enriched_cache[symbol]

tiers = []
body_pcts = []
atr_changes = []

for idx, row in trades.iterrows():
    symbol = row['symbol']
    sig_date = row['signal_date']

    df = get_enriched(symbol)
    if df is None:
        # Can't classify, default to Tier C
        tiers.append('C')
        body_pcts.append(np.nan)
        atr_changes.append(np.nan)
        continue

    # Find signal date row
    mask = df['date'] == sig_date
    if mask.sum() == 0:
        tiers.append('C')
        body_pcts.append(np.nan)
        atr_changes.append(np.nan)
        continue

    sig_idx = df[mask].index[0]
    sig_row = df.loc[sig_idx]

    # body_pct
    high = sig_row['high']
    low = sig_row['low']
    opn = sig_row['open']
    close = sig_row['close']
    rng = high - low
    if rng > 0:
        bp = (close - opn) / rng
    else:
        bp = 0.0

    # atr_change_3d: % change in ATR(14) over 3 days
    if sig_idx >= 3:
        atr_now = sig_row['atr_14']
        atr_3ago = df.loc[sig_idx - 3, 'atr_14']
        if pd.notna(atr_now) and pd.notna(atr_3ago) and atr_3ago > 0:
            atr_chg = (atr_now - atr_3ago) / atr_3ago
        else:
            atr_chg = 0.0
    else:
        atr_chg = 0.0

    strong_candle = bp > 0.23
    atr_rising = atr_chg > 0

    if strong_candle and atr_rising:
        tier = 'A'
    elif strong_candle or atr_rising:
        tier = 'B'
    else:
        tier = 'C'

    tiers.append(tier)
    body_pcts.append(bp)
    atr_changes.append(atr_chg)

trades['tier'] = tiers
trades['calc_body_pct'] = body_pcts
trades['calc_atr_change_3d'] = atr_changes

print(f"\nTier distribution:")
print(trades['tier'].value_counts().sort_index())

###############################################################################
# STEP 2: Tier C — use stock results directly
###############################################################################

tier_c = trades[trades['tier'] == 'C'].copy()
tier_c['source'] = 'stock'
tier_c['final_pnl'] = tier_c['net_pnl']

print(f"\nTier C: {len(tier_c)} trades, stock P&L: ${tier_c['net_pnl'].sum():,.0f}")

###############################################################################
# STEP 3: Tier A+B — simulate options
###############################################################################

tier_ab = trades[trades['tier'].isin(['A', 'B'])].copy()
print(f"\nTier A+B: {len(tier_ab)} trades to process for options")

# Build lookup: (symbol, entry_date) -> list of option bars
opt_lookup = {}
for _, orow in all_opts.iterrows():
    key = (orow['trade_symbol'], orow['trade_entry_date'])
    if key not in opt_lookup:
        opt_lookup[key] = []
    opt_lookup[key].append(orow)

# For each Tier A+B trade, try to get option prices
option_results = []

for idx, row in tier_ab.iterrows():
    symbol = row['symbol']
    entry_date = row['entry_date']
    tier = row['tier']

    # Sizing
    if tier == 'A':
        size = 200_000
    else:
        size = 100_000

    key = (symbol, entry_date)
    opt_bars = opt_lookup.get(key, [])

    if len(opt_bars) == 0:
        # No option data — fallback to stock
        option_results.append({
            'trade_idx': idx,
            'source': 'stock_fallback',
            'final_pnl': row['net_pnl'],
            'opt_entry': np.nan,
            'opt_exit': np.nan,
            'contracts': 0,
        })
        continue

    # Sort bars by bar_date
    bars_df = pd.DataFrame(opt_bars).sort_values('bar_date').reset_index(drop=True)

    # Entry price: morning_price on entry_date
    entry_bars = bars_df[bars_df['bar_date'] == entry_date]
    if len(entry_bars) == 0:
        # Try first bar
        entry_bars = bars_df.head(1)

    entry_price = entry_bars.iloc[0]['morning_price']
    if pd.isna(entry_price) or entry_price <= 0:
        entry_price = entry_bars.iloc[0]['day_open']
    if pd.isna(entry_price) or entry_price <= 0:
        option_results.append({
            'trade_idx': idx,
            'source': 'stock_fallback',
            'final_pnl': row['net_pnl'],
            'opt_entry': np.nan,
            'opt_exit': np.nan,
            'contracts': 0,
        })
        continue

    # Contracts
    contracts = int(size / (entry_price * 100))
    if contracts < 1:
        option_results.append({
            'trade_idx': idx,
            'source': 'stock_fallback',
            'final_pnl': row['net_pnl'],
            'opt_entry': np.nan,
            'opt_exit': np.nan,
            'contracts': 0,
        })
        continue

    # Exit price: 13th trading day from entry
    # bars_df has one row per trading day for this option
    # Find bars from entry_date onward
    future_bars = bars_df[bars_df['bar_date'] >= entry_date].sort_values('bar_date').reset_index(drop=True)

    if len(future_bars) >= 13:
        exit_bar = future_bars.iloc[12]  # 0-indexed, 13th day = index 12
    elif len(future_bars) > 0:
        exit_bar = future_bars.iloc[-1]  # Use last available
    else:
        option_results.append({
            'trade_idx': idx,
            'source': 'stock_fallback',
            'final_pnl': row['net_pnl'],
            'opt_entry': np.nan,
            'opt_exit': np.nan,
            'contracts': 0,
        })
        continue

    exit_price = exit_bar['morning_price']
    if pd.isna(exit_price) or exit_price <= 0:
        exit_price = exit_bar['day_open']
    if pd.isna(exit_price) or exit_price <= 0:
        exit_price = exit_bar['day_close']
    if pd.isna(exit_price) or exit_price <= 0:
        option_results.append({
            'trade_idx': idx,
            'source': 'stock_fallback',
            'final_pnl': row['net_pnl'],
            'opt_entry': np.nan,
            'opt_exit': np.nan,
            'contracts': 0,
        })
        continue

    # P&L
    gross_pnl = contracts * (exit_price - entry_price) * 100
    # Costs: 5 bps each way on notional
    entry_notional = contracts * entry_price * 100
    exit_notional = contracts * exit_price * 100
    costs = entry_notional * 0.0005 + exit_notional * 0.0005
    net_pnl = gross_pnl - costs

    option_results.append({
        'trade_idx': idx,
        'source': 'option',
        'final_pnl': net_pnl,
        'opt_entry': entry_price,
        'opt_exit': exit_price,
        'contracts': contracts,
    })

opt_df = pd.DataFrame(option_results)
print(f"\nOption results: {len(opt_df)} trades")
print(f"  Options used: {(opt_df['source']=='option').sum()}")
print(f"  Stock fallback: {(opt_df['source']=='stock_fallback').sum()}")

# Merge back
tier_ab['source'] = opt_df['source'].values
tier_ab['final_pnl'] = opt_df['final_pnl'].values
tier_ab['opt_entry'] = opt_df['opt_entry'].values
tier_ab['opt_exit'] = opt_df['opt_exit'].values
tier_ab['contracts'] = opt_df['contracts'].values

###############################################################################
# STEP 4: Combine into one portfolio
###############################################################################

combined = pd.concat([tier_c, tier_ab], ignore_index=False).sort_values('entry_date')
combined['year'] = combined['entry_date'].dt.year

total_pnl = combined['final_pnl'].sum()
total_trades = len(combined)
wins = (combined['final_pnl'] > 0).sum()
losses = (combined['final_pnl'] <= 0).sum()
wr = wins / total_trades * 100
gross_wins = combined[combined['final_pnl'] > 0]['final_pnl'].sum()
gross_losses = abs(combined[combined['final_pnl'] <= 0]['final_pnl'].sum())
pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

# Max drawdown
cum_pnl = combined['final_pnl'].cumsum()
running_max = cum_pnl.cummax()
drawdown = cum_pnl - running_max
max_dd = drawdown.min()

print(f"\n=== COMBINED PORTFOLIO ===")
print(f"Total Trades: {total_trades}")
print(f"Total P&L: ${total_pnl:,.0f}")
print(f"Win Rate: {wr:.1f}%")
print(f"Profit Factor: {pf:.2f}")
print(f"Max Drawdown: ${max_dd:,.0f}")

###############################################################################
# DETAILED REPORT
###############################################################################

report_lines = []
def rpt(line=""):
    report_lines.append(line)

rpt("=" * 80)
rpt("HYBRID PORTFOLIO BACKTEST REPORT")
rpt("Tier C = Stock (ATR stop, 20-day hold)")
rpt("Tier A+B = Options (30d 1-month, 13-day hold, no stop)")
rpt("=" * 80)

# A) PORTFOLIO BREAKDOWN
rpt()
rpt("A) PORTFOLIO BREAKDOWN")
rpt("-" * 60)

# Tier C stock
tc = combined[combined['tier'] == 'C']
tc_wins = (tc['final_pnl'] > 0).sum()
tc_losses = (tc['final_pnl'] <= 0).sum()
tc_gw = tc[tc['final_pnl'] > 0]['final_pnl'].sum()
tc_gl = abs(tc[tc['final_pnl'] <= 0]['final_pnl'].sum())
tc_pf = tc_gw / tc_gl if tc_gl > 0 else float('inf')

rpt(f"\n  TIER C (Stock trades):")
rpt(f"    Count:     {len(tc)}")
rpt(f"    Total P&L: ${tc['final_pnl'].sum():>14,.0f}")
rpt(f"    Win Rate:  {tc_wins/len(tc)*100:.1f}%")
rpt(f"    Profit Factor: {tc_pf:.2f}")
rpt(f"    Avg P&L:   ${tc['final_pnl'].mean():>14,.0f}")
rpt(f"    Median P&L:${tc['final_pnl'].median():>14,.0f}")

# Tier A+B with options
tab_opt = combined[(combined['tier'].isin(['A','B'])) & (combined['source'] == 'option')]
if len(tab_opt) > 0:
    tab_opt_wins = (tab_opt['final_pnl'] > 0).sum()
    tab_opt_gl = abs(tab_opt[tab_opt['final_pnl'] <= 0]['final_pnl'].sum())
    tab_opt_gw = tab_opt[tab_opt['final_pnl'] > 0]['final_pnl'].sum()
    tab_opt_pf = tab_opt_gw / tab_opt_gl if tab_opt_gl > 0 else float('inf')
    rpt(f"\n  TIER A+B (Option trades):")
    rpt(f"    Count:     {len(tab_opt)}")
    rpt(f"    Total P&L: ${tab_opt['final_pnl'].sum():>14,.0f}")
    rpt(f"    Win Rate:  {tab_opt_wins/len(tab_opt)*100:.1f}%")
    rpt(f"    Profit Factor: {tab_opt_pf:.2f}")
    rpt(f"    Avg P&L:   ${tab_opt['final_pnl'].mean():>14,.0f}")
    rpt(f"    Median P&L:${tab_opt['final_pnl'].median():>14,.0f}")

# Tier A+B breakdown by tier
for t in ['A', 'B']:
    sub = tab_opt[tab_opt['tier'] == t]
    if len(sub) > 0:
        sw = (sub['final_pnl'] > 0).sum()
        rpt(f"      Tier {t}: {len(sub)} trades, P&L: ${sub['final_pnl'].sum():>12,.0f}, WR: {sw/len(sub)*100:.1f}%, Avg: ${sub['final_pnl'].mean():>10,.0f}")

# Tier A+B stock fallback
tab_fb = combined[(combined['tier'].isin(['A','B'])) & (combined['source'] == 'stock_fallback')]
rpt(f"\n  TIER A+B (Stock fallback — no option data):")
rpt(f"    Count:     {len(tab_fb)}")
rpt(f"    Total P&L: ${tab_fb['final_pnl'].sum():>14,.0f}")
if len(tab_fb) > 0:
    fb_wins = (tab_fb['final_pnl'] > 0).sum()
    rpt(f"    Win Rate:  {fb_wins/len(tab_fb)*100:.1f}%")

# Combined
rpt(f"\n  COMBINED PORTFOLIO:")
rpt(f"    Total Trades: {total_trades}")
rpt(f"    Total P&L:    ${total_pnl:>14,.0f}")
rpt(f"    Win Rate:     {wr:.1f}%")
rpt(f"    Profit Factor:{pf:>8.2f}")
rpt(f"    Avg P&L:      ${combined['final_pnl'].mean():>14,.0f}")
rpt(f"    Max Drawdown: ${max_dd:>14,.0f}")

# B) COMPARISON TABLE
rpt()
rpt()
rpt("B) COMPARISON TABLE")
rpt("-" * 80)

stock_only_pnl = 88_200_000  # given
options_only_pnl = 36_100_000  # given
overlay_pnl = 124_300_000  # given approximate

# Compute stock-only stats from the full trade log
stock_all_wins = (trades['net_pnl'] > 0).sum()
stock_all_wr = stock_all_wins / len(trades) * 100
stock_gw = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
stock_gl = abs(trades[trades['net_pnl'] <= 0]['net_pnl'].sum())
stock_pf = stock_gw / stock_gl if stock_gl > 0 else float('inf')
stock_cum = trades.sort_values('entry_date')['net_pnl'].cumsum()
stock_dd = (stock_cum - stock_cum.cummax()).min()

rpt(f"  {'Metric':<25} {'Hybrid':>15} {'Stock Only':>15} {'Options Only':>15} {'Overlay':>15}")
rpt(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
rpt(f"  {'Total Trades':<25} {total_trades:>15,} {len(trades):>15,} {'~2,082':>15} {'~4,164':>15}")
rpt(f"  {'Total P&L':<25} {'${:,.0f}'.format(total_pnl):>15} {'$88,200,000':>15} {'$36,100,000':>15} {'~$124,300,000':>15}")
rpt(f"  {'Win Rate':<25} {'{:.1f}%'.format(wr):>15} {'{:.1f}%'.format(stock_all_wr):>15} {'N/A':>15} {'N/A':>15}")
rpt(f"  {'Profit Factor':<25} {'{:.2f}'.format(pf):>15} {'{:.2f}'.format(stock_pf):>15} {'N/A':>15} {'N/A':>15}")
rpt(f"  {'Max Drawdown':<25} {'${:,.0f}'.format(max_dd):>15} {'${:,.0f}'.format(stock_dd):>15} {'N/A':>15} {'N/A':>15}")
rpt(f"  {'Avg P&L/Trade':<25} {'${:,.0f}'.format(combined['final_pnl'].mean()):>15} {'${:,.0f}'.format(trades['net_pnl'].mean()):>15} {'N/A':>15} {'N/A':>15}")

# C) YEAR-BY-YEAR
rpt()
rpt()
rpt("C) YEAR-BY-YEAR PERFORMANCE")
rpt("-" * 90)
rpt(f"  {'Year':<6} {'Trades':>7} {'Total P&L':>14} {'WR':>7} {'PF':>7} {'Avg P&L':>12} {'Stock PnL':>12} {'Opt PnL':>12} {'FB PnL':>12}")
rpt(f"  {'----':<6} {'------':>7} {'---------':>14} {'--':>7} {'--':>7} {'-------':>12} {'---------':>12} {'-------':>12} {'------':>12}")

for year in sorted(combined['year'].unique()):
    ydf = combined[combined['year'] == year]
    yw = (ydf['final_pnl'] > 0).sum()
    ywr = yw / len(ydf) * 100 if len(ydf) > 0 else 0
    ygw = ydf[ydf['final_pnl'] > 0]['final_pnl'].sum()
    ygl = abs(ydf[ydf['final_pnl'] <= 0]['final_pnl'].sum())
    ypf = ygw / ygl if ygl > 0 else float('inf')

    y_stock = ydf[ydf['source'] == 'stock']['final_pnl'].sum()
    y_opt = ydf[ydf['source'] == 'option']['final_pnl'].sum()
    y_fb = ydf[ydf['source'] == 'stock_fallback']['final_pnl'].sum()

    rpt(f"  {year:<6} {len(ydf):>7} ${ydf['final_pnl'].sum():>13,.0f} {ywr:>6.1f}% {ypf:>6.2f} ${ydf['final_pnl'].mean():>11,.0f} ${y_stock:>11,.0f} ${y_opt:>11,.0f} ${y_fb:>11,.0f}")

# D) CAPITAL EFFICIENCY
rpt()
rpt()
rpt("D) CAPITAL EFFICIENCY")
rpt("-" * 60)

# For stock trades (Tier C), position_size is in the data
# For option trades, we used $200K (A) or $100K (B)
# Calculate avg capital deployed per day

# Build daily capital deployed
from collections import defaultdict

daily_capital_hybrid = defaultdict(float)
daily_capital_stock = defaultdict(float)

for _, row in combined.iterrows():
    entry = row['entry_date']
    exit_d = row['exit_date']
    if pd.isna(exit_d):
        continue

    dates = pd.bdate_range(entry, exit_d)

    if row['source'] == 'stock' or row['source'] == 'stock_fallback':
        cap = row['position_size']
    elif row['source'] == 'option':
        cap = 200_000 if row['tier'] == 'A' else 100_000
    else:
        cap = row['position_size']

    for d in dates:
        daily_capital_hybrid[d] += cap

# Stock only capital
for _, row in trades.iterrows():
    entry = row['entry_date']
    exit_d = row['exit_date']
    if pd.isna(exit_d):
        continue
    dates = pd.bdate_range(entry, exit_d)
    for d in dates:
        daily_capital_stock[d] += row['position_size']

if daily_capital_hybrid:
    avg_cap_hybrid = np.mean(list(daily_capital_hybrid.values()))
    max_cap_hybrid = max(daily_capital_hybrid.values())
else:
    avg_cap_hybrid = 0
    max_cap_hybrid = 0

if daily_capital_stock:
    avg_cap_stock = np.mean(list(daily_capital_stock.values()))
    max_cap_stock = max(daily_capital_stock.values())
else:
    avg_cap_stock = 0
    max_cap_stock = 0

roc_hybrid = (total_pnl / avg_cap_hybrid * 100) if avg_cap_hybrid > 0 else 0
roc_stock = (trades['net_pnl'].sum() / avg_cap_stock * 100) if avg_cap_stock > 0 else 0

rpt(f"  {'Metric':<35} {'Hybrid':>15} {'Stock Only':>15}")
rpt(f"  {'-'*35} {'-'*15} {'-'*15}")
rpt(f"  {'Avg Daily Capital Deployed':<35} {'${:,.0f}'.format(avg_cap_hybrid):>15} {'${:,.0f}'.format(avg_cap_stock):>15}")
rpt(f"  {'Max Daily Capital Deployed':<35} {'${:,.0f}'.format(max_cap_hybrid):>15} {'${:,.0f}'.format(max_cap_stock):>15}")
rpt(f"  {'Total Return':<35} {'${:,.0f}'.format(total_pnl):>15} {'${:,.0f}'.format(trades['net_pnl'].sum()):>15}")
rpt(f"  {'Return on Avg Capital':<35} {'{:.1f}%'.format(roc_hybrid):>15} {'{:.1f}%'.format(roc_stock):>15}")

# Capital reduction from hybrid
if avg_cap_stock > 0:
    cap_reduction = (1 - avg_cap_hybrid / avg_cap_stock) * 100
    rpt(f"  {'Capital Reduction (Hybrid)':<35} {'{:.1f}%'.format(cap_reduction):>15}")

# E) TOP 10 WINNERS AND LOSERS
rpt()
rpt()
rpt("E) TOP 10 WINNERS AND TOP 10 LOSERS")
rpt("-" * 90)

rpt(f"\n  TOP 10 WINNERS:")
rpt(f"  {'#':>3} {'Symbol':<8} {'Entry':>12} {'Tier':>5} {'Source':>10} {'P&L':>14} {'Return%':>10}")
rpt(f"  {'---':>3} {'------':<8} {'-----':>12} {'----':>5} {'------':>10} {'---':>14} {'-------':>10}")

top_winners = combined.nlargest(10, 'final_pnl')
for i, (_, row) in enumerate(top_winners.iterrows(), 1):
    ret = row.get('return_pct', 0)
    if row['source'] == 'option' and 'opt_entry' in row and row['opt_entry'] > 0:
        ret = (row['opt_exit'] - row['opt_entry']) / row['opt_entry'] * 100
    rpt(f"  {i:>3} {row['symbol']:<8} {str(row['entry_date'].date()):>12} {row['tier']:>5} {row['source']:>10} ${row['final_pnl']:>13,.0f} {ret:>9.1f}%")

rpt(f"\n  TOP 10 LOSERS:")
rpt(f"  {'#':>3} {'Symbol':<8} {'Entry':>12} {'Tier':>5} {'Source':>10} {'P&L':>14} {'Return%':>10}")
rpt(f"  {'---':>3} {'------':<8} {'-----':>12} {'----':>5} {'------':>10} {'---':>14} {'-------':>10}")

top_losers = combined.nsmallest(10, 'final_pnl')
for i, (_, row) in enumerate(top_losers.iterrows(), 1):
    ret = row.get('return_pct', 0)
    if row['source'] == 'option' and 'opt_entry' in row and row['opt_entry'] > 0:
        ret = (row['opt_exit'] - row['opt_entry']) / row['opt_entry'] * 100
    rpt(f"  {i:>3} {row['symbol']:<8} {str(row['entry_date'].date()):>12} {row['tier']:>5} {row['source']:>10} ${row['final_pnl']:>13,.0f} {ret:>9.1f}%")

# F) RISK PROFILE
rpt()
rpt()
rpt("F) RISK PROFILE")
rpt("-" * 60)

# Monthly P&L
combined['month'] = combined['entry_date'].dt.to_period('M')
monthly_pnl = combined.groupby('month')['final_pnl'].sum()
worst_month = monthly_pnl.idxmin()
worst_month_val = monthly_pnl.min()

# Longest drawdown period
cum_pnl_series = combined.sort_values('entry_date')['final_pnl'].cumsum()
cum_pnl_series.index = combined.sort_values('entry_date')['entry_date'].values
running_max_series = cum_pnl_series.cummax()
in_drawdown = cum_pnl_series < running_max_series

# Find longest consecutive drawdown by date
dd_dates = cum_pnl_series.index[in_drawdown]
if len(dd_dates) > 0:
    # Group consecutive drawdown periods
    longest_dd_start = dd_dates[0]
    longest_dd_end = dd_dates[0]
    current_start = dd_dates[0]

    # Actually compute by checking when we recover from drawdown
    sorted_combined = combined.sort_values('entry_date')
    cum = sorted_combined['final_pnl'].cumsum().values
    dates_arr = sorted_combined['entry_date'].values
    rmax = np.maximum.accumulate(cum)

    max_dd_days = 0
    dd_start_idx = 0
    best_dd_start = 0
    best_dd_end = 0

    for i in range(len(cum)):
        if cum[i] >= rmax[i]:
            dd_start_idx = i
        else:
            dd_len = (pd.Timestamp(dates_arr[i]) - pd.Timestamp(dates_arr[dd_start_idx])).days
            if dd_len > max_dd_days:
                max_dd_days = dd_len
                best_dd_start = dd_start_idx
                best_dd_end = i

    longest_dd_str = f"{pd.Timestamp(dates_arr[best_dd_start]).date()} to {pd.Timestamp(dates_arr[best_dd_end]).date()} ({max_dd_days} days)"
else:
    longest_dd_str = "No drawdown"

# Max concurrent positions
daily_positions = defaultdict(int)
for _, row in combined.iterrows():
    entry = row['entry_date']
    exit_d = row['exit_date']
    if pd.isna(exit_d):
        continue
    dates = pd.bdate_range(entry, exit_d)
    for d in dates:
        daily_positions[d] += 1

max_concurrent = max(daily_positions.values()) if daily_positions else 0

# Max concurrent by type
daily_stock_pos = defaultdict(int)
daily_opt_pos = defaultdict(int)
for _, row in combined.iterrows():
    entry = row['entry_date']
    exit_d = row['exit_date']
    if pd.isna(exit_d):
        continue
    dates = pd.bdate_range(entry, exit_d)
    for d in dates:
        if row['source'] == 'option':
            daily_opt_pos[d] += 1
        else:
            daily_stock_pos[d] += 1

max_stock_pos = max(daily_stock_pos.values()) if daily_stock_pos else 0
max_opt_pos = max(daily_opt_pos.values()) if daily_opt_pos else 0

rpt(f"  Max Drawdown:            ${max_dd:>14,.0f}")
rpt(f"  Worst Month:             {worst_month} (${worst_month_val:>12,.0f})")
rpt(f"  Longest Drawdown Period: {longest_dd_str}")
rpt(f"  Max Concurrent Positions:{max_concurrent:>5} (Stock: {max_stock_pos}, Options: {max_opt_pos})")

# Best month
best_month = monthly_pnl.idxmax()
best_month_val = monthly_pnl.max()
rpt(f"  Best Month:              {best_month} (${best_month_val:>12,.0f})")

# Avg monthly P&L
rpt(f"  Avg Monthly P&L:         ${monthly_pnl.mean():>14,.0f}")
rpt(f"  Monthly P&L Std Dev:     ${monthly_pnl.std():>14,.0f}")

# Sharpe-like ratio (monthly)
if monthly_pnl.std() > 0:
    sharpe_monthly = monthly_pnl.mean() / monthly_pnl.std()
    sharpe_annual = sharpe_monthly * np.sqrt(12)
    rpt(f"  Annualized Sharpe Ratio: {sharpe_annual:>8.2f}")

# Percent of months profitable
pct_months_profitable = (monthly_pnl > 0).sum() / len(monthly_pnl) * 100
rpt(f"  Profitable Months:       {(monthly_pnl > 0).sum()}/{len(monthly_pnl)} ({pct_months_profitable:.0f}%)")

###############################################################################
# Write report
###############################################################################

report_text = "\n".join(report_lines)
print(report_text)

with open('/home/ubuntu/daily_data/analysis_results/hybrid_portfolio_report.txt', 'w') as f:
    f.write(report_text)

print(f"\n\nReport saved to /home/ubuntu/daily_data/analysis_results/hybrid_portfolio_report.txt")
