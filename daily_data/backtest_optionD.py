import pandas as pd
import numpy as np
import os
from collections import defaultdict
import math

# ── LOAD DATA ──
trades_df = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv')
enriched_df = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_enriched.csv')

# Merge entry_body_pct from enriched into trades
enriched_sub = enriched_df[['symbol', 'signal_date', 'entry_date', 'entry_body_pct']].copy()
trades_df = trades_df.merge(enriched_sub, on=['symbol', 'signal_date', 'entry_date'], how='left')

# ── QUINTILE ASSIGNMENT ──
def assign_quintile(bp):
    if pd.isna(bp):
        return None
    if bp < -0.53:
        return 'Q1'
    elif bp < -0.17:
        return 'Q2'
    elif bp < 0.23:
        return 'Q3'
    elif bp < 0.57:
        return 'Q4'
    else:
        return 'Q5'

# Note: Q5 boundary in spec says >0.58 but Q4 says <0.57, so Q5 is >=0.57 effectively
# Actually spec says Q4: 0.23 to 0.57, Q5: >0.58. There's a gap 0.57-0.58.
# Use Q4 < 0.57, Q5 >= 0.57 to not lose trades in the gap.
# Re-reading: Q4 0.23 to 0.57 means <=0.57, Q5 > 0.58 means >0.58. Gap is 0.57-0.58.
# Let's use the literal boundaries: Q4 up to 0.57 inclusive, Q5 > 0.57
def assign_quintile(bp):
    if pd.isna(bp):
        return None
    if bp < -0.53:
        return 'Q1'
    elif bp < -0.17:
        return 'Q2'
    elif bp < 0.23:
        return 'Q3'
    elif bp <= 0.57:
        return 'Q4'
    else:
        return 'Q5'

position_sizes = {'Q1': 0, 'Q2': 500000, 'Q3': 1000000, 'Q4': 1500000, 'Q5': 2000000}

trades_df['quintile'] = trades_df['entry_body_pct'].apply(assign_quintile)

# ── LOAD SYMBOL DATA CACHE ──
symbol_data_cache = {}
def get_symbol_data(symbol):
    if symbol not in symbol_data_cache:
        fpath = f'/home/ubuntu/daily_data/data/{symbol}_enriched.csv'
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            symbol_data_cache[symbol] = df
        else:
            symbol_data_cache[symbol] = None
    return symbol_data_cache[symbol]

# ── SIMULATE EACH TRADE ──
results = []
cost_bps = 5  # each way

for idx, row in trades_df.iterrows():
    q = row['quintile']
    if q is None or q == 'Q1':
        # Skip Q1 trades
        continue

    pos_size = position_sizes[q]
    symbol = row['symbol']
    signal_date = row['signal_date']
    entry_date = row['entry_date']

    sdf = get_symbol_data(symbol)
    if sdf is None:
        continue

    # Find signal_date row for low and atr_14
    signal_rows = sdf[sdf['date'] == pd.to_datetime(signal_date)]
    if len(signal_rows) == 0:
        continue
    signal_row = signal_rows.iloc[0]
    signal_day_low = signal_row['low']
    atr_14 = signal_row['atr_14']

    if pd.isna(signal_day_low) or pd.isna(atr_14):
        continue

    stop_price = signal_day_low - 3.0 * atr_14

    # Find entry_date row
    entry_rows = sdf[sdf['date'] == pd.to_datetime(entry_date)]
    if len(entry_rows) == 0:
        continue
    entry_row = entry_rows.iloc[0]
    entry_price = entry_row['open']

    if pd.isna(entry_price) or entry_price <= 0:
        continue

    shares = int(math.floor(pos_size / entry_price))
    if shares <= 0:
        continue

    # Find entry_date index
    entry_idx = entry_rows.index[0]

    # Walk through next 20 trading days
    exit_type = 'TIME'
    exit_price = None
    exit_date = None
    days_held = 0

    # Day 1 is entry_date, we check days 1-20
    for day_offset in range(20):
        current_idx = entry_idx + day_offset
        if current_idx >= len(sdf):
            break

        day_row = sdf.iloc[current_idx]
        days_held = day_offset + 1

        # Check if low hits stop
        if day_row['low'] <= stop_price:
            exit_type = 'STOP'
            exit_price = stop_price
            exit_date = day_row['date']
            break

        # If day 20, exit at close
        if day_offset == 19:
            exit_price = day_row['close']
            exit_date = day_row['date']
            break

    if exit_price is None:
        # Ran out of data
        # Exit at last available close
        last_idx = min(entry_idx + 19, len(sdf) - 1)
        exit_price = sdf.iloc[last_idx]['close']
        exit_date = sdf.iloc[last_idx]['date']
        days_held = last_idx - entry_idx + 1
        exit_type = 'TIME'

    # Compute PnL
    gross_pnl = shares * (exit_price - entry_price)
    entry_cost = shares * entry_price * cost_bps / 10000
    exit_cost = shares * exit_price * cost_bps / 10000
    net_pnl = gross_pnl - entry_cost - exit_cost

    # For stopped trades: what would day 20 close have been?
    day20_close = None
    if exit_type == 'STOP':
        day20_idx = entry_idx + 19
        if day20_idx < len(sdf):
            day20_close = sdf.iloc[day20_idx]['close']

    results.append({
        'symbol': symbol,
        'signal_date': signal_date,
        'entry_date': entry_date,
        'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
        'quintile': q,
        'position_size': pos_size,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'stop_price': stop_price,
        'shares': shares,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'return_pct': (exit_price / entry_price - 1) * 100,
        'days_held': days_held,
        'exit_type': exit_type,
        'day20_close': day20_close,
    })

res_df = pd.DataFrame(results)

# Save trade log
res_df.to_csv('/home/ubuntu/daily_data/analysis_results/backtest_optionD_trades.csv', index=False)

print(f"Total trades simulated: {len(res_df)}")
print(f"Quintile distribution:")
print(res_df['quintile'].value_counts().sort_index())
print(f"\nSkipped Q1 trades: {len(trades_df[trades_df['quintile'] == 'Q1'])}")
print(f"Skipped None quintile: {len(trades_df[trades_df['quintile'].isna()])}")

# ══════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════

report_lines = []
def rpt(s=''):
    report_lines.append(s)

rpt("=" * 80)
rpt("OPTION D BACKTEST: Skip Q1 weak candles, tier-size the rest")
rpt("=" * 80)
rpt()

# ── A) OVERALL STATS ──
total_trades = len(res_df)
total_net_pnl = res_df['net_pnl'].sum()
winners = res_df[res_df['net_pnl'] > 0]
losers = res_df[res_df['net_pnl'] <= 0]
win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
gross_wins = winners['net_pnl'].sum()
gross_losses = abs(losers['net_pnl'].sum())
profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
avg_pnl = res_df['net_pnl'].mean()
median_pnl = res_df['net_pnl'].median()
worst_trade = res_df['net_pnl'].min()
best_trade = res_df['net_pnl'].max()
stop_hits = len(res_df[res_df['exit_type'] == 'STOP'])
stop_rate = stop_hits / total_trades * 100

# Max drawdown
equity_curve = res_df['net_pnl'].cumsum()
running_max = equity_curve.cummax()
drawdown = equity_curve - running_max
max_dd = drawdown.min()
max_dd_end_idx = drawdown.idxmin()
# Find start of this drawdown
peak_idx = equity_curve[:max_dd_end_idx + 1].idxmax()

# Sharpe ratio (per-trade, annualized roughly)
# Use per-trade returns; assume ~250 trading days, avg hold ~10 days => ~25 trades/year
avg_return_pct = res_df['return_pct'].mean()
std_return_pct = res_df['return_pct'].std()
trades_per_year = 250 / res_df['days_held'].mean() if res_df['days_held'].mean() > 0 else 25
sharpe = (avg_return_pct / std_return_pct) * np.sqrt(trades_per_year) if std_return_pct > 0 else 0

# Calmar ratio
years_span = (pd.to_datetime(res_df['exit_date'].iloc[-1]) - pd.to_datetime(res_df['entry_date'].iloc[0])).days / 365.25
annual_return = total_net_pnl / years_span if years_span > 0 else 0
calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

# Capital deployed per day
# For each trade, capital = shares * entry_price, deployed for days_held days
res_df['capital_deployed'] = res_df['shares'] * res_df['entry_price']
res_df['capital_days'] = res_df['capital_deployed'] * res_df['days_held']
total_trading_days = (pd.to_datetime(res_df['exit_date'].max()) - pd.to_datetime(res_df['entry_date'].min())).days
avg_capital_per_day = res_df['capital_days'].sum() / total_trading_days if total_trading_days > 0 else 0

# Max capital on any single day
from datetime import timedelta
date_capital = defaultdict(float)
for _, r in res_df.iterrows():
    ed = pd.to_datetime(r['entry_date'])
    exd = pd.to_datetime(r['exit_date'])
    cap = r['capital_deployed']
    d = ed
    while d <= exd:
        date_capital[d] += cap
        d += timedelta(days=1)
max_capital = max(date_capital.values()) if date_capital else 0

rpt("A) OVERALL STATS")
rpt("-" * 60)
rpt(f"  Total trades:              {total_trades:,}")
rpt(f"  Total net P&L:             ${total_net_pnl:,.0f}")
rpt(f"  Win rate:                  {win_rate:.1f}%")
rpt(f"  Profit factor:             {profit_factor:.2f}")
rpt(f"  Avg P&L per trade:         ${avg_pnl:,.0f}")
rpt(f"  Median P&L per trade:      ${median_pnl:,.0f}")
rpt(f"  Max drawdown:              ${max_dd:,.0f}")
rpt(f"  Worst trade:               ${worst_trade:,.0f}")
rpt(f"  Best trade:                ${best_trade:,.0f}")
rpt(f"  Sharpe ratio:              {sharpe:.2f}")
rpt(f"  Calmar ratio:              {calmar:.2f}")
rpt(f"  Avg capital deployed/day:  ${avg_capital_per_day:,.0f}")
rpt(f"  Max capital deployed:      ${max_capital:,.0f}")
rpt(f"  Stop hit rate:             {stop_rate:.1f}%")
rpt()

# ── B) BY QUINTILE ──
rpt("B) BY QUINTILE (Q2-Q5)")
rpt("-" * 60)
rpt(f"  {'Quintile':<10} {'N':>6} {'Pos Size':>10} {'Total P&L':>14} {'Avg P&L':>12} {'WR':>7} {'PF':>7}")
rpt(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*14} {'-'*12} {'-'*7} {'-'*7}")

for q in ['Q2', 'Q3', 'Q4', 'Q5']:
    qdf = res_df[res_df['quintile'] == q]
    if len(qdf) == 0:
        continue
    n = len(qdf)
    ps = position_sizes[q]
    tp = qdf['net_pnl'].sum()
    ap = qdf['net_pnl'].mean()
    w = len(qdf[qdf['net_pnl'] > 0])
    wr = w / n * 100
    gw = qdf[qdf['net_pnl'] > 0]['net_pnl'].sum()
    gl = abs(qdf[qdf['net_pnl'] <= 0]['net_pnl'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    rpt(f"  {q:<10} {n:>6} ${ps/1e6:.1f}M{' ':>4} ${tp:>12,.0f} ${ap:>10,.0f} {wr:>6.1f}% {pf:>6.2f}")
rpt()

# ── C) YEAR BY YEAR ──
rpt("C) YEAR BY YEAR")
rpt("-" * 60)
res_df['entry_year'] = pd.to_datetime(res_df['entry_date']).dt.year
rpt(f"  {'Year':<6} {'Trades':>7} {'Net P&L':>14} {'WR':>7} {'PF':>7} {'Max DD':>14}")
rpt(f"  {'-'*6} {'-'*7} {'-'*14} {'-'*7} {'-'*7} {'-'*14}")

for year in range(2020, 2027):
    ydf = res_df[res_df['entry_year'] == year]
    if len(ydf) == 0:
        continue
    n = len(ydf)
    tp = ydf['net_pnl'].sum()
    w = len(ydf[ydf['net_pnl'] > 0])
    wr = w / n * 100
    gw = ydf[ydf['net_pnl'] > 0]['net_pnl'].sum()
    gl = abs(ydf[ydf['net_pnl'] <= 0]['net_pnl'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    # Max DD for year
    ec = ydf['net_pnl'].cumsum()
    rm = ec.cummax()
    dd = ec - rm
    mdd = dd.min()
    rpt(f"  {year:<6} {n:>7} ${tp:>12,.0f} {wr:>6.1f}% {pf:>6.2f} ${mdd:>12,.0f}")
rpt()

# ── D) MONTHLY P&L TABLE ──
rpt("D) MONTHLY P&L TABLE")
rpt("-" * 60)
res_df['entry_month'] = pd.to_datetime(res_df['entry_date']).dt.to_period('M')
monthly = res_df.groupby('entry_month')['net_pnl'].sum()

rpt(f"  {'Month':<10} {'Net P&L':>14} {'Trades':>7}")
rpt(f"  {'-'*10} {'-'*14} {'-'*7}")
monthly_counts = res_df.groupby('entry_month')['net_pnl'].count()
for period in monthly.index:
    rpt(f"  {str(period):<10} ${monthly[period]:>12,.0f} {monthly_counts[period]:>7}")
rpt()

# ── E) TOP 10 WINNERS AND LOSERS ──
rpt("E) TOP 10 WINNERS")
rpt("-" * 60)
top10w = res_df.nlargest(10, 'net_pnl')
rpt(f"  {'Symbol':<8} {'Entry Date':<12} {'Exit Date':<12} {'Entry':>10} {'Exit':>10} {'P&L':>14} {'Pos Size':>10}")
rpt(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*14} {'-'*10}")
for _, r in top10w.iterrows():
    rpt(f"  {r['symbol']:<8} {r['entry_date']:<12} {r['exit_date']:<12} ${r['entry_price']:>8,.2f} ${r['exit_price']:>8,.2f} ${r['net_pnl']:>12,.0f} ${r['position_size']/1e6:.1f}M")
rpt()

rpt("   TOP 10 LOSERS")
rpt("-" * 60)
top10l = res_df.nsmallest(10, 'net_pnl')
rpt(f"  {'Symbol':<8} {'Entry Date':<12} {'Exit Date':<12} {'Entry':>10} {'Exit':>10} {'P&L':>14} {'Pos Size':>10}")
rpt(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*14} {'-'*10}")
for _, r in top10l.iterrows():
    rpt(f"  {r['symbol']:<8} {r['entry_date']:<12} {r['exit_date']:<12} ${r['entry_price']:>8,.2f} ${r['exit_price']:>8,.2f} ${r['net_pnl']:>12,.0f} ${r['position_size']/1e6:.1f}M")
rpt()

# ── F) EXIT TYPE BREAKDOWN ──
rpt("F) EXIT TYPE BREAKDOWN")
rpt("-" * 60)
stopped = res_df[res_df['exit_type'] == 'STOP']
timed = res_df[res_df['exit_type'] == 'TIME']
rpt(f"  Stopped exits:      {len(stopped):>6}  Avg P&L: ${stopped['net_pnl'].mean():>12,.0f}")
rpt(f"  Time exits:         {len(timed):>6}  Avg P&L: ${timed['net_pnl'].mean():>12,.0f}")
rpt()

# Of stopped trades: how many would have been profitable at day 20?
stopped_with_d20 = stopped[stopped['day20_close'].notna()]
if len(stopped_with_d20) > 0:
    hypothetical_pnl = stopped_with_d20.apply(
        lambda r: r['shares'] * (r['day20_close'] - r['entry_price']), axis=1
    )
    would_profit = (hypothetical_pnl > 0).sum()
    rpt(f"  Of {len(stopped_with_d20)} stopped trades with day-20 data:")
    rpt(f"    Would have been profitable at day 20: {would_profit} ({would_profit/len(stopped_with_d20)*100:.1f}%)")
    rpt(f"    Would have been unprofitable at day 20: {len(stopped_with_d20) - would_profit} ({(len(stopped_with_d20)-would_profit)/len(stopped_with_d20)*100:.1f}%)")
rpt()

# ── G) DRAWDOWN ANALYSIS ──
rpt("G) DRAWDOWN ANALYSIS — Top 3 Drawdown Periods")
rpt("-" * 60)

# We need to find distinct drawdown periods
equity = res_df['net_pnl'].cumsum().values
dates = pd.to_datetime(res_df['entry_date']).values

# Find all drawdown periods
running_max_arr = np.maximum.accumulate(equity)
dd_arr = equity - running_max_arr

# Find drawdown periods
drawdown_periods = []
in_dd = False
dd_start = 0
dd_depth = 0
dd_trough = 0

for i in range(len(dd_arr)):
    if dd_arr[i] < 0:
        if not in_dd:
            in_dd = True
            dd_start = i
            dd_depth = dd_arr[i]
            dd_trough = i
        else:
            if dd_arr[i] < dd_depth:
                dd_depth = dd_arr[i]
                dd_trough = i
    else:
        if in_dd:
            drawdown_periods.append({
                'start_idx': dd_start,
                'trough_idx': dd_trough,
                'end_idx': i,
                'depth': dd_depth,
                'start_date': str(dates[dd_start])[:10],
                'trough_date': str(dates[dd_trough])[:10],
                'end_date': str(dates[i])[:10],
                'duration_trades': i - dd_start,
            })
            in_dd = False

# If still in drawdown at end
if in_dd:
    drawdown_periods.append({
        'start_idx': dd_start,
        'trough_idx': dd_trough,
        'end_idx': len(dd_arr) - 1,
        'depth': dd_depth,
        'start_date': str(dates[dd_start])[:10],
        'trough_date': str(dates[dd_trough])[:10],
        'end_date': str(dates[-1])[:10],
        'duration_trades': len(dd_arr) - dd_start,
    })

drawdown_periods.sort(key=lambda x: x['depth'])
for i, dp in enumerate(drawdown_periods[:3]):
    rpt(f"  #{i+1}: Depth ${dp['depth']:,.0f}")
    rpt(f"      Start: {dp['start_date']}  Trough: {dp['trough_date']}  End: {dp['end_date']}")
    rpt(f"      Duration: {dp['duration_trades']} trades")
    rpt()

# ── H) COMPARISON TABLE ──
rpt("H) COMPARISON TABLE")
rpt("-" * 80)
rpt(f"  {'Metric':<28} {'Baseline':>16} {'Scheme 2':>16} {'Option D':>16}")
rpt(f"  {'(flat $1M)':28} {'':>16} {'(skip Q1 $1M)':>16} {'(tier-size)':>16}")
rpt(f"  {'-'*28} {'-'*16} {'-'*16} {'-'*16}")
rpt(f"  {'Total Net P&L':<28} {'$37.4M':>16} {'$54.0M':>16} {'${:.1f}M'.format(total_net_pnl/1e6):>16}")
rpt(f"  {'Win Rate':<28} {'51.2%':>16} {'54.8%':>16} {'{:.1f}%'.format(win_rate):>16}")
rpt(f"  {'Profit Factor':<28} {'1.31':>16} {'1.64':>16} {'{:.2f}'.format(profit_factor):>16}")
rpt(f"  {'Max Drawdown':<28} {'-$13.3M':>16} {'-$6.7M':>16} {'${:.1f}M'.format(max_dd/1e6):>16}")
rpt(f"  {'Total Trades':<28} {'2,875':>16} {'2,295':>16} {'{:,}'.format(total_trades):>16}")
rpt(f"  {'Avg P&L/Trade':<28} {'$13,009':>16} {'$23,529':>16} {'${:,.0f}'.format(avg_pnl):>16}")
rpt(f"  {'Sharpe':<28} {'':>16} {'':>16} {'{:.2f}'.format(sharpe):>16}")
rpt(f"  {'Calmar':<28} {'':>16} {'':>16} {'{:.2f}'.format(calmar):>16}")
rpt()

# Save report
with open('/home/ubuntu/daily_data/analysis_results/backtest_optionD_report.txt', 'w') as f:
    f.write('\n'.join(report_lines))

print("\nReport saved.")
print(f"Total Net P&L: ${total_net_pnl:,.0f}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max DD: ${max_dd:,.0f}")
