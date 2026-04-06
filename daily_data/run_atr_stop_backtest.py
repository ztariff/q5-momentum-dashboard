#!/usr/bin/env python3
"""
Trade 1 ATR-Based Stop Loss Backtest
Stop = signal_day_low - 3.0 * ATR(14)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/ubuntu/daily_data/data")
RESULTS_DIR = Path("/home/ubuntu/daily_data/analysis_results")

# ── Load trade log ──
trades_df = pd.read_csv(RESULTS_DIR / "trade1_20d_trades.csv")
trades_df['signal_date'] = pd.to_datetime(trades_df['signal_date'])
trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

print(f"Loaded {len(trades_df)} trades")

# ── Cache enriched data per symbol ──
symbol_cache = {}

def get_symbol_data(symbol):
    if symbol not in symbol_cache:
        fp = DATA_DIR / f"{symbol}_enriched.csv"
        if not fp.exists():
            return None
        df = pd.read_csv(fp, usecols=['date', 'open', 'high', 'low', 'close', 'atr_14'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date')
        symbol_cache[symbol] = df
    return symbol_cache[symbol]

# ── Simulate each trade ──
results = []
skipped = 0

for idx, row in trades_df.iterrows():
    sym = row['symbol']
    sdf = get_symbol_data(sym)
    if sdf is None:
        skipped += 1
        continue

    signal_date = row['signal_date']
    entry_date = row['entry_date']
    original_exit_date = row['exit_date']

    # Step 1: Get signal day low and ATR
    if signal_date not in sdf.index:
        # Find nearest date
        mask = sdf.index <= signal_date
        if mask.sum() == 0:
            skipped += 1
            continue
        signal_date_actual = sdf.index[mask][-1]
    else:
        signal_date_actual = signal_date

    signal_row = sdf.loc[signal_date_actual]
    if isinstance(signal_row, pd.DataFrame):
        signal_row = signal_row.iloc[-1]

    signal_day_low = signal_row['low']
    atr_14 = signal_row['atr_14']

    if pd.isna(atr_14) or pd.isna(signal_day_low):
        skipped += 1
        continue

    # Step 2: stop price
    stop_price = signal_day_low - 3.0 * atr_14

    # Step 3: Entry
    if entry_date not in sdf.index:
        mask = sdf.index >= entry_date
        if mask.sum() == 0:
            skipped += 1
            continue
        entry_date_actual = sdf.index[mask][0]
    else:
        entry_date_actual = entry_date

    entry_row = sdf.loc[entry_date_actual]
    if isinstance(entry_row, pd.DataFrame):
        entry_row = entry_row.iloc[0]

    entry_price = entry_row['open']
    if pd.isna(entry_price) or entry_price <= 0:
        skipped += 1
        continue

    shares = int(np.floor(1_000_000 / entry_price))
    if shares <= 0:
        skipped += 1
        continue

    # Step 4: Walk through trading days
    # Get all trading days from entry to original exit
    trade_days = sdf.loc[entry_date_actual:original_exit_date]
    if len(trade_days) == 0:
        skipped += 1
        continue

    exit_type = "TIME"
    exit_price = None
    actual_exit_date = None
    days_held = 0

    for i, (day_date, day_row) in enumerate(trade_days.iterrows()):
        day_low = day_row['low']
        if pd.isna(day_low):
            continue
        if day_low <= stop_price:
            exit_type = "STOP"
            exit_price = stop_price
            actual_exit_date = day_date
            days_held = i + 1  # 1-based count of trading days held
            break

    if exit_type == "TIME":
        # Exit at close of original exit date
        if original_exit_date in sdf.index:
            exit_row_data = sdf.loc[original_exit_date]
            if isinstance(exit_row_data, pd.DataFrame):
                exit_row_data = exit_row_data.iloc[-1]
            exit_price = exit_row_data['close']
        else:
            # Find nearest date <= original_exit_date
            mask = sdf.index <= original_exit_date
            if mask.sum() > 0:
                nearest = sdf.index[mask][-1]
                exit_row_data = sdf.loc[nearest]
                if isinstance(exit_row_data, pd.DataFrame):
                    exit_row_data = exit_row_data.iloc[-1]
                exit_price = exit_row_data['close']
                actual_exit_date = nearest
            else:
                skipped += 1
                continue
        if actual_exit_date is None:
            actual_exit_date = original_exit_date
        days_held = len(trade_days)

    if exit_price is None or pd.isna(exit_price):
        skipped += 1
        continue

    # Step 6: PnL
    gross_pnl = shares * (exit_price - entry_price)
    cost = shares * (entry_price + exit_price) * 5 / 10000
    net_pnl = gross_pnl - cost
    return_pct = ((exit_price - entry_price) / entry_price) * 100

    # Also store original exit price for counterfactual analysis
    if original_exit_date in sdf.index:
        orig_exit_row = sdf.loc[original_exit_date]
        if isinstance(orig_exit_row, pd.DataFrame):
            orig_exit_row = orig_exit_row.iloc[-1]
        original_exit_price = orig_exit_row['close']
    else:
        original_exit_price = row['exit_price']

    results.append({
        'symbol': sym,
        'signal_date': row['signal_date'],
        'entry_date': entry_date_actual,
        'actual_exit_date': actual_exit_date,
        'entry_price': round(entry_price, 4),
        'exit_price': round(exit_price, 4),
        'stop_price': round(stop_price, 4),
        'signal_day_low': round(signal_day_low, 4),
        'atr_14': round(atr_14, 4),
        'shares': shares,
        'gross_pnl': round(gross_pnl, 2),
        'net_pnl': round(net_pnl, 2),
        'return_pct': round(return_pct, 4),
        'exit_type': exit_type,
        'days_held': days_held,
        'z_score': row['z_score'],
        'original_exit_price': round(original_exit_price, 4),
    })

print(f"Simulated {len(results)} trades, skipped {skipped}")

# ── Save File 1 ──
res_df = pd.DataFrame(results)
# Save without original_exit_price in the main output (keep for analysis)
out_cols = ['symbol','signal_date','entry_date','actual_exit_date','entry_price',
            'exit_price','stop_price','signal_day_low','atr_14','shares',
            'gross_pnl','net_pnl','return_pct','exit_type','days_held','z_score']
res_df[out_cols].to_csv(RESULTS_DIR / "trade1_atr_stop_trades.csv", index=False)
print("Saved trade1_atr_stop_trades.csv")

# ── Build Daily P&L (File 2) ──
# For each trade, distribute realized PnL on exit date
# Track open positions per day

all_dates = set()
for r in results:
    sdf = get_symbol_data(r['symbol'])
    entry = r['entry_date']
    exit_d = r['actual_exit_date']
    days_in_range = sdf.loc[entry:exit_d].index.tolist()
    all_dates.update(days_in_range)

all_dates = sorted(all_dates)

# Build daily arrays
daily_data = []
cum_pnl = 0.0
peak_pnl = 0.0
max_dd = 0.0

for d in all_dates:
    # Count open positions on this date
    n_open = 0
    realized_today = 0.0
    for r in results:
        if r['entry_date'] <= d <= r['actual_exit_date']:
            n_open += 1
        if r['actual_exit_date'] == d:
            realized_today += r['net_pnl']

    cum_pnl += realized_today
    if cum_pnl > peak_pnl:
        peak_pnl = cum_pnl
    current_dd = peak_pnl - cum_pnl

    if current_dd > max_dd:
        max_dd = current_dd

    daily_data.append({
        'date': d,
        'n_positions_open': n_open,
        'realized_pnl_today': round(realized_today, 2),
        'cumulative_pnl': round(cum_pnl, 2),
        'current_drawdown': round(current_dd, 2),
        'max_drawdown_to_date': round(max_dd, 2),
    })

daily_df = pd.DataFrame(daily_data)
daily_df.to_csv(RESULTS_DIR / "trade1_atr_stop_daily.csv", index=False)
print("Saved trade1_atr_stop_daily.csv")

# ══════════════════════════════════════════════════════════════
# ── REPORT (File 3) ──
# ══════════════════════════════════════════════════════════════

lines = []
def L(s=""): lines.append(s)

L("=" * 80)
L("  TRADE 1 — ATR STOP LOSS BACKTEST REPORT")
L("  Stop = Signal Day Low - 3.0 × ATR(14)")
L("=" * 80)
L()

# ── A) OVERALL STATS ──
total_trades = len(res_df)
total_net_pnl = res_df['net_pnl'].sum()
winners = res_df[res_df['net_pnl'] > 0]
losers = res_df[res_df['net_pnl'] <= 0]
n_winners = len(winners)
n_losers = len(losers)
win_rate = n_winners / total_trades * 100

avg_winner = winners['net_pnl'].mean() if len(winners) > 0 else 0
avg_loser = losers['net_pnl'].mean() if len(losers) > 0 else 0

best_idx = res_df['net_pnl'].idxmax()
worst_idx = res_df['net_pnl'].idxmin()
best = res_df.loc[best_idx]
worst = res_df.loc[worst_idx]

sum_winners = winners['net_pnl'].sum()
sum_losers = abs(losers['net_pnl'].sum())
profit_factor = sum_winners / sum_losers if sum_losers > 0 else float('inf')

# Max drawdown from daily
max_dd_val = daily_df['max_drawdown_to_date'].max()

# Max drawdown duration
in_dd = False
dd_start = None
max_dd_duration = 0
current_dd_duration = 0
for _, drow in daily_df.iterrows():
    if drow['current_drawdown'] > 0:
        if not in_dd:
            in_dd = True
            dd_start = drow['date']
            current_dd_duration = 0
        current_dd_duration += 1
    else:
        if in_dd:
            max_dd_duration = max(max_dd_duration, current_dd_duration)
            in_dd = False
            current_dd_duration = 0
if in_dd:
    max_dd_duration = max(max_dd_duration, current_dd_duration)

# Calmar ratio
first_date = daily_df['date'].iloc[0]
last_date = daily_df['date'].iloc[-1]
n_calendar_days = (last_date - first_date).days
n_years = n_calendar_days / 365.25
annual_return = total_net_pnl / n_years if n_years > 0 else 0
calmar = annual_return / max_dd_val if max_dd_val > 0 else float('inf')

# Concurrent positions
avg_concurrent = daily_df['n_positions_open'].mean()
max_concurrent = daily_df['n_positions_open'].max()

# Sharpe (using daily realized PnL)
daily_returns = daily_df['realized_pnl_today']
sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

L("A) OVERALL STATISTICS")
L("-" * 50)
L(f"  Total trades:              {total_trades}")
L(f"  Total net P&L:             ${total_net_pnl:>14,.2f}")
L(f"  Winners:                   {n_winners} ({win_rate:.1f}%)")
L(f"  Losers:                    {n_losers} ({100 - win_rate:.1f}%)")
L(f"  Avg winner $:              ${avg_winner:>14,.2f}")
L(f"  Avg loser $:               ${avg_loser:>14,.2f}")
L(f"  Biggest winner:            ${best['net_pnl']:>14,.2f}  ({best['symbol']} on {best['entry_date'].strftime('%Y-%m-%d') if hasattr(best['entry_date'], 'strftime') else best['entry_date']})")
L(f"  Biggest loser:             ${worst['net_pnl']:>14,.2f}  ({worst['symbol']} on {worst['entry_date'].strftime('%Y-%m-%d') if hasattr(worst['entry_date'], 'strftime') else worst['entry_date']})")
L(f"  Profit factor:             {profit_factor:>14.3f}")
L(f"  Win rate:                  {win_rate:>13.1f}%")
L(f"  Max drawdown:              ${max_dd_val:>14,.2f}")
L(f"  Max drawdown duration:     {max_dd_duration:>10} trading days")
L(f"  Annual return:             ${annual_return:>14,.2f}")
L(f"  Calmar ratio:              {calmar:>14.3f}")
L(f"  Sharpe ratio:              {sharpe:>14.3f}")
L(f"  Avg concurrent positions:  {avg_concurrent:>14.1f}")
L(f"  Max concurrent positions:  {max_concurrent:>10}")
L(f"  Avg days held:             {res_df['days_held'].mean():>14.1f}")
L()

# ── B) EXIT TYPE BREAKDOWN ──
L("B) EXIT TYPE BREAKDOWN")
L("-" * 50)
stop_trades = res_df[res_df['exit_type'] == 'STOP']
time_trades = res_df[res_df['exit_type'] == 'TIME']

L(f"  Stopped out:     {len(stop_trades):>6}  ({len(stop_trades)/total_trades*100:.1f}%)")
L(f"  Time exit:       {len(time_trades):>6}  ({len(time_trades)/total_trades*100:.1f}%)")
L()
L(f"  Avg P&L (STOP):  ${stop_trades['net_pnl'].mean():>14,.2f}")
L(f"  Avg P&L (TIME):  ${time_trades['net_pnl'].mean():>14,.2f}")
L()

# Counterfactual: of stopped trades, how many would have been winners at day 20?
if len(stop_trades) > 0:
    stopped_full = res_df[res_df['exit_type'] == 'STOP'].copy()
    # Use original_exit_price from results list
    orig_prices = {i: results[i]['original_exit_price'] for i in range(len(results))}
    # Map back
    stopped_indices = stopped_full.index.tolist()
    counterfactual_winners = 0
    counterfactual_pnl_diff = 0
    for si in stopped_indices:
        orig_ep = results[si]['original_exit_price']
        entry_p = results[si]['entry_price']
        if orig_ep > entry_p:
            counterfactual_winners += 1

    L(f"  Stopped trades that WOULD have been winners at day 20:")
    L(f"    {counterfactual_winners} of {len(stop_trades)} ({counterfactual_winners/len(stop_trades)*100:.1f}%)")
    L(f"    (i.e., stop removed a profitable trade)")
L()

# ── C) YEAR BY YEAR ──
L("C) YEAR-BY-YEAR BREAKDOWN")
L("-" * 80)
res_df['year'] = pd.to_datetime(res_df['actual_exit_date']).dt.year
L(f"  {'Year':>6}  {'Trades':>7}  {'Net P&L':>14}  {'WR':>7}  {'PF':>8}  {'Max DD':>14}")
L(f"  {'----':>6}  {'------':>7}  {'---------':>14}  {'--':>7}  {'--':>8}  {'------':>14}")

for year in sorted(res_df['year'].unique()):
    ydf = res_df[res_df['year'] == year]
    yw = ydf[ydf['net_pnl'] > 0]
    yl = ydf[ydf['net_pnl'] <= 0]
    ywr = len(yw) / len(ydf) * 100 if len(ydf) > 0 else 0
    ypf = yw['net_pnl'].sum() / abs(yl['net_pnl'].sum()) if len(yl) > 0 and yl['net_pnl'].sum() != 0 else float('inf')

    # Year max DD from daily data
    year_daily = daily_df[pd.to_datetime(daily_df['date']).dt.year == year].copy()
    if len(year_daily) > 0:
        # Recompute DD for just this year's segment
        year_cum = year_daily['cumulative_pnl'].values
        year_peak = np.maximum.accumulate(year_cum)
        year_dd = year_peak - year_cum
        year_max_dd = year_dd.max()
    else:
        year_max_dd = 0

    L(f"  {year:>6}  {len(ydf):>7}  ${ydf['net_pnl'].sum():>13,.2f}  {ywr:>6.1f}%  {ypf:>8.3f}  ${year_max_dd:>13,.2f}")
L()

# ── D) MONTHLY P&L TABLE ──
L("D) MONTHLY P&L TABLE")
L("-" * 80)
res_df['month'] = pd.to_datetime(res_df['actual_exit_date']).dt.to_period('M')
monthly = res_df.groupby('month').agg(
    net_pnl=('net_pnl', 'sum'),
    trades=('net_pnl', 'count')
).reset_index()

L(f"  {'Month':>10}  {'Net P&L':>14}  {'Trades':>7}")
L(f"  {'-----':>10}  {'---------':>14}  {'------':>7}")
for _, mrow in monthly.iterrows():
    L(f"  {str(mrow['month']):>10}  ${mrow['net_pnl']:>13,.2f}  {mrow['trades']:>7}")
L()

# ── E) TOP 10 WINNERS & LOSERS ──
L("E) TOP 10 WINNERS")
L("-" * 110)
L(f"  {'Symbol':>8}  {'Entry Date':>12}  {'Exit Date':>12}  {'Entry $':>10}  {'Exit $':>10}  {'Stop $':>10}  {'Net P&L':>14}  {'Ret%':>8}  {'Exit':>5}  {'Days':>5}")
top10w = res_df.nlargest(10, 'net_pnl')
for _, r in top10w.iterrows():
    ed = r['entry_date'].strftime('%Y-%m-%d') if hasattr(r['entry_date'], 'strftime') else str(r['entry_date'])[:10]
    xd = r['actual_exit_date'].strftime('%Y-%m-%d') if hasattr(r['actual_exit_date'], 'strftime') else str(r['actual_exit_date'])[:10]
    L(f"  {r['symbol']:>8}  {ed:>12}  {xd:>12}  {r['entry_price']:>10.2f}  {r['exit_price']:>10.2f}  {r['stop_price']:>10.2f}  ${r['net_pnl']:>13,.2f}  {r['return_pct']:>7.2f}%  {r['exit_type']:>5}  {r['days_held']:>5}")
L()

L("   TOP 10 LOSERS")
L("-" * 110)
L(f"  {'Symbol':>8}  {'Entry Date':>12}  {'Exit Date':>12}  {'Entry $':>10}  {'Exit $':>10}  {'Stop $':>10}  {'Net P&L':>14}  {'Ret%':>8}  {'Exit':>5}  {'Days':>5}")
top10l = res_df.nsmallest(10, 'net_pnl')
for _, r in top10l.iterrows():
    ed = r['entry_date'].strftime('%Y-%m-%d') if hasattr(r['entry_date'], 'strftime') else str(r['entry_date'])[:10]
    xd = r['actual_exit_date'].strftime('%Y-%m-%d') if hasattr(r['actual_exit_date'], 'strftime') else str(r['actual_exit_date'])[:10]
    L(f"  {r['symbol']:>8}  {ed:>12}  {xd:>12}  {r['entry_price']:>10.2f}  {r['exit_price']:>10.2f}  {r['stop_price']:>10.2f}  ${r['net_pnl']:>13,.2f}  {r['return_pct']:>7.2f}%  {r['exit_type']:>5}  {r['days_held']:>5}")
L()

# ── F) COMPARISON TO BASELINE ──
L("F) COMPARISON TO BASELINE (No Stop)")
L("-" * 70)
baseline_df = pd.read_csv(RESULTS_DIR / "trade1_20d_trades.csv")

b_total_pnl = baseline_df['net_pnl'].sum()
b_winners = baseline_df[baseline_df['net_pnl'] > 0]
b_losers = baseline_df[baseline_df['net_pnl'] <= 0]
b_wr = len(b_winners) / len(baseline_df) * 100
b_pf = b_winners['net_pnl'].sum() / abs(b_losers['net_pnl'].sum()) if b_losers['net_pnl'].sum() != 0 else float('inf')
b_worst = baseline_df['net_pnl'].min()
b_worst_sym = baseline_df.loc[baseline_df['net_pnl'].idxmin(), 'symbol']

# Baseline daily PnL for Sharpe & DD
baseline_df['exit_date_dt'] = pd.to_datetime(baseline_df['exit_date'])
b_daily = baseline_df.groupby('exit_date_dt')['net_pnl'].sum().sort_index()
b_cum = b_daily.cumsum()
b_peak = b_cum.cummax()
b_dd = b_peak - b_cum
b_max_dd = b_dd.max()
b_sharpe = (b_daily.mean() / b_daily.std()) * np.sqrt(252) if b_daily.std() > 0 else 0

# ATR stop metrics
s_worst = res_df['net_pnl'].min()
s_worst_sym = res_df.loc[res_df['net_pnl'].idxmin(), 'symbol']

L(f"  {'Metric':<30}  {'No Stop':>16}  {'ATR Stop':>16}  {'Delta':>16}")
L(f"  {'-'*30}  {'-'*16}  {'-'*16}  {'-'*16}")
L(f"  {'Total Net P&L':<30}  ${b_total_pnl:>15,.2f}  ${total_net_pnl:>15,.2f}  ${total_net_pnl - b_total_pnl:>15,.2f}")
L(f"  {'Win Rate':<30}  {b_wr:>15.1f}%  {win_rate:>15.1f}%  {win_rate - b_wr:>15.1f}%")
L(f"  {'Profit Factor':<30}  {b_pf:>16.3f}  {profit_factor:>16.3f}  {profit_factor - b_pf:>16.3f}")
L(f"  {'Max Drawdown':<30}  ${b_max_dd:>15,.2f}  ${max_dd_val:>15,.2f}  ${max_dd_val - b_max_dd:>15,.2f}")
L(f"  {'Worst Trade':<30}  ${b_worst:>15,.2f}  ${s_worst:>15,.2f}  ${s_worst - b_worst:>15,.2f}")
L(f"  {'Worst Trade Symbol':<30}  {b_worst_sym:>16}  {s_worst_sym:>16}")
L(f"  {'Sharpe Ratio':<30}  {b_sharpe:>16.3f}  {sharpe:>16.3f}  {sharpe - b_sharpe:>16.3f}")
L()

# ── G) DRAWDOWN ANALYSIS ──
L("G) DRAWDOWN ANALYSIS")
L("-" * 80)

# Identify drawdown periods
cum_pnl_arr = daily_df['cumulative_pnl'].values
dates_arr = daily_df['date'].values
peak_arr = np.maximum.accumulate(cum_pnl_arr)
dd_arr = peak_arr - cum_pnl_arr

# Find drawdown periods (contiguous blocks where dd > 0)
dd_periods = []
in_dd = False
dd_start_idx = 0
dd_depth = 0

for i in range(len(dd_arr)):
    if dd_arr[i] > 0:
        if not in_dd:
            in_dd = True
            dd_start_idx = i
            dd_depth = dd_arr[i]
        else:
            dd_depth = max(dd_depth, dd_arr[i])
    else:
        if in_dd:
            dd_periods.append({
                'start_date': dates_arr[dd_start_idx],
                'end_date': dates_arr[i],  # recovery date
                'depth': dd_depth,
                'duration': i - dd_start_idx,
                'recovered': True
            })
            in_dd = False

# If still in drawdown at end
if in_dd:
    dd_periods.append({
        'start_date': dates_arr[dd_start_idx],
        'end_date': dates_arr[-1],
        'depth': dd_depth,
        'duration': len(dd_arr) - dd_start_idx,
        'recovered': False
    })

# Sort by depth
dd_periods.sort(key=lambda x: x['depth'], reverse=True)

L("  Top 5 Drawdown Periods:")
L(f"  {'#':>3}  {'Start':>12}  {'End':>12}  {'Depth':>14}  {'Duration':>10}  {'Recovered':>10}")
for i, ddp in enumerate(dd_periods[:5]):
    sd = pd.Timestamp(ddp['start_date']).strftime('%Y-%m-%d')
    ed = pd.Timestamp(ddp['end_date']).strftime('%Y-%m-%d')
    rec = "Yes" if ddp['recovered'] else "No"
    L(f"  {i+1:>3}  {sd:>12}  {ed:>12}  ${ddp['depth']:>13,.2f}  {ddp['duration']:>7} days  {rec:>10}")
L()

# Longest time between new equity highs
new_high_gaps = []
last_high_idx = 0
for i in range(1, len(cum_pnl_arr)):
    if cum_pnl_arr[i] >= peak_arr[i] and (i == 0 or cum_pnl_arr[i] > peak_arr[i-1]):
        gap = i - last_high_idx
        if gap > 1:
            new_high_gaps.append({
                'from': dates_arr[last_high_idx],
                'to': dates_arr[i],
                'days': gap
            })
        last_high_idx = i

if new_high_gaps:
    longest_gap = max(new_high_gaps, key=lambda x: x['days'])
    gf = pd.Timestamp(longest_gap['from']).strftime('%Y-%m-%d')
    gt = pd.Timestamp(longest_gap['to']).strftime('%Y-%m-%d')
    L(f"  Longest time between new equity highs: {longest_gap['days']} trading days")
    L(f"    From {gf} to {gt}")
else:
    L("  No new equity highs recorded after the first day.")
L()

L("=" * 80)
L("  END OF REPORT")
L("=" * 80)

report = "\n".join(lines)
with open(RESULTS_DIR / "trade1_atr_stop_report.txt", "w") as f:
    f.write(report)

print("Saved trade1_atr_stop_report.txt")
print()
print(report)
