#!/usr/bin/env python3
"""
Options Hold-Period Backtest
Uses actual daily option prices for the SAME contract through the entire hold period.
"""

import pandas as pd
import numpy as np
from math import floor
from datetime import datetime

# ── Load data ──────────────────────────────────────────────────────────────────
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/backtest_optionD_noReentry_trades.csv')
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])

# Filter to 2022+
trades = trades[trades['entry_date'] >= '2022-01-01'].copy()
print(f"Trades after 2022 filter: {len(trades)}")

opts = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options.csv')
opts['trade_entry_date'] = pd.to_datetime(opts['trade_entry_date'])
opts['trade_exit_date'] = pd.to_datetime(opts['trade_exit_date'])
opts['bar_date'] = pd.to_datetime(opts['bar_date'])

# Convert price columns to numeric
for col in ['day_open', 'day_high', 'day_low', 'day_close', 'morning_price']:
    opts[col] = pd.to_numeric(opts[col], errors='coerce')

print(f"Options rows: {len(opts)}")
print(f"Unique option types: {opts['option_type'].unique()}")

# ── Option D sizing by quintile ────────────────────────────────────────────────
quintile_size = {
    'Q1': 0,       # skip
    'Q2': 50000,
    'Q3': 100000,
    'Q4': 150000,
    'Q5': 200000,
}

# ── Build index for fast lookup ────────────────────────────────────────────────
# Key: (trade_symbol, trade_entry_date, option_type, bar_date) -> row
opts_idx = {}
for _, row in opts.iterrows():
    key = (row['trade_symbol'], row['trade_entry_date'], row['option_type'], row['bar_date'])
    opts_idx[key] = row

print(f"Options index entries: {len(opts_idx)}")

# ── Process each trade × option type ──────────────────────────────────────────
option_types = ['30d_1m', '50d_1m', '30d_3m', '50d_3m']
results = []

for _, trade in trades.iterrows():
    sym = trade['symbol']
    entry_dt = trade['entry_date']
    exit_dt = trade['exit_date']
    exit_type = trade['exit_type']
    quintile = trade['body_quintile']
    stock_net_pnl = trade['net_pnl']
    stock_return = trade['return_pct']

    pos_size = quintile_size.get(quintile, 0)
    if pos_size == 0:
        continue  # Q1 skipped

    for otype in option_types:
        # Look up entry day
        entry_key = (sym, entry_dt, otype, entry_dt)
        entry_row = opts_idx.get(entry_key)

        if entry_row is None:
            continue  # no entry day data

        # Determine entry price: morning_price > day_open > day_close
        option_entry_price = None
        if pd.notna(entry_row['morning_price']) and entry_row['morning_price'] > 0:
            option_entry_price = entry_row['morning_price']
        elif pd.notna(entry_row['day_open']) and entry_row['day_open'] > 0:
            option_entry_price = entry_row['day_open']
        elif pd.notna(entry_row['day_close']) and entry_row['day_close'] > 0:
            option_entry_price = entry_row['day_close']

        if option_entry_price is None or option_entry_price <= 0:
            continue

        # Look up exit day
        exit_key = (sym, entry_dt, otype, exit_dt)
        exit_row = opts_idx.get(exit_key)

        if exit_row is None:
            continue  # no exit day data

        # Exit price: always day_close (sell at end of day)
        option_exit_price = exit_row['day_close']
        if pd.isna(option_exit_price) or option_exit_price < 0:
            continue

        # Calculate P&L
        contracts = floor(pos_size / (option_entry_price * 100))
        if contracts < 1:
            continue  # too expensive

        entry_cost = contracts * option_entry_price * 100
        exit_value = contracts * option_exit_price * 100
        transaction_costs = (entry_cost + exit_value) * 5 / 10000
        net_pnl = exit_value - entry_cost - transaction_costs
        option_return_pct = (option_exit_price / option_entry_price - 1) * 100

        results.append({
            'symbol': sym,
            'entry_date': entry_dt,
            'exit_date': exit_dt,
            'exit_type': exit_type,
            'body_quintile': quintile,
            'option_type': otype,
            'option_ticker': entry_row['option_ticker'],
            'strike': entry_row['strike'],
            'expiration': entry_row['expiration'],
            'option_entry_price': option_entry_price,
            'option_exit_price': option_exit_price,
            'option_position_size': pos_size,
            'contracts': contracts,
            'entry_cost': entry_cost,
            'exit_value': exit_value,
            'transaction_costs': transaction_costs,
            'option_net_pnl': net_pnl,
            'option_return_pct': option_return_pct,
            'stock_net_pnl': stock_net_pnl,
            'stock_return_pct': stock_return,
            'days_held': trade['days_held'],
        })

df = pd.DataFrame(results)
print(f"\nTotal option trades executed: {len(df)}")
print(f"By option type:")
print(df.groupby('option_type').size())

# ── Save trade-level CSV ──────────────────────────────────────────────────────
df.to_csv('/home/ubuntu/daily_data/analysis_results/options_holdperiod_trades.csv', index=False)
print("Trade-level CSV saved.")

# ── Helper functions ──────────────────────────────────────────────────────────
def calc_max_drawdown(pnl_series):
    """Calculate max drawdown from a series of trade P&Ls."""
    cumulative = pnl_series.cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    return drawdown.min()

def calc_profit_factor(pnl_series):
    """Profit factor = gross profits / abs(gross losses)."""
    wins = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    if losses == 0:
        return float('inf') if wins > 0 else 0
    return wins / losses

def fmt_money(x):
    if x >= 0:
        return f"${x:,.0f}"
    else:
        return f"-${abs(x):,.0f}"

def fmt_pct(x):
    return f"{x:.1f}%"

# ── Generate Report ───────────────────────────────────────────────────────────
lines = []
def w(s=''):
    lines.append(s)

w("=" * 100)
w("OPTIONS HOLD-PERIOD BACKTEST — ACTUAL CONTRACT TRACKING")
w("=" * 100)
w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
w(f"Strategy: Option D No-Re-Entry, entry_date >= 2022-01-01")
w(f"Total stock trades in filtered universe: {len(trades)}")
w(f"  (Q1 trades skipped per Option D sizing)")
w(f"Option position sizing: Q2=$50K, Q3=$100K, Q4=$150K, Q5=$200K")
w(f"Transaction costs: 5 bps each way")
w(f"Entry price: morning_price > day_open > day_close")
w(f"Exit price: day_close on exit date")
w()
w("KEY: This backtest tracks the SAME option contract from entry to exit.")
w("No comparison of different contracts. Actual hold-period performance.")

# ── A) SUMMARY BY OPTION TYPE ────────────────────────────────────────────────
w()
w("=" * 100)
w("A) SUMMARY BY OPTION TYPE")
w("=" * 100)

# Count trades with any data (entry row exists) vs executed (contracts >= 1)
# We need to re-count trades with data
trades_with_data = {}
for otype in option_types:
    count = 0
    for _, trade in trades.iterrows():
        if quintile_size.get(trade['body_quintile'], 0) == 0:
            continue
        key = (trade['symbol'], trade['entry_date'], otype, trade['entry_date'])
        if key in opts_idx:
            count += 1
    trades_with_data[otype] = count

for otype in option_types:
    sub = df[df['option_type'] == otype]
    if len(sub) == 0:
        w(f"\n--- {otype}: NO TRADES EXECUTED ---")
        w(f"    Trades with entry-day data: {trades_with_data.get(otype, 0)}")
        continue

    w(f"\n{'─' * 80}")
    w(f"  {otype.upper()}")
    w(f"{'─' * 80}")

    n_data = trades_with_data.get(otype, 0)
    n_exec = len(sub)
    total_pnl = sub['option_net_pnl'].sum()
    wins = (sub['option_net_pnl'] > 0).sum()
    wr = wins / n_exec * 100
    pf = calc_profit_factor(sub['option_net_pnl'])
    avg_pnl = sub['option_net_pnl'].mean()
    median_pnl = sub['option_net_pnl'].median()
    max_dd = calc_max_drawdown(sub.sort_values('entry_date')['option_net_pnl'])
    best = sub.loc[sub['option_net_pnl'].idxmax()]
    worst = sub.loc[sub['option_net_pnl'].idxmin()]
    avg_ret = sub['option_return_pct'].mean()
    median_ret = sub['option_return_pct'].median()

    # Corresponding stock P&L for same trades
    stock_pnl = sub['stock_net_pnl'].sum()
    leverage = total_pnl / stock_pnl if stock_pnl != 0 else float('inf')

    avg_entry_cost = sub['entry_cost'].mean()
    avg_contracts = sub['contracts'].mean()

    w(f"  Trades with option data:  {n_data}")
    w(f"  Trades executed:          {n_exec}")
    w(f"  Avg contracts per trade:  {avg_contracts:.1f}")
    w(f"  Avg capital deployed:     {fmt_money(avg_entry_cost)}")
    w()
    w(f"  Total P&L:                {fmt_money(total_pnl)}")
    w(f"  Win Rate:                 {fmt_pct(wr)} ({wins}/{n_exec})")
    w(f"  Profit Factor:            {pf:.2f}")
    w(f"  Avg P&L per trade:        {fmt_money(avg_pnl)}")
    w(f"  Median P&L per trade:     {fmt_money(median_pnl)}")
    w(f"  Avg Option Return:        {fmt_pct(avg_ret)}")
    w(f"  Median Option Return:     {fmt_pct(median_ret)}")
    w()
    w(f"  Max Drawdown:             {fmt_money(max_dd)}")
    w(f"  Best Trade:               {fmt_money(best['option_net_pnl'])} ({best['symbol']} {best['entry_date'].strftime('%Y-%m-%d')}, {fmt_pct(best['option_return_pct'])})")
    w(f"  Worst Trade:              {fmt_money(worst['option_net_pnl'])} ({worst['symbol']} {worst['entry_date'].strftime('%Y-%m-%d')}, {fmt_pct(worst['option_return_pct'])})")
    w()
    w(f"  Corresponding Stock P&L:  {fmt_money(stock_pnl)}")
    w(f"  Leverage Ratio:           {leverage:.2f}x (option P&L / stock P&L)")

# ── B) YEAR-BY-YEAR ──────────────────────────────────────────────────────────
w()
w("=" * 100)
w("B) YEAR-BY-YEAR PERFORMANCE")
w("=" * 100)

df['year'] = df['entry_date'].dt.year

for otype in option_types:
    sub = df[df['option_type'] == otype]
    if len(sub) == 0:
        w(f"\n--- {otype}: NO DATA ---")
        continue

    w(f"\n{'─' * 80}")
    w(f"  {otype.upper()}")
    w(f"{'─' * 80}")
    w(f"  {'Year':<8} {'Trades':>8} {'P&L':>14} {'Win Rate':>10} {'Avg P&L':>12} {'Avg Ret%':>10} {'PF':>8}")
    w(f"  {'─'*8} {'─'*8} {'─'*14} {'─'*10} {'─'*12} {'─'*10} {'─'*8}")

    for year in sorted(sub['year'].unique()):
        yr = sub[sub['year'] == year]
        n = len(yr)
        pnl = yr['option_net_pnl'].sum()
        wr_y = (yr['option_net_pnl'] > 0).sum() / n * 100
        avg = yr['option_net_pnl'].mean()
        avg_r = yr['option_return_pct'].mean()
        pf_y = calc_profit_factor(yr['option_net_pnl'])
        pf_str = f"{pf_y:.2f}" if pf_y != float('inf') else "inf"
        w(f"  {year:<8} {n:>8} {fmt_money(pnl):>14} {fmt_pct(wr_y):>10} {fmt_money(avg):>12} {fmt_pct(avg_r):>10} {pf_str:>8}")

    # Total row
    n = len(sub)
    pnl = sub['option_net_pnl'].sum()
    wr_t = (sub['option_net_pnl'] > 0).sum() / n * 100
    avg = sub['option_net_pnl'].mean()
    avg_r = sub['option_return_pct'].mean()
    pf_t = calc_profit_factor(sub['option_net_pnl'])
    pf_str = f"{pf_t:.2f}" if pf_t != float('inf') else "inf"
    w(f"  {'─'*8} {'─'*8} {'─'*14} {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
    w(f"  {'TOTAL':<8} {n:>8} {fmt_money(pnl):>14} {fmt_pct(wr_t):>10} {fmt_money(avg):>12} {fmt_pct(avg_r):>10} {pf_str:>8}")

# ── C) COMPARISON TABLE ──────────────────────────────────────────────────────
w()
w("=" * 100)
w("C) COMPARISON TABLE — ALL OPTION TYPES + STOCK BASELINE")
w("=" * 100)

# Stock baseline: only trades that are in 2022+ and not Q1
stock_base = trades[trades['body_quintile'] != 'Q1'].copy()
stock_pnl_total = stock_base['net_pnl'].sum()
stock_wr = (stock_base['net_pnl'] > 0).sum() / len(stock_base) * 100
stock_pf = calc_profit_factor(stock_base['net_pnl'])
stock_avg = stock_base['net_pnl'].mean()
stock_avg_ret = stock_base['return_pct'].mean()
stock_dd = calc_max_drawdown(stock_base.sort_values('entry_date')['net_pnl'])

header = f"  {'Metric':<28}"
for otype in option_types:
    header += f" {otype:>14}"
header += f" {'STOCK':>14}"
w(header)
w(f"  {'─'*28}" + f" {'─'*14}" * 5)

# Trades
row = f"  {'Trades Executed':<28}"
for otype in option_types:
    sub = df[df['option_type'] == otype]
    row += f" {len(sub):>14}"
row += f" {len(stock_base):>14}"
w(row)

# Total P&L
row = f"  {'Total P&L':<28}"
for otype in option_types:
    sub = df[df['option_type'] == otype]
    val = sub['option_net_pnl'].sum() if len(sub) > 0 else 0
    row += f" {fmt_money(val):>14}"
row += f" {fmt_money(stock_pnl_total):>14}"
w(row)

# Win Rate
row = f"  {'Win Rate':<28}"
for otype in option_types:
    sub = df[df['option_type'] == otype]
    val = (sub['option_net_pnl'] > 0).sum() / len(sub) * 100 if len(sub) > 0 else 0
    row += f" {fmt_pct(val):>14}"
row += f" {fmt_pct(stock_wr):>14}"
w(row)

# Profit Factor
row = f"  {'Profit Factor':<28}"
for otype in option_types:
    sub = df[df['option_type'] == otype]
    val = calc_profit_factor(sub['option_net_pnl']) if len(sub) > 0 else 0
    val_str = f"{val:.2f}" if val != float('inf') else "inf"
    row += f" {val_str:>14}"
stock_pf_str = f"{stock_pf:.2f}" if stock_pf != float('inf') else "inf"
row += f" {stock_pf_str:>14}"
w(row)

# Avg P&L
row = f"  {'Avg P&L per Trade':<28}"
for otype in option_types:
    sub = df[df['option_type'] == otype]
    val = sub['option_net_pnl'].mean() if len(sub) > 0 else 0
    row += f" {fmt_money(val):>14}"
row += f" {fmt_money(stock_avg):>14}"
w(row)

# Avg Return %
row = f"  {'Avg Return %':<28}"
for otype in option_types:
    sub = df[df['option_type'] == otype]
    val = sub['option_return_pct'].mean() if len(sub) > 0 else 0
    row += f" {fmt_pct(val):>14}"
row += f" {fmt_pct(stock_avg_ret):>14}"
w(row)

# Max Drawdown
row = f"  {'Max Drawdown':<28}"
for otype in option_types:
    sub = df[df['option_type'] == otype]
    val = calc_max_drawdown(sub.sort_values('entry_date')['option_net_pnl']) if len(sub) > 0 else 0
    row += f" {fmt_money(val):>14}"
row += f" {fmt_money(stock_dd):>14}"
w(row)

# Leverage ratio
row = f"  {'Leverage vs Stock':<28}"
for otype in option_types:
    sub = df[df['option_type'] == otype]
    opt_pnl = sub['option_net_pnl'].sum() if len(sub) > 0 else 0
    stk_pnl = sub['stock_net_pnl'].sum() if len(sub) > 0 else 0
    if stk_pnl != 0:
        lev = opt_pnl / stk_pnl
        row += f" {lev:.2f}x".rjust(14)
    else:
        row += f" {'N/A':>14}"
row += f" {'1.00x':>14}"
w(row)

# ── D) TOP 10 WINNERS & LOSERS for 50d_1m ────────────────────────────────────
w()
w("=" * 100)
w("D) TOP 10 WINNERS & TOP 10 LOSERS — 50d_1m (ATM 1-MONTH)")
w("=" * 100)

sub_50d1m = df[df['option_type'] == '50d_1m'].copy()

if len(sub_50d1m) > 0:
    w(f"\n  TOP 10 WINNERS")
    w(f"  {'─'*130}")
    w(f"  {'Symbol':<8} {'Entry':>12} {'Exit':>12} {'Option Ticker':<30} {'Opt Entry':>10} {'Opt Exit':>10} {'Opt Ret%':>10} {'Opt P&L':>12} {'Stk Ret%':>10} {'Exit Type':>10}")
    w(f"  {'─'*130}")

    top10 = sub_50d1m.nlargest(10, 'option_net_pnl')
    for _, r in top10.iterrows():
        w(f"  {r['symbol']:<8} {r['entry_date'].strftime('%Y-%m-%d'):>12} {r['exit_date'].strftime('%Y-%m-%d'):>12} {str(r['option_ticker']):<30} {r['option_entry_price']:>10.2f} {r['option_exit_price']:>10.2f} {fmt_pct(r['option_return_pct']):>10} {fmt_money(r['option_net_pnl']):>12} {fmt_pct(r['stock_return_pct']):>10} {r['exit_type']:>10}")

    w(f"\n  TOP 10 LOSERS")
    w(f"  {'─'*130}")
    w(f"  {'Symbol':<8} {'Entry':>12} {'Exit':>12} {'Option Ticker':<30} {'Opt Entry':>10} {'Opt Exit':>10} {'Opt Ret%':>10} {'Opt P&L':>12} {'Stk Ret%':>10} {'Exit Type':>10}")
    w(f"  {'─'*130}")

    bot10 = sub_50d1m.nsmallest(10, 'option_net_pnl')
    for _, r in bot10.iterrows():
        w(f"  {r['symbol']:<8} {r['entry_date'].strftime('%Y-%m-%d'):>12} {r['exit_date'].strftime('%Y-%m-%d'):>12} {str(r['option_ticker']):<30} {r['option_entry_price']:>10.2f} {r['option_exit_price']:>10.2f} {fmt_pct(r['option_return_pct']):>10} {fmt_money(r['option_net_pnl']):>12} {fmt_pct(r['stock_return_pct']):>10} {r['exit_type']:>10}")
else:
    w("  No 50d_1m trades executed.")

# ── E) COMBINED PORTFOLIO ─────────────────────────────────────────────────────
w()
w("=" * 100)
w("E) COMBINED PORTFOLIO — ALL 4 OPTION TYPES SIMULTANEOUSLY")
w("=" * 100)
w(f"  Running all 4 option types per trade = up to $400K option exposure per trade")
w(f"  (scaled by quintile: Q2=$200K, Q3=$400K, Q4=$600K, Q5=$800K total)")

# Group by trade (symbol + entry_date), sum P&L across option types
combo = df.groupby(['symbol', 'entry_date']).agg(
    combined_pnl=('option_net_pnl', 'sum'),
    n_types=('option_type', 'count'),
    stock_pnl=('stock_net_pnl', 'first'),
).reset_index()

combo_total = combo['combined_pnl'].sum()
combo_wins = (combo['combined_pnl'] > 0).sum()
combo_n = len(combo)
combo_wr = combo_wins / combo_n * 100
combo_pf = calc_profit_factor(combo['combined_pnl'])
combo_avg = combo['combined_pnl'].mean()
combo_dd = calc_max_drawdown(combo.sort_values('entry_date')['combined_pnl'])
combo_stock = combo['stock_pnl'].sum()
combo_lev = combo_total / combo_stock if combo_stock != 0 else float('inf')

w(f"\n  Unique trades with any option data:  {combo_n}")
w(f"  Avg option types per trade:          {combo['n_types'].mean():.1f}")
w()
w(f"  Combined Option P&L:                 {fmt_money(combo_total)}")
w(f"  Win Rate:                            {fmt_pct(combo_wr)} ({combo_wins}/{combo_n})")
w(f"  Profit Factor:                       {combo_pf:.2f}")
w(f"  Avg P&L per trade:                   {fmt_money(combo_avg)}")
w(f"  Max Drawdown:                        {fmt_money(combo_dd)}")
w()
w(f"  Corresponding Stock P&L:             {fmt_money(combo_stock)}")
w(f"  Leverage vs Stock:                   {combo_lev:.2f}x")

# By year
w(f"\n  Year-by-Year Combined:")
combo['year'] = combo['entry_date'].dt.year
w(f"  {'Year':<8} {'Trades':>8} {'P&L':>14} {'Win Rate':>10} {'Avg P&L':>12}")
w(f"  {'─'*8} {'─'*8} {'─'*14} {'─'*10} {'─'*12}")
for year in sorted(combo['year'].unique()):
    yr = combo[combo['year'] == year]
    n = len(yr)
    pnl = yr['combined_pnl'].sum()
    wr_y = (yr['combined_pnl'] > 0).sum() / n * 100
    avg = yr['combined_pnl'].mean()
    w(f"  {year:<8} {n:>8} {fmt_money(pnl):>14} {fmt_pct(wr_y):>10} {fmt_money(avg):>12}")

# ── Additional stats ──────────────────────────────────────────────────────────
w()
w("=" * 100)
w("F) DATA COVERAGE NOTES")
w("=" * 100)
w()

for otype in option_types:
    sub = df[df['option_type'] == otype]
    data_count = trades_with_data.get(otype, 0)
    exec_count = len(sub)
    w(f"  {otype}: {data_count} trades with entry-day data, {exec_count} executed (contracts >= 1)")
    if data_count > 0 and exec_count < data_count:
        skip_expensive = data_count - exec_count
        w(f"    -> {skip_expensive} skipped (option too expensive for position size, contracts < 1)")

# Non-Q1 trades total
non_q1 = len(trades[trades['body_quintile'] != 'Q1'])
w(f"\n  Total non-Q1 stock trades (2022+): {non_q1}")
w(f"  Trades with at least one option type executed: {combo_n}")
w(f"  Coverage rate: {combo_n/non_q1*100:.1f}%")

w()
w("=" * 100)

# ── Write report ──────────────────────────────────────────────────────────────
report = '\n'.join(lines)
with open('/home/ubuntu/daily_data/analysis_results/options_holdperiod_backtest.txt', 'w') as f:
    f.write(report)

print("\n" + report)
print(f"\nReport saved to: /home/ubuntu/daily_data/analysis_results/options_holdperiod_backtest.txt")
print(f"Trades saved to: /home/ubuntu/daily_data/analysis_results/options_holdperiod_trades.csv")
