#!/usr/bin/env python3
"""
Full options backtest: Option D no-re-entry trades × 4 option types
Using hold_period_options data from 2020-2026.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/backtest_optionD_noReentry_trades.csv')
opts = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options.csv')

print(f"Trade log: {len(trades)} trades")
print(f"Options data: {len(opts)} rows, {opts[['trade_symbol','trade_entry_date']].drop_duplicates().shape[0]} unique trades")
print(f"Option types: {sorted(opts['option_type'].unique())}")

# Parse dates
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])
opts['trade_entry_date'] = pd.to_datetime(opts['trade_entry_date'])
opts['trade_exit_date'] = pd.to_datetime(opts['trade_exit_date'])
opts['bar_date'] = pd.to_datetime(opts['bar_date'])

# ── Option D sizing on the OPTION position ─────────────────────────────
quintile_size = {'Q2': 50_000, 'Q3': 100_000, 'Q4': 150_000, 'Q5': 200_000}

OPTION_TYPES = ['30d_1m', '50d_1m', '30d_3m', '50d_3m']

# ── Process each trade × option type ───────────────────────────────────
results = []

for _, t in trades.iterrows():
    sym = t['symbol']
    edate = t['entry_date']
    xdate = t['exit_date']
    bq = t['body_quintile']

    if bq not in quintile_size:
        continue

    opt_pos_size = quintile_size[bq]

    # Stock info for comparison
    stock_return_pct = t['return_pct']
    stock_pnl = t['net_pnl']

    for otype in OPTION_TYPES:
        # Get matching option rows
        mask = (
            (opts['trade_symbol'] == sym) &
            (opts['trade_entry_date'] == edate) &
            (opts['option_type'] == otype)
        )
        orows = opts[mask].copy()

        if orows.empty:
            continue

        # Entry day
        entry_rows = orows[orows['bar_date'] == edate]
        if entry_rows.empty:
            continue
        er = entry_rows.iloc[0]

        # Entry price: morning_price if available and > 0, else day_open, else day_close
        mp = er['morning_price']
        if pd.notna(mp) and mp > 0:
            option_entry_price = mp
        elif pd.notna(er['day_open']) and er['day_open'] > 0:
            option_entry_price = er['day_open']
        else:
            option_entry_price = er['day_close']

        if pd.isna(option_entry_price) or option_entry_price <= 0:
            continue

        # Exit day
        exit_rows = orows[orows['bar_date'] == xdate]
        if exit_rows.empty:
            continue
        xr = exit_rows.iloc[0]
        option_exit_price = xr['day_close']

        if pd.isna(option_exit_price):
            continue

        # Contracts and P&L
        contracts = int(opt_pos_size / (option_entry_price * 100))
        if contracts < 1:
            continue

        entry_cost = contracts * option_entry_price * 100
        exit_value = contracts * option_exit_price * 100
        transaction_costs = (entry_cost + exit_value) * 5 / 10000
        net_pnl = exit_value - entry_cost - transaction_costs
        option_return_pct = (option_exit_price / option_entry_price - 1) * 100

        option_ticker = er['option_ticker'] if pd.notna(er['option_ticker']) else ''

        results.append({
            'symbol': sym,
            'entry_date': edate,
            'exit_date': xdate,
            'option_type': otype,
            'option_ticker': option_ticker,
            'option_entry_price': round(option_entry_price, 4),
            'option_exit_price': round(option_exit_price, 4),
            'option_return_pct': round(option_return_pct, 2),
            'contracts': contracts,
            'position_size': opt_pos_size,
            'option_pnl': round(net_pnl, 2),
            'stock_return_pct': round(stock_return_pct, 2),
            'stock_pnl': round(stock_pnl, 2),
            'exit_type': t['exit_type'],
            'body_quintile': bq,
            'entry_cost': round(entry_cost, 2),
            'exit_value': round(exit_value, 2),
            'year': edate.year,
        })

df = pd.DataFrame(results)
print(f"\nTotal option trade results: {len(df)}")
print(f"By option type:\n{df['option_type'].value_counts().sort_index()}")

# ── Save trade-level CSV ───────────────────────────────────────────────
csv_cols = ['symbol','entry_date','exit_date','option_type','option_ticker',
            'option_entry_price','option_exit_price','option_return_pct',
            'contracts','position_size','option_pnl','stock_return_pct',
            'stock_pnl','exit_type','body_quintile']
df[csv_cols].to_csv('/home/ubuntu/daily_data/analysis_results/options_full_trades.csv', index=False)
print("Saved options_full_trades.csv")

# ── Helper functions ───────────────────────────────────────────────────
def calc_max_drawdown(pnl_series):
    """Max drawdown from cumulative P&L."""
    cum = pnl_series.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return dd.min()

def calc_profit_factor(pnl_series):
    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0
    return gross_profit / gross_loss

def calc_max_consecutive_losers(pnl_series):
    max_streak = 0
    current = 0
    for p in pnl_series:
        if p < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak

def fmt(v, prefix='$', decimals=0):
    if prefix == '$':
        if abs(v) >= 1e6:
            return f"${v/1e6:,.{decimals}f}M" if decimals else f"${v/1e6:,.1f}M"
        return f"${v:,.{decimals}f}"
    elif prefix == '%':
        return f"{v:,.2f}%"
    return f"{v:,.{decimals}f}"

# ── BUILD REPORT ───────────────────────────────────────────────────────
lines = []
def w(s=''):
    lines.append(s)

w("=" * 100)
w("OPTIONS FULL BACKTEST — Option D No-Re-Entry × Hold-Period Options (2020–2026)")
w("=" * 100)
w()
w(f"Trade log: {len(trades)} trades (Option D no-re-entry)")
w(f"Options data: {len(opts)} rows across {opts[['trade_symbol','trade_entry_date']].drop_duplicates().shape[0]} unique trades")
w(f"Date range: ALL years 2020–2026 (no date filter)")
w(f"Option position sizing: Q2=$50K, Q3=$100K, Q4=$150K, Q5=$200K")
w(f"Transaction costs: 5 bps on entry + exit notional")
w()

# ════════════════════════════════════════════════════════════════════════
# A) SUMMARY BY OPTION TYPE
# ════════════════════════════════════════════════════════════════════════
w("=" * 100)
w("A) SUMMARY BY OPTION TYPE")
w("=" * 100)

for otype in OPTION_TYPES:
    sub = df[df['option_type'] == otype].copy()

    # How many trades had data (even if contracts < 1)?
    # We need to recount from the raw matching
    mask_data = opts[opts['option_type'] == otype][['trade_symbol','trade_entry_date']].drop_duplicates()
    trades_with_data_count = 0
    for _, t in trades.iterrows():
        if t['body_quintile'] not in quintile_size:
            continue
        m = (mask_data['trade_symbol'] == t['symbol']) & (mask_data['trade_entry_date'] == t['entry_date'])
        if m.any():
            trades_with_data_count += 1

    w(f"\n{'─' * 80}")
    w(f"  {otype.upper()}")
    w(f"{'─' * 80}")

    n_exec = len(sub)
    w(f"  Trades with option data:    {trades_with_data_count}")
    w(f"  Trades executed (ctrs>=1):  {n_exec}")

    if n_exec == 0:
        w("  ** No executed trades **")
        continue

    total_pnl = sub['option_pnl'].sum()
    winners = (sub['option_pnl'] > 0).sum()
    losers = (sub['option_pnl'] <= 0).sum()
    wr = winners / n_exec * 100
    pf = calc_profit_factor(sub['option_pnl'])
    avg_pnl = sub['option_pnl'].mean()
    med_pnl = sub['option_pnl'].median()
    avg_ret = sub['option_return_pct'].mean()
    med_ret = sub['option_return_pct'].median()
    max_dd = calc_max_drawdown(sub.sort_values('entry_date')['option_pnl'])
    best = sub.loc[sub['option_pnl'].idxmax()]
    worst = sub.loc[sub['option_pnl'].idxmin()]

    # Stock P&L for same subset
    stock_total = sub['stock_pnl'].sum()
    leverage = total_pnl / stock_total if stock_total != 0 else float('inf')

    w(f"  Total P&L:                  {fmt(total_pnl)}")
    w(f"  Winners / Losers:           {winners} / {losers}")
    w(f"  Win Rate:                   {wr:.1f}%")
    w(f"  Profit Factor:              {pf:.2f}")
    w(f"  Avg P&L per trade:          {fmt(avg_pnl)}")
    w(f"  Median P&L per trade:       {fmt(med_pnl)}")
    w(f"  Avg option return:          {avg_ret:.2f}%")
    w(f"  Median option return:       {med_ret:.2f}%")
    w(f"  Max Drawdown:               {fmt(max_dd)}")
    w(f"  Best trade:                 {best['symbol']} {best['entry_date'].strftime('%Y-%m-%d')} → {fmt(best['option_pnl'])} ({best['option_return_pct']:.1f}%)")
    w(f"  Worst trade:                {worst['symbol']} {worst['entry_date'].strftime('%Y-%m-%d')} → {fmt(worst['option_pnl'])} ({worst['option_return_pct']:.1f}%)")
    w(f"  Stock P&L (same trades):    {fmt(stock_total)}")
    w(f"  Leverage ratio (opt/stock): {leverage:.2f}x")

# ════════════════════════════════════════════════════════════════════════
# B) YEAR-BY-YEAR FOR EACH OPTION TYPE
# ════════════════════════════════════════════════════════════════════════
w()
w("=" * 100)
w("B) YEAR-BY-YEAR BREAKDOWN")
w("=" * 100)

for otype in OPTION_TYPES:
    sub = df[df['option_type'] == otype].copy()
    w(f"\n  {otype.upper()}")
    w(f"  {'Year':<6} {'Trades':>7} {'P&L':>14} {'WR':>8} {'PF':>8} {'Avg P&L':>12} {'Avg Ret%':>10}")
    w(f"  {'─'*6} {'─'*7} {'─'*14} {'─'*8} {'─'*8} {'─'*12} {'─'*10}")

    for year in sorted(sub['year'].unique()):
        ys = sub[sub['year'] == year]
        n = len(ys)
        pnl = ys['option_pnl'].sum()
        wr_ = (ys['option_pnl'] > 0).sum() / n * 100
        pf_ = calc_profit_factor(ys['option_pnl'])
        avg_ = ys['option_pnl'].mean()
        avgr = ys['option_return_pct'].mean()
        w(f"  {year:<6} {n:>7} {fmt(pnl):>14} {wr_:>7.1f}% {pf_:>8.2f} {fmt(avg_):>12} {avgr:>9.2f}%")

    # Total row
    n = len(sub)
    pnl = sub['option_pnl'].sum()
    wr_ = (sub['option_pnl'] > 0).sum() / n * 100
    pf_ = calc_profit_factor(sub['option_pnl'])
    avg_ = sub['option_pnl'].mean()
    avgr = sub['option_return_pct'].mean()
    w(f"  {'TOTAL':<6} {n:>7} {fmt(pnl):>14} {wr_:>7.1f}% {pf_:>8.2f} {fmt(avg_):>12} {avgr:>9.2f}%")

# ════════════════════════════════════════════════════════════════════════
# C) COMPARISON TABLE — all 4 option types + stock
# ════════════════════════════════════════════════════════════════════════
w()
w("=" * 100)
w("C) COMPARISON TABLE — ALL OPTION TYPES + STOCK")
w("=" * 100)

header = f"  {'Metric':<30}"
for otype in OPTION_TYPES:
    header += f" {otype:>14}"
header += f" {'STOCK':>14}"
w(header)
w("  " + "─" * (30 + 15 * 5))

metrics_data = {}
for otype in OPTION_TYPES:
    sub = df[df['option_type'] == otype]
    n = len(sub)
    metrics_data[otype] = {
        'Trades': n,
        'Total P&L': sub['option_pnl'].sum(),
        'Win Rate': (sub['option_pnl'] > 0).sum() / n * 100 if n > 0 else 0,
        'Profit Factor': calc_profit_factor(sub['option_pnl']) if n > 0 else 0,
        'Avg P&L': sub['option_pnl'].mean() if n > 0 else 0,
        'Median P&L': sub['option_pnl'].median() if n > 0 else 0,
        'Avg Return %': sub['option_return_pct'].mean() if n > 0 else 0,
        'Median Return %': sub['option_return_pct'].median() if n > 0 else 0,
        'Max Drawdown': calc_max_drawdown(sub.sort_values('entry_date')['option_pnl']) if n > 0 else 0,
    }

# Stock metrics (using trades that have ANY option data)
all_opt_trades = df[['symbol','entry_date','stock_pnl','stock_return_pct']].drop_duplicates(subset=['symbol','entry_date'])
sn = len(all_opt_trades)
metrics_data['STOCK'] = {
    'Trades': sn,
    'Total P&L': all_opt_trades['stock_pnl'].sum(),
    'Win Rate': (all_opt_trades['stock_pnl'] > 0).sum() / sn * 100 if sn > 0 else 0,
    'Profit Factor': calc_profit_factor(all_opt_trades['stock_pnl']) if sn > 0 else 0,
    'Avg P&L': all_opt_trades['stock_pnl'].mean() if sn > 0 else 0,
    'Median P&L': all_opt_trades['stock_pnl'].median() if sn > 0 else 0,
    'Avg Return %': all_opt_trades['stock_return_pct'].mean() if sn > 0 else 0,
    'Median Return %': all_opt_trades['stock_return_pct'].median() if sn > 0 else 0,
    'Max Drawdown': calc_max_drawdown(all_opt_trades.sort_values('entry_date')['stock_pnl']) if sn > 0 else 0,
}

for metric in ['Trades','Total P&L','Win Rate','Profit Factor','Avg P&L','Median P&L','Avg Return %','Median Return %','Max Drawdown']:
    row = f"  {metric:<30}"
    all_keys = OPTION_TYPES + ['STOCK']
    for k in all_keys:
        v = metrics_data[k][metric]
        if metric == 'Trades':
            row += f" {v:>14,}"
        elif metric in ['Win Rate','Avg Return %','Median Return %']:
            row += f" {v:>13.2f}%"
        elif metric == 'Profit Factor':
            row += f" {v:>14.2f}"
        else:
            row += f" {fmt(v):>14}"
    w(row)

# ════════════════════════════════════════════════════════════════════════
# D) TOP 15 WINNERS AND TOP 15 LOSERS
# ════════════════════════════════════════════════════════════════════════
w()
w("=" * 100)
w("D) TOP 15 WINNERS AND TOP 15 LOSERS BY OPTION TYPE")
w("=" * 100)

detail_cols = ['symbol','entry_date','exit_date','option_ticker','option_entry_price',
               'option_exit_price','option_return_pct','option_pnl','stock_return_pct',
               'exit_type','body_quintile','position_size']

for otype in OPTION_TYPES:
    sub = df[df['option_type'] == otype].copy()

    w(f"\n{'─' * 95}")
    w(f"  {otype.upper()} — TOP 15 WINNERS")
    w(f"{'─' * 95}")
    top15 = sub.nlargest(15, 'option_pnl')
    w(f"  {'Symbol':<7} {'Entry':>10} {'Exit':>10} {'OptTicker':<28} {'OptEnt':>7} {'OptExit':>7} {'OptRet%':>8} {'OptP&L':>12} {'StkRet%':>8} {'ExType':>6} {'BQ':>3} {'PosSize':>8}")
    for _, r in top15.iterrows():
        ticker_short = r['option_ticker'][:27] if len(str(r['option_ticker'])) > 27 else r['option_ticker']
        w(f"  {r['symbol']:<7} {r['entry_date'].strftime('%Y-%m-%d'):>10} {r['exit_date'].strftime('%Y-%m-%d'):>10} {ticker_short:<28} {r['option_entry_price']:>7.2f} {r['option_exit_price']:>7.2f} {r['option_return_pct']:>7.1f}% {fmt(r['option_pnl']):>12} {r['stock_return_pct']:>7.2f}% {r['exit_type']:>6} {r['body_quintile']:>3} {fmt(r['position_size']):>8}")

    w(f"\n  {otype.upper()} — TOP 15 LOSERS")
    w(f"{'─' * 95}")
    bot15 = sub.nsmallest(15, 'option_pnl')
    w(f"  {'Symbol':<7} {'Entry':>10} {'Exit':>10} {'OptTicker':<28} {'OptEnt':>7} {'OptExit':>7} {'OptRet%':>8} {'OptP&L':>12} {'StkRet%':>8} {'ExType':>6} {'BQ':>3} {'PosSize':>8}")
    for _, r in bot15.iterrows():
        ticker_short = r['option_ticker'][:27] if len(str(r['option_ticker'])) > 27 else r['option_ticker']
        w(f"  {r['symbol']:<7} {r['entry_date'].strftime('%Y-%m-%d'):>10} {r['exit_date'].strftime('%Y-%m-%d'):>10} {ticker_short:<28} {r['option_entry_price']:>7.2f} {r['option_exit_price']:>7.2f} {r['option_return_pct']:>7.1f}% {fmt(r['option_pnl']):>12} {r['stock_return_pct']:>7.2f}% {r['exit_type']:>6} {r['body_quintile']:>3} {fmt(r['position_size']):>8}")

# ════════════════════════════════════════════════════════════════════════
# E) COMBINED PORTFOLIO — all 4 types simultaneously
# ════════════════════════════════════════════════════════════════════════
w()
w("=" * 100)
w("E) COMBINED PORTFOLIO — ALL 4 OPTION TYPES SIMULTANEOUSLY")
w("=" * 100)

# Sort all trades by entry_date for drawdown calc
combined = df.sort_values('entry_date').copy()
total_pnl = combined['option_pnl'].sum()
n_total = len(combined)
winners = (combined['option_pnl'] > 0).sum()
wr = winners / n_total * 100
pf = calc_profit_factor(combined['option_pnl'])
max_dd = calc_max_drawdown(combined['option_pnl'])

w(f"  Total trades (all 4 types): {n_total:,}")
w(f"  Total P&L:                  {fmt(total_pnl)}")
w(f"  Win Rate:                   {wr:.1f}%")
w(f"  Profit Factor:              {pf:.2f}")
w(f"  Avg P&L per trade:          {fmt(combined['option_pnl'].mean())}")
w(f"  Median P&L per trade:       {fmt(combined['option_pnl'].median())}")
w(f"  Max Drawdown:               {fmt(max_dd)}")
w()

# Year-by-year combined
w(f"  {'Year':<6} {'Trades':>7} {'P&L':>14} {'WR':>8} {'PF':>8}")
w(f"  {'─'*6} {'─'*7} {'─'*14} {'─'*8} {'─'*8}")
for year in sorted(combined['year'].unique()):
    ys = combined[combined['year'] == year]
    n = len(ys)
    pnl = ys['option_pnl'].sum()
    wr_ = (ys['option_pnl'] > 0).sum() / n * 100
    pf_ = calc_profit_factor(ys['option_pnl'])
    w(f"  {year:<6} {n:>7} {fmt(pnl):>14} {wr_:>7.1f}% {pf_:>8.2f}")

# ════════════════════════════════════════════════════════════════════════
# F) STOCK-ONLY COMPARISON (same subset of trades)
# ════════════════════════════════════════════════════════════════════════
w()
w("=" * 100)
w("F) STOCK-ONLY COMPARISON — TRADES WITH OPTION DATA (APPLES-TO-APPLES)")
w("=" * 100)

# Unique trades that appear in at least one option type
stock_sub = df[['symbol','entry_date','stock_pnl','stock_return_pct','exit_type','body_quintile','year']].drop_duplicates(subset=['symbol','entry_date']).sort_values('entry_date')

sn = len(stock_sub)
stotal = stock_sub['stock_pnl'].sum()
swr = (stock_sub['stock_pnl'] > 0).sum() / sn * 100
spf = calc_profit_factor(stock_sub['stock_pnl'])
sdd = calc_max_drawdown(stock_sub['stock_pnl'])

w(f"  Trades in common:           {sn:,}")
w(f"  Total Stock P&L:            {fmt(stotal)}")
w(f"  Win Rate:                   {swr:.1f}%")
w(f"  Profit Factor:              {spf:.2f}")
w(f"  Avg P&L per trade:          {fmt(stock_sub['stock_pnl'].mean())}")
w(f"  Median P&L per trade:       {fmt(stock_sub['stock_pnl'].median())}")
w(f"  Max Drawdown:               {fmt(sdd)}")
w()

# Per-type stock comparison
w(f"  Stock P&L for trades in each option type:")
w(f"  {'Option Type':<12} {'Trades':>7} {'Stock P&L':>14} {'Stock WR':>10} {'Stock PF':>10}")
w(f"  {'─'*12} {'─'*7} {'─'*14} {'─'*10} {'─'*10}")
for otype in OPTION_TYPES:
    sub = df[df['option_type'] == otype][['symbol','entry_date','stock_pnl']].drop_duplicates(subset=['symbol','entry_date'])
    n = len(sub)
    sp = sub['stock_pnl'].sum()
    sw = (sub['stock_pnl'] > 0).sum() / n * 100 if n > 0 else 0
    sf = calc_profit_factor(sub['stock_pnl']) if n > 0 else 0
    w(f"  {otype:<12} {n:>7} {fmt(sp):>14} {sw:>9.1f}% {sf:>10.2f}")

w()
w(f"  Year-by-year stock performance (trades with any option data):")
w(f"  {'Year':<6} {'Trades':>7} {'Stock P&L':>14} {'WR':>8} {'PF':>8}")
w(f"  {'─'*6} {'─'*7} {'─'*14} {'─'*8} {'─'*8}")
for year in sorted(stock_sub['year'].unique()):
    ys = stock_sub[stock_sub['year'] == year]
    n = len(ys)
    pnl = ys['stock_pnl'].sum()
    wr_ = (ys['stock_pnl'] > 0).sum() / n * 100
    pf_ = calc_profit_factor(ys['stock_pnl'])
    w(f"  {year:<6} {n:>7} {fmt(pnl):>14} {wr_:>7.1f}% {pf_:>8.2f}")

# ════════════════════════════════════════════════════════════════════════
# G) RISK METRICS
# ════════════════════════════════════════════════════════════════════════
w()
w("=" * 100)
w("G) RISK METRICS")
w("=" * 100)

w(f"\n  Max Consecutive Losers:")
w(f"  {'Option Type':<12} {'Max Consec Losers':>18}")
w(f"  {'─'*12} {'─'*18}")
for otype in OPTION_TYPES:
    sub = df[df['option_type'] == otype].sort_values('entry_date')
    mcl = calc_max_consecutive_losers(sub['option_pnl'])
    w(f"  {otype:<12} {mcl:>18}")
# Stock
mcl_stock = calc_max_consecutive_losers(stock_sub['stock_pnl'])
w(f"  {'STOCK':<12} {mcl_stock:>18}")

# Largest single-trade loss per type
w(f"\n  Largest Single-Trade Loss:")
w(f"  {'Option Type':<12} {'Worst Trade P&L':>16} {'Symbol':>8} {'Date':>12}")
w(f"  {'─'*12} {'─'*16} {'─'*8} {'─'*12}")
for otype in OPTION_TYPES:
    sub = df[df['option_type'] == otype]
    if len(sub) > 0:
        worst_idx = sub['option_pnl'].idxmin()
        worst_row = sub.loc[worst_idx]
        w(f"  {otype:<12} {fmt(worst_row['option_pnl']):>16} {worst_row['symbol']:>8} {worst_row['entry_date'].strftime('%Y-%m-%d'):>12}")

# Calmar ratio: annualized return / abs(max drawdown)
w(f"\n  Calmar Ratio (annualized P&L / |max drawdown|):")
w(f"  {'Option Type':<12} {'Total P&L':>14} {'Max DD':>14} {'Years':>6} {'Ann P&L':>14} {'Calmar':>8}")
w(f"  {'─'*12} {'─'*14} {'─'*14} {'─'*6} {'─'*14} {'─'*8}")

# Calculate year span
min_year = df['year'].min()
max_year = df['year'].max()
year_span = max_year - min_year + 1

for otype in OPTION_TYPES:
    sub = df[df['option_type'] == otype].sort_values('entry_date')
    if len(sub) == 0:
        continue
    tp = sub['option_pnl'].sum()
    dd = calc_max_drawdown(sub['option_pnl'])
    ann = tp / year_span
    calmar = ann / abs(dd) if dd != 0 else float('inf')
    w(f"  {otype:<12} {fmt(tp):>14} {fmt(dd):>14} {year_span:>6} {fmt(ann):>14} {calmar:>8.2f}")

# Stock calmar
tp_s = stock_sub['stock_pnl'].sum()
dd_s = calc_max_drawdown(stock_sub['stock_pnl'])
ann_s = tp_s / year_span
calmar_s = ann_s / abs(dd_s) if dd_s != 0 else float('inf')
w(f"  {'STOCK':<12} {fmt(tp_s):>14} {fmt(dd_s):>14} {year_span:>6} {fmt(ann_s):>14} {calmar_s:>8.2f}")

# Combined portfolio calmar
w(f"\n  Combined portfolio (all 4 types):")
ann_c = total_pnl / year_span
calmar_c = ann_c / abs(max_dd) if max_dd != 0 else float('inf')
w(f"  Total P&L: {fmt(total_pnl)}   Max DD: {fmt(max_dd)}   Ann P&L: {fmt(ann_c)}   Calmar: {calmar_c:.2f}")

# Largest single-day portfolio loss (aggregate all option trades exiting on same day)
w(f"\n  Largest Single-Day Portfolio Loss (by exit_date, all option types combined):")
daily_pnl = df.groupby('exit_date')['option_pnl'].sum()
worst_day = daily_pnl.idxmin()
worst_day_pnl = daily_pnl.min()
best_day = daily_pnl.idxmax()
best_day_pnl = daily_pnl.max()
w(f"  Worst day: {worst_day.strftime('%Y-%m-%d')} → {fmt(worst_day_pnl)}")
w(f"  Best day:  {best_day.strftime('%Y-%m-%d')} → {fmt(best_day_pnl)}")

w()
w("=" * 100)
w("END OF REPORT")
w("=" * 100)

# ── Write report ───────────────────────────────────────────────────────
report = '\n'.join(lines)
with open('/home/ubuntu/daily_data/analysis_results/options_full_backtest.txt', 'w') as f:
    f.write(report)

print("\nSaved options_full_backtest.txt")
print("\n" + report)
