#!/usr/bin/env python3
"""Options backtest for Trade 1 with $100K base and Option D sizing."""

import pandas as pd
import numpy as np
import math
from collections import defaultdict

# ─── Load data ───
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv')
options = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/options_bars_clean.csv')
enriched = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_enriched.csv')

print(f"Trades: {len(trades)}, Options rows: {len(options)}, Enriched: {len(enriched)}")

# ─── Merge body_pct from enriched into trades ───
# Use symbol + entry_date as key
enriched_sub = enriched[['symbol', 'entry_date', 'entry_body_pct']].copy()
trades = trades.merge(enriched_sub, on=['symbol', 'entry_date'], how='left')
print(f"Trades with body_pct: {trades['entry_body_pct'].notna().sum()}")

# ─── Option D sizing ───
BASE = 100_000

def get_option_d_size(body_pct):
    """Return position size based on body_pct quintile, or None to skip."""
    if pd.isna(body_pct):
        return None  # skip if no data
    if body_pct <= -0.53:      # Q1: skip
        return None
    elif body_pct <= -0.17:    # Q2: 0.5x
        return BASE * 0.5
    elif body_pct <= 0.23:     # Q3: 1.0x
        return BASE * 1.0
    elif body_pct <= 0.57:     # Q4: 1.5x
        return BASE * 1.5
    else:                      # Q5: 2.0x
        return BASE * 2.0

def get_quintile_label(body_pct):
    if pd.isna(body_pct):
        return 'NA'
    if body_pct <= -0.53:
        return 'Q1'
    elif body_pct <= -0.17:
        return 'Q2'
    elif body_pct <= 0.23:
        return 'Q3'
    elif body_pct <= 0.57:
        return 'Q4'
    else:
        return 'Q5'

trades['quintile'] = trades['entry_body_pct'].apply(get_quintile_label)
trades['position_size'] = trades['entry_body_pct'].apply(get_option_d_size)

# ─── Build options lookup: (symbol, date, date_type, delta) → avg_price ───
# Filter to rows with valid avg_price
options['avg_price'] = pd.to_numeric(options['avg_price'], errors='coerce')
options_valid = options[options['avg_price'].notna() & (options['avg_price'] > 0)].copy()

# Build lookup dict
opt_lookup = {}
for _, row in options_valid.iterrows():
    key = (row['symbol'], str(row['date']), row['date_type'], int(row['delta_target']))
    opt_lookup[key] = row['avg_price']

print(f"Options lookup entries: {len(opt_lookup)}")

# ─── Run backtest for each delta ───
DELTAS = [80, 70, 60, 50, 40, 30, 20, 10]

all_results = {}

for delta in DELTAS:
    results = []

    for _, t in trades.iterrows():
        sym = t['symbol']
        entry_date = str(t['entry_date'])
        exit_date = str(t['exit_date'])
        pos_size = t['position_size']
        quintile = t['quintile']

        # Look up entry and exit option prices
        entry_key = (sym, entry_date, 'entry', delta)
        exit_key = (sym, exit_date, 'exit', delta)

        entry_price = opt_lookup.get(entry_key)
        exit_price = opt_lookup.get(exit_key)

        has_data = (entry_price is not None and exit_price is not None)

        rec = {
            'symbol': sym,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'quintile': quintile,
            'body_pct': t['entry_body_pct'],
            'position_size': pos_size,
            'stock_entry': t['entry_price'],
            'stock_exit': t['exit_price'],
            'stock_return_pct': t['return_pct'],
            'has_option_data': has_data,
            'entry_option_price': entry_price,
            'exit_option_price': exit_price,
            'skipped_q1': quintile == 'Q1',
            'contracts': 0,
            'entry_cost': 0,
            'exit_value': 0,
            'costs': 0,
            'net_pnl': 0,
            'option_return_pct': 0,
            'executed': False,
            # For stock comparison
            'stock_pnl_optD': 0,
            'stock_executed': False,
        }

        if has_data and pos_size is not None and not (isinstance(pos_size, float) and math.isnan(pos_size)) and entry_price > 0:
            contracts = math.floor(pos_size / (entry_price * 100))
            if contracts >= 1:
                entry_cost = contracts * entry_price * 100
                exit_value = contracts * exit_price * 100
                costs = (entry_cost + exit_value) * 5 / 10000
                net_pnl = exit_value - entry_cost - costs
                option_return_pct = (exit_price / entry_price - 1) * 100

                rec['contracts'] = contracts
                rec['entry_cost'] = entry_cost
                rec['exit_value'] = exit_value
                rec['costs'] = costs
                rec['net_pnl'] = net_pnl
                rec['option_return_pct'] = option_return_pct
                rec['executed'] = True

        # Stock comparison: same trade, same Option D sizing, stock basis
        if has_data and pos_size is not None and not (isinstance(pos_size, float) and math.isnan(pos_size)):
            stock_entry = t['entry_price']
            stock_exit = t['exit_price']
            if stock_entry > 0:
                shares = math.floor(pos_size / stock_entry)
                if shares >= 1:
                    s_entry_cost = shares * stock_entry
                    s_exit_value = shares * stock_exit
                    s_costs = (s_entry_cost + s_exit_value) * 5 / 10000
                    rec['stock_pnl_optD'] = s_exit_value - s_entry_cost - s_costs
                    rec['stock_executed'] = True

        results.append(rec)

    all_results[delta] = pd.DataFrame(results)

# ─── Generate report ───
report_lines = []
def rpt(line=''):
    report_lines.append(line)

rpt("=" * 100)
rpt("OPTIONS BACKTEST REPORT — Trade 1, $100K Base, Option D Sizing")
rpt("=" * 100)
rpt()
rpt(f"Total trades in trade log: {len(trades)}")
rpt(f"Quintile distribution:")
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'NA']:
    cnt = (trades['quintile'] == q).sum()
    rpt(f"  {q}: {cnt} trades")
rpt()
rpt("Option D sizing ($100K base):")
rpt("  Q1 (body_pct <= -0.53): SKIP")
rpt("  Q2 (-0.53 to -0.17):   $50K  (0.5x)")
rpt("  Q3 (-0.17 to +0.23):   $100K (1.0x)")
rpt("  Q4 (+0.23 to +0.57):   $150K (1.5x)")
rpt("  Q5 (body_pct > +0.57): $200K (2.0x)")
rpt()

# ─── Summary table data ───
summary_rows = []

for delta in DELTAS:
    df = all_results[delta]

    with_data = df[df['has_option_data']].copy()
    after_filter = with_data[~with_data['skipped_q1']].copy()
    executed = after_filter[after_filter['executed']].copy()

    rpt("=" * 100)
    rpt(f"DELTA {delta}")
    rpt("=" * 100)
    rpt(f"  Trades with options data (entry+exit):  {len(with_data)}")
    rpt(f"  After Option D filter (Q1 skipped):     {len(after_filter)}")
    rpt(f"  Trades actually executed (contracts≥1):  {len(executed)}")

    if len(executed) == 0:
        rpt("  No executed trades at this delta.")
        rpt()
        summary_rows.append({
            'delta': delta, 'trades': 0, 'total_pnl': 0, 'wr': 0, 'pf': 0,
            'avg_pnl': 0, 'max_dd': 0, 'stock_pnl': 0, 'leverage': 0
        })
        continue

    total_pnl = executed['net_pnl'].sum()
    winners = executed[executed['net_pnl'] > 0]
    losers = executed[executed['net_pnl'] < 0]
    win_rate = len(winners) / len(executed) * 100
    gross_wins = winners['net_pnl'].sum() if len(winners) > 0 else 0
    gross_losses = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    avg_pnl = total_pnl / len(executed)
    biggest_winner = executed['net_pnl'].max()
    biggest_loser = executed['net_pnl'].min()

    # Max drawdown
    cumulative = executed['net_pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    # Stock comparison for same executed trades
    stock_comp = executed[executed['stock_executed']]
    stock_total_pnl = stock_comp['stock_pnl_optD'].sum()
    leverage_ratio = total_pnl / stock_total_pnl if stock_total_pnl != 0 else float('inf')

    rpt(f"  Total P&L:              ${total_pnl:,.2f}")
    rpt(f"  Win Rate:               {win_rate:.1f}%")
    rpt(f"  Profit Factor:          {profit_factor:.2f}")
    rpt(f"  Avg P&L per trade:      ${avg_pnl:,.2f}")
    rpt(f"  Biggest Winner:         ${biggest_winner:,.2f}")
    rpt(f"  Biggest Loser:          ${biggest_loser:,.2f}")
    rpt(f"  Max Drawdown:           ${max_dd:,.2f}")
    rpt(f"  Stock P&L (same trades): ${stock_total_pnl:,.2f}")
    rpt(f"  Leverage Ratio (opt/stk): {leverage_ratio:.2f}x")
    rpt()

    # Year by year
    executed = executed.copy()
    executed['year'] = pd.to_datetime(executed['entry_date']).dt.year
    rpt(f"  Year-by-Year Breakdown:")
    rpt(f"  {'Year':<8} {'Trades':>7} {'P&L':>14} {'WR':>8} {'Avg P&L':>12}")
    rpt(f"  {'-'*8} {'-'*7} {'-'*14} {'-'*8} {'-'*12}")
    for year in sorted(executed['year'].unique()):
        yr = executed[executed['year'] == year]
        yr_pnl = yr['net_pnl'].sum()
        yr_wr = (yr['net_pnl'] > 0).sum() / len(yr) * 100
        yr_avg = yr_pnl / len(yr)
        rpt(f"  {year:<8} {len(yr):>7} ${yr_pnl:>13,.2f} {yr_wr:>7.1f}% ${yr_avg:>11,.2f}")
    rpt()

    summary_rows.append({
        'delta': delta,
        'trades': len(executed),
        'total_pnl': total_pnl,
        'wr': win_rate,
        'pf': profit_factor,
        'avg_pnl': avg_pnl,
        'max_dd': max_dd,
        'stock_pnl': stock_total_pnl,
        'leverage': leverage_ratio
    })

# ─── Summary comparison table ───
rpt()
rpt("=" * 100)
rpt("SUMMARY COMPARISON TABLE")
rpt("=" * 100)
rpt()
header = f"{'Delta':>6} | {'Trades':>7} | {'Total P&L':>14} | {'WR':>7} | {'PF':>6} | {'Avg P&L':>11} | {'MaxDD':>14} | {'Stock P&L':>14} | {'Lev Ratio':>10}"
rpt(header)
rpt("-" * len(header))
for r in summary_rows:
    pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else 'inf'
    lev_str = f"{r['leverage']:.2f}x" if r['leverage'] != float('inf') else 'inf'
    rpt(f"{r['delta']:>6} | {r['trades']:>7} | ${r['total_pnl']:>13,.2f} | {r['wr']:>6.1f}% | {pf_str:>6} | ${r['avg_pnl']:>10,.2f} | ${r['max_dd']:>13,.2f} | ${r['stock_pnl']:>13,.2f} | {lev_str:>10}")

# ─── 50-delta top winners and losers ───
rpt()
rpt("=" * 100)
rpt("50-DELTA: 10 BIGGEST OPTION WINNERS")
rpt("=" * 100)
rpt()

df50 = all_results[50]
executed50 = df50[df50['executed']].copy()

if len(executed50) > 0:
    top_winners = executed50.nlargest(10, 'net_pnl')
    rpt(f"{'#':>3} {'Symbol':<7} {'Entry Date':<12} {'Exit Date':<12} {'Q':>3} {'BodyPct':>8} {'PosSize':>9} {'Ctrs':>5} {'EntPx':>8} {'ExPx':>8} {'OptRet%':>8} {'Net P&L':>12} {'StockRet%':>9}")
    rpt("-" * 120)
    for i, (_, r) in enumerate(top_winners.iterrows(), 1):
        rpt(f"{i:>3} {r['symbol']:<7} {r['entry_date']:<12} {r['exit_date']:<12} {r['quintile']:>3} {r['body_pct']:>8.2f} ${r['position_size']:>8,.0f} {r['contracts']:>5} ${r['entry_option_price']:>7.2f} ${r['exit_option_price']:>7.2f} {r['option_return_pct']:>7.1f}% ${r['net_pnl']:>11,.2f} {r['stock_return_pct']:>8.2f}%")

    rpt()
    rpt("=" * 100)
    rpt("50-DELTA: 10 BIGGEST OPTION LOSERS")
    rpt("=" * 100)
    rpt()
    top_losers = executed50.nsmallest(10, 'net_pnl')
    rpt(f"{'#':>3} {'Symbol':<7} {'Entry Date':<12} {'Exit Date':<12} {'Q':>3} {'BodyPct':>8} {'PosSize':>9} {'Ctrs':>5} {'EntPx':>8} {'ExPx':>8} {'OptRet%':>8} {'Net P&L':>12} {'StockRet%':>9}")
    rpt("-" * 120)
    for i, (_, r) in enumerate(top_losers.iterrows(), 1):
        rpt(f"{i:>3} {r['symbol']:<7} {r['entry_date']:<12} {r['exit_date']:<12} {r['quintile']:>3} {r['body_pct']:>8.2f} ${r['position_size']:>8,.0f} {r['contracts']:>5} ${r['entry_option_price']:>7.2f} ${r['exit_option_price']:>7.2f} {r['option_return_pct']:>7.1f}% ${r['net_pnl']:>11,.2f} {r['stock_return_pct']:>8.2f}%")

rpt()
rpt("=" * 100)
rpt("END OF REPORT")
rpt("=" * 100)

# Write report
report_text = '\n'.join(report_lines)
with open('/home/ubuntu/daily_data/analysis_results/options_backtest_100k_report.txt', 'w') as f:
    f.write(report_text)

print("\nReport saved.")
print(report_text)
