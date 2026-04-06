#!/usr/bin/env python3
"""
Full backtest with ATR-stop simulation for 3 schemes:
  Baseline: flat $1M
  Scheme 1: size by entry_body_pct quintile (downsize weak, upsize strong)
  Scheme 2: skip Q1, flat $1M for Q2-Q5
"""

import pandas as pd
import numpy as np
import math
import os
import sys
from collections import defaultdict

# ── Load trades and enriched ──────────────────────────────────────────────
trades = pd.read_csv("/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv")
enriched = pd.read_csv("/home/ubuntu/daily_data/analysis_results/trade1_20d_enriched.csv")

print(f"Trades loaded: {len(trades)}")
print(f"Enriched loaded: {len(enriched)}")

# Join entry_body_pct from enriched onto trades
enriched_subset = enriched[['symbol', 'signal_date', 'entry_body_pct']].copy()
trades = trades.merge(enriched_subset, on=['symbol', 'signal_date'], how='left')
print(f"Trades after merge: {len(trades)}")
print(f"entry_body_pct nulls: {trades['entry_body_pct'].isna().sum()}")

# Drop rows with missing entry_body_pct
trades = trades.dropna(subset=['entry_body_pct']).reset_index(drop=True)
print(f"Trades after dropping NaN body_pct: {len(trades)}")

# ── Compute quintiles on entry_body_pct ───────────────────────────────────
trades['body_quintile'] = pd.qcut(trades['entry_body_pct'], 5, labels=['Q1','Q2','Q3','Q4','Q5'])
boundaries = trades.groupby('body_quintile')['entry_body_pct'].agg(['min','max','count'])
print("\nentry_body_pct quintile boundaries:")
print(boundaries)

# ── Cache symbol daily data ───────────────────────────────────────────────
symbol_data_cache = {}

def get_symbol_data(symbol):
    if symbol not in symbol_data_cache:
        fpath = f"/home/ubuntu/daily_data/data/{symbol}_enriched.csv"
        if not os.path.exists(fpath):
            symbol_data_cache[symbol] = None
            return None
        df = pd.read_csv(fpath, usecols=['date','open','high','low','close','atr_14'])
        df['date'] = df['date'].astype(str)
        df = df.sort_values('date').reset_index(drop=True)
        symbol_data_cache[symbol] = df
    return symbol_data_cache[symbol]

# ── Simulate one trade ────────────────────────────────────────────────────
def simulate_trade(symbol, signal_date, entry_date, position_size):
    """
    Returns dict with trade results or None if data missing.
    """
    df = get_symbol_data(symbol)
    if df is None:
        return None

    signal_date_str = str(signal_date)
    entry_date_str = str(entry_date)

    # Find signal_date row for low and atr_14
    sig_rows = df[df['date'] == signal_date_str]
    if len(sig_rows) == 0:
        return None
    sig_row = sig_rows.iloc[0]
    signal_low = sig_row['low']
    signal_atr = sig_row['atr_14']
    if pd.isna(signal_low) or pd.isna(signal_atr):
        return None

    stop_price = signal_low - 3.0 * signal_atr

    # Find entry_date row for entry price (open)
    entry_rows = df[df['date'] == entry_date_str]
    if len(entry_rows) == 0:
        return None
    entry_row = entry_rows.iloc[0]
    entry_price = entry_row['open']
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    shares = math.floor(position_size / entry_price)
    if shares <= 0:
        return None

    # Find entry_date index and walk next 20 trading days
    entry_idx = entry_rows.index[0]
    # Days from entry_date (inclusive) up to 20 trading days
    end_idx = min(entry_idx + 20, len(df))  # entry_idx + 20 means day index 0..19

    exit_price = None
    exit_date = None
    stop_hit = False

    for i in range(entry_idx, end_idx):
        row = df.iloc[i]
        day_low = row['low']

        if i == entry_idx:
            # On entry day, check if low hits stop (after open)
            if day_low <= stop_price:
                exit_price = stop_price
                exit_date = row['date']
                stop_hit = True
                break
        else:
            if day_low <= stop_price:
                exit_price = stop_price
                exit_date = row['date']
                stop_hit = True
                break

    if exit_price is None:
        # Exit at close of day 20 (index entry_idx + 19)
        day20_idx = entry_idx + 19
        if day20_idx >= len(df):
            day20_idx = len(df) - 1
        exit_row = df.iloc[day20_idx]
        exit_price = exit_row['close']
        exit_date = exit_row['date']

    gross_pnl = shares * (exit_price - entry_price)
    costs = shares * (entry_price + exit_price) * 5.0 / 10000.0
    net_pnl = gross_pnl - costs

    return {
        'symbol': symbol,
        'signal_date': signal_date_str,
        'entry_date': entry_date_str,
        'exit_date': exit_date,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'stop_price': stop_price,
        'stop_hit': stop_hit,
        'shares': shares,
        'position_size': shares * entry_price,
        'gross_pnl': gross_pnl,
        'costs': costs,
        'net_pnl': net_pnl,
    }

# ── Sizing maps ──────────────────────────────────────────────────────────
scheme1_sizes = {'Q1': 250_000, 'Q2': 500_000, 'Q3': 1_000_000, 'Q4': 1_500_000, 'Q5': 2_000_000}
scheme2_sizes = {'Q1': 0, 'Q2': 1_000_000, 'Q3': 1_000_000, 'Q4': 1_000_000, 'Q5': 1_000_000}
baseline_size = 1_000_000

# ── Run all three schemes ────────────────────────────────────────────────
results = {}

for scheme_name, size_map in [('Baseline', None), ('Scheme1', scheme1_sizes), ('Scheme2', scheme2_sizes)]:
    print(f"\n{'='*60}")
    print(f"Running {scheme_name}...")
    trade_results = []
    skipped = 0
    data_missing = 0

    for idx, row in trades.iterrows():
        quintile = row['body_quintile']

        if size_map is None:
            pos_size = baseline_size
        else:
            pos_size = size_map[quintile]

        if pos_size == 0:
            skipped += 1
            continue

        result = simulate_trade(row['symbol'], row['signal_date'], row['entry_date'], pos_size)
        if result is None:
            data_missing += 1
            continue

        result['body_quintile'] = quintile
        result['entry_body_pct'] = row['entry_body_pct']
        trade_results.append(result)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(trades)} trades...")

    print(f"  Completed: {len(trade_results)} trades, {skipped} skipped, {data_missing} data missing")
    results[scheme_name] = pd.DataFrame(trade_results)

# ── Compute metrics ──────────────────────────────────────────────────────
def compute_metrics(df, label):
    if len(df) == 0:
        return {}

    total_trades = len(df)
    total_net = df['net_pnl'].sum()
    wins = df[df['net_pnl'] > 0]
    losses = df[df['net_pnl'] <= 0]
    win_rate = len(wins) / total_trades * 100

    gross_wins = wins['net_pnl'].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 1
    profit_factor = gross_wins / gross_losses if gross_losses != 0 else float('inf')

    avg_pnl = df['net_pnl'].mean()
    worst_trade = df['net_pnl'].min()
    best_trade = df['net_pnl'].max()

    # Sort by entry_date for chronological cumulative P&L
    df_sorted = df.sort_values('entry_date').reset_index(drop=True)
    cum_pnl = df_sorted['net_pnl'].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # Sharpe: annualized from trade returns
    # Each trade is ~20 days, so ~252/20 = 12.6 trades per year
    trade_returns = df_sorted['net_pnl'] / df_sorted['position_size']
    mean_ret = trade_returns.mean()
    std_ret = trade_returns.std()
    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(252 / 20)
    else:
        sharpe = 0

    # Average capital deployed per day
    # For each trade, it's open for some number of days
    # We need to compute across all calendar trading days
    df_sorted['entry_date_dt'] = pd.to_datetime(df_sorted['entry_date'])
    df_sorted['exit_date_dt'] = pd.to_datetime(df_sorted['exit_date'])

    all_dates = set()
    for _, r in df_sorted.iterrows():
        # Generate business days between entry and exit
        dates = pd.bdate_range(r['entry_date_dt'], r['exit_date_dt'])
        all_dates.update(dates)

    if len(all_dates) > 0:
        daily_capital = {}
        for d in all_dates:
            mask = (df_sorted['entry_date_dt'] <= d) & (df_sorted['exit_date_dt'] >= d)
            daily_capital[d] = df_sorted.loc[mask, 'position_size'].sum()
        avg_capital = np.mean(list(daily_capital.values()))
    else:
        avg_capital = 0

    # Stop hit rate
    stop_hits = df['stop_hit'].sum()
    stop_rate = stop_hits / total_trades * 100

    return {
        'label': label,
        'total_trades': total_trades,
        'total_net_pnl': total_net,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_pnl': avg_pnl,
        'max_drawdown': max_dd,
        'worst_trade': worst_trade,
        'best_trade': best_trade,
        'sharpe': sharpe,
        'avg_capital_deployed': avg_capital,
        'stop_hit_rate': stop_rate,
    }

def compute_yearly(df):
    df = df.copy()
    df['year'] = pd.to_datetime(df['entry_date']).dt.year
    yearly = []
    for year, gdf in df.groupby('year'):
        n = len(gdf)
        net = gdf['net_pnl'].sum()
        wr = (gdf['net_pnl'] > 0).sum() / n * 100
        wins_sum = gdf.loc[gdf['net_pnl'] > 0, 'net_pnl'].sum()
        losses_sum = abs(gdf.loc[gdf['net_pnl'] <= 0, 'net_pnl'].sum())
        pf = wins_sum / losses_sum if losses_sum > 0 else float('inf')
        yearly.append({'year': year, 'trades': n, 'net_pnl': net, 'win_rate': wr, 'profit_factor': pf})
    return pd.DataFrame(yearly)

# Compute metrics for all schemes
metrics = {}
for name, df in results.items():
    print(f"\nComputing metrics for {name}...")
    metrics[name] = compute_metrics(df, name)

# ── Print results ────────────────────────────────────────────────────────
def fmt_money(v):
    if v >= 0:
        return f"${v:,.0f}"
    else:
        return f"-${abs(v):,.0f}"

def fmt_pct(v):
    return f"{v:.1f}%"

print("\n" + "="*80)
print("BACKTEST SIZING COMPARISON REPORT")
print("="*80)

header = f"{'Metric':<30} {'Baseline':>18} {'Scheme 1':>18} {'Scheme 2':>18}"
print(header)
print("-"*84)

for key, label in [
    ('total_trades', 'Total Trades'),
    ('total_net_pnl', 'Total Net P&L'),
    ('win_rate', 'Win Rate'),
    ('profit_factor', 'Profit Factor'),
    ('avg_pnl', 'Avg P&L / Trade'),
    ('max_drawdown', 'Max Drawdown'),
    ('worst_trade', 'Worst Trade'),
    ('best_trade', 'Best Trade'),
    ('sharpe', 'Sharpe (Ann.)'),
    ('avg_capital_deployed', 'Avg Capital Deployed'),
    ('stop_hit_rate', 'Stop Hit Rate'),
]:
    vals = []
    for scheme in ['Baseline', 'Scheme1', 'Scheme2']:
        v = metrics[scheme].get(key, 0)
        if key in ('total_net_pnl', 'avg_pnl', 'max_drawdown', 'worst_trade', 'best_trade', 'avg_capital_deployed'):
            vals.append(fmt_money(v))
        elif key in ('win_rate', 'stop_hit_rate'):
            vals.append(fmt_pct(v))
        elif key == 'profit_factor':
            vals.append(f"{v:.2f}")
        elif key == 'sharpe':
            vals.append(f"{v:.3f}")
        elif key == 'total_trades':
            vals.append(f"{v:,}")
        else:
            vals.append(f"{v}")
    print(f"{label:<30} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

# Year by year
for scheme_name in ['Baseline', 'Scheme1', 'Scheme2']:
    print(f"\n--- {scheme_name} Year-by-Year ---")
    ydf = compute_yearly(results[scheme_name])
    print(f"{'Year':<6} {'Trades':>8} {'Net P&L':>16} {'WR':>8} {'PF':>8}")
    for _, r in ydf.iterrows():
        print(f"{int(r['year']):<6} {int(r['trades']):>8} {fmt_money(r['net_pnl']):>16} {fmt_pct(r['win_rate']):>8} {r['profit_factor']:>8.2f}")

# ── Save trade logs ──────────────────────────────────────────────────────
out_dir = "/home/ubuntu/daily_data/analysis_results"

results['Scheme1'].to_csv(f"{out_dir}/backtest_scheme1_trades.csv", index=False)
print(f"\nSaved: {out_dir}/backtest_scheme1_trades.csv")

results['Scheme2'].to_csv(f"{out_dir}/backtest_scheme2_trades.csv", index=False)
print(f"Saved: {out_dir}/backtest_scheme2_trades.csv")

# ── Save report ──────────────────────────────────────────────────────────
report_lines = []
report_lines.append("=" * 84)
report_lines.append("BACKTEST SIZING COMPARISON REPORT")
report_lines.append("=" * 84)
report_lines.append("")
report_lines.append("STRATEGY: Q5 z-score entry, 20-day hold, ATR(14) x3.0 stop from signal_day_low")
report_lines.append("COSTS: 5 bps each way")
report_lines.append("")
report_lines.append("SIZING SCHEMES:")
report_lines.append("  Baseline: Flat $1M per trade, all trades")
report_lines.append("  Scheme 1: Downsize weak candles, upsize strong candles")
report_lines.append("    Q1=$250K, Q2=$500K, Q3=$1M, Q4=$1.5M, Q5=$2M")
report_lines.append("  Scheme 2: Skip Q1 (weak candles), flat $1M for Q2-Q5")
report_lines.append("")
report_lines.append("entry_body_pct quintile boundaries:")
report_lines.append(boundaries.to_string())
report_lines.append("")

header = f"{'Metric':<30} {'Baseline':>18} {'Scheme 1':>18} {'Scheme 2':>18}"
report_lines.append(header)
report_lines.append("-" * 84)

for key, label in [
    ('total_trades', 'Total Trades'),
    ('total_net_pnl', 'Total Net P&L'),
    ('win_rate', 'Win Rate'),
    ('profit_factor', 'Profit Factor'),
    ('avg_pnl', 'Avg P&L / Trade'),
    ('max_drawdown', 'Max Drawdown'),
    ('worst_trade', 'Worst Trade'),
    ('best_trade', 'Best Trade'),
    ('sharpe', 'Sharpe (Ann.)'),
    ('avg_capital_deployed', 'Avg Capital Deployed'),
    ('stop_hit_rate', 'Stop Hit Rate'),
]:
    vals = []
    for scheme in ['Baseline', 'Scheme1', 'Scheme2']:
        v = metrics[scheme].get(key, 0)
        if key in ('total_net_pnl', 'avg_pnl', 'max_drawdown', 'worst_trade', 'best_trade', 'avg_capital_deployed'):
            vals.append(fmt_money(v))
        elif key in ('win_rate', 'stop_hit_rate'):
            vals.append(fmt_pct(v))
        elif key == 'profit_factor':
            vals.append(f"{v:.2f}")
        elif key == 'sharpe':
            vals.append(f"{v:.3f}")
        elif key == 'total_trades':
            vals.append(f"{v:,}")
        else:
            vals.append(f"{v}")
    report_lines.append(f"{label:<30} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

for scheme_name in ['Baseline', 'Scheme1', 'Scheme2']:
    report_lines.append("")
    report_lines.append(f"--- {scheme_name} Year-by-Year ---")
    ydf = compute_yearly(results[scheme_name])
    report_lines.append(f"{'Year':<6} {'Trades':>8} {'Net P&L':>16} {'WR':>8} {'PF':>8}")
    for _, r in ydf.iterrows():
        report_lines.append(f"{int(r['year']):<6} {int(r['trades']):>8} {fmt_money(r['net_pnl']):>16} {fmt_pct(r['win_rate']):>8} {r['profit_factor']:>8.2f}")

report_text = "\n".join(report_lines)

with open(f"{out_dir}/backtest_sizing_report.txt", 'w') as f:
    f.write(report_text)
print(f"\nSaved: {out_dir}/backtest_sizing_report.txt")

print("\nDONE.")
