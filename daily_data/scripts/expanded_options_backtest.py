#!/usr/bin/env python3
"""
Expanded Options Backtest: Merge original + backfill options data,
run hybrid and options-only backtests with tier classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/home/ubuntu/daily_data/data')
RESULTS_DIR = Path('/home/ubuntu/daily_data/analysis_results')
OPTIONS_DIR = DATA_DIR / 'trade_options'

# ─── STEP 1: Load and merge options data ───────────────────────────────────────

print("=== STEP 1: Loading and merging options data ===")

# Load original (30d_1m only)
opts_orig = pd.read_csv(OPTIONS_DIR / 'hold_period_options.csv')
opts_orig_30d = opts_orig[opts_orig['option_type'] == '30d_1m'].copy()
print(f"Original 30d_1m rows: {len(opts_orig_30d)}")

# Load backfill
opts_backfill = pd.read_csv(OPTIONS_DIR / 'hold_period_options_backfill.csv')
print(f"Backfill rows: {len(opts_backfill)}")

# Standardize backfill columns to match original
# Backfill has extra cols: actual_delta, search_attempt - drop them for merge
common_cols = [c for c in opts_orig_30d.columns if c in opts_backfill.columns]
opts_backfill_clean = opts_backfill[common_cols].copy()
opts_backfill_clean['option_type'] = '30d_1m'  # treat as equivalent

# Combine: original 30d_1m + backfill
opts_combined = pd.concat([opts_orig_30d, opts_backfill_clean], ignore_index=True)
print(f"Combined rows: {len(opts_combined)}")

# Parse dates
for col in ['trade_entry_date', 'trade_exit_date', 'bar_date']:
    opts_combined[col] = pd.to_datetime(opts_combined[col])

# ─── Load trade log ────────────────────────────────────────────────────────────

trades = pd.read_csv(RESULTS_DIR / 'backtest_optionD_noReentry_trades.csv')
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['signal_date'] = pd.to_datetime(trades['signal_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])
print(f"Trade log: {len(trades)} trades")

# ─── Assess coverage ──────────────────────────────────────────────────────────

# For each trade, check if we have entry day + 13th bar day prices
def assess_coverage(opts_df, trades_df):
    """Check which trades have usable entry+exit option prices."""
    coverage = {}

    for _, trade in trades_df.iterrows():
        sym = trade['symbol']
        edate = trade['entry_date']
        key = (sym, edate)

        # Get option bars for this trade
        mask = (opts_df['trade_symbol'] == sym) & (opts_df['trade_entry_date'] == edate)
        trade_opts = opts_df[mask].dropna(subset=['day_close'])

        if len(trade_opts) == 0:
            coverage[key] = {'has_data': False}
            continue

        # Sort by bar_date
        trade_opts = trade_opts.sort_values('bar_date')
        bars = trade_opts['bar_date'].unique()

        # Entry day = first bar_date that matches entry_date
        entry_bars = trade_opts[trade_opts['bar_date'] == edate]
        if len(entry_bars) == 0:
            # Try first available bar
            entry_bars = trade_opts.iloc[:1]

        # Need at least entry price
        if len(entry_bars) == 0:
            coverage[key] = {'has_data': False}
            continue

        entry_price = entry_bars.iloc[0]['morning_price']
        if pd.isna(entry_price) or entry_price <= 0:
            entry_price = entry_bars.iloc[0]['day_open']
        if pd.isna(entry_price) or entry_price <= 0:
            entry_price = entry_bars.iloc[0]['day_close']
        if pd.isna(entry_price) or entry_price <= 0:
            coverage[key] = {'has_data': False}
            continue

        # Exit = 13th bar_date in sorted order (index 12)
        sorted_bars = sorted(bars)
        if len(sorted_bars) >= 13:
            exit_date = sorted_bars[12]
            exit_bars = trade_opts[trade_opts['bar_date'] == exit_date]
            exit_price = exit_bars.iloc[0]['day_close']
        else:
            # Use last available bar
            exit_date = sorted_bars[-1]
            exit_bars = trade_opts[trade_opts['bar_date'] == exit_date]
            exit_price = exit_bars.iloc[0]['day_close']

        if pd.isna(exit_price):
            coverage[key] = {'has_data': False}
            continue

        coverage[key] = {
            'has_data': True,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_bar_date': entry_bars.iloc[0]['bar_date'],
            'exit_bar_date': exit_date,
            'n_bars': len(sorted_bars),
            'option_ticker': entry_bars.iloc[0]['option_ticker'],
        }

    return coverage

# Coverage with original only
print("\nAssessing coverage with original 30d_1m only...")
coverage_orig = assess_coverage(opts_orig_30d.assign(
    trade_entry_date=pd.to_datetime(opts_orig_30d['trade_entry_date']),
    trade_exit_date=pd.to_datetime(opts_orig_30d['trade_exit_date']),
    bar_date=pd.to_datetime(opts_orig_30d['bar_date'])
), trades)
orig_with_data = sum(1 for v in coverage_orig.values() if v['has_data'])

# Coverage with combined
print("Assessing coverage with combined (original + backfill)...")
coverage_combined = assess_coverage(opts_combined, trades)
combined_with_data = sum(1 for v in coverage_combined.values() if v['has_data'])

# Still missing
missing_trades = []
for _, trade in trades.iterrows():
    key = (trade['symbol'], trade['entry_date'])
    if not coverage_combined.get(key, {}).get('has_data', False):
        missing_trades.append(trade)

missing_symbols = sorted(set(t['symbol'] for t in missing_trades))

print(f"\n--- COVERAGE SUMMARY ---")
print(f"Before backfill: {orig_with_data} trades with 30d_1m data")
print(f"After backfill:  {combined_with_data} trades with call option data")
print(f"Still missing:   {len(missing_trades)} trades")
print(f"Missing symbols: {missing_symbols}")

# ─── STEP 2: Compute tier classification ──────────────────────────────────────

print("\n=== STEP 2: Computing tier classification for all 2,082 trades ===")

tier_data = []
for _, trade in trades.iterrows():
    sym = trade['symbol']
    signal_day = trade['signal_date']
    entry_day = trade['entry_date']

    enriched_path = DATA_DIR / f'{sym}_enriched.csv'
    if not enriched_path.exists():
        # No enriched data — default Tier C
        tier_data.append({'symbol': sym, 'entry_date': entry_day, 'tier': 'C',
                         'body_pct': np.nan, 'atr_change_3d': np.nan})
        continue

    df = pd.read_csv(enriched_path, parse_dates=['date'])

    # Find signal day row
    sig_row = df[df['date'] == signal_day]
    if len(sig_row) == 0:
        tier_data.append({'symbol': sym, 'entry_date': entry_day, 'tier': 'C',
                         'body_pct': np.nan, 'atr_change_3d': np.nan})
        continue

    sig_idx = sig_row.index[0]
    row = sig_row.iloc[0]

    # body_pct from enriched data
    body_pct = row.get('body_pct', np.nan)
    if pd.isna(body_pct):
        # Calculate manually
        high = row['high']
        low = row['low']
        if high != low:
            body_pct = (row['close'] - row['open']) / (high - low)
        else:
            body_pct = 0

    # atr_change_3d: % change in ATR(14) over 3 days
    atr_now = row.get('atr_14', np.nan)
    if sig_idx >= 3 and not pd.isna(atr_now):
        atr_3d_ago = df.loc[sig_idx - 3, 'atr_14'] if 'atr_14' in df.columns else np.nan
        if not pd.isna(atr_3d_ago) and atr_3d_ago > 0:
            atr_change_3d = (atr_now - atr_3d_ago) / atr_3d_ago * 100
        else:
            atr_change_3d = np.nan
    else:
        atr_change_3d = np.nan

    # Tier classification
    cond_body = body_pct > 0.23 if not pd.isna(body_pct) else False
    cond_atr = atr_change_3d > 0 if not pd.isna(atr_change_3d) else False

    if cond_body and cond_atr:
        tier = 'A'
    elif cond_body or cond_atr:
        tier = 'B'
    else:
        tier = 'C'

    tier_data.append({
        'symbol': sym, 'entry_date': entry_day, 'tier': tier,
        'body_pct': body_pct, 'atr_change_3d': atr_change_3d
    })

tier_df = pd.DataFrame(tier_data)
tier_counts = tier_df['tier'].value_counts().sort_index()
print(f"Tier A: {tier_counts.get('A', 0)}")
print(f"Tier B: {tier_counts.get('B', 0)}")
print(f"Tier C: {tier_counts.get('C', 0)}")

# Merge tier into trades
trades = trades.merge(tier_df[['symbol', 'entry_date', 'tier', 'body_pct', 'atr_change_3d']],
                      on=['symbol', 'entry_date'], how='left')

# ─── STEP 3: Run backtests ────────────────────────────────────────────────────

print("\n=== STEP 3: Running backtests ===")

COST_BPS = 5  # 5 bps each way

def compute_option_pnl(entry_price, exit_price, tier, n_contracts=None):
    """Compute option trade P&L with tier-based sizing."""
    if tier == 'A':
        size = 200_000
    elif tier == 'B':
        size = 100_000
    else:
        return None, None, None, None  # Tier C doesn't trade options

    if n_contracts is None:
        n_contracts = int(size / (entry_price * 100))

    if n_contracts < 1:
        return None, None, None, None

    notional_entry = n_contracts * entry_price * 100
    notional_exit = n_contracts * exit_price * 100
    cost = (notional_entry + notional_exit) * COST_BPS / 10_000
    gross_pnl = notional_exit - notional_entry
    net_pnl = gross_pnl - cost
    ret_pct = (exit_price - entry_price) / entry_price * 100

    return net_pnl, n_contracts, size, ret_pct


# ─── BACKTEST 1: HYBRID ───────────────────────────────────────────────────────

print("\n--- Backtest 1: HYBRID ---")
hybrid_results = []

for _, trade in trades.iterrows():
    sym = trade['symbol']
    edate = trade['entry_date']
    tier = trade['tier']
    key = (sym, edate)

    result = {
        'symbol': sym,
        'entry_date': edate,
        'exit_date': trade['exit_date'],
        'signal_date': trade['signal_date'],
        'tier': tier,
        'body_pct': trade.get('body_pct', np.nan),
        'atr_change_3d': trade.get('atr_change_3d', np.nan),
        'stock_pnl': trade['net_pnl'],
        'stock_return': trade['return_pct'],
    }

    if tier == 'C':
        # Stock trade
        result['source'] = 'stock'
        result['pnl'] = trade['net_pnl']
        result['return_pct'] = trade['return_pct']
        result['contracts'] = 0
        result['position_size'] = trade['position_size']
        result['option_entry'] = np.nan
        result['option_exit'] = np.nan
        result['option_ticker'] = ''
    elif tier in ('A', 'B'):
        cov = coverage_combined.get(key, {})
        if cov.get('has_data', False):
            entry_p = cov['entry_price']
            exit_p = cov['exit_price']
            pnl, ctrs, size, ret = compute_option_pnl(entry_p, exit_p, tier)
            if pnl is not None:
                result['source'] = 'option'
                result['pnl'] = pnl
                result['return_pct'] = ret
                result['contracts'] = ctrs
                result['position_size'] = size
                result['option_entry'] = entry_p
                result['option_exit'] = exit_p
                result['option_ticker'] = cov.get('option_ticker', '')
            else:
                # Can't execute (< 1 contract)
                result['source'] = 'stock_fallback'
                result['pnl'] = trade['net_pnl']
                result['return_pct'] = trade['return_pct']
                result['contracts'] = 0
                result['position_size'] = trade['position_size']
                result['option_entry'] = entry_p
                result['option_exit'] = exit_p
                result['option_ticker'] = cov.get('option_ticker', '')
        else:
            # No option data — fall back to stock
            result['source'] = 'stock_fallback'
            result['pnl'] = trade['net_pnl']
            result['return_pct'] = trade['return_pct']
            result['contracts'] = 0
            result['position_size'] = trade['position_size']
            result['option_entry'] = np.nan
            result['option_exit'] = np.nan
            result['option_ticker'] = ''

    hybrid_results.append(result)

hybrid_df = pd.DataFrame(hybrid_results)

# ─── BACKTEST 2: OPTIONS ONLY ─────────────────────────────────────────────────

print("--- Backtest 2: OPTIONS ONLY ---")
optonly_results = []

for _, trade in trades.iterrows():
    sym = trade['symbol']
    edate = trade['entry_date']
    tier = trade['tier']
    key = (sym, edate)

    if tier == 'C':
        continue  # Skip Tier C entirely

    cov = coverage_combined.get(key, {})
    if not cov.get('has_data', False):
        continue  # Skip if no option data

    entry_p = cov['entry_price']
    exit_p = cov['exit_price']
    pnl, ctrs, size, ret = compute_option_pnl(entry_p, exit_p, tier)

    if pnl is None:
        continue  # Can't execute

    optonly_results.append({
        'symbol': sym,
        'entry_date': edate,
        'exit_date': trade['exit_date'],
        'signal_date': trade['signal_date'],
        'tier': tier,
        'body_pct': trade.get('body_pct', np.nan),
        'atr_change_3d': trade.get('atr_change_3d', np.nan),
        'source': 'option',
        'pnl': pnl,
        'return_pct': ret,
        'contracts': ctrs,
        'position_size': size,
        'option_entry': entry_p,
        'option_exit': exit_p,
        'option_ticker': cov.get('option_ticker', ''),
        'stock_pnl': trade['net_pnl'],
        'stock_return': trade['return_pct'],
    })

optonly_df = pd.DataFrame(optonly_results)

# ─── Helper functions for metrics ─────────────────────────────────────────────

def calc_metrics(df, pnl_col='pnl'):
    """Calculate backtest metrics from a DataFrame with a pnl column."""
    if len(df) == 0:
        return {}

    pnls = df[pnl_col].values
    total = pnls.sum()
    wins = (pnls > 0).sum()
    losses = (pnls <= 0).sum()
    wr = wins / len(pnls) * 100 if len(pnls) > 0 else 0
    avg = pnls.mean()

    win_total = pnls[pnls > 0].sum() if wins > 0 else 0
    loss_total = abs(pnls[pnls <= 0].sum()) if losses > 0 else 1
    pf = win_total / loss_total if loss_total > 0 else float('inf')

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    maxdd = drawdown.min()

    # Sharpe (annualized, assuming ~250 trading days, avg 1 trade/day is rough)
    if len(pnls) > 1 and pnls.std() > 0:
        # Use monthly returns approach
        df_copy = df.copy()
        df_copy['entry_date'] = pd.to_datetime(df_copy['entry_date'])
        df_copy['month'] = df_copy['entry_date'].dt.to_period('M')
        monthly = df_copy.groupby('month')[pnl_col].sum()
        if len(monthly) > 1 and monthly.std() > 0:
            sharpe = monthly.mean() / monthly.std() * np.sqrt(12)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Avg capital deployed
    if 'position_size' in df.columns:
        avg_capital = df['position_size'].mean()
    else:
        avg_capital = 0

    return {
        'trades': len(pnls),
        'total_pnl': total,
        'avg_pnl': avg,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': maxdd,
        'sharpe': sharpe,
        'avg_capital': avg_capital,
    }

# ─── Compute metrics ──────────────────────────────────────────────────────────

hybrid_metrics = calc_metrics(hybrid_df)
optonly_metrics = calc_metrics(optonly_df)

# Prior run metrics (from the hybrid_portfolio_report.txt)
prior_hybrid = {
    'trades': 2082, 'total_pnl': 60_713_181, 'avg_pnl': 29_161,
    'win_rate': 42.5, 'profit_factor': 1.56, 'max_drawdown': -7_379_292,
    'sharpe': 1.42, 'avg_capital': 11_632_821,
}

# Prior options-only: from options_2filter_backtest.txt (Version A — tiered)
prior_optonly = {
    'trades': 675, 'total_pnl': 36_110_144, 'avg_pnl': 53_497,
    'win_rate': 37.9, 'profit_factor': 1.92, 'max_drawdown': -2_691_548,
    'sharpe': 1.94, 'avg_capital': 0,
}

# Stock only baseline
stock_only = {
    'trades': 2082, 'total_pnl': 88_200_000, 'avg_pnl': 42_364,
    'win_rate': 55.3, 'profit_factor': 2.03, 'max_drawdown': -5_812_059,
    'sharpe': 0, 'avg_capital': 30_373_387,
}

# Compute stock-only Sharpe from the actual trade data
stock_metrics = calc_metrics(trades.rename(columns={'net_pnl': 'pnl'}).assign(pnl=trades['net_pnl']))
stock_only['sharpe'] = stock_metrics['sharpe']
stock_only['total_pnl'] = stock_metrics['total_pnl']
stock_only['avg_pnl'] = stock_metrics['avg_pnl']
stock_only['win_rate'] = stock_metrics['win_rate']
stock_only['profit_factor'] = stock_metrics['profit_factor']
stock_only['max_drawdown'] = stock_metrics['max_drawdown']

# ─── Year-by-year ─────────────────────────────────────────────────────────────

def year_by_year(df, pnl_col='pnl'):
    df = df.copy()
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['year'] = df['entry_date'].dt.year
    results = []
    for year in sorted(df['year'].unique()):
        ydf = df[df['year'] == year]
        m = calc_metrics(ydf, pnl_col)
        m['year'] = year
        results.append(m)
    return results

hybrid_yby = year_by_year(hybrid_df)
optonly_yby = year_by_year(optonly_df)

# ─── Top 10 winners/losers ────────────────────────────────────────────────────

def top_n(df, n=10, pnl_col='pnl'):
    top_w = df.nlargest(n, pnl_col)
    top_l = df.nsmallest(n, pnl_col)
    return top_w, top_l

hybrid_top_w, hybrid_top_l = top_n(hybrid_df)
optonly_top_w, optonly_top_l = top_n(optonly_df)

# ─── Coverage details ─────────────────────────────────────────────────────────

tier_ab = trades[trades['tier'].isin(['A', 'B'])]
tier_ab_with_opts = sum(1 for _, t in tier_ab.iterrows()
                        if coverage_combined.get((t['symbol'], t['entry_date']), {}).get('has_data', False))
tier_ab_with_opts_orig = sum(1 for _, t in tier_ab.iterrows()
                             if coverage_orig.get((t['symbol'], t['entry_date']), {}).get('has_data', False))

# Tier A+B that actually executed in options-only
tier_ab_executed = len(optonly_df)

# Missing Tier A+B trades
missing_ab = []
for _, t in tier_ab.iterrows():
    key = (t['symbol'], t['entry_date'])
    if not coverage_combined.get(key, {}).get('has_data', False):
        missing_ab.append(t)

# Count hybrid fallbacks
hybrid_stock_fb = hybrid_df[hybrid_df['source'] == 'stock_fallback']
hybrid_option = hybrid_df[hybrid_df['source'] == 'option']
hybrid_stock = hybrid_df[hybrid_df['source'] == 'stock']

# ─── GENERATE REPORT ──────────────────────────────────────────────────────────

print("\n=== Generating report ===")

report_lines = []
def w(line=''):
    report_lines.append(line)

w("=" * 100)
w("EXPANDED OPTIONS BACKTEST — Original + Backfill Data (2020-2026)")
w("=" * 100)
w()

# A) COVERAGE SUMMARY
w("A) COVERAGE SUMMARY")
w("-" * 80)
w(f"  Total trades:                          {len(trades):,}")
w(f"  Tier A:                                {tier_counts.get('A', 0):,}")
w(f"  Tier B:                                {tier_counts.get('B', 0):,}")
w(f"  Tier C:                                {tier_counts.get('C', 0):,}")
w(f"  Tier A+B total:                        {len(tier_ab):,}")
w()
w(f"  Before backfill (30d_1m only):")
w(f"    Trades with option data:             {orig_with_data:,}")
w(f"    Tier A+B with option data:           {tier_ab_with_opts_orig:,}")
w(f"    Coverage rate (A+B):                 {tier_ab_with_opts_orig/len(tier_ab)*100:.1f}%")
w()
w(f"  After backfill (original + backfill):")
w(f"    Trades with option data:             {combined_with_data:,}")
w(f"    Tier A+B with option data:           {tier_ab_with_opts:,}")
w(f"    Tier A+B executed (contracts >= 1):  {tier_ab_executed:,}")
w(f"    Coverage rate (A+B):                 {tier_ab_with_opts/len(tier_ab)*100:.1f}%")
w()
w(f"  Improvement from backfill:             +{combined_with_data - orig_with_data} trades")
w(f"  Still missing (all trades):            {len(missing_trades)} trades")
w(f"  Still missing (Tier A+B):              {len(missing_ab)} trades")
w()
w(f"  Hybrid composition:")
w(f"    Tier C (stock):                      {len(hybrid_stock):,} trades")
w(f"    Tier A+B (options):                  {len(hybrid_option):,} trades")
w(f"    Tier A+B (stock fallback):           {len(hybrid_stock_fb):,} trades")
w()

# B) COMPARISON TABLE
w()
w("B) COMPARISON TABLE — ALL VERSIONS")
w("-" * 120)
w(f"{'Version':<40} {'Trades':>7} {'Total PnL':>14} {'Avg PnL':>10} {'WR':>6} {'PF':>6} {'MaxDD':>14} {'Sharpe':>7} {'Avg Capital':>14}")
w("-" * 120)

versions = [
    ("Hybrid (expanded)", hybrid_metrics),
    ("Hybrid (prior run)", prior_hybrid),
    ("Options only w/ tier (expanded)", optonly_metrics),
    ("Options only w/ tier (prior run)", prior_optonly),
    ("Stock only baseline", stock_only),
]

for name, m in versions:
    w(f"{name:<40} {m['trades']:>7,} ${m['total_pnl']:>13,.0f} ${m['avg_pnl']:>9,.0f} {m['win_rate']:>5.1f}% {m['profit_factor']:>5.2f} ${m['max_drawdown']:>13,.0f} {m['sharpe']:>6.2f} ${m.get('avg_capital', 0):>13,.0f}")

w()
w("  Notes:")
w("  - Prior hybrid used body_quintile sizing; expanded uses tier A/B/C classification")
w("  - Prior options-only is from 2-filter tiered backtest (A=$200K, B=$100K, skip C)")
w("  - Stock only = Option D no-re-entry baseline with ATR stop + 20-day hold")
w()

# C) YEAR-BY-YEAR
w()
w("C) YEAR-BY-YEAR PERFORMANCE")
w("=" * 100)
w()
w("  HYBRID (expanded):")
w(f"  {'Year':>6} {'Trades':>7} {'Total PnL':>14} {'Avg PnL':>10} {'WR':>6} {'PF':>6} {'MaxDD':>14} {'Sharpe':>7}")
w(f"  {'-'*6} {'-'*7} {'-'*14} {'-'*10} {'-'*6} {'-'*6} {'-'*14} {'-'*7}")
for m in hybrid_yby:
    w(f"  {m['year']:>6} {m['trades']:>7,} ${m['total_pnl']:>13,.0f} ${m['avg_pnl']:>9,.0f} {m['win_rate']:>5.1f}% {m['profit_factor']:>5.2f} ${m['max_drawdown']:>13,.0f} {m['sharpe']:>6.2f}")

w()
w("  OPTIONS ONLY w/ tier (expanded):")
w(f"  {'Year':>6} {'Trades':>7} {'Total PnL':>14} {'Avg PnL':>10} {'WR':>6} {'PF':>6} {'MaxDD':>14} {'Sharpe':>7}")
w(f"  {'-'*6} {'-'*7} {'-'*14} {'-'*10} {'-'*6} {'-'*6} {'-'*14} {'-'*7}")
for m in optonly_yby:
    w(f"  {m['year']:>6} {m['trades']:>7,} ${m['total_pnl']:>13,.0f} ${m['avg_pnl']:>9,.0f} {m['win_rate']:>5.1f}% {m['profit_factor']:>5.2f} ${m['max_drawdown']:>13,.0f} {m['sharpe']:>6.2f}")

w()

# D) TOP 10 WINNERS / LOSERS
w()
w("D) TOP 10 WINNERS / LOSERS")
w("=" * 100)
w()

def format_top(label, df_top):
    w(f"  {label}:")
    w(f"  {'#':>3} {'Symbol':<8} {'Entry':>12} {'Tier':>5} {'Source':>10} {'PnL':>14} {'Return%':>10}")
    w(f"  {'-'*3} {'-'*8} {'-'*12} {'-'*5} {'-'*10} {'-'*14} {'-'*10}")
    for i, (_, r) in enumerate(df_top.iterrows(), 1):
        edate_str = pd.to_datetime(r['entry_date']).strftime('%Y-%m-%d')
        w(f"  {i:>3} {r['symbol']:<8} {edate_str:>12} {r['tier']:>5} {r['source']:>10} ${r['pnl']:>13,.0f} {r['return_pct']:>9.1f}%")
    w()

format_top("HYBRID — TOP 10 WINNERS", hybrid_top_w)
format_top("HYBRID — TOP 10 LOSERS", hybrid_top_l)
format_top("OPTIONS ONLY — TOP 10 WINNERS", optonly_top_w)
format_top("OPTIONS ONLY — TOP 10 LOSERS", optonly_top_l)

# E) STILL MISSING TRADES
w()
w("E) STILL-MISSING TRADES (Tier A+B without option data)")
w("=" * 100)
w(f"  These {len(missing_ab)} Tier A+B trades have no option data.")
w(f"  In Hybrid backtest they fall back to stock P&L; in Options-only they are skipped.")
w()
w(f"  {'Symbol':<8} {'Entry Date':>12} {'Tier':>5} {'Stock PnL':>14} {'Stock Ret%':>10}")
w(f"  {'-'*8} {'-'*12} {'-'*5} {'-'*14} {'-'*10}")

missing_ab_df = pd.DataFrame(missing_ab)
if len(missing_ab_df) > 0:
    missing_ab_df = missing_ab_df.sort_values('entry_date')
    total_missing_pnl = 0
    for _, r in missing_ab_df.iterrows():
        edate_str = pd.to_datetime(r['entry_date']).strftime('%Y-%m-%d')
        w(f"  {r['symbol']:<8} {edate_str:>12} {r['tier']:>5} ${r['net_pnl']:>13,.0f} {r['return_pct']:>9.1f}%")
        total_missing_pnl += r['net_pnl']
    w()
    w(f"  Total stock P&L left on table (options-only): ${total_missing_pnl:>13,.0f}")

w()
w(f"  Missing symbols ({len(missing_symbols)}): {', '.join(missing_symbols)}")

# Also list ALL missing trades (including Tier C that are just stock)
w()
w(f"  All missing trades (any tier): {len(missing_trades)} trades across {len(missing_symbols)} symbols")

# Save report
report_path = RESULTS_DIR / 'expanded_options_backtest.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"Report saved to {report_path}")

# ─── Save trade detail CSV ────────────────────────────────────────────────────

# Combine hybrid and options-only into one CSV with a backtest column
hybrid_df['backtest'] = 'hybrid'
optonly_df['backtest'] = 'options_only'
all_trades_out = pd.concat([hybrid_df, optonly_df], ignore_index=True)
trades_path = RESULTS_DIR / 'expanded_options_trades.csv'
all_trades_out.to_csv(trades_path, index=False)
print(f"Trade details saved to {trades_path}")

# Print summary to console
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Hybrid (expanded):  {hybrid_metrics['trades']:,} trades, ${hybrid_metrics['total_pnl']:,.0f} total PnL, "
      f"WR={hybrid_metrics['win_rate']:.1f}%, PF={hybrid_metrics['profit_factor']:.2f}, "
      f"Sharpe={hybrid_metrics['sharpe']:.2f}")
print(f"Options only (exp): {optonly_metrics['trades']:,} trades, ${optonly_metrics['total_pnl']:,.0f} total PnL, "
      f"WR={optonly_metrics['win_rate']:.1f}%, PF={optonly_metrics['profit_factor']:.2f}, "
      f"Sharpe={optonly_metrics['sharpe']:.2f}")
print(f"Stock baseline:     {stock_only['trades']:,} trades, ${stock_only['total_pnl']:,.0f} total PnL")
