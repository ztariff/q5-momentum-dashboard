#!/usr/bin/env python3
"""
ATR-based Risk Sizing Backtest for the HYBRID strategy.
Instead of fixed notional sizing, every trade risks the same dollar amount per 1 ATR move.
"""

import pandas as pd
import numpy as np
import math
import os
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_RISK = 50_000          # $50K risk per 1 ATR
COST_BPS = 5                # 5 bps each way
DATA_DIR = "/home/ubuntu/daily_data/data"
RESULTS_DIR = "/home/ubuntu/daily_data/analysis_results"
OPTION_TYPE = "30d_1m"      # ATM 30-day monthly calls
OPTION_HOLD_DAYS = 13       # Hold options 13 calendar/trading days
STOCK_HOLD_DAYS = 20        # Hold stocks 20 trading days

# ── LOAD DATA ───────────────────────────────────────────────────────────────
print("Loading trade log...")
trades = pd.read_csv(f"{RESULTS_DIR}/backtest_optionD_noReentry_trades.csv")
trades['signal_date'] = pd.to_datetime(trades['signal_date'])
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])

print("Loading enriched trade data for body_pct...")
enriched = pd.read_csv(f"{RESULTS_DIR}/trade1_20d_enriched.csv")
enriched['signal_date'] = pd.to_datetime(enriched['signal_date'])
enriched['entry_date'] = pd.to_datetime(enriched['entry_date'])

# Build lookup: (symbol, signal_date) -> body_pct
enriched_lookup = {}
for _, row in enriched.iterrows():
    key = (row['symbol'], row['signal_date'])
    enriched_lookup[key] = {
        'body_pct': row['entry_body_pct'],
        'entry_atr_14': row['entry_atr_14'],
    }

print("Loading options data (merged)...")
opts1 = pd.read_csv(f"{DATA_DIR}/trade_options/hold_period_options.csv")
opts2 = pd.read_csv(f"{DATA_DIR}/trade_options/hold_period_options_backfill.csv")
opts3 = pd.read_csv(f"{DATA_DIR}/trade_options/hold_period_options_monthly_replace.csv")

# Standardize columns – backfill has extra cols
common_cols = opts1.columns.tolist()
for df in [opts2, opts3]:
    for c in df.columns:
        if c not in common_cols and c not in ['actual_delta', 'search_attempt']:
            pass
opts2_trimmed = opts2[[c for c in common_cols if c in opts2.columns]]
opts3_trimmed = opts3[[c for c in common_cols if c in opts3.columns]]

options_all = pd.concat([opts1, opts2_trimmed, opts3_trimmed], ignore_index=True)
options_all['trade_entry_date'] = pd.to_datetime(options_all['trade_entry_date'])
options_all['bar_date'] = pd.to_datetime(options_all['bar_date'])

# Filter to 30d_1m only
options_30d = options_all[options_all['option_type'] == OPTION_TYPE].copy()
print(f"  Options rows (30d_1m): {len(options_30d)}")

# Build options lookup: (symbol, entry_date) -> list of (bar_date, open, high, low, close, morning_price)
# sorted by bar_date
options_by_trade = {}
for _, row in options_30d.iterrows():
    key = (row['trade_symbol'], row['trade_entry_date'])
    if key not in options_by_trade:
        options_by_trade[key] = []
    options_by_trade[key].append({
        'bar_date': row['bar_date'],
        'day_open': row['day_open'],
        'day_high': row['day_high'],
        'day_low': row['day_low'],
        'day_close': row['day_close'],
        'morning_price': row.get('morning_price', np.nan),
    })

for key in options_by_trade:
    options_by_trade[key].sort(key=lambda x: x['bar_date'])

# ── LOAD DAILY DATA PER SYMBOL ─────────────────────────────────────────────
print("Loading daily enriched data per symbol...")
symbols = trades['symbol'].unique()
daily_data = {}
for sym in symbols:
    fpath = f"{DATA_DIR}/{sym}_enriched.csv"
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        daily_data[sym] = df
    else:
        print(f"  WARNING: no daily data for {sym}")

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def get_signal_day_data(sym, signal_date):
    """Get ATR(14), body_pct, atr_change_3d from daily data on signal day."""
    if sym not in daily_data:
        return None
    df = daily_data[sym]
    idx = df.index[df['date'] == signal_date]
    if len(idx) == 0:
        return None
    i = idx[0]
    atr_14 = df.loc[i, 'atr_14']
    body_pct = df.loc[i, 'body_pct']
    low = df.loc[i, 'low']

    # atr_change_3d: ATR today vs ATR 3 trading days ago
    if i >= 3:
        atr_3d_ago = df.loc[i - 3, 'atr_14']
        atr_change_3d = atr_14 - atr_3d_ago
    else:
        atr_change_3d = 0.0

    return {
        'atr_14': atr_14,
        'body_pct': body_pct,
        'low': low,
        'atr_change_3d': atr_change_3d,
    }

def get_daily_prices(sym, entry_date, num_days=25):
    """Get daily OHLC for hold period starting from entry_date."""
    if sym not in daily_data:
        return []
    df = daily_data[sym]
    start_idx = df.index[df['date'] >= entry_date]
    if len(start_idx) == 0:
        return []
    i = start_idx[0]
    end_i = min(i + num_days, len(df))
    rows = []
    for j in range(i, end_i):
        rows.append({
            'date': df.loc[j, 'date'],
            'open': df.loc[j, 'open'],
            'high': df.loc[j, 'high'],
            'low': df.loc[j, 'low'],
            'close': df.loc[j, 'close'],
        })
    return rows

def compute_tier(body_pct, atr_change_3d):
    strong_candle = body_pct > 0.23
    atr_rising = atr_change_3d > 0
    if strong_candle and atr_rising:
        return 'A'
    elif strong_candle or atr_rising:
        return 'B'
    else:
        return 'C'

def body_quintile_from_pct(body_pct):
    """Map body_pct to quintile like the original backtest."""
    # Use the quintile from the original trade log if available;
    # otherwise estimate. We'll use the original trade log quintiles.
    pass

def get_body_multiplier(body_quintile):
    """Risk multiplier by body quintile for Tier C stocks."""
    mapping = {
        'Q1': None,   # skip
        'Q2': 0.5,
        'Q3': 1.0,
        'Q4': 1.5,
        'Q5': 2.0,
    }
    return mapping.get(body_quintile)

# ── SIMULATE ────────────────────────────────────────────────────────────────
print("\nSimulating 2,082 trades with ATR risk sizing...")

results = []
skipped = 0
missing_data = 0

for idx, trade in trades.iterrows():
    sym = trade['symbol']
    signal_date = trade['signal_date']
    entry_date = trade['entry_date']
    body_quintile = trade['body_quintile']
    original_entry_price = trade['entry_price']

    # Get signal day data
    sig_data = get_signal_day_data(sym, signal_date)
    if sig_data is None:
        # Try enriched lookup
        ekey = (sym, signal_date)
        if ekey in enriched_lookup:
            # Use entry_atr_14 from enriched (which is entry day, close enough)
            atr_14 = enriched_lookup[ekey]['entry_atr_14']
            body_pct = enriched_lookup[ekey]['body_pct']
            sig_data = {
                'atr_14': atr_14,
                'body_pct': body_pct,
                'atr_change_3d': 0.0,  # can't compute, default neutral
                'low': None,
            }
        else:
            missing_data += 1
            continue

    atr_14 = sig_data['atr_14']
    body_pct_val = sig_data['body_pct']
    atr_change_3d = sig_data['atr_change_3d']
    signal_low = sig_data['low']

    if pd.isna(atr_14) or atr_14 <= 0:
        missing_data += 1
        continue

    # Compute tier
    tier = compute_tier(body_pct_val, atr_change_3d)

    # Determine trade type and sizing
    trade_type = None
    risk_budget = None
    tier_mult = None

    if tier == 'C':
        # Stock trade with body quintile multiplier on risk
        mult = get_body_multiplier(body_quintile)
        if mult is None:
            skipped += 1
            continue  # Q1 skip
        risk_budget = BASE_RISK * mult
        trade_type = 'stock_tierC'
        tier_mult = mult

    elif tier in ('A', 'B'):
        # Check if options data exists
        opt_key = (sym, entry_date)
        if opt_key in options_by_trade and len(options_by_trade[opt_key]) > 0:
            trade_type = 'option_tier' + tier
            if tier == 'A':
                risk_budget = BASE_RISK * 2.0
                tier_mult = 2.0
            else:
                risk_budget = BASE_RISK * 1.0
                tier_mult = 1.0
        else:
            # Fallback to stock with ATR risk sizing
            if tier == 'A':
                risk_budget = BASE_RISK * 2.0
                tier_mult = 2.0
            else:
                risk_budget = BASE_RISK * 1.0
                tier_mult = 1.0
            trade_type = 'stock_fallback_tier' + tier

    # ── EXECUTE TRADE ──
    result = {
        'symbol': sym,
        'signal_date': signal_date,
        'entry_date': entry_date,
        'body_quintile': body_quintile,
        'body_pct': body_pct_val,
        'atr_14': atr_14,
        'atr_change_3d': atr_change_3d,
        'tier': tier,
        'trade_type': trade_type,
        'tier_mult': tier_mult,
        'risk_budget': risk_budget,
    }

    if trade_type.startswith('option'):
        # ── OPTIONS TRADE ──
        opt_bars = options_by_trade[(sym, entry_date)]

        # Entry: first bar_date = entry_date, use morning_price or day_open
        entry_bar = None
        for b in opt_bars:
            if b['bar_date'] >= entry_date:
                entry_bar = b
                break

        if entry_bar is None or pd.isna(entry_bar['day_open']) or entry_bar['day_open'] <= 0:
            # Fallback to stock
            trade_type = 'stock_fallback_tier' + tier
            result['trade_type'] = trade_type
            # Fall through to stock logic below
        else:
            opt_entry_price = entry_bar['morning_price'] if not pd.isna(entry_bar.get('morning_price', np.nan)) and entry_bar['morning_price'] > 0 else entry_bar['day_open']

            contracts = math.floor(risk_budget / (opt_entry_price * 100))
            if contracts <= 0:
                # Too expensive, fallback
                trade_type = 'stock_fallback_tier' + tier
                result['trade_type'] = trade_type
            else:
                # Hold 13 trading days from entry
                # Find exit bar: 13th trading day bar
                entry_idx_in_bars = None
                for bi, b in enumerate(opt_bars):
                    if b['bar_date'] >= entry_date:
                        entry_idx_in_bars = bi
                        break

                exit_idx = entry_idx_in_bars + OPTION_HOLD_DAYS
                if exit_idx < len(opt_bars):
                    exit_bar = opt_bars[exit_idx]
                    opt_exit_price = exit_bar['day_close']
                else:
                    # Use last available bar
                    exit_bar = opt_bars[-1]
                    opt_exit_price = exit_bar['day_close']

                if pd.isna(opt_exit_price) or opt_exit_price < 0:
                    opt_exit_price = 0.0

                position_cost = contracts * opt_entry_price * 100
                exit_value = contracts * opt_exit_price * 100

                cost_entry = position_cost * COST_BPS / 10000
                cost_exit = exit_value * COST_BPS / 10000

                gross_pnl = exit_value - position_cost
                net_pnl = gross_pnl - cost_entry - cost_exit

                days_held = (exit_bar['bar_date'] - entry_bar['bar_date']).days

                result.update({
                    'entry_price': opt_entry_price,
                    'exit_price': opt_exit_price,
                    'exit_date': exit_bar['bar_date'],
                    'shares_or_contracts': contracts,
                    'position_size': position_cost,
                    'stop_price': np.nan,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'return_pct': (net_pnl / position_cost * 100) if position_cost > 0 else 0,
                    'days_held': days_held,
                    'exit_type': 'TIME_OPT',
                })
                results.append(result)
                continue

    # ── STOCK TRADE (Tier C, fallback A/B, or option fallback) ──
    if trade_type.startswith('stock') or trade_type.startswith('option'):
        # Recategorize if we fell through from options
        if trade_type.startswith('option'):
            trade_type = 'stock_fallback_tier' + tier
            result['trade_type'] = trade_type

        shares = math.floor(risk_budget / atr_14)
        if shares <= 0:
            missing_data += 1
            continue

        # Get daily prices for hold period
        daily_prices = get_daily_prices(sym, entry_date, num_days=STOCK_HOLD_DAYS + 5)
        if len(daily_prices) == 0:
            missing_data += 1
            continue

        entry_price = daily_prices[0]['open']
        if pd.isna(entry_price) or entry_price <= 0:
            entry_price = original_entry_price

        position_size = shares * entry_price

        # Stop loss: signal_day_low - 3 * ATR(14)
        if signal_low is not None and not pd.isna(signal_low):
            stop_price = signal_low - 3 * atr_14
        else:
            # Use original trade stop price as fallback
            stop_price = trade['stop_price'] if not pd.isna(trade['stop_price']) else entry_price - 3 * atr_14

        # Simulate: check daily low vs stop, exit at 20d close or stop
        exit_price = None
        exit_date = None
        exit_type = None
        days_held = 0

        for di, dp in enumerate(daily_prices):
            if di == 0:
                continue  # entry day, don't check stop on entry day open

            days_held = di

            if dp['low'] <= stop_price:
                exit_price = stop_price
                exit_date = dp['date']
                exit_type = 'STOP'
                break

            if di >= STOCK_HOLD_DAYS:
                exit_price = dp['close']
                exit_date = dp['date']
                exit_type = 'TIME'
                break

        if exit_price is None:
            # Ran out of data, use last available
            if len(daily_prices) > 1:
                last = daily_prices[-1]
                exit_price = last['close']
                exit_date = last['date']
                exit_type = 'TIME_TRUNC'
                days_held = len(daily_prices) - 1
            else:
                missing_data += 1
                continue

        gross_pnl = shares * (exit_price - entry_price)
        cost_entry = shares * entry_price * COST_BPS / 10000
        cost_exit = shares * abs(exit_price) * COST_BPS / 10000
        net_pnl = gross_pnl - cost_entry - cost_exit

        result.update({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_date': exit_date,
            'shares_or_contracts': shares,
            'position_size': position_size,
            'stop_price': stop_price,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return_pct': (net_pnl / position_size * 100) if position_size > 0 else 0,
            'days_held': days_held,
            'exit_type': exit_type,
        })
        results.append(result)

print(f"\nCompleted: {len(results)} trades simulated")
print(f"Skipped (Q1): {skipped}")
print(f"Missing data: {missing_data}")

# ── BUILD RESULTS DATAFRAME ────────────────────────────────────────────────
df = pd.DataFrame(results)
df['exit_date'] = pd.to_datetime(df['exit_date'])
df['signal_date'] = pd.to_datetime(df['signal_date'])
df['entry_date'] = pd.to_datetime(df['entry_date'])
df['year'] = df['entry_date'].dt.year

# Save trade details
df.to_csv(f"{RESULTS_DIR}/atr_risk_sizing_trades.csv", index=False)
print(f"Trade details saved to {RESULTS_DIR}/atr_risk_sizing_trades.csv")

# ── ANALYSIS ────────────────────────────────────────────────────────────────
def calc_stats(subset, label=""):
    """Calculate comprehensive stats for a subset of trades."""
    n = len(subset)
    if n == 0:
        return {'label': label, 'trades': 0}

    total_pnl = subset['net_pnl'].sum()
    winners = subset[subset['net_pnl'] > 0]
    losers = subset[subset['net_pnl'] <= 0]
    wr = len(winners) / n * 100

    gross_win = winners['net_pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 1
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
    avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0

    # Max drawdown (cumulative)
    cum = subset['net_pnl'].cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = dd.min()

    # Sharpe (annualized, assuming ~15 trading days avg hold)
    avg_daily_ret = subset['return_pct'].mean() / subset['days_held'].mean() if subset['days_held'].mean() > 0 else 0
    std_daily_ret = (subset['return_pct'] / subset['days_held'].clip(lower=1)).std()
    sharpe = (avg_daily_ret / std_daily_ret * np.sqrt(252)) if std_daily_ret > 0 else 0

    return {
        'label': label,
        'trades': n,
        'total_pnl': total_pnl,
        'win_rate': wr,
        'profit_factor': pf,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'avg_pnl': total_pnl / n,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_position_size': subset['position_size'].mean(),
        'median_position_size': subset['position_size'].median(),
        'max_position_size': subset['position_size'].max(),
        'min_position_size': subset['position_size'].min(),
        'avg_risk_budget': subset['risk_budget'].mean(),
    }

# ── GENERATE REPORT ────────────────────────────────────────────────────────
report_lines = []
def rprint(s=""):
    report_lines.append(s)

rprint("=" * 90)
rprint("ATR-BASED RISK SIZING BACKTEST — HYBRID STRATEGY")
rprint("=" * 90)
rprint(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
rprint(f"Base risk per trade: ${BASE_RISK:,.0f} per 1 ATR")
rprint(f"Trades simulated: {len(df)}")
rprint(f"Trades skipped (Q1): {skipped}")
rprint(f"Missing data: {missing_data}")
rprint()

# ── A) OVERALL STATS ──
overall = calc_stats(df, "ATR Risk Sizing")
rprint("=" * 90)
rprint("A) OVERALL STATS")
rprint("=" * 90)
rprint(f"  Total Trades:         {overall['trades']:,}")
rprint(f"  Total Net P&L:        ${overall['total_pnl']:,.2f}")
rprint(f"  Win Rate:             {overall['win_rate']:.1f}%")
rprint(f"  Profit Factor:        {overall['profit_factor']:.2f}")
rprint(f"  Max Drawdown:         ${overall['max_dd']:,.2f}")
rprint(f"  Sharpe Ratio:         {overall['sharpe']:.2f}")
rprint(f"  Avg P&L/Trade:        ${overall['avg_pnl']:,.2f}")
rprint(f"  Avg Win:              ${overall['avg_win']:,.2f}")
rprint(f"  Avg Loss:             ${overall['avg_loss']:,.2f}")
rprint()
rprint(f"  Avg Position Size:    ${overall['avg_position_size']:,.2f}")
rprint(f"  Median Position Size: ${overall['median_position_size']:,.2f}")
rprint(f"  Max Position Size:    ${overall['max_position_size']:,.2f}")
rprint(f"  Min Position Size:    ${overall['min_position_size']:,.2f}")
rprint(f"  Avg Risk Budget:      ${overall['avg_risk_budget']:,.2f}")

# Position size analysis
stock_trades = df[df['trade_type'].str.startswith('stock')]
if len(stock_trades) > 0:
    rprint()
    rprint("  Stock Trades Position Size Stats:")
    rprint(f"    Mean:   ${stock_trades['position_size'].mean():,.2f}")
    rprint(f"    Median: ${stock_trades['position_size'].median():,.2f}")
    rprint(f"    Std:    ${stock_trades['position_size'].std():,.2f}")
    rprint(f"    Max:    ${stock_trades['position_size'].max():,.2f}")
    rprint(f"    Min:    ${stock_trades['position_size'].min():,.2f}")

opt_trades = df[df['trade_type'].str.startswith('option')]
if len(opt_trades) > 0:
    rprint()
    rprint("  Options Trades Position Size Stats:")
    rprint(f"    Mean:   ${opt_trades['position_size'].mean():,.2f}")
    rprint(f"    Median: ${opt_trades['position_size'].median():,.2f}")
    rprint(f"    Max:    ${opt_trades['position_size'].max():,.2f}")
    rprint(f"    Min:    ${opt_trades['position_size'].min():,.2f}")

# ── B) COMPARISON TABLE ──
rprint()
rprint("=" * 90)
rprint("B) COMPARISON TABLE")
rprint("=" * 90)
rprint()
rprint(f"{'Metric':<30} {'ATR Risk Sizing':>20} {'Notional Hybrid':>20} {'Stock-Only Notional':>20}")
rprint("-" * 90)

# Known results from prior backtests
notional_hybrid = {
    'total_pnl': 80_700_000,
    'label': 'Notional Hybrid',
}
stock_only = {
    'total_pnl': 88_200_000,
    'label': 'Stock-Only Notional',
}

rprint(f"{'Total Net P&L':<30} ${overall['total_pnl']:>19,.0f} ${notional_hybrid['total_pnl']:>19,.0f} ${stock_only['total_pnl']:>19,.0f}")
rprint(f"{'Win Rate':<30} {overall['win_rate']:>19.1f}% {'—':>20} {'—':>20}")
rprint(f"{'Profit Factor':<30} {overall['profit_factor']:>20.2f} {'—':>20} {'—':>20}")
rprint(f"{'Max Drawdown':<30} ${overall['max_dd']:>19,.0f} {'—':>20} {'—':>20}")
rprint(f"{'Sharpe Ratio':<30} {overall['sharpe']:>20.2f} {'—':>20} {'—':>20}")
rprint(f"{'Avg Position Size':<30} ${overall['avg_position_size']:>19,.0f} {'—':>20} {'—':>20}")
rprint(f"{'Avg Risk Budget':<30} ${overall['avg_risk_budget']:>19,.0f} {'—':>20} {'—':>20}")

# ── C) YEAR-BY-YEAR ──
rprint()
rprint("=" * 90)
rprint("C) YEAR-BY-YEAR BREAKDOWN")
rprint("=" * 90)
rprint()
rprint(f"{'Year':<8} {'Trades':>8} {'Net P&L':>16} {'WR':>8} {'PF':>8} {'Avg P&L':>14} {'Avg PosSize':>14} {'MaxDD':>16}")
rprint("-" * 90)

for year in sorted(df['year'].unique()):
    ys = df[df['year'] == year]
    ystats = calc_stats(ys, str(year))
    rprint(f"{year:<8} {ystats['trades']:>8} ${ystats['total_pnl']:>14,.0f} {ystats['win_rate']:>7.1f}% {ystats['profit_factor']:>7.2f} ${ystats['avg_pnl']:>12,.0f} ${ystats['avg_position_size']:>12,.0f} ${ystats['max_dd']:>14,.0f}")

# ── D) POSITION SIZE DISTRIBUTION ──
rprint()
rprint("=" * 90)
rprint("D) POSITION SIZE DISTRIBUTION")
rprint("=" * 90)
rprint()

# Histogram buckets
bins = [0, 100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000, float('inf')]
labels_bins = ['<100K', '100K-250K', '250K-500K', '500K-750K', '750K-1M', '1M-1.5M', '1.5M-2M', '2M-3M', '3M-5M', '5M-10M', '>10M']
df['pos_bucket'] = pd.cut(df['position_size'], bins=bins, labels=labels_bins, right=False)

rprint(f"{'Bucket':<15} {'Count':>8} {'Pct':>8} {'Avg P&L':>14}")
rprint("-" * 50)
for label in labels_bins:
    bucket = df[df['pos_bucket'] == label]
    if len(bucket) > 0:
        rprint(f"{label:<15} {len(bucket):>8} {len(bucket)/len(df)*100:>7.1f}% ${bucket['net_pnl'].mean():>12,.0f}")

rprint()
rprint("Top 10 LARGEST Positions:")
rprint(f"{'Symbol':<8} {'Entry Date':<12} {'PosSize':>14} {'ATR':>10} {'Price':>10} {'Shares':>10} {'Tier':>6} {'P&L':>14}")
rprint("-" * 90)
top10_large = df.nlargest(10, 'position_size')
for _, r in top10_large.iterrows():
    rprint(f"{r['symbol']:<8} {str(r['entry_date'])[:10]:<12} ${r['position_size']:>12,.0f} {r['atr_14']:>9.2f} ${r['entry_price']:>8.2f} {r['shares_or_contracts']:>10,} {r['tier']:>6} ${r['net_pnl']:>12,.0f}")

rprint()
rprint("Top 10 SMALLEST Positions:")
rprint(f"{'Symbol':<8} {'Entry Date':<12} {'PosSize':>14} {'ATR':>10} {'Price':>10} {'Shares/Cts':>10} {'Tier':>6} {'P&L':>14}")
rprint("-" * 90)
top10_small = df.nsmallest(10, 'position_size')
for _, r in top10_small.iterrows():
    rprint(f"{r['symbol']:<8} {str(r['entry_date'])[:10]:<12} ${r['position_size']:>12,.0f} {r['atr_14']:>9.2f} ${r['entry_price']:>8.2f} {r['shares_or_contracts']:>10,} {r['tier']:>6} ${r['net_pnl']:>12,.0f}")

# ── E) RISK NORMALIZATION CHECK ──
rprint()
rprint("=" * 90)
rprint("E) RISK NORMALIZATION CHECK")
rprint("=" * 90)
rprint()

stopped = df[df['exit_type'] == 'STOP'].copy()
if len(stopped) > 0:
    stopped['dollar_loss'] = stopped['net_pnl']
    stopped['expected_max_loss'] = -3 * stopped['risk_budget']
    stopped['loss_vs_expected'] = stopped['dollar_loss'] / stopped['expected_max_loss']  # should be near 1.0 if normalized

    rprint(f"Stopped Trades: {len(stopped)}")
    rprint(f"  Avg Dollar Loss:        ${stopped['dollar_loss'].mean():,.2f}")
    rprint(f"  Median Dollar Loss:     ${stopped['dollar_loss'].median():,.2f}")
    rprint(f"  Std Dollar Loss:        ${stopped['dollar_loss'].std():,.2f}")
    rprint(f"  Avg Expected Max Loss:  ${stopped['expected_max_loss'].mean():,.2f}")
    rprint()
    rprint(f"  Loss / Expected Max Loss (should cluster near 1.0 if stop hit exactly):")
    rprint(f"    Mean ratio:  {stopped['loss_vs_expected'].mean():.3f}")
    rprint(f"    Median ratio:{stopped['loss_vs_expected'].median():.3f}")
    rprint(f"    Std ratio:   {stopped['loss_vs_expected'].std():.3f}")

    rprint()
    rprint("  Distribution of actual losses on stopped trades:")
    loss_bins = [0, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 500_000, float('inf')]
    loss_labels = ['<25K', '25-50K', '50-75K', '75-100K', '100-150K', '150-200K', '200-300K', '300-500K', '>500K']
    stopped['loss_bucket'] = pd.cut(stopped['dollar_loss'].abs(), bins=loss_bins, labels=loss_labels, right=False)
    rprint(f"  {'Loss Range':<15} {'Count':>8} {'Pct':>8}")
    rprint("  " + "-" * 35)
    for lb in loss_labels:
        cnt = len(stopped[stopped['loss_bucket'] == lb])
        if cnt > 0:
            rprint(f"  {lb:<15} {cnt:>8} {cnt/len(stopped)*100:>7.1f}%")

    rprint()
    rprint("  Comparison: Risk normalization vs notional sizing")
    rprint("    Notional sizing: losses vary by stock price (fixed $ notional, variable risk)")
    rprint("    ATR sizing: losses should cluster by tier (risk budget controls exposure)")

    # Show by tier
    for t in ['A', 'B', 'C']:
        ts = stopped[stopped['tier'] == t]
        if len(ts) > 0:
            rprint(f"    Tier {t} stopped: n={len(ts)}, avg_loss=${ts['dollar_loss'].mean():,.0f}, std=${ts['dollar_loss'].std():,.0f}")

# ── F) TOP 10 WINNERS / LOSERS ──
rprint()
rprint("=" * 90)
rprint("F) TOP 10 WINNERS AND LOSERS")
rprint("=" * 90)
rprint()

rprint("TOP 10 WINNERS:")
rprint(f"{'Symbol':<8} {'Entry Date':<12} {'Type':<20} {'PosSize':>12} {'Net P&L':>14} {'Return':>8} {'Exit':>6}")
rprint("-" * 85)
for _, r in df.nlargest(10, 'net_pnl').iterrows():
    rprint(f"{r['symbol']:<8} {str(r['entry_date'])[:10]:<12} {r['trade_type']:<20} ${r['position_size']:>10,.0f} ${r['net_pnl']:>12,.0f} {r['return_pct']:>7.1f}% {r['exit_type']:>6}")

rprint()
rprint("TOP 10 LOSERS:")
rprint(f"{'Symbol':<8} {'Entry Date':<12} {'Type':<20} {'PosSize':>12} {'Net P&L':>14} {'Return':>8} {'Exit':>6}")
rprint("-" * 85)
for _, r in df.nsmallest(10, 'net_pnl').iterrows():
    rprint(f"{r['symbol']:<8} {str(r['entry_date'])[:10]:<12} {r['trade_type']:<20} ${r['position_size']:>10,.0f} ${r['net_pnl']:>12,.0f} {r['return_pct']:>7.1f}% {r['exit_type']:>6}")

# ── G) BY TIER ──
rprint()
rprint("=" * 90)
rprint("G) BREAKDOWN BY TIER AND TRADE TYPE")
rprint("=" * 90)
rprint()

rprint(f"{'Category':<25} {'Trades':>8} {'Net P&L':>16} {'WR':>8} {'PF':>8} {'Avg P&L':>14} {'Avg PosSize':>14}")
rprint("-" * 95)

for tt in ['option_tierA', 'option_tierB', 'stock_tierC', 'stock_fallback_tierA', 'stock_fallback_tierB']:
    subset = df[df['trade_type'] == tt]
    if len(subset) > 0:
        s = calc_stats(subset, tt)
        rprint(f"{tt:<25} {s['trades']:>8} ${s['total_pnl']:>14,.0f} {s['win_rate']:>7.1f}% {s['profit_factor']:>7.2f} ${s['avg_pnl']:>12,.0f} ${s['avg_position_size']:>12,.0f}")

# By tier overall
rprint()
rprint("By Tier (aggregated):")
rprint(f"{'Tier':<10} {'Trades':>8} {'Net P&L':>16} {'WR':>8} {'PF':>8} {'Avg P&L':>14}")
rprint("-" * 70)
for t in ['A', 'B', 'C']:
    subset = df[df['tier'] == t]
    if len(subset) > 0:
        s = calc_stats(subset, f"Tier {t}")
        rprint(f"{'Tier '+t:<10} {s['trades']:>8} ${s['total_pnl']:>14,.0f} {s['win_rate']:>7.1f}% {s['profit_factor']:>7.2f} ${s['avg_pnl']:>12,.0f}")

# Tier C by body quintile
rprint()
rprint("Tier C by Body Quintile:")
rprint(f"{'Quintile':<10} {'Trades':>8} {'Net P&L':>16} {'WR':>8} {'PF':>8} {'Avg P&L':>14} {'Avg Risk':>12}")
rprint("-" * 80)
for q in ['Q2', 'Q3', 'Q4', 'Q5']:
    subset = df[(df['trade_type'] == 'stock_tierC') & (df['body_quintile'] == q)]
    if len(subset) > 0:
        s = calc_stats(subset, q)
        rprint(f"{q:<10} {s['trades']:>8} ${s['total_pnl']:>14,.0f} {s['win_rate']:>7.1f}% {s['profit_factor']:>7.2f} ${s['avg_pnl']:>12,.0f} ${s['avg_risk_budget']:>10,.0f}")

# ── SAVE REPORT ──
report_text = "\n".join(report_lines)
with open(f"{RESULTS_DIR}/atr_risk_sizing_backtest.txt", 'w') as f:
    f.write(report_text)

print(f"\nReport saved to {RESULTS_DIR}/atr_risk_sizing_backtest.txt")
print(report_text)
