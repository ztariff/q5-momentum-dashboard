import pandas as pd
import numpy as np
import os
from datetime import datetime

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
trades_df = pd.read_csv('/home/ubuntu/daily_data/analysis_results/backtest_optionD_noReentry_trades.csv')
options_df = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options.csv')

# Filter options to 30d_1m only
options_df = options_df[options_df['option_type'] == '30d_1m'].copy()
options_df['bar_date'] = pd.to_datetime(options_df['bar_date'])
options_df['trade_entry_date'] = pd.to_datetime(options_df['trade_entry_date'])

trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
trades_df['signal_date'] = pd.to_datetime(trades_df['signal_date'])

print(f"Total trades: {len(trades_df)}")
print(f"Options rows (30d_1m): {len(options_df)}")

# ─── STEP 1: Compute ATR features ───────────────────────────────────────────
# Cache enriched data per symbol
enriched_cache = {}

def load_enriched(symbol):
    if symbol not in enriched_cache:
        path = f'/home/ubuntu/daily_data/data/{symbol}_enriched.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            enriched_cache[symbol] = df
        else:
            enriched_cache[symbol] = None
    return enriched_cache[symbol]

results = []

for idx, trade in trades_df.iterrows():
    symbol = trade['symbol']
    entry_date = trade['entry_date']
    signal_date = trade['signal_date']

    edf = load_enriched(symbol)
    if edf is None:
        continue

    # Find signal day: the trading day before entry_date
    # Use the signal_date from the trade log
    signal_row_mask = edf['date'] == signal_date
    if signal_row_mask.sum() == 0:
        # Fallback: find the trading day before entry_date
        before_entry = edf[edf['date'] < entry_date]
        if len(before_entry) == 0:
            continue
        signal_idx = before_entry.index[-1]
    else:
        signal_idx = edf[signal_row_mask].index[0]

    signal_row = edf.loc[signal_idx]

    # ATR features
    atr_14 = signal_row.get('atr_14', np.nan)

    # atr_3d_ago: ATR(14) from 3 trading days before signal day
    if signal_idx >= 3:
        atr_3d_ago = edf.loc[signal_idx - 3, 'atr_14']
    else:
        atr_3d_ago = np.nan

    # atr_change_3d
    if pd.notna(atr_14) and pd.notna(atr_3d_ago) and atr_3d_ago != 0:
        atr_change_3d = (atr_14 - atr_3d_ago) / atr_3d_ago * 100
    else:
        atr_change_3d = np.nan

    # atr_20d_mean: mean of ATR(14) over 20 trading days ending on signal day
    start_20d = max(0, signal_idx - 19)
    atr_20d_vals = edf.loc[start_20d:signal_idx, 'atr_14']
    atr_20d_mean = atr_20d_vals.mean() if len(atr_20d_vals) > 0 else np.nan

    # atr_vs_avg
    if pd.notna(atr_14) and pd.notna(atr_20d_mean) and atr_20d_mean != 0:
        atr_vs_avg = atr_14 / atr_20d_mean
    else:
        atr_vs_avg = np.nan

    # close_location
    close_location = signal_row.get('close_location', np.nan)

    # body_pct
    body_pct_val = signal_row.get('body_pct', np.nan)

    # daily_return_pct
    daily_return_pct = signal_row.get('daily_return_pct', np.nan)

    # ─── STEP 2: Tier classification ────────────────────────────────────
    conditions = []
    c1 = pd.notna(body_pct_val) and body_pct_val > 0.23  # body_pct > 0.23
    c2 = pd.notna(close_location) and close_location > 0.5  # close in upper half
    c3 = pd.notna(atr_change_3d) and atr_change_3d > 0  # ATR rising
    c4 = pd.notna(atr_vs_avg) and atr_vs_avg < 1.0  # ATR below 20d avg (compression)

    conditions_met = sum([c1, c2, c3, c4])

    if conditions_met == 4:
        tier = 1
        position_size = 200000
    elif conditions_met == 3:
        tier = 2
        position_size = 150000
    elif conditions_met == 2:
        tier = 3
        position_size = 100000
    else:
        tier = 4
        position_size = 0  # skip

    results.append({
        'symbol': symbol,
        'signal_date': signal_date,
        'entry_date': entry_date,
        'exit_date': trade['exit_date'],
        'body_quintile': trade['body_quintile'],
        'atr_14': atr_14,
        'atr_3d_ago': atr_3d_ago,
        'atr_change_3d': atr_change_3d,
        'atr_20d_mean': atr_20d_mean,
        'atr_vs_avg': atr_vs_avg,
        'close_location': close_location,
        'body_pct': body_pct_val,
        'daily_return_pct': daily_return_pct,
        'c1_body': c1,
        'c2_close_loc': c2,
        'c3_atr_rising': c3,
        'c4_atr_compressed': c4,
        'conditions_met': conditions_met,
        'tier': tier,
        'tier_position_size': position_size,
    })

rdf = pd.DataFrame(results)
print(f"Trades with features computed: {len(rdf)}")
print(f"Tier distribution:\n{rdf['tier'].value_counts().sort_index()}")

# ─── STEP 3: Simulate options trades ────────────────────────────────────────

def get_option_prices(symbol, entry_date, options_df):
    """Get entry price on entry_date and exit price on 13th trading day."""
    # Filter options for this trade
    mask = (options_df['trade_symbol'] == symbol) & (options_df['trade_entry_date'] == entry_date)
    trade_opts = options_df[mask].sort_values('bar_date').reset_index(drop=True)

    if len(trade_opts) == 0:
        return None, None, None

    option_ticker = trade_opts.iloc[0]['option_ticker']

    # Entry price: first bar_date (should be entry_date)
    # Use morning_price > day_open > day_close priority
    entry_row = trade_opts.iloc[0]
    entry_price = entry_row['morning_price']
    if pd.isna(entry_price) or entry_price <= 0:
        entry_price = entry_row['day_open']
    if pd.isna(entry_price) or entry_price <= 0:
        entry_price = entry_row['day_close']

    # Exit price: 13th trading day (bar index 12, 0-indexed) -> day_close
    if len(trade_opts) >= 13:
        exit_row = trade_opts.iloc[12]  # 13th bar (0-indexed = 12)
        exit_price = exit_row['day_close']
    else:
        # Use last available bar
        exit_row = trade_opts.iloc[-1]
        exit_price = exit_row['day_close']

    return entry_price, exit_price, option_ticker

def simulate_trade(entry_price, exit_price, position_size):
    """Simulate option trade with given position size."""
    if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
        return None, None, None

    contracts = int(position_size / (entry_price * 100))
    if contracts < 1:
        return None, None, None

    # Cost: 5 bps each way
    entry_cost = contracts * entry_price * 100 * 0.0005
    exit_cost = contracts * exit_price * 100 * 0.0005
    gross_pnl = contracts * (exit_price - entry_price) * 100
    net_pnl = gross_pnl - entry_cost - exit_cost

    return contracts, net_pnl, (exit_price - entry_price) / entry_price * 100

# Process all trades
trade_details = []

for idx, row in rdf.iterrows():
    entry_price, exit_price, opt_ticker = get_option_prices(
        row['symbol'], row['entry_date'], options_df
    )

    # Version A: Filtered + Tiered
    if row['tier'] <= 3:
        tier_size = row['tier_position_size']
        a_contracts, a_pnl, a_ret = simulate_trade(entry_price, exit_price, tier_size)
    else:
        a_contracts, a_pnl, a_ret = None, None, None

    # Version B: Filtered Flat (skip tier 4, flat $100K)
    if row['tier'] <= 3:
        b_contracts, b_pnl, b_ret = simulate_trade(entry_price, exit_price, 100000)
    else:
        b_contracts, b_pnl, b_ret = None, None, None

    # Version C: Unfiltered (all trades, flat $100K)
    c_contracts, c_pnl, c_ret = simulate_trade(entry_price, exit_price, 100000)

    trade_details.append({
        **row,
        'option_ticker': opt_ticker,
        'option_entry_price': entry_price,
        'option_exit_price': exit_price,
        'option_return_pct': a_ret if a_ret is not None else (c_ret if c_ret is not None else np.nan),
        # Version A
        'a_contracts': a_contracts,
        'a_pnl': a_pnl,
        'a_return_pct': a_ret,
        # Version B
        'b_contracts': b_contracts,
        'b_pnl': b_pnl,
        'b_return_pct': b_ret,
        # Version C
        'c_contracts': c_contracts,
        'c_pnl': c_pnl,
        'c_return_pct': c_ret,
    })

tdf = pd.DataFrame(trade_details)
print(f"\nTrades with option data: {tdf['option_entry_price'].notna().sum()}")

# Save trade details
tdf.to_csv('/home/ubuntu/daily_data/analysis_results/options_atr_filtered_trades.csv', index=False)
print("Trade details saved.")

# ─── STEP 4: Analysis & Report ──────────────────────────────────────────────

def calc_stats(pnl_series, ret_series=None):
    """Calculate trading stats from a PnL series."""
    pnl = pnl_series.dropna()
    if len(pnl) == 0:
        return {'trades': 0, 'total_pnl': 0, 'avg_pnl': 0, 'wr': 0, 'pf': 0, 'max_dd': 0, 'sharpe': 0}

    trades = len(pnl)
    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    wr = (pnl > 0).sum() / trades * 100

    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = wins / losses if losses > 0 else float('inf')

    # Max drawdown from cumulative PnL
    cum_pnl = pnl.cumsum()
    running_max = cum_pnl.cummax()
    dd = cum_pnl - running_max
    max_dd = dd.min()

    # Sharpe (annualized, using per-trade returns)
    if ret_series is not None:
        rets = ret_series.dropna()
    else:
        rets = pnl
    if len(rets) > 1 and rets.std() > 0:
        trades_per_year = 252 / 13  # ~19.4 trades per year if continuous
        sharpe = (rets.mean() / rets.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    return {
        'trades': trades,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'wr': wr,
        'pf': pf,
        'max_dd': max_dd,
        'sharpe': sharpe
    }

# ─── Build report ───────────────────────────────────────────────────────────
lines = []
lines.append("=" * 90)
lines.append("   ATR-FILTERED 30-DELTA 1-MONTH OPTION BACKTEST (13-DAY HOLD, NO STOP)")
lines.append("=" * 90)
lines.append(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append(f"   Total stock trades in log: {len(trades_df)}")
lines.append(f"   Trades with ATR features: {len(rdf)}")
lines.append(f"   Trades with option data: {tdf['option_entry_price'].notna().sum()}")
lines.append("")

# ─── 1) TIER DISTRIBUTION ───────────────────────────────────────────────────
lines.append("=" * 90)
lines.append("  1) TIER DISTRIBUTION")
lines.append("=" * 90)
lines.append("")
lines.append(f"  {'Tier':<8} {'Desc':<20} {'Size':<10} {'Trades':>7} {'Pct':>7} {'Avg Opt Ret':>12} {'WR':>7} {'Avg PnL':>12}")
lines.append("  " + "-" * 84)

for tier_num in [1, 2, 3, 4]:
    tier_mask = tdf['tier'] == tier_num
    tier_trades = tdf[tier_mask]
    desc = {1: 'BEST (2x)', 2: 'GOOD (1.5x)', 3: 'NEUTRAL (1x)', 4: 'WORST (skip)'}[tier_num]
    size = {1: '$200K', 2: '$150K', 3: '$100K', 4: 'SKIP'}[tier_num]

    # Use option return (common across versions)
    opt_rets = tier_trades['option_return_pct'].dropna()
    avg_ret = opt_rets.mean() if len(opt_rets) > 0 else 0
    wr = (opt_rets > 0).sum() / len(opt_rets) * 100 if len(opt_rets) > 0 else 0

    # Use version A PnL for tiers 1-3, version C for tier 4
    if tier_num <= 3:
        pnl_col = tier_trades['a_pnl'].dropna()
    else:
        pnl_col = tier_trades['c_pnl'].dropna()
    avg_pnl = pnl_col.mean() if len(pnl_col) > 0 else 0

    n = len(tier_trades)
    pct = n / len(tdf) * 100

    lines.append(f"  Tier {tier_num:<3} {desc:<20} {size:<10} {n:>7} {pct:>6.1f}% {avg_ret:>+11.2f}% {wr:>6.1f}% {avg_pnl:>+11.0f}")

lines.append("")

# ─── 2) COMPARISON TABLE ────────────────────────────────────────────────────
lines.append("=" * 90)
lines.append("  2) STRATEGY COMPARISON")
lines.append("=" * 90)
lines.append("")

versions = {
    'A) Filtered+Tiered': ('a_pnl', 'a_return_pct'),
    'B) Filtered Flat': ('b_pnl', 'b_return_pct'),
    'C) Unfiltered': ('c_pnl', 'c_return_pct'),
}

lines.append(f"  {'Version':<25} {'Trades':>7} {'Total PnL':>14} {'Avg PnL':>10} {'WR':>7} {'PF':>7} {'MaxDD':>14} {'Sharpe':>8}")
lines.append("  " + "-" * 92)

for vname, (pnl_col, ret_col) in versions.items():
    pnl = tdf[pnl_col].dropna()
    rets = tdf[ret_col].dropna()
    stats = calc_stats(pnl, rets)
    lines.append(f"  {vname:<25} {stats['trades']:>7} {stats['total_pnl']:>+14,.0f} {stats['avg_pnl']:>+10,.0f} {stats['wr']:>6.1f}% {stats['pf']:>6.2f} {stats['max_dd']:>+14,.0f} {stats['sharpe']:>+7.2f}")

lines.append("")

# ─── 3) YEAR-BY-YEAR for Filtered+Tiered ────────────────────────────────────
lines.append("=" * 90)
lines.append("  3) YEAR-BY-YEAR: FILTERED + TIERED (Version A)")
lines.append("=" * 90)
lines.append("")

tdf['year'] = pd.to_datetime(tdf['entry_date']).dt.year

lines.append(f"  {'Year':<8} {'Trades':>7} {'Total PnL':>14} {'Avg PnL':>10} {'WR':>7} {'PF':>7} {'MaxDD':>14}")
lines.append("  " + "-" * 68)

for year in sorted(tdf['year'].unique()):
    yr_mask = tdf['year'] == year
    pnl = tdf.loc[yr_mask, 'a_pnl'].dropna()
    if len(pnl) == 0:
        continue
    stats = calc_stats(pnl)
    lines.append(f"  {year:<8} {stats['trades']:>7} {stats['total_pnl']:>+14,.0f} {stats['avg_pnl']:>+10,.0f} {stats['wr']:>6.1f}% {stats['pf']:>6.2f} {stats['max_dd']:>+14,.0f}")

lines.append("")

# ─── 4) TOP 10 WINNERS / LOSERS ─────────────────────────────────────────────
lines.append("=" * 90)
lines.append("  4) TOP 10 WINNERS & LOSERS: FILTERED + TIERED (Version A)")
lines.append("=" * 90)
lines.append("")

valid_a = tdf[tdf['a_pnl'].notna()].copy()

lines.append("  TOP 10 WINNERS:")
lines.append(f"  {'Symbol':<8} {'Entry Date':<12} {'Tier':>5} {'Contracts':>10} {'Entry':>8} {'Exit':>8} {'Return':>8} {'PnL':>12}")
lines.append("  " + "-" * 72)
top_winners = valid_a.nlargest(10, 'a_pnl')
for _, r in top_winners.iterrows():
    lines.append(f"  {r['symbol']:<8} {str(r['entry_date'])[:10]:<12} {int(r['tier']):>5} {int(r['a_contracts']):>10} {r['option_entry_price']:>8.2f} {r['option_exit_price']:>8.2f} {r['a_return_pct']:>+7.1f}% {r['a_pnl']:>+12,.0f}")

lines.append("")
lines.append("  TOP 10 LOSERS:")
lines.append(f"  {'Symbol':<8} {'Entry Date':<12} {'Tier':>5} {'Contracts':>10} {'Entry':>8} {'Exit':>8} {'Return':>8} {'PnL':>12}")
lines.append("  " + "-" * 72)
top_losers = valid_a.nsmallest(10, 'a_pnl')
for _, r in top_losers.iterrows():
    lines.append(f"  {r['symbol']:<8} {str(r['entry_date'])[:10]:<12} {int(r['tier']):>5} {int(r['a_contracts']):>10} {r['option_entry_price']:>8.2f} {r['option_exit_price']:>8.2f} {r['a_return_pct']:>+7.1f}% {r['a_pnl']:>+12,.0f}")

lines.append("")

# ─── 5) FILTER EFFECTIVENESS ────────────────────────────────────────────────
lines.append("=" * 90)
lines.append("  5) INDIVIDUAL FILTER EFFECTIVENESS")
lines.append("=" * 90)
lines.append("")

# Use option_return_pct (common metric regardless of sizing)
valid_opts = tdf[tdf['option_return_pct'].notna()].copy()

filter_names = {
    'c1_body': 'body_pct > 0.23',
    'c2_close_loc': 'close_location > 0.5',
    'c3_atr_rising': 'atr_change_3d > 0',
    'c4_atr_compressed': 'atr_vs_avg < 1.0',
}

lines.append(f"  {'Filter':<25} {'State':<8} {'Trades':>7} {'Avg Ret':>10} {'WR':>7} {'Med Ret':>10}")
lines.append("  " + "-" * 68)

for col, desc in filter_names.items():
    for state, label in [(True, 'TRUE'), (False, 'FALSE')]:
        mask = valid_opts[col] == state
        subset = valid_opts.loc[mask, 'option_return_pct']
        if len(subset) > 0:
            avg = subset.mean()
            wr = (subset > 0).sum() / len(subset) * 100
            med = subset.median()
            lines.append(f"  {desc:<25} {label:<8} {len(subset):>7} {avg:>+9.2f}% {wr:>6.1f}% {med:>+9.2f}%")
        else:
            lines.append(f"  {desc:<25} {label:<8} {'N/A':>7}")
    lines.append("")

# ─── 6) ATR + BODY INTERACTION ──────────────────────────────────────────────
lines.append("=" * 90)
lines.append("  6) ATR COMPRESSION BREAKOUT vs BODY FILTER INTERACTION")
lines.append("=" * 90)
lines.append("")
lines.append("  ATR filter = (atr_change_3d > 0) AND (atr_vs_avg < 1.0)")
lines.append("  Body filter = (body_pct > 0.23)")
lines.append("")

valid_opts['atr_filter'] = valid_opts['c3_atr_rising'] & valid_opts['c4_atr_compressed']
valid_opts['body_filter'] = valid_opts['c1_body']

combos = [
    (True, True, 'ATR PASS + Body PASS'),
    (True, False, 'ATR PASS + Body FAIL'),
    (False, True, 'ATR FAIL + Body PASS'),
    (False, False, 'ATR FAIL + Body FAIL'),
]

lines.append(f"  {'Combination':<30} {'Trades':>7} {'Avg Ret':>10} {'WR':>7} {'Med Ret':>10} {'Total PnL(C)':>14}")
lines.append("  " + "-" * 80)

for atr_state, body_state, label in combos:
    mask = (valid_opts['atr_filter'] == atr_state) & (valid_opts['body_filter'] == body_state)
    subset = valid_opts[mask]
    rets = subset['option_return_pct'].dropna()
    pnl_c = subset['c_pnl'].dropna()
    if len(rets) > 0:
        avg = rets.mean()
        wr = (rets > 0).sum() / len(rets) * 100
        med = rets.median()
        total_pnl = pnl_c.sum()
        lines.append(f"  {label:<30} {len(rets):>7} {avg:>+9.2f}% {wr:>6.1f}% {med:>+9.2f}% {total_pnl:>+14,.0f}")
    else:
        lines.append(f"  {label:<30} {'N/A':>7}")

lines.append("")
lines.append("  Does ATR compression breakout work independently of body_pct?")
lines.append("")

# Check ATR filter independently
atr_pass = valid_opts[valid_opts['atr_filter'] == True]['option_return_pct']
atr_fail = valid_opts[valid_opts['atr_filter'] == False]['option_return_pct']
if len(atr_pass) > 0 and len(atr_fail) > 0:
    lines.append(f"  ATR filter alone: PASS avg {atr_pass.mean():+.2f}%, WR {(atr_pass>0).sum()/len(atr_pass)*100:.1f}% ({len(atr_pass)} trades)")
    lines.append(f"                    FAIL avg {atr_fail.mean():+.2f}%, WR {(atr_fail>0).sum()/len(atr_fail)*100:.1f}% ({len(atr_fail)} trades)")
    lines.append(f"  Edge (PASS - FAIL): {atr_pass.mean() - atr_fail.mean():+.2f}% avg return")

lines.append("")
lines.append("=" * 90)

report = "\n".join(lines)

# Save report
with open('/home/ubuntu/daily_data/analysis_results/options_atr_filtered_backtest.txt', 'w') as f:
    f.write(report)

print("\n" + report)
print("\nReport saved to /home/ubuntu/daily_data/analysis_results/options_atr_filtered_backtest.txt")
print("Trade details saved to /home/ubuntu/daily_data/analysis_results/options_atr_filtered_trades.csv")
