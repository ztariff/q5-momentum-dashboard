import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Load trades ──
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv')
trades['signal_date'] = pd.to_datetime(trades['signal_date'])
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])
print(f"Loaded {len(trades)} trades")

# ── Cache enriched data per symbol ──
data_dir = Path('/home/ubuntu/daily_data/data')
symbol_data = {}
symbols_needed = trades['symbol'].unique()
print(f"Loading enriched data for {len(symbols_needed)} symbols...")

for sym in symbols_needed:
    fpath = data_dir / f"{sym}_enriched.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, usecols=['date', 'open', 'high', 'low', 'close', 'atr_14'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        symbol_data[sym] = df
    else:
        print(f"  WARNING: {fpath} not found")

print(f"Loaded data for {len(symbol_data)} symbols")

# ── Constants ──
NOTIONAL = 1_000_000
COST_BPS = 5  # each way
MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TRADING_DAYS = 252

# ── For each trade, gather: signal_day ATR, signal_day low, and the 20 forward days ──
print("Processing trades...")

results = []  # list of dicts per trade

for idx, row in trades.iterrows():
    sym = row['symbol']
    if sym not in symbol_data:
        continue

    sdf = symbol_data[sym]

    # Find signal_date row
    sig_mask = sdf['date'] == row['signal_date']
    if sig_mask.sum() == 0:
        continue
    sig_idx = sdf.index[sig_mask][0]
    sig_row = sdf.loc[sig_idx]

    atr = sig_row['atr_14']
    signal_low = sig_row['low']

    if pd.isna(atr) or pd.isna(signal_low) or atr <= 0:
        continue

    entry_price = row['entry_price']
    shares = row['shares']

    # Get forward 20 trading days starting from entry_date
    entry_mask = sdf['date'] >= row['entry_date']
    forward = sdf.loc[entry_mask].head(20)

    if len(forward) == 0:
        continue

    # Baseline exit: close of last day
    baseline_exit = forward.iloc[-1]['close']
    cost = NOTIONAL * COST_BPS / 10000 * 2  # entry + exit
    baseline_pnl = shares * (baseline_exit - entry_price) - cost

    # For each stop config, simulate
    trade_result = {
        'symbol': sym,
        'signal_date': row['signal_date'],
        'entry_date': row['entry_date'],
        'entry_price': entry_price,
        'shares': shares,
        'atr': atr,
        'signal_low': signal_low,
        'baseline_exit': baseline_exit,
        'baseline_pnl': baseline_pnl,
        'baseline_return': baseline_pnl / NOTIONAL,
    }

    # Pre-extract lows and closes as arrays for speed
    fwd_lows = forward['low'].values
    fwd_closes = forward['close'].values

    for mult in MULTIPLIERS:
        # Reference A: stop below entry price
        stop_a = entry_price - atr * mult
        # Reference B: stop below signal day low
        stop_b = signal_low - atr * mult

        for ref, stop_price in [('A', stop_a), ('B', stop_b)]:
            key = f"{ref}_{mult}"

            # Walk forward
            stopped = False
            for d in range(len(fwd_lows)):
                if fwd_lows[d] <= stop_price:
                    # Stopped out - exit at stop price
                    exit_price = stop_price
                    pnl = shares * (exit_price - entry_price) - cost
                    trade_result[f'{key}_pnl'] = pnl
                    trade_result[f'{key}_return'] = pnl / NOTIONAL
                    trade_result[f'{key}_stopped'] = True
                    trade_result[f'{key}_stop_day'] = d + 1
                    stopped = True
                    break

            if not stopped:
                trade_result[f'{key}_pnl'] = baseline_pnl
                trade_result[f'{key}_return'] = baseline_pnl / NOTIONAL
                trade_result[f'{key}_stopped'] = False
                trade_result[f'{key}_stop_day'] = 0

    results.append(trade_result)

    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{len(trades)} trades")

print(f"Completed: {len(results)} trades with valid data")

# ── Build results DataFrame ──
rdf = pd.DataFrame(results)

# ── Compute stats for each configuration ──
configs = ['baseline'] + [f"{ref}_{mult}" for ref in ['A', 'B'] for mult in MULTIPLIERS]

def compute_stats(pnls, returns, label, stopped=None):
    """Compute standard stats for a configuration."""
    n = len(pnls)
    wins = (pnls > 0).sum()
    losses = (pnls <= 0).sum()
    win_rate = wins / n * 100

    gross_wins = pnls[pnls > 0].sum()
    gross_losses = abs(pnls[pnls <= 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Max drawdown on cumulative P&L
    cum_pnl = pnls.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # Sharpe (annualized from trade-level returns)
    mean_ret = returns.mean()
    std_ret = returns.std()
    # ~12.6 non-overlapping 20-day periods per year
    periods_per_year = TRADING_DAYS / 20
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0

    stats = {
        'label': label,
        'n_trades': n,
        'stopped_pct': (stopped.sum() / n * 100) if stopped is not None else 0.0,
        'mean_pnl': pnls.mean(),
        'median_pnl': pnls.median(),
        'total_pnl': pnls.sum(),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'worst_trade': pnls.min(),
        'max_drawdown': max_dd,
        'sharpe': sharpe,
    }
    return stats

all_stats = []

# Baseline
bl_pnls = rdf['baseline_pnl']
bl_rets = rdf['baseline_return']
all_stats.append(compute_stats(bl_pnls, bl_rets, 'BASELINE (no stop)'))

for ref in ['A', 'B']:
    for mult in MULTIPLIERS:
        key = f"{ref}_{mult}"
        label_ref = "Entry Price" if ref == 'A' else "Signal Low"
        label = f"Ref {ref}: {label_ref} - {mult}×ATR"
        pnls = rdf[f'{key}_pnl']
        rets = rdf[f'{key}_return']
        stopped = rdf[f'{key}_stopped']
        all_stats.append(compute_stats(pnls, rets, label, stopped))

stats_df = pd.DataFrame(all_stats)

# ── Year-by-year for top 3 configs ──
# Rank by total PnL
stats_df_sorted = stats_df.sort_values('total_pnl', ascending=False)
# Top 3 non-baseline
top3_labels = stats_df_sorted[stats_df_sorted['label'] != 'BASELINE (no stop)'].head(3)['label'].tolist()
# Also include baseline for comparison
yby_configs = ['BASELINE (no stop)'] + top3_labels

# Map label back to key
label_to_key = {'BASELINE (no stop)': 'baseline'}
for ref in ['A', 'B']:
    for mult in MULTIPLIERS:
        key = f"{ref}_{mult}"
        label_ref = "Entry Price" if ref == 'A' else "Signal Low"
        label = f"Ref {ref}: {label_ref} - {mult}×ATR"
        label_to_key[label] = key

rdf['year'] = rdf['entry_date'].dt.year
years = sorted(rdf['year'].unique())

yby_data = {}
for label in yby_configs:
    key = label_to_key[label]
    pnl_col = f'{key}_pnl' if key != 'baseline' else 'baseline_pnl'
    ret_col = f'{key}_return' if key != 'baseline' else 'baseline_return'

    yearly = {}
    for yr in years:
        mask = rdf['year'] == yr
        yr_pnls = rdf.loc[mask, pnl_col]
        yr_rets = rdf.loc[mask, ret_col]
        yearly[yr] = {
            'n': len(yr_pnls),
            'total_pnl': yr_pnls.sum(),
            'mean_pnl': yr_pnls.mean(),
            'win_rate': (yr_pnls > 0).sum() / len(yr_pnls) * 100,
            'sharpe': (yr_rets.mean() / yr_rets.std() * np.sqrt(TRADING_DAYS/20)) if yr_rets.std() > 0 else 0,
        }
    yby_data[label] = yearly

# ── Write Report ──
out = '/home/ubuntu/daily_data/analysis_results/atr_stop_analysis_report.txt'

with open(out, 'w') as f:
    f.write("=" * 120 + "\n")
    f.write("ATR-BASED STOP LOSS ANALYSIS FOR 20-DAY MOMENTUM TRADE\n")
    f.write("=" * 120 + "\n\n")

    f.write(f"Universe: {len(results)} trades with valid ATR data (of {len(trades)} total)\n")
    f.write(f"Trade spec: Entry at OPEN of entry_date, exit at CLOSE 20 trading days later\n")
    f.write(f"Position size: $1M per trade, 5 bps cost each way\n")
    f.write(f"Stop mechanics: If day's LOW breaches stop, exit at stop price\n\n")

    f.write("Reference A: Stop = entry_price - (ATR_14 × multiplier)\n")
    f.write("Reference B: Stop = signal_day_low - (ATR_14 × multiplier)\n")
    f.write("  (Reference B gives wider stops since signal_day_low <= entry_price typically)\n\n")

    # Summary table
    f.write("-" * 120 + "\n")
    f.write("SUMMARY OF ALL CONFIGURATIONS\n")
    f.write("-" * 120 + "\n\n")

    header = f"{'Configuration':<40} {'N':>5} {'Stop%':>6} {'MeanPnL':>12} {'MedPnL':>12} {'TotalPnL':>14} {'WinR%':>6} {'PF':>6} {'Worst':>14} {'MaxDD':>14} {'Sharpe':>7}\n"
    f.write(header)
    f.write("-" * 120 + "\n")

    for _, r in stats_df.iterrows():
        line = f"{r['label']:<40} {r['n_trades']:>5} {r['stopped_pct']:>5.1f}% {r['mean_pnl']:>12,.0f} {r['median_pnl']:>12,.0f} {r['total_pnl']:>14,.0f} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} {r['worst_trade']:>14,.0f} {r['max_drawdown']:>14,.0f} {r['sharpe']:>7.3f}\n"
        f.write(line)

    f.write("\n")

    # Ranked by total PnL
    f.write("-" * 120 + "\n")
    f.write("CONFIGURATIONS RANKED BY TOTAL P&L\n")
    f.write("-" * 120 + "\n\n")

    for rank, (_, r) in enumerate(stats_df_sorted.iterrows(), 1):
        bl_pnl = stats_df.loc[stats_df['label'] == 'BASELINE (no stop)', 'total_pnl'].iloc[0]
        delta = r['total_pnl'] - bl_pnl
        marker = " <<<" if r['label'] == 'BASELINE (no stop)' else ""
        f.write(f"  {rank:>2}. {r['label']:<40}  Total: ${r['total_pnl']:>14,.0f}   vs Baseline: ${delta:>+14,.0f}   Sharpe: {r['sharpe']:.3f}{marker}\n")

    f.write("\n")

    # Reference A detail
    f.write("-" * 120 + "\n")
    f.write("REFERENCE A: STOP BELOW ENTRY PRICE\n")
    f.write("-" * 120 + "\n\n")

    f.write(f"{'Multiplier':<12} {'Stop%':>6} {'MeanPnL':>12} {'TotalPnL':>14} {'WinR%':>6} {'PF':>6} {'MaxDD':>14} {'Sharpe':>7}  {'vs Baseline':>14}\n")
    f.write("-" * 100 + "\n")

    bl_total = stats_df.loc[stats_df['label'] == 'BASELINE (no stop)', 'total_pnl'].iloc[0]

    for mult in MULTIPLIERS:
        r = stats_df[stats_df['label'].str.contains(f"Ref A.*{mult}×ATR")].iloc[0]
        delta = r['total_pnl'] - bl_total
        f.write(f"  {mult:<10} {r['stopped_pct']:>5.1f}% {r['mean_pnl']:>12,.0f} {r['total_pnl']:>14,.0f} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} {r['max_drawdown']:>14,.0f} {r['sharpe']:>7.3f}  {delta:>+14,.0f}\n")

    f.write("\n")

    # Reference B detail
    f.write("-" * 120 + "\n")
    f.write("REFERENCE B: STOP BELOW SIGNAL DAY'S LOW\n")
    f.write("-" * 120 + "\n\n")

    f.write(f"{'Multiplier':<12} {'Stop%':>6} {'MeanPnL':>12} {'TotalPnL':>14} {'WinR%':>6} {'PF':>6} {'MaxDD':>14} {'Sharpe':>7}  {'vs Baseline':>14}\n")
    f.write("-" * 100 + "\n")

    for mult in MULTIPLIERS:
        r = stats_df[stats_df['label'].str.contains(f"Ref B.*{mult}×ATR")].iloc[0]
        delta = r['total_pnl'] - bl_total
        f.write(f"  {mult:<10} {r['stopped_pct']:>5.1f}% {r['mean_pnl']:>12,.0f} {r['total_pnl']:>14,.0f} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} {r['max_drawdown']:>14,.0f} {r['sharpe']:>7.3f}  {delta:>+14,.0f}\n")

    f.write("\n")

    # Stop-out analysis
    f.write("-" * 120 + "\n")
    f.write("STOP-OUT FREQUENCY AND IMPACT\n")
    f.write("-" * 120 + "\n\n")

    for ref in ['A', 'B']:
        ref_label = "Entry Price" if ref == 'A' else "Signal Low"
        f.write(f"  Reference {ref} ({ref_label}):\n")
        for mult in MULTIPLIERS:
            key = f"{ref}_{mult}"
            stopped = rdf[f'{key}_stopped']
            n_stopped = stopped.sum()
            pct = n_stopped / len(rdf) * 100

            # PnL of stopped vs not-stopped
            stopped_pnl = rdf.loc[rdf[f'{key}_stopped'], f'{key}_pnl']
            not_stopped_pnl = rdf.loc[~rdf[f'{key}_stopped'], f'{key}_pnl']

            # Average stop day
            stop_days = rdf.loc[rdf[f'{key}_stopped'], f'{key}_stop_day']
            avg_day = stop_days.mean() if len(stop_days) > 0 else 0

            # How many stopped trades would have been winners without stop?
            stopped_baseline = rdf.loc[rdf[f'{key}_stopped'], 'baseline_pnl']
            would_have_won = (stopped_baseline > 0).sum()

            f.write(f"    {mult}×ATR: {n_stopped:>5} stopped ({pct:>5.1f}%)  avg stop day: {avg_day:>4.1f}  ")
            f.write(f"avg stopped PnL: ${stopped_pnl.mean():>10,.0f}  ")
            f.write(f"would-have-won: {would_have_won}/{n_stopped} ({would_have_won/n_stopped*100:.0f}%)\n" if n_stopped > 0 else "\n")
        f.write("\n")

    # Year-by-year
    f.write("-" * 120 + "\n")
    f.write("YEAR-BY-YEAR: BASELINE + TOP 3 CONFIGURATIONS\n")
    f.write("-" * 120 + "\n\n")

    for label in yby_configs:
        f.write(f"  {label}\n")
        f.write(f"  {'Year':<6} {'N':>5} {'TotalPnL':>14} {'MeanPnL':>12} {'WinR%':>6} {'Sharpe':>7}\n")
        f.write("  " + "-" * 60 + "\n")
        for yr in years:
            d = yby_data[label][yr]
            f.write(f"  {yr:<6} {d['n']:>5} {d['total_pnl']:>14,.0f} {d['mean_pnl']:>12,.0f} {d['win_rate']:>5.1f}% {d['sharpe']:>7.3f}\n")
        f.write("\n")

    # Key findings
    f.write("=" * 120 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("=" * 120 + "\n\n")

    best_config = stats_df_sorted.iloc[0]
    best_non_bl = stats_df_sorted[stats_df_sorted['label'] != 'BASELINE (no stop)'].iloc[0]

    f.write(f"  Baseline total P&L:            ${bl_total:>14,.0f}\n")
    f.write(f"  Best config total P&L:         ${best_config['total_pnl']:>14,.0f}  ({best_config['label']})\n")
    f.write(f"  Best non-baseline config P&L:  ${best_non_bl['total_pnl']:>14,.0f}  ({best_non_bl['label']})\n\n")

    beats_baseline = stats_df[stats_df['total_pnl'] > bl_total]
    n_beats = len(beats_baseline[beats_baseline['label'] != 'BASELINE (no stop)'])

    f.write(f"  Configs that beat baseline on total P&L: {n_beats} out of 12\n\n")

    if n_beats > 0:
        f.write("  Configurations beating baseline:\n")
        for _, r in beats_baseline[beats_baseline['label'] != 'BASELINE (no stop)'].iterrows():
            delta = r['total_pnl'] - bl_total
            f.write(f"    - {r['label']}: ${r['total_pnl']:>14,.0f}  (+${delta:>10,.0f})\n")
    else:
        f.write("  NO stop configuration produces higher total P&L than baseline.\n")
        f.write("  ATR-based stops, like fixed-bps stops, reduce total P&L.\n")

    f.write("\n")

    # Risk-adjusted comparison
    bl_sharpe = stats_df.loc[stats_df['label'] == 'BASELINE (no stop)', 'sharpe'].iloc[0]
    bl_maxdd = stats_df.loc[stats_df['label'] == 'BASELINE (no stop)', 'max_drawdown'].iloc[0]
    bl_worst = stats_df.loc[stats_df['label'] == 'BASELINE (no stop)', 'worst_trade'].iloc[0]

    better_sharpe = stats_df[(stats_df['sharpe'] > bl_sharpe) & (stats_df['label'] != 'BASELINE (no stop)')]
    better_dd = stats_df[(stats_df['max_drawdown'] > bl_maxdd) & (stats_df['label'] != 'BASELINE (no stop)')]
    better_worst = stats_df[(stats_df['worst_trade'] > bl_worst) & (stats_df['label'] != 'BASELINE (no stop)')]

    f.write("  Risk-adjusted comparison vs baseline:\n")
    f.write(f"    Configs with better Sharpe ratio:   {len(better_sharpe)} of 12\n")
    f.write(f"    Configs with smaller max drawdown:  {len(better_dd)} of 12\n")
    f.write(f"    Configs with better worst trade:    {len(better_worst)} of 12\n\n")

    f.write(f"  Baseline Sharpe: {bl_sharpe:.3f}   Max DD: ${bl_maxdd:>12,.0f}   Worst trade: ${bl_worst:>12,.0f}\n")

    if len(better_sharpe) > 0:
        best_sharpe_row = better_sharpe.sort_values('sharpe', ascending=False).iloc[0]
        f.write(f"  Best Sharpe:     {best_sharpe_row['sharpe']:.3f}   ({best_sharpe_row['label']})\n")
    if len(better_dd) > 0:
        best_dd_row = better_dd.sort_values('max_drawdown', ascending=False).iloc[0]
        f.write(f"  Best Max DD:     ${best_dd_row['max_drawdown']:>12,.0f}   ({best_dd_row['label']})\n")
    if len(better_worst) > 0:
        best_worst_row = better_worst.sort_values('worst_trade', ascending=False).iloc[0]
        f.write(f"  Best Worst Tr:   ${best_worst_row['worst_trade']:>12,.0f}   ({best_worst_row['label']})\n")

    f.write("\n" + "=" * 120 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 120 + "\n")

print(f"\nReport saved to {out}")
print(f"\nQuick summary:")
print(f"  Baseline total P&L: ${bl_total:,.0f}")
print(f"  Best config: {best_config['label']} at ${best_config['total_pnl']:,.0f}")
print(f"  Configs beating baseline: {n_beats}/12")
