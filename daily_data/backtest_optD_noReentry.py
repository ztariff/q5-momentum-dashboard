#!/usr/bin/env python3
"""
Option D Backtest with NO RE-ENTRY rule.
Rebuilds signals from scratch. If a position in symbol X is open, skip new signals for X.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/ubuntu/daily_data/data")
OUT_DIR = Path("/home/ubuntu/daily_data/analysis_results")

# ── 1. Load manifest ──
manifest = pd.read_csv(DATA_DIR / "manifest.csv")
symbols = manifest[manifest['status'] == 'ok']['symbol'].tolist()
print(f"Universe: {len(symbols)} symbols")

# ── 2. Load all data and compute signals ──
all_frames = []
for sym in symbols:
    fp = DATA_DIR / f"{sym}_enriched.csv"
    df = pd.read_csv(fp, usecols=['date', 'open', 'high', 'low', 'close', 'atr_14', 'body_pct'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # SMA50
    df['sma50'] = df['close'].rolling(50, min_periods=50).mean()

    # 10-day slope of SMA50 (percentage change)
    df['sma50_slope10'] = (df['sma50'] - df['sma50'].shift(10)) / df['sma50'].shift(10) * 100

    # 252-day rolling z-score of the slope (min_periods=60, no shift)
    roll = df['sma50_slope10'].rolling(252, min_periods=60)
    df['slope_z'] = (df['sma50_slope10'] - roll.mean()) / roll.std()

    df['symbol'] = sym
    all_frames.append(df[['date', 'symbol', 'open', 'high', 'low', 'close', 'atr_14', 'body_pct', 'slope_z']].dropna(subset=['slope_z']))

print("Data loaded, computing cross-sectional quintiles...")

big = pd.concat(all_frames, ignore_index=True)
del all_frames

# ── 3. Cross-sectional quintile sort per day ──
def assign_quintiles(slope_z_series):
    if len(slope_z_series) < 20:
        return pd.Series(np.nan, index=slope_z_series.index)
    try:
        return pd.qcut(slope_z_series, 5, labels=[1,2,3,4,5]).astype(float)
    except ValueError:
        return pd.Series(np.nan, index=slope_z_series.index)

big['quintile'] = big.groupby('date')['slope_z'].transform(assign_quintiles)
big = big.dropna(subset=['quintile'])
big['quintile'] = big['quintile'].astype(int)

print(f"Total rows with quintiles: {len(big)}")

# ── 4. Detect Q5 transitions ──
big = big.sort_values(['symbol', 'date']).reset_index(drop=True)
big['prev_quintile'] = big.groupby('symbol')['quintile'].shift(1)
big['entered_q5'] = (big['quintile'] == 5) & (big['prev_quintile'] != 5)

signals = big[big['entered_q5']].copy()
# Filter to 2020+ to match original backtest period
signals = signals[signals['date'] >= '2020-01-01']
signals = signals.sort_values('date').reset_index(drop=True)
print(f"Total Q5 entry signals (2020+): {len(signals)}")

# ── 5. Build per-symbol lookup from raw data (not filtered by quintile) ──
print("Building per-symbol lookup tables...")
sym_data = {}
for sym in symbols:
    fp = DATA_DIR / f"{sym}_enriched.csv"
    sdf = pd.read_csv(fp, usecols=['date', 'open', 'high', 'low', 'close', 'atr_14', 'body_pct'])
    sdf['date'] = pd.to_datetime(sdf['date'])
    sdf = sdf.sort_values('date').reset_index(drop=True)
    sym_data[sym] = sdf

# ── 6. Body_pct quintile breakpoints ──
# Collect ENTRY day body_pct for all signals (entry = next day after signal)
print("Computing entry-day body_pct for quintile breaks...")
entry_body_pcts = []
for _, row in signals.iterrows():
    sym = row['symbol']
    signal_date = row['date']
    sdf = sym_data.get(sym)
    if sdf is None:
        continue
    sig_idx_arr = sdf.index[sdf['date'] == signal_date]
    if len(sig_idx_arr) == 0:
        continue
    sig_pos = sig_idx_arr[0]
    if sig_pos + 1 < len(sdf):
        entry_body = sdf.loc[sig_pos + 1, 'body_pct']
        if not pd.isna(entry_body):
            entry_body_pcts.append(entry_body)

entry_body_series = pd.Series(entry_body_pcts)
_, bp_bins = pd.qcut(entry_body_series, 5, labels=[1,2,3,4,5], retbins=True)
print(f"body_pct quintile bin edges (entry day): {bp_bins}")
print(f"  Q1 range: [{bp_bins[0]:.4f}, {bp_bins[1]:.4f}]")
print(f"  Total signals with entry body: {len(entry_body_pcts)}")

def get_body_quintile_and_size(body_pct):
    """Return (body_quintile_label, position_size) or (label, 0) for skip."""
    if pd.isna(body_pct):
        return 'Q3', 1_000_000  # default if missing
    if body_pct <= bp_bins[1]:  # Q1 → skip
        return 'Q1', 0
    elif body_pct <= bp_bins[2]:  # Q2
        return 'Q2', 500_000
    elif body_pct <= bp_bins[3]:  # Q3
        return 'Q3', 1_000_000
    elif body_pct <= bp_bins[4]:  # Q4
        return 'Q4', 1_500_000
    else:  # Q5
        return 'Q5', 2_000_000

# ── 7. Walk through signals and generate trades with NO RE-ENTRY ──
print("Running backtest with no-re-entry rule...")

trades = []
open_positions = {}  # symbol -> exit_date
skipped_reentry = []
skipped_q1 = 0

signals = signals.sort_values('date').reset_index(drop=True)

for idx, row in signals.iterrows():
    sym = row['symbol']
    signal_date = row['date']
    body = row['body_pct']

    # Look up symbol data
    sdf = sym_data.get(sym)
    if sdf is None:
        continue

    # Find signal day index
    sig_idx_arr = sdf.index[sdf['date'] == signal_date]
    if len(sig_idx_arr) == 0:
        continue
    sig_pos = sig_idx_arr[0]

    signal_day_low = sdf.loc[sig_pos, 'low']
    signal_day_atr = sdf.loc[sig_pos, 'atr_14']

    if pd.isna(signal_day_atr) or pd.isna(signal_day_low):
        continue

    stop_price = signal_day_low - 3.0 * signal_day_atr

    # Entry: next day's open
    if sig_pos + 1 >= len(sdf):
        continue
    entry_pos = sig_pos + 1
    entry_date = sdf.loc[entry_pos, 'date']
    entry_price = sdf.loc[entry_pos, 'open']

    if pd.isna(entry_price) or entry_price <= 0:
        continue

    # Use ENTRY day's body_pct for sizing (matches original implementation)
    entry_body = sdf.loc[entry_pos, 'body_pct']
    bq_label, pos_size = get_body_quintile_and_size(entry_body)

    # Skip Q1 body_pct
    if pos_size == 0:
        skipped_q1 += 1
        continue

    # Check no-re-entry: is there an open position in this symbol?
    if sym in open_positions and open_positions[sym] > signal_date:
        skipped_reentry.append({
            'symbol': sym,
            'signal_date': signal_date,
            'body_quintile': bq_label,
            'position_size': pos_size,
            'reason': 'reentry_blocked',
            'existing_exit': open_positions[sym]
        })
        continue

    shares = int(pos_size / entry_price)
    if shares <= 0:
        continue

    # Walk forward up to 19 bars (entry day = day 1, exit on day 20 = entry+19)
    exit_type = 'TIME'
    exit_price = None
    exit_date = None
    days_held = 0
    day20_close = None

    for d in range(1, 20):  # d=1..19; day 20 = entry+19
        check_pos = entry_pos + d
        if check_pos >= len(sdf):
            # Ran out of data; exit at last available close
            last_pos = len(sdf) - 1
            exit_price = sdf.loc[last_pos, 'close']
            exit_date = sdf.loc[last_pos, 'date']
            days_held = last_pos - entry_pos + 1  # +1 because entry day = day 1
            exit_type = 'DATA_END'
            break

        day_low = sdf.loc[check_pos, 'low']

        if d == 19:
            day20_close = sdf.loc[check_pos, 'close']

        # Check stop
        if not pd.isna(day_low) and day_low <= stop_price:
            exit_price = stop_price
            exit_date = sdf.loc[check_pos, 'date']
            days_held = d + 1  # entry day = day 1
            exit_type = 'STOP'
            break

        if d == 19:
            exit_price = sdf.loc[check_pos, 'close']
            exit_date = sdf.loc[check_pos, 'date']
            days_held = 20
            exit_type = 'TIME'
            break

    if exit_price is None:
        continue

    # Costs: 5 bps each way
    cost_entry = entry_price * shares * 0.0005
    cost_exit = exit_price * shares * 0.0005
    gross_pnl = (exit_price - entry_price) * shares
    net_pnl = gross_pnl - cost_entry - cost_exit
    ret_pct = net_pnl / pos_size * 100

    # Track open position exit date for no-re-entry
    open_positions[sym] = exit_date

    trades.append({
        'symbol': sym,
        'signal_date': signal_date,
        'entry_date': entry_date,
        'exit_date': exit_date,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'stop_price': stop_price,
        'shares': shares,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'return_pct': ret_pct,
        'days_held': days_held,
        'exit_type': exit_type,
        'body_quintile': bq_label,
        'position_size': pos_size,
        'day20_close': day20_close
    })

print(f"\nDone. {len(trades)} trades, {len(skipped_reentry)} blocked by re-entry rule, {skipped_q1} skipped Q1.")

# ── 8. Save trades ──
trades_df = pd.DataFrame(trades)
trades_df = trades_df.sort_values(['entry_date', 'symbol']).reset_index(drop=True)
trades_df.to_csv(OUT_DIR / "backtest_optionD_noReentry_trades.csv", index=False)

skipped_df = pd.DataFrame(skipped_reentry)
print(f"Skipped re-entry signals: {len(skipped_df)}")

# ── 9. Load original Option D trades for blocked trade comparison ──
orig_trades = pd.read_csv(OUT_DIR / "backtest_optionD_trades.csv")
orig_trades['signal_date'] = pd.to_datetime(orig_trades['signal_date'])

# ── 10. Generate Report ──
print("Generating report...")

report_lines = []
def rpt(s=''):
    report_lines.append(s)

rpt("=" * 90)
rpt("OPTION D BACKTEST — NO RE-ENTRY RULE")
rpt("=" * 90)
rpt()
rpt("Strategy: Q5 z-scored SMA50 10-day slope entry, ATR(14)x3 stop, 20-day hold")
rpt("Sizing: Skip body_pct Q1, Q2=$500K, Q3=$1M, Q4=$1.5M, Q5=$2M")
rpt("NEW: If a position in symbol X is open, do NOT open another. Skip the signal.")
rpt(f"body_pct quintile bin edges: {bp_bins}")
rpt()

# ── A) OVERALL STATS ──
rpt("-" * 90)
rpt("A) OVERALL STATS")
rpt("-" * 90)

n_trades = len(trades_df)
total_pnl = trades_df['net_pnl'].sum()
winners = (trades_df['net_pnl'] > 0).sum()
losers = (trades_df['net_pnl'] <= 0).sum()
wr = winners / n_trades * 100 if n_trades > 0 else 0
avg_pnl = trades_df['net_pnl'].mean()
median_pnl = trades_df['net_pnl'].median()
best = trades_df['net_pnl'].max()
worst = trades_df['net_pnl'].min()
avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winners > 0 else 0
avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losers > 0 else 0
pf = abs(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() / trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum()) if losers > 0 else float('inf')

# Max drawdown on cumulative P&L
cum_pnl = trades_df.sort_values('exit_date')['net_pnl'].cumsum()
running_max = cum_pnl.cummax()
drawdown = cum_pnl - running_max
max_dd = drawdown.min()

# Sharpe (annualized, trade-level return-based to match original methodology)
trades_df['exit_date_dt'] = pd.to_datetime(trades_df['exit_date'])
trade_returns = trades_df['return_pct'] / 100  # decimal returns
avg_hold = trades_df['days_held'].mean()
sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252 / avg_hold) if trade_returns.std() > 0 else 0

# Calmar = annualized return / |max DD|
years = (trades_df['exit_date_dt'].max() - trades_df['exit_date_dt'].min()).days / 365.25
annual_ret = total_pnl / years if years > 0 else 0
calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

# Avg concurrent positions
entry_dates_np = pd.to_datetime(trades_df['entry_date']).values
exit_dates_np = pd.to_datetime(trades_df['exit_date']).values
sample_dates = pd.bdate_range(trades_df['entry_date'].min(), trades_df['exit_date'].max())
concurrent = []
for d in sample_dates:
    d_np = np.datetime64(d)
    cnt = ((entry_dates_np <= d_np) & (exit_dates_np >= d_np)).sum()
    concurrent.append(cnt)
avg_concurrent = np.mean(concurrent) if concurrent else 0

rpt(f"Total trades:                    {n_trades}")
rpt(f"Winners / Losers:                {winners} / {losers}")
rpt(f"Win rate:                        {wr:.1f}%")
rpt(f"Profit factor:                   {pf:.2f}")
rpt(f"Total P&L:                       ${total_pnl:,.0f}")
rpt(f"Avg P&L per trade:               ${avg_pnl:,.0f}")
rpt(f"Median P&L per trade:            ${median_pnl:,.0f}")
rpt(f"Avg winner:                      ${avg_win:,.0f}")
rpt(f"Avg loser:                       ${avg_loss:,.0f}")
rpt(f"Best trade:                      ${best:,.0f}")
rpt(f"Worst trade:                     ${worst:,.0f}")
rpt(f"Max drawdown:                    ${max_dd:,.0f}")
rpt(f"Sharpe ratio (annualized):       {sharpe:.2f}")
rpt(f"Calmar ratio:                    {calmar:.2f}")
rpt(f"Avg concurrent positions:        {avg_concurrent:.1f}")
rpt(f"Trades skipped (Q1 body):        {skipped_q1}")
rpt(f"Trades skipped (re-entry block): {len(skipped_df)}")
rpt()

# ── B) YEAR BY YEAR ──
rpt("-" * 90)
rpt("B) YEAR-BY-YEAR PERFORMANCE")
rpt("-" * 90)

trades_df['year'] = pd.to_datetime(trades_df['exit_date']).dt.year
rpt(f"{'Year':<6} {'Trades':>7} {'P&L':>14} {'WR%':>7} {'PF':>7} {'MaxDD':>14}")
rpt("-" * 60)

for yr in sorted(trades_df['year'].unique()):
    yt = trades_df[trades_df['year'] == yr].sort_values('exit_date')
    yn = len(yt)
    ypnl = yt['net_pnl'].sum()
    ywr = (yt['net_pnl'] > 0).sum() / yn * 100 if yn > 0 else 0
    yw = yt[yt['net_pnl'] > 0]['net_pnl'].sum()
    yl = yt[yt['net_pnl'] <= 0]['net_pnl'].sum()
    ypf = abs(yw / yl) if yl != 0 else float('inf')
    ycum = yt['net_pnl'].cumsum()
    ydd = (ycum - ycum.cummax()).min()
    rpt(f"{yr:<6} {yn:>7} ${ypnl:>12,.0f} {ywr:>6.1f}% {ypf:>7.2f} ${ydd:>12,.0f}")

rpt()

# ── C) MONTHLY P&L TABLE ──
rpt("-" * 90)
rpt("C) MONTHLY P&L TABLE")
rpt("-" * 90)

trades_df['month'] = pd.to_datetime(trades_df['exit_date']).dt.month
monthly = trades_df.pivot_table(index='year', columns='month', values='net_pnl', aggfunc='sum', fill_value=0)
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
hdr = f"{'Year':<6}" + "".join(f"{m:>10}" for m in months) + f"{'TOTAL':>12}"
rpt(hdr)
rpt("-" * len(hdr))
for yr in sorted(monthly.index):
    row_str = f"{yr:<6}"
    yr_total = 0
    for m in range(1, 13):
        v = monthly.loc[yr, m] if m in monthly.columns else 0
        yr_total += v
        if v == 0:
            row_str += f"{'--':>10}"
        else:
            row_str += f"{v/1000:>9.0f}K"
    row_str += f"{yr_total/1e6:>11.1f}M"
    rpt(row_str)
rpt()

# ── D) TOP 10 WINNERS and LOSERS ──
rpt("-" * 90)
rpt("D) TOP 10 WINNERS")
rpt("-" * 90)
top10w = trades_df.nlargest(10, 'net_pnl')
rpt(f"{'Symbol':<8} {'Signal Date':<12} {'Entry':>8} {'Exit':>8} {'P&L':>12} {'Ret%':>8} {'Type':<6} {'Size':>10}")
for _, t in top10w.iterrows():
    sd = pd.to_datetime(t['signal_date']).strftime('%Y-%m-%d')
    rpt(f"{t['symbol']:<8} {sd:<12} {t['entry_price']:>8.2f} {t['exit_price']:>8.2f} ${t['net_pnl']:>10,.0f} {t['return_pct']:>7.2f}% {t['exit_type']:<6} ${t['position_size']:>8,}")

rpt()
rpt("-" * 90)
rpt("D) TOP 10 LOSERS")
rpt("-" * 90)
top10l = trades_df.nsmallest(10, 'net_pnl')
rpt(f"{'Symbol':<8} {'Signal Date':<12} {'Entry':>8} {'Exit':>8} {'P&L':>12} {'Ret%':>8} {'Type':<6} {'Size':>10}")
for _, t in top10l.iterrows():
    sd = pd.to_datetime(t['signal_date']).strftime('%Y-%m-%d')
    rpt(f"{t['symbol']:<8} {sd:<12} {t['entry_price']:>8.2f} {t['exit_price']:>8.2f} ${t['net_pnl']:>10,.0f} {t['return_pct']:>7.2f}% {t['exit_type']:<6} ${t['position_size']:>8,}")
rpt()

# ── E) EXIT TYPE BREAKDOWN ──
rpt("-" * 90)
rpt("E) EXIT TYPE BREAKDOWN")
rpt("-" * 90)
for et in sorted(trades_df['exit_type'].unique()):
    sub = trades_df[trades_df['exit_type'] == et]
    rpt(f"  {et:<10}: {len(sub):>5} trades, avg P&L ${sub['net_pnl'].mean():>10,.0f}, "
        f"WR {(sub['net_pnl']>0).mean()*100:.1f}%, total ${sub['net_pnl'].sum():>12,.0f}")
rpt()

# ── F) COMPARISON TABLE ──
rpt("-" * 90)
rpt("F) COMPARISON TABLE")
rpt("-" * 90)
rpt(f"{'Strategy':<35} {'Trades':>7} {'Total P&L':>14} {'WR%':>7} {'MaxDD':>14} {'PF':>7} {'Sharpe':>7}")
rpt("-" * 95)
rpt(f"{'Baseline flat $1M no filter':<35} {'2,875':>7} {'$37,400,000':>14} {'51.2%':>7} {'$-13,300,000':>14} {'1.31':>7} {'--':>7}")
rpt(f"{'Option D with re-entry':<35} {'2,308':>7} {'$92,900,000':>14} {'54.8%':>7} {'$-6,400,000':>14} {'1.97':>7} {'0.60':>7}")
rpt(f"{'Option D NO re-entry':<35} {n_trades:>7,} ${total_pnl:>13,.0f} {wr:>6.1f}% ${max_dd:>13,.0f} {pf:>7.2f} {sharpe:>7.2f}")
rpt()

# ── G) BLOCKED TRADES ANALYSIS ──
rpt("-" * 90)
rpt("G) BLOCKED TRADES — RE-ENTRY RULE")
rpt("-" * 90)
rpt(f"Total signals blocked by no-re-entry: {len(skipped_df)}")
rpt()

if len(skipped_df) > 0:
    skipped_df['signal_date'] = pd.to_datetime(skipped_df['signal_date'])

    matched = skipped_df.merge(
        orig_trades[['symbol', 'signal_date', 'net_pnl', 'return_pct', 'exit_type', 'entry_price', 'exit_price']],
        on=['symbol', 'signal_date'],
        how='left',
        suffixes=('', '_orig')
    )

    matched_found = matched.dropna(subset=['net_pnl'])
    rpt(f"Matched to original Option D trades: {len(matched_found)} / {len(skipped_df)}")
    if len(matched_found) > 0:
        rpt(f"Total P&L of blocked trades (had they run): ${matched_found['net_pnl'].sum():,.0f}")
        blocked_winners = (matched_found['net_pnl'] > 0).sum()
        blocked_losers = (matched_found['net_pnl'] <= 0).sum()
        rpt(f"Blocked winners / losers: {blocked_winners} / {blocked_losers}")
        rpt(f"Blocked WR: {blocked_winners/len(matched_found)*100:.1f}%")
        rpt(f"Avg P&L of blocked trades: ${matched_found['net_pnl'].mean():,.0f}")
        rpt()
        rpt(f"{'Symbol':<8} {'Signal Date':<12} {'Would-Be P&L':>14} {'Ret%':>8} {'Exit':>6}")
        rpt("-" * 55)
        for _, b in matched_found.sort_values('signal_date').iterrows():
            sd = b['signal_date'].strftime('%Y-%m-%d')
            pnl_str = f"${b['net_pnl']:>12,.0f}" if not pd.isna(b['net_pnl']) else "N/A"
            ret_str = f"{b['return_pct']:>7.2f}%" if not pd.isna(b['return_pct']) else "N/A"
            et_str = b['exit_type'] if not pd.isna(b['exit_type']) else "N/A"
            rpt(f"{b['symbol']:<8} {sd:<12} {pnl_str:>14} {ret_str:>8} {et_str:>6}")

    unmatched = matched[matched['net_pnl'].isna()]
    if len(unmatched) > 0:
        rpt()
        rpt(f"Blocked signals NOT found in original trade log (were Q1-skipped or different signal): {len(unmatched)}")
        for _, u in unmatched.iterrows():
            sd = u['signal_date'].strftime('%Y-%m-%d')
            rpt(f"  {u['symbol']:<8} {sd}")

rpt()

# ── H) CUMULATIVE P&L ON SPECIFIC DATES ──
rpt("-" * 90)
rpt("H) CUMULATIVE P&L ON SPECIFIC DATES")
rpt("-" * 90)

# Build cumulative P&L by exit_date
daily_pnl_cum = trades_df.groupby('exit_date_dt')['net_pnl'].sum().sort_index().cumsum()

check_dates = [
    '2020-03-31', '2020-06-11', '2020-06-30', '2020-12-31',
    '2021-06-30', '2021-12-31', '2022-06-30', '2022-12-31',
]
for y in range(2020, 2027):
    d = f"{y}-12-31"
    if d not in check_dates:
        check_dates.append(d)
check_dates = sorted(set(check_dates))

rpt(f"{'Date':<14} {'Cumulative P&L':>18} {'Trades to Date':>16}")
rpt("-" * 50)

for ds in check_dates:
    dt = pd.to_datetime(ds)
    mask = daily_pnl_cum.index <= dt
    if mask.any():
        cum = daily_pnl_cum[mask].iloc[-1]
        ntrades = (trades_df['exit_date_dt'] <= dt).sum()
    else:
        cum = 0
        ntrades = 0
    rpt(f"{ds:<14} ${cum:>16,.0f} {ntrades:>16}")

rpt()
rpt("=" * 90)
rpt("END OF REPORT")
rpt("=" * 90)

# Write report
report_text = "\n".join(report_lines)
with open(OUT_DIR / "backtest_optionD_noReentry_report.txt", 'w') as f:
    f.write(report_text)

print("\nReport written to:")
print(f"  {OUT_DIR / 'backtest_optionD_noReentry_trades.csv'}")
print(f"  {OUT_DIR / 'backtest_optionD_noReentry_report.txt'}")
print(f"\nTotal P&L: ${total_pnl:,.0f}")
print(f"Trades: {n_trades}, Blocked: {len(skipped_df)}, Q1 skipped: {skipped_q1}")
