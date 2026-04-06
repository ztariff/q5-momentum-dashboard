import pandas as pd
import numpy as np
import math
from collections import defaultdict

# ── Load data ──
opts = pd.read_csv('/home/ubuntu/daily_data/data/trade_options/hold_period_options.csv')
trades = pd.read_csv('/home/ubuntu/daily_data/analysis_results/backtest_optionD_noReentry_trades.csv')

# ── Filter to 30d_1m only ──
opts_30 = opts[opts['option_type'] == '30d_1m'].copy()
opts_30['bar_date'] = pd.to_datetime(opts_30['bar_date'])
opts_30['trade_entry_date'] = pd.to_datetime(opts_30['trade_entry_date'])
# Drop rows with no bar_date (no data contracts)
opts_30 = opts_30.dropna(subset=['bar_date'])

trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['exit_date'] = pd.to_datetime(trades['exit_date'])

# ── Option D sizing ──
size_map = {'Q2': 50_000, 'Q3': 100_000, 'Q4': 150_000, 'Q5': 200_000}

# ── Group option bars by (symbol, entry_date), sorted by bar_date ──
opts_30_sorted = opts_30.sort_values(['trade_symbol', 'trade_entry_date', 'bar_date'])
grouped = {}
for (sym, edate), grp in opts_30_sorted.groupby(['trade_symbol', 'trade_entry_date']):
    grouped[(sym, edate)] = grp.reset_index(drop=True)

# ── For each trade, compute entry price and all possible exit prices ──
trade_data = []  # list of dicts: one per trade with entry info + bars

for _, row in trades.iterrows():
    sym = row['symbol']
    edate = row['entry_date']
    quintile = row['body_quintile']
    exit_type = row['exit_type']

    if quintile not in size_map:
        continue

    key = (sym, edate)
    if key not in grouped:
        continue

    bars = grouped[key]
    if len(bars) == 0:
        continue

    # Entry price: bar_date == entry_date
    entry_bars = bars[bars['bar_date'] == edate]
    if len(entry_bars) == 0:
        continue

    eb = entry_bars.iloc[0]
    # morning_price > day_open > day_close
    entry_price = eb['morning_price']
    if pd.isna(entry_price) or entry_price <= 0:
        entry_price = eb['day_open']
    if pd.isna(entry_price) or entry_price <= 0:
        entry_price = eb['day_close']
    if pd.isna(entry_price) or entry_price <= 0:
        continue

    pos_size = size_map[quintile]
    contracts = math.floor(pos_size / (entry_price * 100))
    if contracts < 1:
        continue

    # All bars sorted by bar_date ascending (already sorted)
    bar_list = bars[['bar_date', 'day_close', 'day_high']].values.tolist()

    year = edate.year

    trade_data.append({
        'symbol': sym,
        'entry_date': edate,
        'entry_price': entry_price,
        'contracts': contracts,
        'pos_size': pos_size,
        'quintile': quintile,
        'exit_type': exit_type,
        'year': year,
        'bars': bar_list  # list of [bar_date, day_close, day_high]
    })

print(f"Total trades with valid 30d_1m option data: {len(trade_data)}")

# ── Compute P&L for each hold time 1-20, both close and high exit ──
max_hold = 20

# Results storage
results_close = defaultdict(list)  # hold_day -> list of trade dicts
results_high = defaultdict(list)

for td in trade_data:
    entry_price = td['entry_price']
    contracts = td['contracts']
    bars = td['bars']

    for n in range(1, max_hold + 1):
        # N=1 means exit on bar index 0 (entry day itself)
        bar_idx = n - 1
        if bar_idx >= len(bars):
            break

        exit_close = bars[bar_idx][1]  # day_close
        exit_high = bars[bar_idx][2]   # day_high

        if pd.isna(exit_close) or pd.isna(exit_high):
            continue

        # Transaction costs: 5 bps each way on notional
        notional_entry = contracts * entry_price * 100
        notional_exit_close = contracts * exit_close * 100
        notional_exit_high = contracts * exit_high * 100

        cost_close = notional_entry * 0.0005 + notional_exit_close * 0.0005
        cost_high = notional_entry * 0.0005 + notional_exit_high * 0.0005

        gross_pnl_close = contracts * (exit_close - entry_price) * 100
        gross_pnl_high = contracts * (exit_high - entry_price) * 100

        net_pnl_close = gross_pnl_close - cost_close
        net_pnl_high = gross_pnl_high - cost_high

        ret_close = (exit_close - entry_price) / entry_price
        ret_high = (exit_high - entry_price) / entry_price

        rec = {
            'symbol': td['symbol'],
            'entry_date': td['entry_date'],
            'contracts': contracts,
            'entry_price': entry_price,
            'exit_price': exit_close,
            'net_pnl': net_pnl_close,
            'return_pct': ret_close * 100,
            'year': td['year'],
            'exit_type': td['exit_type'],
            'quintile': td['quintile'],
            'pos_size': td['pos_size'],
        }
        results_close[n].append(rec)

        rec_high = dict(rec)
        rec_high['exit_price'] = exit_high
        rec_high['net_pnl'] = net_pnl_high
        rec_high['return_pct'] = ret_high * 100
        results_high[n].append(rec_high)


# ── Helper functions ──
def compute_stats(records):
    if not records:
        return None
    pnls = [r['net_pnl'] for r in records]
    rets = [r['return_pct'] for r in records]
    n = len(records)
    total = sum(pnls)
    avg = total / n
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown from cumulative PnL
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    avg_ret = np.mean(rets)
    med_ret = np.median(rets)

    # Sharpe (daily-ish): mean/std of per-trade returns
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = avg_ret / std_ret if std_ret > 0 else 0.0

    return {
        'trades': n,
        'total_pnl': total,
        'avg_pnl': avg,
        'wr': wr,
        'pf': pf,
        'max_dd': max_dd,
        'avg_ret': avg_ret,
        'med_ret': med_ret,
        'sharpe': sharpe,
    }

def fmt(v, fmt_str='.2f'):
    return f"{v:{fmt_str}}"

def fmt_dollar(v):
    if v >= 0:
        return f"${v:,.0f}"
    else:
        return f"-${abs(v):,.0f}"

# ── Section A: Main table (close exit) ──
lines = []
lines.append("=" * 120)
lines.append("OPTION HOLD TIME ANALYSIS: 30-Delta 1-Month (30d_1m)")
lines.append("Testing hold times from 1 to 20 trading days")
lines.append("=" * 120)
lines.append("")

lines.append("SECTION A: EXIT AT DAY CLOSE")
lines.append("-" * 120)
header = f"{'Hold':>5} | {'Trades':>6} | {'Total PnL':>14} | {'Avg PnL':>11} | {'WR%':>6} | {'PF':>6} | {'MaxDD':>14} | {'Avg Ret%':>9} | {'Med Ret%':>9} | {'Sharpe':>7}"
lines.append(header)
lines.append("-" * 120)

all_stats_close = {}
for n in range(1, max_hold + 1):
    s = compute_stats(results_close[n])
    all_stats_close[n] = s
    if s:
        line = f"{n:>5} | {s['trades']:>6} | {fmt_dollar(s['total_pnl']):>14} | {fmt_dollar(s['avg_pnl']):>11} | {s['wr']:>5.1f}% | {s['pf']:>6.2f} | {fmt_dollar(s['max_dd']):>14} | {s['avg_ret']:>8.2f}% | {s['med_ret']:>8.2f}% | {s['sharpe']:>7.3f}"
        lines.append(line)

# ── Section B: Exit at day high (ceiling) ──
lines.append("")
lines.append("SECTION B: EXIT AT DAY HIGH (CEILING / BEST POSSIBLE INTRADAY EXIT)")
lines.append("-" * 120)
header = f"{'Hold':>5} | {'Trades':>6} | {'Total PnL':>14} | {'Avg PnL':>11} | {'WR%':>6} | {'PF':>6} | {'Avg Ret%':>9} | {'Sharpe':>7}"
lines.append(header)
lines.append("-" * 120)

all_stats_high = {}
for n in range(1, max_hold + 1):
    s = compute_stats(results_high[n])
    all_stats_high[n] = s
    if s:
        line = f"{n:>5} | {s['trades']:>6} | {fmt_dollar(s['total_pnl']):>14} | {fmt_dollar(s['avg_pnl']):>11} | {s['wr']:>5.1f}% | {s['pf']:>6.2f} | {s['avg_ret']:>8.2f}% | {s['sharpe']:>7.3f}"
        lines.append(line)

# ── Section C: Year-by-year for top 3 by total PnL and by Sharpe ──
lines.append("")
lines.append("SECTION C: YEAR-BY-YEAR BREAKDOWN FOR TOP HOLD TIMES")
lines.append("-" * 120)

# Top 3 by total PnL
ranked_pnl = sorted([(n, s['total_pnl']) for n, s in all_stats_close.items() if s], key=lambda x: -x[1])
top3_pnl = [x[0] for x in ranked_pnl[:3]]

# Top 3 by Sharpe
ranked_sharpe = sorted([(n, s['sharpe']) for n, s in all_stats_close.items() if s], key=lambda x: -x[1])
top3_sharpe = [x[0] for x in ranked_sharpe[:3]]

top_hold_days = sorted(set(top3_pnl + top3_sharpe))

lines.append(f"Top 3 by Total PnL: {top3_pnl}")
lines.append(f"Top 3 by Sharpe:    {top3_sharpe}")
lines.append("")

for n in top_hold_days:
    lines.append(f"  Hold = {n} days:")
    records = results_close[n]
    years = sorted(set(r['year'] for r in records))
    header_yr = f"    {'Year':>6} | {'Trades':>6} | {'Total PnL':>14} | {'Avg PnL':>11} | {'WR%':>6} | {'PF':>6} | {'Avg Ret%':>9}"
    lines.append(header_yr)
    lines.append("    " + "-" * 80)
    for yr in years:
        yr_recs = [r for r in records if r['year'] == yr]
        s = compute_stats(yr_recs)
        if s:
            line = f"    {yr:>6} | {s['trades']:>6} | {fmt_dollar(s['total_pnl']):>14} | {fmt_dollar(s['avg_pnl']):>11} | {s['wr']:>5.1f}% | {s['pf']:>6.2f} | {s['avg_ret']:>8.2f}%"
            lines.append(line)
    lines.append("")

# ── Section D: Compare best option hold time to stock strategy ──
lines.append("SECTION D: BEST OPTION HOLD TIME vs STOCK STRATEGY (SAME TRADES)")
lines.append("-" * 120)

best_hold = top3_pnl[0]
lines.append(f"Best option hold time: {best_hold} days")
lines.append("")

# Get the trades that participated in the best hold time
opt_trades_best = results_close[best_hold]
opt_trade_keys = set((r['symbol'], r['entry_date']) for r in opt_trades_best)

# Match to stock trades
stock_matches = []
for _, row in trades.iterrows():
    key = (row['symbol'], row['entry_date'])
    if key in opt_trade_keys:
        stock_matches.append(row)

stock_df = pd.DataFrame(stock_matches)

# Stock stats
stock_pnls = stock_df['net_pnl'].values
stock_rets = stock_df['return_pct'].values

opt_total = sum(r['net_pnl'] for r in opt_trades_best)
opt_avg = opt_total / len(opt_trades_best)
opt_wr = sum(1 for r in opt_trades_best if r['net_pnl'] > 0) / len(opt_trades_best) * 100
opt_avg_ret = np.mean([r['return_pct'] for r in opt_trades_best])

stk_total = stock_pnls.sum()
stk_avg = stock_pnls.mean()
stk_wr = (stock_pnls > 0).sum() / len(stock_pnls) * 100
stk_avg_ret = stock_rets.mean()

opt_label = f'Option (Hold={best_hold})'
lines.append(f"  {'Metric':<20} | {opt_label:>20} | {'Stock Strategy':>20}")
lines.append("  " + "-" * 65)
lines.append(f"  {'Trades':<20} | {len(opt_trades_best):>20} | {len(stock_df):>20}")
lines.append(f"  {'Total PnL':<20} | {fmt_dollar(opt_total):>20} | {fmt_dollar(stk_total):>20}")
lines.append(f"  {'Avg PnL':<20} | {fmt_dollar(opt_avg):>20} | {fmt_dollar(stk_avg):>20}")
lines.append(f"  {'Win Rate':<20} | {opt_wr:>19.1f}% | {stk_wr:>19.1f}%")
lines.append(f"  {'Avg Return %':<20} | {opt_avg_ret:>19.2f}% | {stk_avg_ret:>19.2f}%")

# Also compare best Sharpe hold time if different
best_sharpe_hold = top3_sharpe[0]
if best_sharpe_hold != best_hold:
    lines.append("")
    lines.append(f"  Also: best Sharpe hold time = {best_sharpe_hold} days")
    opt_trades_s = results_close[best_sharpe_hold]
    s = compute_stats(opt_trades_s)
    lines.append(f"    Total PnL: {fmt_dollar(s['total_pnl'])}, Avg Ret: {s['avg_ret']:.2f}%, Sharpe: {s['sharpe']:.3f}")

# ── Section E: Cumulative average return by hold day ──
lines.append("")
lines.append("SECTION E: CUMULATIVE AVERAGE OPTION RETURN % BY HOLD DAY")
lines.append("-" * 120)
lines.append("(Shows where the return curve peaks)")
lines.append("")
lines.append(f"{'Hold':>5} | {'Avg Return %':>12} | {'Bar Chart'}")
lines.append("-" * 60)

max_ret = max(s['avg_ret'] for s in all_stats_close.values() if s)
min_ret = min(s['avg_ret'] for s in all_stats_close.values() if s)
range_ret = max_ret - min_ret if max_ret != min_ret else 1

for n in range(1, max_hold + 1):
    s = all_stats_close[n]
    if s:
        # Scale bar to 50 chars
        bar_len = int((s['avg_ret'] - min_ret) / range_ret * 50) if range_ret > 0 else 0
        bar = '#' * max(bar_len, 0)
        marker = " <-- PEAK" if s['avg_ret'] == max_ret else ""
        lines.append(f"{n:>5} | {s['avg_ret']:>11.2f}% | {bar}{marker}")

# ── Section F: Best hold time per year ──
lines.append("")
lines.append("SECTION F: OPTIMAL HOLD TIME BY YEAR")
lines.append("-" * 120)
lines.append("")

all_years = sorted(set(r['year'] for r in trade_data))
lines.append(f"{'Year':>6} | {'Best Hold (PnL)':>15} | {'Total PnL':>14} | {'Best Hold (Sharpe)':>18} | {'Sharpe':>7}")
lines.append("-" * 70)

for yr in all_years:
    best_n_pnl = None
    best_pnl = -float('inf')
    best_n_sharpe = None
    best_sharpe_val = -float('inf')

    for n in range(1, max_hold + 1):
        yr_recs = [r for r in results_close[n] if r['year'] == yr]
        if not yr_recs:
            continue
        s = compute_stats(yr_recs)
        if s and s['total_pnl'] > best_pnl:
            best_pnl = s['total_pnl']
            best_n_pnl = n
        if s and s['sharpe'] > best_sharpe_val:
            best_sharpe_val = s['sharpe']
            best_n_sharpe = n

    if best_n_pnl:
        lines.append(f"{yr:>6} | {best_n_pnl:>15} | {fmt_dollar(best_pnl):>14} | {best_n_sharpe:>18} | {best_sharpe_val:>7.3f}")

# ── Section G: Best hold time by exit_type ──
lines.append("")
lines.append("SECTION G: OPTIMAL HOLD TIME BY STOCK EXIT TYPE")
lines.append("-" * 120)

for etype in ['STOP', 'TIME']:
    lines.append(f"\n  Stock exit_type = {etype}:")
    header_g = f"  {'Hold':>5} | {'Trades':>6} | {'Total PnL':>14} | {'Avg PnL':>11} | {'WR%':>6} | {'PF':>6} | {'Avg Ret%':>9} | {'Sharpe':>7}"
    lines.append(header_g)
    lines.append("  " + "-" * 90)

    best_n_e = None
    best_pnl_e = -float('inf')

    for n in range(1, max_hold + 1):
        recs = [r for r in results_close[n] if r['exit_type'] == etype]
        if not recs:
            continue
        s = compute_stats(recs)
        if s:
            line = f"  {n:>5} | {s['trades']:>6} | {fmt_dollar(s['total_pnl']):>14} | {fmt_dollar(s['avg_pnl']):>11} | {s['wr']:>5.1f}% | {s['pf']:>6.2f} | {s['avg_ret']:>8.2f}% | {s['sharpe']:>7.3f}"
            lines.append(line)
            if s['total_pnl'] > best_pnl_e:
                best_pnl_e = s['total_pnl']
                best_n_e = n

    lines.append(f"\n  >> Best hold time for {etype} trades: {best_n_e} days (Total PnL: {fmt_dollar(best_pnl_e)})")

# ── Write output ──
lines.append("")
lines.append("=" * 120)
lines.append("END OF ANALYSIS")
lines.append("=" * 120)

output = '\n'.join(lines)
with open('/home/ubuntu/daily_data/analysis_results/option_hold_time_analysis.txt', 'w') as f:
    f.write(output)

print(output)
