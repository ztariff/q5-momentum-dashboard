"""
Enhanced options bars downloader.
Writes to options_bars_enhanced.csv.
Does NOT touch options_bars_final.csv or the running process (PID 532171).

Price window priority:
  1. 9:30-9:36 ET  -> price_window = "930-936"
  2. 9:36-10:00 ET -> price_window = "936-1000"
  3. 10:00-11:00 ET -> price_window = "1000-1100"
  4. First bar of day -> price_window = "first_bar"
  5. No data at all -> price_window = ""
"""

import os, sys, time, math, csv, glob, logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRADES_CSV  = '/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv'
OUT_CSV     = '/home/ubuntu/daily_data/data/trade_options/options_bars_enhanced.csv'
LOG_FILE    = '/home/ubuntu/options-data/download_log.txt'
API_KEY     = 'cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF'
SLEEP_SEC   = 0.12   # slightly over 0.1 for safety

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELTA_Z = {
    80:  0.842,
    70:  0.524,
    60:  0.253,
    50:  0.0,
    40: -0.253,
    30: -0.524,
    20: -0.842,
    10: -1.282,
}

COLUMNS = [
    'symbol', 'date', 'date_type',
    'spot_unadjusted',
    'delta_target', 'option_ticker', 'strike', 'expiration', 'dte',
    'avg_price', 'n_bars', 'total_volume', 'price_window',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_dst(dt: datetime) -> bool:
    """Return True if Eastern time is in DST on date dt."""
    yr = dt.year
    mar = datetime(yr, 3, 8) + timedelta(days=(6 - datetime(yr, 3, 8).weekday()) % 7)
    nov = datetime(yr, 11, 1) + timedelta(days=(6 - datetime(yr, 11, 1).weekday()) % 7)
    return mar <= dt < nov


def et_offset(dt: datetime) -> int:
    """Hours to add to ET to get UTC (4 during DST, 5 otherwise)."""
    return 4 if is_dst(dt) else 5


def monthly_expiry(date_str: str) -> datetime:
    """3rd Friday of the month ~30 days out."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    target = dt + timedelta(days=30)
    yr, mo = target.year, target.month
    first = datetime(yr, mo, 1)
    fri1  = first + timedelta(days=(4 - first.weekday()) % 7)
    fri3  = fri1  + timedelta(days=14)
    if (fri3 - dt).days < 20:
        mo += 1
        if mo > 12:
            mo, yr = 1, yr + 1
        first = datetime(yr, mo, 1)
        fri1  = first + timedelta(days=(4 - first.weekday()) % 7)
        fri3  = fri1  + timedelta(days=14)
    return fri3


def round_strike(price: float, spot: float) -> float:
    if spot < 25:
        inc = 0.5
    elif spot < 200:
        inc = 1.0
    elif spot < 500:
        inc = 2.5
    else:
        inc = 5.0
    return round(price / inc) * inc


def get_unadj_close(symbol: str, date_str: str):
    url = (
        f'https://api.polygon.io/v1/open-close/{symbol}/{date_str}'
        f'?adjusted=false&apiKey={API_KEY}'
    )
    try:
        r = requests.get(url, timeout=10).json()
        if r.get('status') == 'OK' and r.get('close'):
            return float(r['close'])
    except Exception as e:
        log.warning(f'spot fetch error {symbol} {date_str}: {e}')
    return None


def _ms_window(dt: datetime, utc_off: int, h_start: int, m_start: int, h_end: int, m_end: int):
    """Return (start_ms, end_ms) for an ET time window in UTC milliseconds."""
    base = datetime(dt.year, dt.month, dt.day)
    start_ms = int((base.replace(hour=h_start + utc_off, minute=m_start)).timestamp() * 1000)
    end_ms   = int((base.replace(hour=h_end   + utc_off, minute=m_end  )).timestamp() * 1000)
    return start_ms, end_ms


def _vwap(bars):
    """Volume-weighted average close price from a list of bar dicts."""
    total_vol = sum(b.get('v', 0) for b in bars)
    if total_vol > 0:
        return sum(b['c'] * b.get('v', 1) for b in bars) / total_vol
    return float(np.mean([b['c'] for b in bars]))


def get_bars(ticker: str, date_str: str, utc_off: int):
    """
    Fetch 1-min bars and apply the priority window logic.

    Returns (avg_price, n_bars, total_volume, price_window)
    where avg_price is None when no bars exist at all.
    """
    url = (
        f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute'
        f'/{date_str}/{date_str}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}'
    )
    try:
        r = requests.get(url, timeout=10).json()
    except Exception as e:
        log.warning(f'bars fetch error {ticker} {date_str}: {e}')
        return None, 0, 0, ''

    bars = r.get('results', [])
    if not bars:
        return None, 0, 0, ''

    dt = datetime.strptime(date_str, '%Y-%m-%d')

    # Window definitions: (label, h_start, m_start, h_end, m_end)
    windows = [
        ('930-936',   9,  30, 9,  36),
        ('936-1000',  9,  36, 10,  0),
        ('1000-1100', 10,  0, 11,  0),
    ]

    for label, hs, ms, he, me in windows:
        start_ms, end_ms = _ms_window(dt, utc_off, hs, ms, he, me)
        window_bars = [b for b in bars if start_ms <= b['t'] < end_ms]
        if window_bars:
            avg_p     = round(_vwap(window_bars), 4)
            n         = len(window_bars)
            total_vol = sum(b.get('v', 0) for b in window_bars)
            return avg_p, n, total_vol, label

    # Fallback: first bar of the day
    b0        = bars[0]
    avg_p     = round(b0['c'], 4)
    total_vol = int(b0.get('v', 0))
    return avg_p, 1, total_vol, 'first_bar'


# ---------------------------------------------------------------------------
# Load rvol lookup
# ---------------------------------------------------------------------------
def load_rvol():
    rvol = {}
    for f in glob.glob('/home/ubuntu/daily_data/data/*_enriched.csv'):
        sym = os.path.basename(f).replace('_enriched.csv', '')
        try:
            df = pd.read_csv(f, usecols=['date', 'rvol_20d'])
            for _, row in df.iterrows():
                if pd.notna(row['rvol_20d']):
                    rvol[(sym, str(row['date']))] = row['rvol_20d']
        except Exception:
            pass
    return rvol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info('=== Enhanced options bars download started ===')

    # Load trades
    trades = pd.read_csv(TRADES_CSV)
    trades = trades[trades['entry_date'] >= '2022-01-01'].reset_index(drop=True)
    log.info(f'Trades (2022+): {len(trades)}')

    # Build unique task list
    seen = set()
    tasks = []
    for _, t in trades.iterrows():
        for dt_type, dcol in [('entry', 'entry_date'), ('exit', 'exit_date')]:
            k = (t['symbol'], str(t[dcol]), dt_type)
            if k not in seen:
                seen.add(k)
                tasks.append({'symbol': t['symbol'], 'date': str(t[dcol]), 'date_type': dt_type})
    log.info(f'Unique tasks: {len(tasks)}')

    # Load rvol
    rvol_lookup = load_rvol()
    log.info(f'Rvol entries: {len(rvol_lookup):,}')

    # Resume: load already-completed (symbol, date, date_type) groups
    done_keys = set()
    file_exists = os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 100
    if file_exists:
        try:
            existing = pd.read_csv(OUT_CSV, usecols=['symbol', 'date', 'date_type'])
            for _, r in existing.iterrows():
                done_keys.add((str(r['symbol']), str(r['date']), str(r['date_type'])))
            log.info(f'Resuming — already done: {len(done_keys)} (symbol,date,date_type) combos')
        except Exception as e:
            log.warning(f'Could not read existing output: {e}')
            file_exists = False

    # Open output file
    outf   = open(OUT_CSV, 'a', newline='')
    writer = csv.writer(outf)
    if not file_exists:
        writer.writerow(COLUMNS)

    ok = 0; nodata = 0; skipped = 0

    for i, task in enumerate(tasks):
        sym      = task['symbol']
        date_str = task['date']
        dt_type  = task['date_type']

        if (sym, date_str, dt_type) in done_keys:
            skipped += 1
            continue

        if (i + 1) % 50 == 0:
            log.info(f'[{i+1}/{len(tasks)}] {sym} {date_str} {dt_type} | ok={ok} nodata={nodata} skip={skipped}')
            outf.flush()

        # Spot price
        spot = get_unadj_close(sym, date_str)
        time.sleep(SLEEP_SEC)

        if not spot or spot <= 0:
            log.debug(f'No spot for {sym} {date_str}')
            for d in DELTA_Z:
                writer.writerow([sym, date_str, dt_type, '', d, '', '', '', '', '', 0, 0, ''])
            nodata += 1
            continue

        dt      = datetime.strptime(date_str, '%Y-%m-%d')
        utc_off = et_offset(dt)
        vol     = rvol_lookup.get((sym, date_str), 30)
        expiry  = monthly_expiry(date_str)
        dte     = (expiry - dt).days
        sig_sqrt_t = (max(vol, 15) / 100.0) * math.sqrt(max(dte, 1) / 365.0)

        got_any = False
        for delta, z in DELTA_Z.items():
            strike = round_strike(spot * math.exp(-z * sig_sqrt_t), spot)
            ticker = f'O:{sym}{expiry.strftime("%y%m%d")}C{int(strike * 1000):08d}'

            avg_p, nb, tv, pw = get_bars(ticker, date_str, utc_off)
            time.sleep(SLEEP_SEC)

            if avg_p is not None:
                got_any = True

            writer.writerow([
                sym, date_str, dt_type,
                spot, delta, ticker, strike,
                expiry.strftime('%Y-%m-%d'), dte,
                avg_p if avg_p is not None else '',
                nb, tv, pw,
            ])

        if got_any:
            ok += 1
        else:
            nodata += 1

    outf.flush()
    outf.close()
    log.info(f'=== DONE: ok={ok} nodata={nodata} skipped={skipped} ===')
    log.info(f'Output: {OUT_CSV}')


if __name__ == '__main__':
    main()
