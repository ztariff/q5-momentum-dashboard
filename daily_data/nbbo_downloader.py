#!/usr/bin/env python3
"""OPTIONS NBBO DOWNLOADER - AUTHORITATIVE VERSION"""
import os, csv, math, time, fcntl, shutil, calendar, logging, requests, pandas as pd
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

API_KEY      = "cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF"
TRADES_CSV   = "/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv"
ENRICHED_DIR = "/home/ubuntu/daily_data/data"
STAGING      = "/tmp/nbbo_staging.csv"
OUTPUT       = "/home/ubuntu/daily_data/data/trade_options/trade_options_nbbo.csv"
LOCK_F       = "/tmp/nbbo_main.lock"
LOG_F        = "/home/ubuntu/options-data/download_log.txt"
FLUSH_N      = 80

DELTAS = [80, 70, 60, 50, 40, 30, 20, 10]
ZVALS  = {80:0.842, 70:0.524, 60:0.253, 50:0.0, 40:-0.253, 30:-0.524, 20:-0.842, 10:-1.282}
COLS   = ["symbol","date","date_type","spot_price","rvol_20d","delta_target",
          "option_ticker","strike","expiration","dte",
          "avg_bid","avg_ask","avg_mid","spread","n_quotes"]

os.makedirs(os.path.dirname(LOG_F), exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_F), logging.StreamHandler()])
log = logging.getLogger(__name__)

def acquire_lock():
    lf = open(LOCK_F, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lf.write(str(os.getpid())); lf.flush(); return lf
    except IOError:
        lf.close(); return None

def third_friday(y, m):
    c = calendar.monthcalendar(y, m)
    fr = [w[4] for w in c if w[4]]
    return date(y, m, fr[2])

def nearest_expiry(d):
    tgt = d + timedelta(30); cands = []
    for dm in [-1, 0, 1, 2]:
        m = tgt.month + dm; y = tgt.year
        while m > 12: m -= 12; y += 1
        while m < 1:  m += 12; y -= 1
        try:
            e = third_friday(y, m)
            if e > d + timedelta(6): cands.append(e)
        except: pass
    return min(cands, key=lambda e: abs((e-d).days - 30))

def is_edt(d):
    def nth_sun(y, m, n):
        c = calendar.monthcalendar(y, m); s = [w[6] for w in c if w[6]]
        return date(y, m, s[n-1])
    return nth_sun(d.year,3,2) <= d < nth_sun(d.year,11,1)

def open_ns(d):
    off = 4 if is_edt(d) else 5
    s = datetime(d.year,d.month,d.day, 9+off,31,0, tzinfo=timezone.utc)
    e = datetime(d.year,d.month,d.day, 9+off,36,0, tzinfo=timezone.utc)
    return int(s.timestamp()*1e9), int(e.timestamp()*1e9)

def round_strike(raw, spot):
    inc = 0.5 if spot<25 else (1.0 if spot<200 else (2.5 if spot<500 else 5.0))
    return round(raw/inc)*inc

def mk_ticker(sym, exp, k):
    return f"O:{sym}{exp.strftime('%y%m%d')}C{round(k*1000):08d}"

def fetch_quotes(tk, s_ns, e_ns):
    url = (f"https://api.polygon.io/v3/quotes/{tk}"
           f"?timestamp.gte={s_ns}&timestamp.lte={e_ns}"
           f"&limit=500&sort=timestamp&apiKey={API_KEY}")
    res = []
    while url:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 429: time.sleep(5); continue
            r.raise_for_status(); data = r.json()
        except Exception as ex:
            log.error(f"Fetch {tk}: {ex}"); return res
        for q in data.get("results", []):
            b, a = q.get("bid_price"), q.get("ask_price")
            if b is not None and a is not None and b >= 0 and a >= 0:
                res.append((float(b), float(a)))
        url = data.get("next_url")
        if url: url += f"&apiKey={API_KEY}"; time.sleep(0.08)
    return res

def stats(qs):
    if not qs: return None, None, None, None
    bs = [q[0] for q in qs]; aks = [q[1] for q in qs]
    ab = sum(bs)/len(bs); aa = sum(aks)/len(aks)
    return ab, aa, (ab+aa)/2, aa-ab

def load_rvol(sym, dt_str):
    p = Path(ENRICHED_DIR) / f"{sym}_enriched.csv"
    if not p.exists(): return None
    try:
        df = pd.read_csv(p, usecols=["date","close","rvol_20d"]).dropna()
        df["date"] = df["date"].astype(str)
        r = df[df["date"] <= dt_str].sort_values("date").tail(1)
        if r.empty: return None
        return float(r["rvol_20d"].iloc[0]) / 100.0
    except: return None

def load_done():
    done = set()
    if not Path(STAGING).exists(): return done
    try:
        with open(STAGING) as f:
            rd = csv.DictReader(f)
            if rd.fieldnames and "spot_price" in rd.fieldnames and "rvol_20d" in rd.fieldnames:
                for row in rd:
                    done.add((row["symbol"], row["date"], row["date_type"], str(row["delta_target"])))
            else:
                log.info("Staging wrong schema, wiping")
                Path(STAGING).unlink()
    except Exception as ex:
        log.warning(f"load_done: {ex}")
    return done

def sync():
    try: shutil.copy2(STAGING, OUTPUT)
    except Exception as ex: log.error(f"sync fail: {ex}")

def main():
    lk = acquire_lock()
    if lk is None:
        log.warning("Lock held - exiting"); return
    try:
        run()
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
        try: os.remove(LOCK_F)
        except: pass

def run():
    log.info(f"=== NBBO start PID={os.getpid()} ===")
    trades = pd.read_csv(TRADES_CSV)
    log.info(f"Trades: {len(trades)}")
    work = []
    for _, r in trades.iterrows():
        sym = r["symbol"]
        work.append((sym, str(r["entry_date"]), "entry", float(r["entry_price"])))
        work.append((sym, str(r["exit_date"]),  "exit",  float(r["exit_price"])))
    log.info(f"Work items: {len(work)}")
    done = load_done()
    log.info(f"Already done: {len(done)}")
    need_hdr = not Path(STAGING).exists()
    fh = open(STAGING, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    if need_hdr: wr.writeheader(); fh.flush()
    rvol_cache = {}; buf = []; total = api_n = empty_n = 0
    for idx, (sym, dt_str, dtype, spot) in enumerate(work):
        need = [d for d in DELTAS if (sym, dt_str, dtype, str(d)) not in done]
        if not need:
            total += len(DELTAS); continue
        if sym not in rvol_cache: rvol_cache[sym] = {}
        if dt_str not in rvol_cache[sym]:
            rv = load_rvol(sym, dt_str)
            rvol_cache[sym][dt_str] = rv if rv else 0.20
        rvol = rvol_cache[sym][dt_str]
        try:
            td = date.fromisoformat(dt_str)
        except:
            log.error(f"bad date {dt_str}"); continue
        exp = nearest_expiry(td); dte = (exp - td).days; T = dte / 365.0
        s_ns, e_ns = open_ns(td)
        for delta in need:
            z  = ZVALS[delta]
            k  = round_strike(spot * math.exp(-z * rvol * math.sqrt(T)), spot)
            tk = mk_ticker(sym, exp, k)
            qs = fetch_quotes(tk, s_ns, e_ns)
            api_n += 1
            time.sleep(0.08 + 0.04 * (api_n % 3 == 0))
            ab, aa, am, sp = stats(qs)
            buf.append({
                "symbol": sym, "date": dt_str, "date_type": dtype,
                "spot_price": round(spot,4), "rvol_20d": round(rvol*100,4),
                "delta_target": delta, "option_ticker": tk, "strike": k,
                "expiration": exp.isoformat(), "dte": dte,
                "avg_bid":  round(ab,4) if ab is not None else "",
                "avg_ask":  round(aa,4) if aa is not None else "",
                "avg_mid":  round(am,4) if am is not None else "",
                "spread":   round(sp,4) if sp is not None else "",
                "n_quotes": len(qs),
            })
            done.add((sym, dt_str, dtype, str(delta)))
            total += 1
            if not qs: empty_n += 1
        if len(buf) >= FLUSH_N:
            wr.writerows(buf); fh.flush(); buf = []; sync()
            log.info(f"item {idx+1}/{len(work)} rows={total} api={api_n} empty={empty_n}")
    if buf: wr.writerows(buf); fh.flush()
    fh.close(); sync()
    log.info(f"=== DONE rows={total} api={api_n} empty={empty_n} ===")

if __name__ == "__main__":
    main()
