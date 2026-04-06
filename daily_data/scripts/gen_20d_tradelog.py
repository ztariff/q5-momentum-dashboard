#!/usr/bin/env python3
"""Generate the clean 20d no-stop trade log for options download."""

import os, glob, numpy as np, pandas as pd

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT = "/home/ubuntu/daily_data/analysis_results/trade1_20d_trades.csv"

files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
all_dfs = []
for f in files:
    sym = os.path.basename(f).replace("_enriched.csv", "")
    df = pd.read_csv(f, usecols=["date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = sym
    c = df["close"]
    sma50 = c.rolling(50).mean()
    raw_slope = 100 * (sma50 / sma50.shift(10) - 1)
    roll_m = raw_slope.rolling(252, min_periods=60).mean()
    roll_s = raw_slope.rolling(252, min_periods=60).std()
    df["z_signal"] = (raw_slope - roll_m) / roll_s.replace(0, np.nan)
    all_dfs.append(df)
pooled = pd.concat(all_dfs, ignore_index=True)

pooled["quintile"] = np.nan
for date, ddf in pooled.groupby("date"):
    valid = ddf["z_signal"].dropna()
    if len(valid) < 20: continue
    try:
        q = pd.qcut(valid, 5, labels=[1,2,3,4,5], duplicates="drop")
        pooled.loc[q.index, "quintile"] = q.values
    except: continue

pooled = pooled.sort_values(["symbol", "date"]).reset_index(drop=True)
pooled["prev_q"] = pooled.groupby("symbol")["quintile"].shift(1)
pooled["entered_q5"] = (pooled["quintile"] == 5) & (pooled["prev_q"] != 5)

price_lookup = {}
for _, row in pooled.iterrows():
    price_lookup[(row["symbol"], row["date"])] = {
        "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"]
    }

entries = pooled[pooled["entered_q5"] & (pooled["date"].dt.year >= 2020)]
entry_by_date = entries.groupby("date").apply(lambda x: list(zip(x["symbol"], x["z_signal"]))).to_dict()
all_dates = sorted(pooled[pooled["date"].dt.year >= 2020]["date"].unique())

positions = []
trades = []
for i, today in enumerate(all_dates):
    yesterday = all_dates[i-1] if i > 0 else None
    if yesterday and yesterday in entry_by_date:
        for sym, z in entry_by_date[yesterday]:
            prices = price_lookup.get((sym, today))
            if prices is None or prices["open"] <= 0 or pd.isna(prices["open"]): continue
            ep = prices["open"]
            shares = int(1_000_000 / ep)
            positions.append({"symbol": sym, "signal_date": str(yesterday)[:10], "entry_date": today,
                            "entry_price": ep, "shares": shares, "days_held": 0, "z": z})
    surviving = []
    for pos in positions:
        pos["days_held"] += 1
        prices = price_lookup.get((pos["symbol"], today))
        if prices is None:
            surviving.append(pos)
            continue
        if pos["days_held"] >= 20:
            xp = prices["close"]
            gross = pos["shares"] * (xp - pos["entry_price"])
            cost = pos["shares"] * (pos["entry_price"] + xp) * 5 / 10000
            net = gross - cost
            trades.append({
                "symbol": pos["symbol"],
                "signal_date": pos["signal_date"],
                "entry_date": str(pos["entry_date"])[:10],
                "exit_date": str(today)[:10],
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(xp, 4),
                "shares": pos["shares"],
                "gross_pnl": round(gross, 2),
                "net_pnl": round(net, 2),
                "return_pct": round((xp / pos["entry_price"] - 1) * 100, 2),
                "days_held": pos["days_held"],
                "z_score": round(pos["z"], 4),
            })
        else:
            surviving.append(pos)
    positions = surviving

tdf = pd.DataFrame(trades)
tdf.to_csv(OUT, index=False)
print(f"Saved {len(tdf)} trades to {OUT}")
print(f"Total net P&L: ${tdf['net_pnl'].sum():+,.0f}")
print(f"Date range: {tdf['entry_date'].min()} to {tdf['exit_date'].max()}")
