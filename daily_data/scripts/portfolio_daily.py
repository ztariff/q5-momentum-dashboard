#!/usr/bin/env python3
"""
Generate a day-by-day portfolio CSV showing:
- Every date
- Which names are long, which are short
- Per-leg and L/S returns
- Cumulative P&L
- Turnover
- Running Sharpe

Strategy: z_sma50_slope_10d, quintile sort, equal-weight, daily rebalance
Period: Full test (2020-2026)
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"

SIGNAL = "z_sma50_slope_10d"
HOLD = 1
N_QUANTILES = 5
COST_BPS_RT = 10  # round-trip


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_enriched.csv")))
    all_dfs = []
    for f in files:
        sym = os.path.basename(f).replace("_enriched.csv", "")
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = sym

        c = df["close"]
        # Forward 1d return
        df["fwd_1d_bps"] = 10000 * (c.shift(-1) / c - 1)
        # Lagged rolling mean for excess
        lagged = df["fwd_1d_bps"].shift(1)
        df["exc_1d_bps"] = df["fwd_1d_bps"] - lagged.rolling(60, min_periods=30).mean()

        # Signal
        sma50 = c.rolling(50).mean()
        raw_slope = 100 * (sma50 / sma50.shift(10) - 1)
        roll_m = raw_slope.rolling(252, min_periods=60).mean()
        roll_s = raw_slope.rolling(252, min_periods=60).std()
        df[SIGNAL] = (raw_slope - roll_m) / roll_s.replace(0, np.nan)

        df["daily_return_pct"] = c.pct_change() * 100

        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def main():
    print("Loading data...")
    pooled = load_data()
    test = pooled[pooled["date"].dt.year >= 2020].copy()
    print(f"Test: {test['date'].min().date()} to {test['date'].max().date()}")
    print(f"Symbols: {test['symbol'].nunique()}\n")

    rows = []
    prev_long_set = set()
    prev_short_set = set()
    cum_gross = 0
    cum_net = 0
    cum_long = 0
    cum_short = 0
    all_net = []

    dates = sorted(test["date"].unique())

    for date in dates:
        ddf = test[test["date"] == date]
        valid = ddf[[SIGNAL, "exc_1d_bps", "fwd_1d_bps", "symbol", "close"]].dropna()

        if len(valid) < 20:
            continue

        try:
            valid["q"] = pd.qcut(valid[SIGNAL], N_QUANTILES,
                                 labels=range(1, N_QUANTILES + 1), duplicates="drop")
        except ValueError:
            continue
        if valid["q"].nunique() < N_QUANTILES:
            continue

        q1 = valid[valid["q"] == 1]
        q5 = valid[valid["q"] == N_QUANTILES]

        long_names = sorted(q1["symbol"].tolist())
        short_names = sorted(q5["symbol"].tolist())
        long_set = set(long_names)
        short_set = set(short_names)

        # Returns
        long_exc = q1["exc_1d_bps"].mean()
        short_exc = -q5["exc_1d_bps"].mean()
        ls_exc = long_exc + short_exc

        long_raw = q1["fwd_1d_bps"].mean()
        short_raw = -q5["fwd_1d_bps"].mean()
        ls_raw = long_raw + short_raw

        # Turnover
        if prev_long_set:
            long_overlap = len(long_set & prev_long_set)
            short_overlap = len(short_set & prev_short_set)
            long_turnover = 1 - long_overlap / max(len(long_set), 1)
            short_turnover = 1 - short_overlap / max(len(short_set), 1)
            names_changed_long = len(long_set - prev_long_set)
            names_changed_short = len(short_set - prev_short_set)
        else:
            long_turnover = 1.0
            short_turnover = 1.0
            names_changed_long = len(long_set)
            names_changed_short = len(short_set)

        avg_turnover = (long_turnover + short_turnover) / 2
        cost_bps = avg_turnover * COST_BPS_RT

        net_ls = ls_exc - cost_bps

        cum_gross += ls_exc
        cum_net += net_ls
        cum_long += long_exc
        cum_short += short_exc
        all_net.append(net_ls)

        # Running Sharpe (annualized)
        if len(all_net) > 20:
            arr = np.array(all_net)
            running_sr = arr.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
        else:
            running_sr = np.nan

        # Max drawdown
        cum_series = np.cumsum(all_net)
        peak = np.maximum.accumulate(cum_series)
        dd = peak - cum_series
        max_dd = dd[-1] if len(dd) > 0 else 0
        max_dd_ever = dd.max() if len(dd) > 0 else 0

        rows.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "day_num": len(all_net),
            # Returns
            "long_exc_bps": round(long_exc, 2),
            "short_exc_bps": round(short_exc, 2),
            "ls_gross_bps": round(ls_exc, 2),
            "ls_net_bps": round(net_ls, 2),
            "long_raw_bps": round(long_raw, 2),
            "short_raw_bps": round(short_raw, 2),
            "ls_raw_bps": round(ls_raw, 2),
            # Cumulative
            "cum_gross_bps": round(cum_gross, 1),
            "cum_net_bps": round(cum_net, 1),
            "cum_long_bps": round(cum_long, 1),
            "cum_short_bps": round(cum_short, 1),
            # Risk
            "running_sharpe": round(running_sr, 2) if not np.isnan(running_sr) else "",
            "current_dd_bps": round(float(dd[-1]), 1),
            "max_dd_bps": round(float(max_dd_ever), 1),
            # Turnover
            "long_turnover_pct": round(100 * long_turnover, 1),
            "short_turnover_pct": round(100 * short_turnover, 1),
            "names_changed_long": names_changed_long,
            "names_changed_short": names_changed_short,
            "cost_bps": round(cost_bps, 2),
            # Portfolio composition
            "n_long": len(long_names),
            "n_short": len(short_names),
            "long_names": "|".join(long_names),
            "short_names": "|".join(short_names),
        })

        prev_long_set = long_set
        prev_short_set = short_set

    # Save
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "portfolio_daily.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Date range: {out_df['date'].iloc[0]} to {out_df['date'].iloc[-1]}")
    print(f"Final cum net: {out_df['cum_net_bps'].iloc[-1]:+.0f} bps")
    print(f"Final Sharpe: {out_df['running_sharpe'].iloc[-1]}")
    print(f"Max DD: {out_df['max_dd_bps'].iloc[-1]:.0f} bps")

    # Print a small sample
    print(f"\nFirst 10 days:")
    sample_cols = ["date", "ls_net_bps", "cum_net_bps", "running_sharpe",
                   "long_turnover_pct", "n_long", "n_short"]
    print(out_df[sample_cols].head(10).to_string(index=False))

    print(f"\nLast 10 days:")
    print(out_df[sample_cols].tail(10).to_string(index=False))

    # Monthly summary
    out_df["month"] = pd.to_datetime(out_df["date"]).dt.to_period("M")
    monthly = out_df.groupby("month").agg(
        days=("ls_net_bps", "count"),
        net_total=("ls_net_bps", "sum"),
        net_mean=("ls_net_bps", "mean"),
        win_rate=("ls_net_bps", lambda x: 100 * (x > 0).mean()),
        avg_turnover=("long_turnover_pct", "mean"),
    ).round(1)

    monthly_path = os.path.join(OUT_DIR, "portfolio_monthly.csv")
    monthly.to_csv(monthly_path)
    print(f"\nMonthly summary saved: {monthly_path}")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
