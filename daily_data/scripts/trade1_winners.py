#!/usr/bin/env python3
"""
Show the biggest winners and losers from Trade 1 (Q5 momentum entries).
Use the -75 stop / +500 TP / 5d hold spec.
Also show the no-stop baseline to see what the uncapped winners look like.
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"


def load_base():
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

    # Quintiles
    pooled["quintile"] = np.nan
    for date, ddf in pooled.groupby("date"):
        valid = ddf["z_signal"].dropna()
        if len(valid) < 20:
            continue
        try:
            q = pd.qcut(valid, 5, labels=[1,2,3,4,5], duplicates="drop")
            pooled.loc[q.index, "quintile"] = q.values
        except ValueError:
            continue

    pooled = pooled.sort_values(["symbol", "date"]).reset_index(drop=True)
    pooled["prev_q"] = pooled.groupby("symbol")["quintile"].shift(1)
    pooled["entered_q5"] = (pooled["quintile"] == 5) & (pooled["prev_q"] != 5)
    return pooled


def build_trade_log(pooled, entry_mask, stop_bps, tp_bps, max_hold):
    """Build full trade log with day-by-day detail."""
    entries = pooled[entry_mask].copy()
    pooled_idx = pooled.set_index(["symbol", "date"]).sort_index()

    trades = []
    for _, row in entries.iterrows():
        sym = row["symbol"]
        entry_date = row["date"]
        entry_close = row["close"]
        z = row["z_signal"]

        if entry_close <= 0 or pd.isna(entry_close):
            continue

        try:
            sym_data = pooled_idx.loc[sym]
        except KeyError:
            continue

        future = sym_data[sym_data.index > entry_date].head(max_hold)
        if len(future) < 1:
            continue

        # Entry at next open
        entry_price = future.iloc[0]["open"]
        if entry_price <= 0 or pd.isna(entry_price):
            continue

        # Walk through days
        exit_day = None
        exit_price = None
        exit_type = None
        pnl_bps = None
        path = []

        for d in range(len(future)):
            day_low = future.iloc[d]["low"]
            day_high = future.iloc[d]["high"]
            day_close = future.iloc[d]["close"]
            day_date = future.index[d]

            low_bps = 10000 * (day_low / entry_price - 1)
            high_bps = 10000 * (day_high / entry_price - 1)
            close_bps = 10000 * (day_close / entry_price - 1)

            path.append(f"d{d+1}:{close_bps:+.0f}")

            # Check stop (conservative: check before TP)
            if stop_bps is not None and low_bps <= stop_bps:
                exit_day = d + 1
                exit_price = entry_price * (1 + stop_bps / 10000)
                exit_type = "STOP"
                pnl_bps = stop_bps
                break

            # Check TP
            if tp_bps is not None and high_bps >= tp_bps:
                exit_day = d + 1
                exit_price = entry_price * (1 + tp_bps / 10000)
                exit_type = "TARGET"
                pnl_bps = tp_bps
                break

        if exit_type is None:
            # Time exit at close of last day
            exit_day = len(future)
            exit_price = future.iloc[-1]["close"]
            exit_type = "TIME"
            pnl_bps = 10000 * (exit_price / entry_price - 1)

        # Also compute uncapped return (what would have happened with no stop/TP)
        if len(future) >= max_hold:
            uncapped = 10000 * (future.iloc[max_hold - 1]["close"] / entry_price - 1)
        else:
            uncapped = 10000 * (future.iloc[-1]["close"] / entry_price - 1)

        # Max favorable and adverse during the hold
        all_lows = [10000 * (future.iloc[d]["low"] / entry_price - 1) for d in range(min(max_hold, len(future)))]
        all_highs = [10000 * (future.iloc[d]["high"] / entry_price - 1) for d in range(min(max_hold, len(future)))]
        mae = min(all_lows) if all_lows else 0
        mfe = max(all_highs) if all_highs else 0

        trades.append({
            "symbol": sym,
            "signal_date": entry_date.strftime("%Y-%m-%d"),
            "entry_date": future.index[0].strftime("%Y-%m-%d") if hasattr(future.index[0], 'strftime') else str(future.index[0])[:10],
            "entry_price": round(entry_price, 2),
            "z_score": round(z, 2),
            "exit_type": exit_type,
            "exit_day": exit_day,
            "exit_price": round(exit_price, 2),
            "pnl_bps": round(pnl_bps, 1),
            "pnl_dollar_1k": round(pnl_bps / 10000 * entry_price * 1000, 0),
            "uncapped_bps": round(uncapped, 1),
            "mae_bps": round(mae, 1),
            "mfe_bps": round(mfe, 1),
            "path": " ".join(path[:max_hold]),
        })

    return pd.DataFrame(trades)


def main():
    print("Loading data...")
    pooled = load_base()

    # Full period (2020-2026)
    mask = pooled["entered_q5"] & (pooled["date"].dt.year >= 2020)
    print(f"Q5 entries 2020-2026: {mask.sum():,}")

    print("Building trade log (-75 stop, +500 TP, 5d hold)...")
    trades = build_trade_log(pooled, mask, stop_bps=-75, tp_bps=500, max_hold=5)
    print(f"Trades: {len(trades):,}\n")

    # Also build uncapped version
    print("Building uncapped trade log (no stop, no TP, 10d hold)...")
    trades_uncapped = build_trade_log(pooled, mask, stop_bps=None, tp_bps=None, max_hold=10)

    # Save full logs
    trades.to_csv(os.path.join(OUT_DIR, "trade1_log.csv"), index=False)
    trades_uncapped.to_csv(os.path.join(OUT_DIR, "trade1_uncapped_log.csv"), index=False)

    # Report
    report = []
    report.append("=" * 120)
    report.append("TRADE 1: BIGGEST WINNERS AND LOSERS")
    report.append("=" * 120)

    # ── Winners (strategy spec: -75/+500/5d) ──
    report.append(f"\n{'═'*120}")
    report.append("TOP 30 WINNERS (strategy: -75 stop / +500 TP / 5d hold)")
    report.append(f"{'═'*120}")
    winners = trades.sort_values("pnl_bps", ascending=False).head(30)
    report.append(f"{'Symbol':<8} {'Signal':>12} {'Entry$':>9} {'Z':>6} {'Exit':>6} {'Day':>4} {'PnL bps':>8} {'$/1Ksh':>8} {'Type':>7} {'Path'}")
    report.append("─" * 120)
    for _, t in winners.iterrows():
        report.append(f"{t['symbol']:<8} {t['signal_date']:>12} {t['entry_price']:>9.2f} {t['z_score']:>+6.2f} "
                      f"{t['exit_price']:>9.2f} {t['exit_day']:>4} {t['pnl_bps']:>+8.1f} {t['pnl_dollar_1k']:>+8.0f} "
                      f"{t['exit_type']:>7} {t['path']}")

    # ── Biggest uncapped winners ──
    report.append(f"\n{'═'*120}")
    report.append("TOP 30 UNCAPPED WINNERS (no stop, no TP, 10d hold — what you COULD have made)")
    report.append(f"{'═'*120}")
    unc_winners = trades_uncapped.sort_values("pnl_bps", ascending=False).head(30)
    report.append(f"{'Symbol':<8} {'Signal':>12} {'Entry$':>9} {'Z':>6} {'PnL bps':>8} {'$/1Ksh':>8} {'MAE':>8} {'MFE':>8} {'Path'}")
    report.append("─" * 120)
    for _, t in unc_winners.iterrows():
        report.append(f"{t['symbol']:<8} {t['signal_date']:>12} {t['entry_price']:>9.2f} {t['z_score']:>+6.2f} "
                      f"{t['pnl_bps']:>+8.1f} {t['pnl_dollar_1k']:>+8.0f} {t['mae_bps']:>+8.1f} {t['mfe_bps']:>+8.1f} "
                      f"{t['path']}")

    # ── Losers ──
    report.append(f"\n{'═'*120}")
    report.append("TOP 30 LOSERS (strategy: capped at -75 bps by stop)")
    report.append(f"{'═'*120}")
    report.append(f"  All losers exit at -75 bps. But what WOULD have happened without the stop?")
    stopped = trades[trades["exit_type"] == "STOP"].copy()
    stopped = stopped.sort_values("uncapped_bps", ascending=True).head(30)
    report.append(f"\n{'Symbol':<8} {'Signal':>12} {'Entry$':>9} {'Z':>6} {'Stopped@':>9} {'Uncapped':>9} {'Saved':>7} {'Path'}")
    report.append("─" * 120)
    for _, t in stopped.iterrows():
        saved = abs(t["uncapped_bps"]) - 75
        report.append(f"{t['symbol']:<8} {t['signal_date']:>12} {t['entry_price']:>9.2f} {t['z_score']:>+6.2f} "
                      f"{t['pnl_bps']:>+9.1f} {t['uncapped_bps']:>+9.1f} {saved:>+7.0f} {t['path']}")

    # ── Summary stats ──
    report.append(f"\n{'═'*120}")
    report.append("SUMMARY STATS")
    report.append(f"{'═'*120}")

    targets_hit = trades[trades["exit_type"] == "TARGET"]
    stops_hit = trades[trades["exit_type"] == "STOP"]
    time_exits = trades[trades["exit_type"] == "TIME"]

    report.append(f"  Total trades:  {len(trades):,}")
    report.append(f"  Targets hit:   {len(targets_hit):>5} ({100*len(targets_hit)/len(trades):.1f}%) — mean +{targets_hit['pnl_bps'].mean():.0f} bps")
    report.append(f"  Stops hit:     {len(stops_hit):>5} ({100*len(stops_hit)/len(trades):.1f}%) — mean {stops_hit['pnl_bps'].mean():.0f} bps")
    report.append(f"  Time exits:    {len(time_exits):>5} ({100*len(time_exits)/len(trades):.1f}%) — mean {time_exits['pnl_bps'].mean():+.0f} bps")

    # Time exits breakdown
    if len(time_exits) > 0:
        te_pos = time_exits[time_exits["pnl_bps"] > 0]
        te_neg = time_exits[time_exits["pnl_bps"] <= 0]
        report.append(f"\n  Time exits that were positive: {len(te_pos)} ({100*len(te_pos)/len(time_exits):.0f}%), mean +{te_pos['pnl_bps'].mean():.0f} bps")
        report.append(f"  Time exits that were negative: {len(te_neg)} ({100*len(te_neg)/len(time_exits):.0f}%), mean {te_neg['pnl_bps'].mean():.0f} bps")

    # What % of stopped trades WOULD have been profitable at day 5?
    if len(stopped) > 0:
        would_win = stops_hit[stops_hit["uncapped_bps"] > 0]
        report.append(f"\n  Stopped trades that WOULD have been profitable at day 5: {len(would_win)}/{len(stops_hit)} ({100*len(would_win)/len(stops_hit):.0f}%)")
        report.append(f"  → The stop is sacrificing {len(would_win)} winning trades to protect against {len(stops_hit)-len(would_win)} that kept falling")

    # ── By symbol: which names appear most in winners? ──
    report.append(f"\n{'═'*120}")
    report.append("BY SYMBOL — Which names generate the most trades and P&L?")
    report.append(f"{'═'*120}")
    sym_stats = trades.groupby("symbol").agg(
        n_trades=("pnl_bps", "count"),
        total_pnl=("pnl_bps", "sum"),
        mean_pnl=("pnl_bps", "mean"),
        wr=("pnl_bps", lambda x: 100 * (x > 0).mean()),
        targets=("exit_type", lambda x: (x == "TARGET").sum()),
    ).sort_values("total_pnl", ascending=False)

    report.append(f"\n  Top 20 by total P&L:")
    report.append(f"  {'Symbol':<8} {'Trades':>7} {'Total bps':>10} {'Mean':>7} {'WR':>5} {'Targets':>8}")
    report.append(f"  {'─'*50}")
    for sym, row in sym_stats.head(20).iterrows():
        report.append(f"  {sym:<8} {row['n_trades']:>7} {row['total_pnl']:>+10.0f} {row['mean_pnl']:>+7.1f} {row['wr']:>5.0f} {row['targets']:>8}")

    report.append(f"\n  Bottom 10 by total P&L:")
    for sym, row in sym_stats.tail(10).iterrows():
        report.append(f"  {sym:<8} {row['n_trades']:>7} {row['total_pnl']:>+10.0f} {row['mean_pnl']:>+7.1f} {row['wr']:>5.0f} {row['targets']:>8}")

    report.append("")
    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "trade1_winners_report.txt"), "w") as f:
        f.write(report_text)
    print(f"\nReport: {os.path.join(OUT_DIR, 'trade1_winners_report.txt')}")
    print(f"Trade log: {os.path.join(OUT_DIR, 'trade1_log.csv')}")
    print(f"Uncapped log: {os.path.join(OUT_DIR, 'trade1_uncapped_log.csv')}")


if __name__ == "__main__":
    main()
