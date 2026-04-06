#!/usr/bin/env python3
"""
Trade 1 dollar P&L: test multiple configurations honestly.

1. No stop, no TP, hold 5/7/10/15/20 days
2. No stop, no TP, exit when stock leaves Q5
3. Wide stops (-200, -300, -500) with no TP
4. No stop, with TP at various levels
5. The original portfolio approach (always in, quintile sort)

$1M per trade, 5 bps each way.
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"

POSITION_SIZE = 1_000_000
COST_EACH_WAY = 5  # bps


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


def simulate(pooled, stop_bps, tp_bps, max_hold, label):
    """Run full dollar simulation."""
    price_lookup = {}
    for _, row in pooled.iterrows():
        price_lookup[(row["symbol"], row["date"])] = {
            "open": row["open"], "high": row["high"],
            "low": row["low"], "close": row["close"],
        }

    entries = pooled[pooled["entered_q5"] & (pooled["date"].dt.year >= 2020)]
    entry_by_date = entries.groupby("date").apply(
        lambda x: list(zip(x["symbol"], x["z_signal"]))
    ).to_dict()

    all_dates = sorted(pooled[pooled["date"].dt.year >= 2020]["date"].unique())

    positions = []
    trades = []

    for i, today in enumerate(all_dates):
        yesterday = all_dates[i-1] if i > 0 else None
        if yesterday and yesterday in entry_by_date:
            for sym, z in entry_by_date[yesterday]:
                prices = price_lookup.get((sym, today))
                if prices is None or prices["open"] <= 0 or pd.isna(prices["open"]):
                    continue
                ep = prices["open"]
                shares = int(POSITION_SIZE / ep)
                positions.append({
                    "symbol": sym, "entry_date": today, "entry_price": ep,
                    "shares": shares, "notional": shares * ep, "days_held": 0,
                    "stop_price": ep * (1 + stop_bps/10000) if stop_bps else None,
                    "tp_price": ep * (1 + tp_bps/10000) if tp_bps else None,
                })

        surviving = []
        for pos in positions:
            pos["days_held"] += 1
            prices = price_lookup.get((pos["symbol"], today))
            if prices is None:
                surviving.append(pos)
                continue

            exit_price = None
            exit_type = None

            if pos["stop_price"] and prices["low"] <= pos["stop_price"]:
                exit_price = pos["stop_price"]
                exit_type = "STOP"
            elif pos["tp_price"] and prices["high"] >= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_type = "TARGET"
            elif pos["days_held"] >= max_hold:
                exit_price = prices["close"]
                exit_type = "TIME"

            if exit_price:
                cost = pos["shares"] * (pos["entry_price"] + exit_price) * COST_EACH_WAY / 10000
                gross = pos["shares"] * (exit_price - pos["entry_price"])
                net = gross - cost
                trades.append({
                    "symbol": pos["symbol"],
                    "entry_date": pos["entry_date"],
                    "exit_date": today,
                    "net_pnl": net,
                    "gross_pnl": gross,
                    "pnl_bps": 10000 * (exit_price / pos["entry_price"] - 1),
                    "exit_type": exit_type,
                    "days_held": pos["days_held"],
                    "shares": pos["shares"],
                    "entry_price": pos["entry_price"],
                })
            else:
                surviving.append(pos)

        positions = surviving

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return None

    total = tdf["net_pnl"].sum()
    n = len(tdf)
    w = tdf[tdf["net_pnl"] > 0]
    l = tdf[tdf["net_pnl"] <= 0]
    wr = 100 * len(w) / n
    pf = w["net_pnl"].sum() / abs(l["net_pnl"].sum()) if len(l) > 0 and l["net_pnl"].sum() != 0 else float("inf")

    # Drawdown on cumulative
    cum = tdf["net_pnl"].cumsum().values
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()

    # Year by year
    tdf["year"] = pd.to_datetime(tdf["entry_date"]).dt.year
    yearly = {}
    for yr, ydf in tdf.groupby("year"):
        yw = ydf[ydf["net_pnl"] > 0]
        yearly[yr] = {
            "n": len(ydf),
            "pnl": ydf["net_pnl"].sum(),
            "wr": 100 * len(yw) / len(ydf),
            "positive": ydf["net_pnl"].sum() > 0,
        }

    return {
        "label": label,
        "stop": stop_bps,
        "tp": tp_bps,
        "hold": max_hold,
        "n_trades": n,
        "total_pnl": total,
        "wr": wr,
        "pf": pf,
        "avg_pnl": tdf["net_pnl"].mean(),
        "avg_winner": w["net_pnl"].mean() if len(w) > 0 else 0,
        "avg_loser": l["net_pnl"].mean() if len(l) > 0 else 0,
        "max_dd": max_dd,
        "worst": tdf["net_pnl"].min(),
        "best": tdf["net_pnl"].max(),
        "yearly": yearly,
        "pos_years": sum(1 for v in yearly.values() if v["positive"]),
    }


def main():
    print("Loading data...")
    pooled = load_base()
    print(f"Rows: {len(pooled):,}\n")

    configs = [
        # No stop, no TP, various holds
        (None, None, 5, "No stop, no TP, 5d"),
        (None, None, 7, "No stop, no TP, 7d"),
        (None, None, 10, "No stop, no TP, 10d"),
        (None, None, 15, "No stop, no TP, 15d"),
        (None, None, 20, "No stop, no TP, 20d"),
        # Wide stops, no TP
        (-200, None, 10, "Stop -200, no TP, 10d"),
        (-300, None, 10, "Stop -300, no TP, 10d"),
        (-500, None, 10, "Stop -500, no TP, 10d"),
        (-200, None, 20, "Stop -200, no TP, 20d"),
        (-300, None, 20, "Stop -300, no TP, 20d"),
        (-500, None, 20, "Stop -500, no TP, 20d"),
        # The original bad config for comparison
        (-75, 500, 5, "Stop -75, TP +500, 5d (ORIGINAL)"),
        # No stop, with TP
        (None, 300, 10, "No stop, TP +300, 10d"),
        (None, 500, 10, "No stop, TP +500, 10d"),
        (None, 300, 20, "No stop, TP +300, 20d"),
        (None, 500, 20, "No stop, TP +500, 20d"),
        # Wide stop + wide TP
        (-300, 500, 10, "Stop -300, TP +500, 10d"),
        (-500, 500, 20, "Stop -500, TP +500, 20d"),
        (-300, None, 5, "Stop -300, no TP, 5d"),
        (-500, None, 5, "Stop -500, no TP, 5d"),
    ]

    results = []
    for stop, tp, hold, label in configs:
        print(f"  Testing: {label}...")
        r = simulate(pooled, stop, tp, hold, label)
        if r:
            results.append(r)

    # Report
    report = []
    report.append("=" * 120)
    report.append("TRADE 1: HONEST DOLLAR P&L — ALL CONFIGURATIONS")
    report.append(f"$1M per trade, 5 bps each way")
    report.append("=" * 120)

    # Sort by total P&L
    results.sort(key=lambda x: x["total_pnl"], reverse=True)

    report.append(f"\n{'Label':<35} {'Trades':>6} {'Total P&L':>14} {'AvgPnL':>9} {'WR':>5} {'PF':>5} {'MaxDD':>12} {'Worst':>10} {'Yrs+':>5}")
    report.append("─" * 120)

    for r in results:
        report.append(
            f"{r['label']:<35} {r['n_trades']:>6} ${r['total_pnl']:>+13,.0f} ${r['avg_pnl']:>+8,.0f} "
            f"{r['wr']:>5.1f} {r['pf']:>5.2f} ${r['max_dd']:>11,.0f} ${r['worst']:>+9,.0f} "
            f"{r['pos_years']:>2}/{len(r['yearly'])}"
        )

    # Year by year for top 5
    report.append(f"\n\n{'═'*120}")
    report.append("YEAR-BY-YEAR for top 5 configurations:")
    report.append(f"{'═'*120}")

    for r in results[:5]:
        report.append(f"\n  {r['label']}:")
        for yr in sorted(r["yearly"].keys()):
            yd = r["yearly"][yr]
            status = "+" if yd["positive"] else "-"
            report.append(f"    {yr}: {yd['n']:>4} trades, ${yd['pnl']:>+12,.0f}, WR={yd['wr']:.0f}% [{status}]")

    report.append("")
    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "trade1_all_configs_report.txt"), "w") as f:
        f.write(report_text)
    print(f"\nReport: {os.path.join(OUT_DIR, 'trade1_all_configs_report.txt')}")


if __name__ == "__main__":
    main()
