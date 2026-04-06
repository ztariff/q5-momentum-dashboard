#!/usr/bin/env python3
"""
Trade 1 P&L simulation: $1M per trade entry.

Rules:
- Buy $1M worth at next open when stock enters Q5
- Stop at -75 bps, TP at +500 bps, max hold 5 days
- Multiple positions can be open simultaneously
- Track daily portfolio P&L, capital deployed, drawdown
- Include 10 bps round-trip transaction cost (5 bps each way)
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "/home/ubuntu/daily_data/data"
OUT_DIR = "/home/ubuntu/daily_data/analysis_results"

POSITION_SIZE = 1_000_000  # $1M per trade
COST_BPS_EACH_WAY = 5
STOP_BPS = -75
TP_BPS = 500
MAX_HOLD = 5


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
            q = pd.qcut(valid, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            pooled.loc[q.index, "quintile"] = q.values
        except ValueError:
            continue

    pooled = pooled.sort_values(["symbol", "date"]).reset_index(drop=True)
    pooled["prev_q"] = pooled.groupby("symbol")["quintile"].shift(1)
    pooled["entered_q5"] = (pooled["quintile"] == 5) & (pooled["prev_q"] != 5)
    return pooled


def simulate_portfolio(pooled):
    """
    Walk through calendar days. On each day:
    1. Check for new Q5 entries from yesterday → open positions at today's open
    2. For existing positions, check today's low/high for stop/TP
    3. For positions at max_hold, exit at today's close
    4. Record daily P&L
    """
    # Build lookup: (symbol, date) -> {open, high, low, close}
    price_lookup = {}
    for _, row in pooled.iterrows():
        price_lookup[(row["symbol"], row["date"])] = {
            "open": row["open"], "high": row["high"],
            "low": row["low"], "close": row["close"],
        }

    # Get entry signals (signal fires at close of this day, enter next day)
    entries = pooled[pooled["entered_q5"] & (pooled["date"].dt.year >= 2020)].copy()
    entry_by_date = entries.groupby("date").apply(
        lambda x: list(zip(x["symbol"], x["z_signal"]))
    ).to_dict()

    all_dates = sorted(pooled[pooled["date"].dt.year >= 2020]["date"].unique())

    # Active positions: list of dicts
    positions = []
    daily_records = []
    trade_log = []

    for i, today in enumerate(all_dates):
        # 1. Open new positions from yesterday's signals
        yesterday = all_dates[i - 1] if i > 0 else None
        if yesterday and yesterday in entry_by_date:
            for sym, z in entry_by_date[yesterday]:
                prices = price_lookup.get((sym, today))
                if prices is None or prices["open"] <= 0 or pd.isna(prices["open"]):
                    continue
                entry_price = prices["open"]
                shares = int(POSITION_SIZE / entry_price)
                entry_cost = shares * entry_price * COST_BPS_EACH_WAY / 10000

                positions.append({
                    "symbol": sym,
                    "entry_date": today,
                    "entry_price": entry_price,
                    "shares": shares,
                    "notional": shares * entry_price,
                    "stop_price": entry_price * (1 + STOP_BPS / 10000),
                    "tp_price": entry_price * (1 + TP_BPS / 10000),
                    "days_held": 0,
                    "entry_cost": entry_cost,
                    "z_score": z,
                })

        # 2. Process existing positions
        closed_today = []
        daily_realized = 0
        daily_unrealized = 0
        positions_open = 0
        capital_deployed = 0

        surviving = []
        for pos in positions:
            pos["days_held"] += 1
            prices = price_lookup.get((pos["symbol"], today))

            if prices is None:
                # No price data, carry forward
                surviving.append(pos)
                continue

            day_low = prices["low"]
            day_high = prices["high"]
            day_close = prices["close"]

            exit_price = None
            exit_type = None

            # Check stop
            if day_low <= pos["stop_price"]:
                exit_price = pos["stop_price"]
                exit_type = "STOP"
            # Check TP
            elif day_high >= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_type = "TARGET"
            # Check time exit
            elif pos["days_held"] >= MAX_HOLD:
                exit_price = day_close
                exit_type = "TIME"

            if exit_price is not None:
                exit_cost = pos["shares"] * exit_price * COST_BPS_EACH_WAY / 10000
                gross_pnl = pos["shares"] * (exit_price - pos["entry_price"])
                net_pnl = gross_pnl - pos["entry_cost"] - exit_cost
                pnl_bps = 10000 * (exit_price / pos["entry_price"] - 1)

                daily_realized += net_pnl
                closed_today.append(pos)

                trade_log.append({
                    "symbol": pos["symbol"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": today.strftime("%Y-%m-%d") if hasattr(today, 'strftime') else str(today)[:10],
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": pos["shares"],
                    "notional": round(pos["notional"], 0),
                    "gross_pnl": round(gross_pnl, 0),
                    "net_pnl": round(net_pnl, 0),
                    "pnl_bps": round(pnl_bps, 1),
                    "exit_type": exit_type,
                    "days_held": pos["days_held"],
                    "z_score": round(pos["z_score"], 2),
                })
            else:
                # Still open — mark to market
                mtm = pos["shares"] * (day_close - pos["entry_price"])
                daily_unrealized += mtm
                capital_deployed += pos["notional"]
                surviving.append(pos)

        positions = surviving
        positions_open = len(positions)
        capital_deployed += sum(p["notional"] for p in positions)

        daily_records.append({
            "date": today.strftime("%Y-%m-%d") if hasattr(today, 'strftime') else str(today)[:10],
            "realized_pnl": round(daily_realized, 0),
            "unrealized_pnl": round(daily_unrealized, 0),
            "total_pnl": round(daily_realized + daily_unrealized, 0),
            "positions_open": positions_open,
            "capital_deployed": round(capital_deployed, 0),
            "trades_closed": len(closed_today),
            "new_entries": len(entry_by_date.get(yesterday, [])) if yesterday else 0,
        })

    return pd.DataFrame(daily_records), pd.DataFrame(trade_log)


def main():
    print("Loading data...")
    pooled = load_base()
    print(f"Rows: {len(pooled):,}")

    print("Running portfolio simulation ($1M per trade)...")
    daily, trades = simulate_portfolio(pooled)
    print(f"Daily records: {len(daily)}, Trades: {len(trades)}\n")

    # Cumulative P&L
    daily["cum_realized"] = daily["realized_pnl"].cumsum()
    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M")

    report = []
    report.append("=" * 100)
    report.append("TRADE 1 P&L: $1M per trade entry")
    report.append(f"Stop: {STOP_BPS} bps | TP: +{TP_BPS} bps | Max hold: {MAX_HOLD}d | Cost: {COST_BPS_EACH_WAY} bps/side")
    report.append("=" * 100)

    # Overall
    total_realized = daily["cum_realized"].iloc[-1]
    n_trades = len(trades)
    winners = trades[trades["net_pnl"] > 0]
    losers = trades[trades["net_pnl"] <= 0]

    report.append(f"\n  OVERALL (2020-2026):")
    report.append(f"  Total realized P&L:   ${total_realized:,.0f}")
    report.append(f"  Total trades:         {n_trades:,}")
    report.append(f"  Winners:              {len(winners):,} ({100*len(winners)/n_trades:.1f}%)")
    report.append(f"  Losers:               {len(losers):,} ({100*len(losers)/n_trades:.1f}%)")
    report.append(f"  Avg winner:           ${winners['net_pnl'].mean():,.0f}")
    report.append(f"  Avg loser:            ${losers['net_pnl'].mean():,.0f}")
    report.append(f"  Biggest winner:       ${winners['net_pnl'].max():,.0f} ({winners.loc[winners['net_pnl'].idxmax(), 'symbol']})")
    report.append(f"  Biggest loser:        ${losers['net_pnl'].min():,.0f} ({losers.loc[losers['net_pnl'].idxmin(), 'symbol']})")
    report.append(f"  Profit factor:        {winners['net_pnl'].sum() / abs(losers['net_pnl'].sum()):.2f}")
    report.append(f"  Avg trades/day:       {trades.groupby('entry_date').size().mean():.1f}")
    report.append(f"  Max concurrent pos:   {daily['positions_open'].max()}")
    report.append(f"  Avg capital deployed: ${daily['capital_deployed'].mean():,.0f}")
    report.append(f"  Max capital deployed: ${daily['capital_deployed'].max():,.0f}")

    # Drawdown
    cum = daily["cum_realized"].values
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max()
    max_dd_idx = np.argmax(dd)
    max_dd_date = daily.iloc[max_dd_idx]["date"]

    report.append(f"\n  Max drawdown:         ${max_dd:,.0f} (at {max_dd_date})")
    report.append(f"  Return / MaxDD:       {total_realized / max_dd:.2f}x" if max_dd > 0 else "")

    # Year by year
    report.append(f"\n  YEAR-BY-YEAR:")
    report.append(f"  {'Year':>6} {'Trades':>7} {'Realized':>12} {'Winners':>8} {'WR':>5} {'AvgWin':>9} {'AvgLose':>9} {'MaxPos':>7} {'PF':>5}")
    report.append(f"  {'─'*80}")

    trades["year"] = pd.to_datetime(trades["entry_date"]).dt.year
    for yr in sorted(trades["year"].unique()):
        yt = trades[trades["year"] == yr]
        yw = yt[yt["net_pnl"] > 0]
        yl = yt[yt["net_pnl"] <= 0]
        yr_pnl = yt["net_pnl"].sum()
        pf = yw["net_pnl"].sum() / abs(yl["net_pnl"].sum()) if len(yl) > 0 and yl["net_pnl"].sum() != 0 else float("inf")
        yr_daily = daily[daily["year"] == yr]
        max_pos = yr_daily["positions_open"].max()

        report.append(f"  {yr:>6} {len(yt):>7} ${yr_pnl:>+11,.0f} {len(yw):>8} {100*len(yw)/len(yt):>5.1f} "
                      f"${yw['net_pnl'].mean():>+8,.0f} ${yl['net_pnl'].mean():>+8,.0f} {max_pos:>7} {pf:>5.2f}")

    # Monthly P&L
    report.append(f"\n  MONTHLY P&L:")
    monthly = trades.copy()
    monthly["month"] = pd.to_datetime(monthly["exit_date"]).dt.to_period("M")
    monthly_pnl = monthly.groupby("month")["net_pnl"].sum()

    pos_months = (monthly_pnl > 0).sum()
    neg_months = (monthly_pnl <= 0).sum()
    report.append(f"  Positive months: {pos_months}/{pos_months + neg_months} ({100*pos_months/(pos_months+neg_months):.0f}%)")
    report.append(f"  Best month:  ${monthly_pnl.max():>+12,.0f} ({monthly_pnl.idxmax()})")
    report.append(f"  Worst month: ${monthly_pnl.min():>+12,.0f} ({monthly_pnl.idxmin()})")
    report.append(f"  Avg month:   ${monthly_pnl.mean():>+12,.0f}")

    # Quarterly
    report.append(f"\n  QUARTERLY P&L:")
    monthly["quarter"] = pd.to_datetime(monthly["exit_date"]).dt.to_period("Q")
    quarterly_pnl = monthly.groupby("quarter")["net_pnl"].sum()
    report.append(f"  {'Quarter':>8} {'P&L':>12} {'Trades':>7}")
    report.append(f"  {'─'*30}")
    quarterly_trades = monthly.groupby("quarter").size()
    for q in quarterly_pnl.index:
        report.append(f"  {str(q):>8} ${quarterly_pnl[q]:>+11,.0f} {quarterly_trades.get(q, 0):>7}")

    # By exit type
    report.append(f"\n  BY EXIT TYPE:")
    for etype in ["TARGET", "STOP", "TIME"]:
        et = trades[trades["exit_type"] == etype]
        if len(et) == 0:
            continue
        report.append(f"  {etype:<8}: {len(et):>5} trades, total ${et['net_pnl'].sum():>+12,.0f}, "
                      f"avg ${et['net_pnl'].mean():>+8,.0f}")

    report.append("")
    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUT_DIR, "trade1_1m_pnl_report.txt"), "w") as f:
        f.write(report_text)

    daily.to_csv(os.path.join(OUT_DIR, "trade1_1m_daily.csv"), index=False)
    trades.to_csv(os.path.join(OUT_DIR, "trade1_1m_trades.csv"), index=False)
    print(f"\nReport: {os.path.join(OUT_DIR, 'trade1_1m_pnl_report.txt')}")


if __name__ == "__main__":
    main()
