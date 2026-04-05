import { NextResponse } from 'next/server';
import { parseTrades } from '@/lib/csvParser';
import { fetchMultiplePrices } from '@/lib/polygon';
import { Position } from '@/lib/types';
import path from 'path';
import fs from 'fs';

export async function GET() {
  try {
    const csvPath = path.join(process.cwd(), 'public', 'data', 'hybrid_all_trades_calendar.csv');
    const csvText = fs.readFileSync(csvPath, 'utf-8');
    const trades = parseTrades(csvText);

    const today = new Date().toISOString().split('T')[0]; // e.g. '2026-04-03'

    // Helper: count trading days from entry to today (approximate)
    function tradingDaysSinceEntry(entryDate: string): number {
      const start = new Date(entryDate);
      const end = new Date(today);
      let count = 0;
      const cur = new Date(start);
      while (cur < end) {
        cur.setDate(cur.getDate() + 1);
        const day = cur.getDay();
        if (day !== 0 && day !== 6) count++;
      }
      return count;
    }

    // A position is "open" if its exit_date has not passed yet.
    // Treat the backtest as live: positions with exit_date >= today are still open.
    let openTrades = trades.filter(t => t.exit_date >= today);
    let isRecentlyClosed = false;

    // Fallback: if zero open positions (all trades have already exited), show
    // trades from the last 5 trading days as "recently closed"
    if (openTrades.length === 0) {
      isRecentlyClosed = true;

      // Compute the last 5 trading days before today
      const recentCutoff = new Date(today);
      let tradingDaysBack = 0;
      while (tradingDaysBack < 5) {
        recentCutoff.setDate(recentCutoff.getDate() - 1);
        const day = recentCutoff.getDay();
        if (day !== 0 && day !== 6) tradingDaysBack++;
      }
      const cutoffStr = recentCutoff.toISOString().split('T')[0];

      openTrades = trades.filter(t => t.exit_date >= cutoffStr);
    }

    function buildPosition(t: ReturnType<typeof parseTrades>[number]): Position {
      const isOption = t.instrument?.toUpperCase().includes('OPTION');
      const maxHoldDays = isOption ? 13 : 20;

      // Compute days held/remaining from entry date to today
      const daysHeld = tradingDaysSinceEntry(t.entry_date);
      const daysRemaining = Math.max(0, maxHoldDays - daysHeld);

      // Unrealized P&L proxy: use exit_price (or entry_price if not yet set) as current price.
      // For options: exit_price is the option price at expiry/exit — do NOT use live stock price.
      const exitPriceProxy = t.exit_price || t.entry_price;
      const defaultUnrealized = isOption
        ? (exitPriceProxy - t.entry_price) * t.shares_or_contracts * 100
        : (exitPriceProxy - t.entry_price) * t.shares_or_contracts;
      const defaultUnrealizedPct = t.entry_price > 0
        ? ((exitPriceProxy / t.entry_price) - 1) * 100
        : 0;

      // Approximate stop prices:
      // Stocks: 10% trailing stop below entry (proxy for 3×ATR stop from signal day)
      // Options: 50% loss stop on option premium (standard risk management)
      const stopPrice = isOption
        ? t.entry_price * 0.50
        : t.entry_price * 0.90;

      const currentPriceProxy = exitPriceProxy;
      const distanceToStop = currentPriceProxy > 0
        ? ((currentPriceProxy - stopPrice) / currentPriceProxy) * 100
        : 100;

      return {
        ...t,
        days_held: daysHeld,
        days_remaining: daysRemaining,
        scheduled_exit_date: t.exit_date,
        max_hold_days: maxHoldDays,
        is_open: !isRecentlyClosed,
        unrealized_pnl: defaultUnrealized,
        unrealized_pnl_pct: defaultUnrealizedPct,
        current_price: currentPriceProxy,
        stop_price: stopPrice,
        distance_to_stop_pct: distanceToStop,
        distance_to_stop_dollar: currentPriceProxy - stopPrice,
      } as Position;
    }

    const positionsToShow: Position[] = openTrades.map(buildPosition);

    // Fetch live stock prices for STOCK positions only.
    // Options use the exit_price from the CSV — applying live stock prices to options
    // would give wildly inflated P&L (e.g. NFLX stock at $850 vs $4.63 option entry).
    const stockSymbols = [
      ...new Set(
        positionsToShow
          .filter(p => !p.instrument?.toUpperCase().includes('OPTION'))
          .map(p => p.symbol)
      ),
    ];
    const prices = await fetchMultiplePrices(stockSymbols);

    // Enrich STOCK positions with live prices where available
    const enriched = positionsToShow.map(pos => {
      const isOption = pos.instrument?.toUpperCase().includes('OPTION');

      // Options: never update with stock price — P&L stays at exit_price proxy
      if (isOption) return pos;

      const livePrice = prices.get(pos.symbol);
      if (!livePrice) return pos;

      const unrealizedPnl = (livePrice - pos.entry_price) * pos.shares_or_contracts;
      const unrealizedPct = pos.entry_price > 0
        ? ((livePrice / pos.entry_price) - 1) * 100
        : 0;

      // Recompute distance to stop with live price
      const stopPrice = pos.stop_price ?? pos.entry_price * 0.90;
      const distanceToStop = livePrice > 0
        ? ((livePrice - stopPrice) / livePrice) * 100
        : 100;

      return {
        ...pos,
        current_price: livePrice,
        unrealized_pnl: unrealizedPnl,
        unrealized_pnl_pct: unrealizedPct,
        distance_to_stop_pct: distanceToStop,
        distance_to_stop_dollar: livePrice - stopPrice,
      };
    });

    // summary.total_positions = count of truly open positions
    const totalPositions = enriched.length;

    // Total unrealized = sum across only these open positions
    const totalUnrealized = enriched.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);

    // Realized P&L = sum of net_pnl from ALL historical closed trades
    const closedTrades = trades.filter(t => t.exit_price > 0 && t.net_pnl !== 0);
    const totalRealized = closedTrades.reduce((s, t) => s + t.net_pnl, 0);

    const currentMonth = today.substring(0, 7); // e.g. "2026-04"

    // Month unrealized: only open positions entered this month
    const monthUnrealized = enriched
      .filter(p => p.entry_date.startsWith(currentMonth))
      .reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);

    // Month realized: closed trades where exit happened this month
    const monthRealized = closedTrades
      .filter(t => (t.exit_date || '').startsWith(currentMonth))
      .reduce((s, t) => s + t.net_pnl, 0);

    // Risk alerts only apply to open positions
    // (enriched array already contains only open/recently-closed positions)

    return NextResponse.json({
      positions: enriched,
      count: enriched.length,
      is_demo: false,
      demo_note: isRecentlyClosed
        ? 'No positions with future exit dates found — showing recently closed positions from the last 5 trading days.'
        : null,
      summary: {
        total_positions: totalPositions,
        total_unrealized: totalUnrealized,
        total_realized: totalRealized,
        month_unrealized: monthUnrealized,
        month_realized: monthRealized,
        current_month: currentMonth,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error('Error in /api/positions:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
