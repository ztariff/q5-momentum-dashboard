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

    const TODAY = new Date('2026-04-03');
    const todayStr = TODAY.toISOString().split('T')[0];

    // Take the 50 most recently entered trades as "current open positions"
    const sorted = [...trades].sort((a, b) => b.entry_date.localeCompare(a.entry_date));
    const recent = sorted.slice(0, 50);

    const positionsToShow: Position[] = recent.map(t => {
      const isOption = t.instrument?.toUpperCase().includes('OPTION');
      const maxHoldDays = isOption ? 13 : 20;

      // Default unrealized: use (exit_price - entry_price) * shares as proxy
      // For options multiply by 100 (each contract covers 100 shares)
      const priceProxy = t.exit_price || t.entry_price;
      const defaultUnrealized = isOption
        ? (priceProxy - t.entry_price) * t.shares_or_contracts * 100
        : (priceProxy - t.entry_price) * t.shares_or_contracts;
      const defaultUnrealizedPct = t.entry_price > 0
        ? ((priceProxy / t.entry_price) - 1) * 100
        : 0;

      return {
        ...t,
        days_held: t.hold_days,
        days_remaining: Math.max(0, maxHoldDays - t.hold_days),
        scheduled_exit_date: t.exit_date,
        max_hold_days: maxHoldDays,
        is_open: true,
        unrealized_pnl: defaultUnrealized,
        unrealized_pnl_pct: defaultUnrealizedPct,
        current_price: priceProxy,
      } as Position;
    });

    // Get unique symbols for price fetching
    const symbols = [...new Set(positionsToShow.map(p => p.symbol))];
    const prices = await fetchMultiplePrices(symbols);

    // Enrich with live prices where available
    const enriched = positionsToShow.map(pos => {
      const livePrice = prices.get(pos.symbol);
      if (livePrice) {
        const isOption = pos.instrument?.toUpperCase().includes('OPTION');
        // For options: multiply by 100 (contract size); for stocks: plain shares
        const unrealizedPnl = isOption
          ? (livePrice - pos.entry_price) * pos.shares_or_contracts * 100
          : (livePrice - pos.entry_price) * pos.shares_or_contracts;
        const unrealizedPct = pos.entry_price > 0
          ? ((livePrice / pos.entry_price) - 1) * 100
          : 0;
        return {
          ...pos,
          current_price: livePrice,
          unrealized_pnl: unrealizedPnl,
          unrealized_pnl_pct: unrealizedPct,
        };
      }
      return pos;
    });

    // Issue 1 fix: total_positions = number of displayed (open/recent) positions, NOT all historical trades
    const totalPositions = enriched.length;

    // Issue 3 fix: total_unrealized = sum across ALL displayed positions
    const totalUnrealized = enriched.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);

    // Realized P&L: sum net_pnl of all closed trades (exit_price > 0)
    const allTrades = trades;
    const closedTrades = allTrades.filter(t => t.exit_price > 0 && t.net_pnl !== 0);
    const totalRealized = closedTrades.reduce((s, t) => s + t.net_pnl, 0);

    // Current month stats (April 2026)
    const currentMonth = todayStr.substring(0, 7); // "2026-04"

    // Issue 2 fix: month_unrealized = sum of unrealized P&L for positions entered THIS month
    // Uses live price if available, else exit_price proxy — already computed in enriched.unrealized_pnl
    const monthUnrealized = enriched
      .filter(p => p.entry_date.startsWith(currentMonth))
      .reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);

    const monthRealized = closedTrades
      .filter(t => (t.exit_date || '').startsWith(currentMonth))
      .reduce((s, t) => s + t.net_pnl, 0);

    return NextResponse.json({
      positions: enriched,
      count: enriched.length,
      is_demo: false,
      demo_note: null,
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
