import { NextResponse } from 'next/server';
import { parseTrades } from '@/lib/csvParser';
import { fetchMultiplePrices } from '@/lib/polygon';
import { Position } from '@/lib/types';
import path from 'path';
import fs from 'fs';

const SCALE_FACTOR = 0.0666; // Target $40K daily vol = 1/15th of backtest size

// Scale sizing_rule text to reflect new dollar amounts
function scaleSizingRule(rule: string): string {
  return rule
    .replace(/\$2[,.]?000K|\$2M/g, '$133K')
    .replace(/\$1[,.]?500K|\$1\.5M/g, '$100K')
    .replace(/\$1[,.]?000K|\$1M/g, '$67K')
    .replace(/\$500K/g, '$33K')
    .replace(/\$200K/g, '$13K')
    .replace(/\$100K/g, '$7K');
}

/**
 * Compute the proper exit date for a trade: entry_date + holdDays business days.
 * Options hold 13 biz days; stocks hold 20 biz days.
 */
function computeProperExitDate(entryDate: string, instrument: string): string {
  const entry = new Date(entryDate);
  const isOption = instrument?.toUpperCase().includes('OPTION');
  const holdDays = isOption ? 13 : 20;
  let bizDays = 0;
  const d = new Date(entry);
  while (bizDays < holdDays) {
    d.setDate(d.getDate() + 1);
    const dow = d.getDay();
    if (dow !== 0 && dow !== 6) bizDays++; // skip weekends
  }
  return d.toISOString().split('T')[0];
}

export async function GET() {
  try {
    const csvPath = path.join(process.cwd(), 'public', 'data', 'hybrid_all_trades_calendar.csv');
    const csvText = fs.readFileSync(csvPath, 'utf-8');
    const rawTrades = parseTrades(csvText);

    // Apply scale factor to position sizes, shares, and P&L
    const trades = rawTrades.map(t => ({
      ...t,
      position_size: Math.round(t.position_size * SCALE_FACTOR),
      shares_or_contracts: Math.floor(t.shares_or_contracts * SCALE_FACTOR),
      net_pnl: t.net_pnl * SCALE_FACTOR,
      sizing_rule: scaleSizingRule(t.sizing_rule || ''),
    }));

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

    // Helper: count business days between two date strings (start exclusive, end inclusive)
    function bizDaysBetween(startDate: string, endDate: string): number {
      const start = new Date(startDate);
      const end = new Date(endDate);
      if (end <= start) return 0;
      let count = 0;
      const cur = new Date(start);
      while (cur < end) {
        cur.setDate(cur.getDate() + 1);
        const dow = cur.getDay();
        if (dow !== 0 && dow !== 6) count++;
      }
      return count;
    }

    // For each trade, determine whether it was backtest-truncated.
    // If the proper exit date (entry + hold days) is later than the CSV exit date,
    // the trade was cut short by the backtest end — it is actually still OPEN.
    type TradeWithProperExit = (typeof trades)[number] & {
      proper_exit_date: string;
      csv_exit_date: string;
      is_truncated: boolean;
    };

    const tradesWithProper: TradeWithProperExit[] = trades.map(t => {
      const proper_exit_date = computeProperExitDate(t.entry_date, t.instrument);
      const csv_exit_date = t.exit_date;
      const is_truncated = proper_exit_date > csv_exit_date;
      return { ...t, proper_exit_date, csv_exit_date, is_truncated };
    });

    // Open positions: those whose proper exit date is today or in the future
    const openTrades = tradesWithProper.filter(t => t.proper_exit_date >= today);

    let isRecentlyClosed = false;
    let positionsSource: TradeWithProperExit[] = openTrades;

    // Fallback: if zero open positions, show recently closed trades
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

      // Use proper_exit_date for recently-closed fallback as well
      positionsSource = tradesWithProper.filter(t => t.proper_exit_date >= cutoffStr);
    }

    function buildPosition(t: TradeWithProperExit): Position {
      const isOption = t.instrument?.toUpperCase().includes('OPTION');
      const maxHoldDays = isOption ? 13 : 20;

      // Use the proper exit date as the canonical exit date.
      // For truncated (open) trades this will be in the future.
      const canonicalExitDate = t.proper_exit_date;

      // Days held so far (from entry to today)
      const daysHeld = tradingDaysSinceEntry(t.entry_date);

      // Days remaining = business days from today to proper exit
      const daysRemaining = isRecentlyClosed
        ? 0
        : Math.max(0, bizDaysBetween(today, canonicalExitDate));

      // For truncated/open positions: last known price from the CSV is the
      // most recent mark-to-market price. For normally-exited trades: use exit_price.
      const lastKnownPrice = t.exit_price || t.entry_price;

      // Unrealized P&L for open positions (mark-to-market vs entry)
      // shares_or_contracts is already scaled.
      const unrealizedPnl = isOption
        ? (lastKnownPrice - t.entry_price) * t.shares_or_contracts * 100
        : (lastKnownPrice - t.entry_price) * t.shares_or_contracts;
      const unrealizedPnlPct = t.entry_price > 0
        ? ((lastKnownPrice / t.entry_price) - 1) * 100
        : 0;

      // Approximate stop prices:
      // Options: 50% loss stop on option premium
      // Stocks: 10% trailing stop below entry (proxy for 3×ATR stop)
      const stopPrice = isOption
        ? t.entry_price * 0.50
        : t.entry_price * 0.90;

      const currentPriceProxy = lastKnownPrice;
      const distanceToStop = currentPriceProxy > 0
        ? ((currentPriceProxy - stopPrice) / currentPriceProxy) * 100
        : 100;

      const isOpen = !isRecentlyClosed;

      return {
        ...t,
        // Override exit_date with the proper (potentially future) exit date
        exit_date: canonicalExitDate,
        // For open positions the exit_price is the last known price (mark to market)
        exit_price: isOpen && t.is_truncated ? lastKnownPrice : t.exit_price,
        days_held: daysHeld,
        days_remaining: daysRemaining,
        scheduled_exit_date: canonicalExitDate,
        max_hold_days: maxHoldDays,
        is_open: isOpen,
        unrealized_pnl: unrealizedPnl,
        unrealized_pnl_pct: unrealizedPnlPct,
        current_price: currentPriceProxy,
        stop_price: stopPrice,
        distance_to_stop_pct: distanceToStop,
        distance_to_stop_dollar: currentPriceProxy - stopPrice,
      } as Position;
    }

    const positionsToShow: Position[] = positionsSource.map(buildPosition);

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

      // Options: never update with stock price — P&L stays at last known price proxy
      if (isOption) return pos;

      const livePrice = prices.get(pos.symbol);
      if (!livePrice) return pos;

      // shares_or_contracts is already scaled — unrealized P&L reflects scaled size
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

    // Realized P&L = sum of net_pnl from ALL historical closed trades (already scaled)
    // Exclude truncated (still-open) trades from realized totals
    const closedTrades = trades.filter(t => {
      const proper = computeProperExitDate(t.entry_date, t.instrument);
      const isTruncated = proper > t.exit_date;
      return !isTruncated && t.exit_price > 0 && t.net_pnl !== 0;
    });
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
