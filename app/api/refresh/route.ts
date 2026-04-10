import { NextResponse } from 'next/server';
import { fetchOptionSnapshots, fetchCurrentPrice } from '@/lib/polygon';
import path from 'path';
import fs from 'fs';

// Price-only refresh: updates current_price and unrealized P&L for active
// positions using Polygon option snapshots. Does NOT touch CSVs, signals,
// prev_quintiles, or any other state. Safe to run anytime, including during
// market hours.

interface LivePosition {
  symbol: string;
  status: string;
  instrument: string;
  option_ticker?: string;
  entry_date: string;
  entry_price: number;
  current_price: number;
  exit_date: string;
  shares_or_contracts: number;
  unrealized_pnl: number;
  unrealized_pct: number;
  days_held: number;
  days_remaining: number;
  [key: string]: unknown;
}

interface LiveState {
  last_refresh: string;
  market_date: string;
  positions: LivePosition[];
  summary: Record<string, number>;
  [key: string]: unknown;
}

function bizDaysBetween(startStr: string, endStr: string): number {
  const start = new Date(startStr);
  const end = new Date(endStr);
  if (end <= start) return 0;
  let count = 0;
  const cur = new Date(start);
  while (cur < end) {
    cur.setDate(cur.getDate() + 1);
    if (cur.getDay() !== 0 && cur.getDay() !== 6) count++;
  }
  return count;
}

export async function GET() {
  const liveStatePath = path.join(process.cwd(), 'public', 'data', 'live_state.json');

  let liveState: LiveState | null = null;
  try {
    if (fs.existsSync(liveStatePath)) {
      liveState = JSON.parse(fs.readFileSync(liveStatePath, 'utf-8')) as LiveState;
    }
  } catch {
    return NextResponse.json({ error: 'Failed to read live_state.json' }, { status: 500 });
  }

  if (!liveState) {
    return NextResponse.json({ error: 'live_state.json not found' }, { status: 404 });
  }

  const today = new Date().toISOString().split('T')[0];

  // Collect active option positions (skip pending, skipped, and closed)
  const activeOptions = liveState.positions.filter(
    p => (p.status === 'active' || p.status === 'exit_today') &&
         p.instrument?.toUpperCase().includes('OPTION') &&
         p.option_ticker
  );

  const optionTickers = activeOptions.map(p => p.option_ticker as string);

  // Fetch option snapshots in batches
  const optionPrices = await fetchOptionSnapshots(optionTickers);

  // Also fetch stock prices for any stock positions (legacy support)
  const stockPositions = liveState.positions.filter(
    p => (p.status === 'active' || p.status === 'exit_today') &&
         !p.instrument?.toUpperCase().includes('OPTION')
  );
  const stockPriceMap = new Map<string, number>();
  for (const pos of stockPositions) {
    const price = await fetchCurrentPrice(pos.symbol);
    if (price !== null) stockPriceMap.set(pos.symbol, price);
  }

  // Update positions
  const updatedPositions: LivePosition[] = liveState.positions.map(pos => {
    // Skip non-active positions — preserve as-is
    if (pos.status !== 'active' && pos.status !== 'exit_today') {
      return pos;
    }

    const isOption = pos.instrument?.toUpperCase().includes('OPTION');
    let currentPrice = pos.current_price;

    if (isOption && pos.option_ticker) {
      const fresh = optionPrices.get(pos.option_ticker);
      if (fresh !== undefined) currentPrice = fresh;
    } else if (!isOption) {
      const fresh = stockPriceMap.get(pos.symbol);
      if (fresh !== undefined) currentPrice = fresh;
    }

    // Recompute P&L
    const entry = pos.entry_price;
    const qty = pos.shares_or_contracts;
    const unrealizedPnl = isOption
      ? (currentPrice - entry) * qty * 100
      : (currentPrice - entry) * qty;
    const unrealizedPct = entry > 0 ? ((currentPrice / entry) - 1) * 100 : 0;

    // Update days held/remaining
    const daysHeld = bizDaysBetween(pos.entry_date, today);
    const daysRemaining = Math.max(0, bizDaysBetween(today, pos.exit_date));

    return {
      ...pos,
      current_price: Math.round(currentPrice * 100) / 100,
      unrealized_pnl: Math.round(unrealizedPnl),
      unrealized_pct: Math.round(unrealizedPct * 100) / 100,
      days_held: daysHeld,
      days_remaining: daysRemaining,
    };
  });

  // Recompute summary totals (exclude skipped and pending)
  const activeForSummary = updatedPositions.filter(
    p => p.status === 'active' || p.status === 'exit_today'
  );
  const totalUnrealized = activeForSummary.reduce((s, p) => s + (p.unrealized_pnl || 0), 0);

  const updatedState: LiveState = {
    ...liveState,
    last_refresh: new Date().toISOString(),
    positions: updatedPositions,
    summary: {
      ...liveState.summary,
      total_unrealized: Math.round(totalUnrealized),
      month_unrealized: Math.round(totalUnrealized),
      year_total: (liveState.summary?.year_realized || 0) + Math.round(totalUnrealized),
    },
  };

  try {
    fs.writeFileSync(liveStatePath, JSON.stringify(updatedState, null, 2));
  } catch (err) {
    return NextResponse.json({ error: `Failed to write live_state.json: ${err}` }, { status: 500 });
  }

  return NextResponse.json({
    success: true,
    positions_updated: activeForSummary.length,
    options_priced: optionPrices.size,
    stocks_priced: stockPriceMap.size,
    total_unrealized: Math.round(totalUnrealized),
    last_refresh: updatedState.last_refresh,
  });
}
