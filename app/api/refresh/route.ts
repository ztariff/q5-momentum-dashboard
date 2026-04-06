import { NextResponse } from 'next/server';
import { fetchCurrentPrice } from '@/lib/polygon';
import path from 'path';
import fs from 'fs';

// ─── Types ────────────────────────────────────────────────────────────────────

interface LivePosition {
  symbol: string;
  status: 'active' | 'exit_today' | 'pending';
  instrument: string;
  tier: string;
  option_ticker: string;
  entry_date: string;
  entry_price: number;
  current_price: number;
  stop_price: number;
  exit_date: string;
  position_size: number;
  shares_or_contracts: number;
  unrealized_pnl: number;
  unrealized_pct: number;
  distance_to_stop_pct: number;
  days_held: number;
  days_remaining: number;
  body_pct: number;
  atr_change_3d: number;
}

interface LiveState {
  last_refresh: string;
  market_date: string;
  positions: LivePosition[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  pending_signals: any[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  recently_closed: any[];
  summary: {
    total_positions: number;
    total_pending: number;
    total_unrealized: number;
    realized_pnl: number;
    month_realized: number;
    month_unrealized: number;
    win_rate: number;
    profit_factor: number;
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  risk_alerts: any[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

function tradingDaysSince(entryDate: string, today: string): number {
  return bizDaysBetween(entryDate, today);
}

// ─── Main refresh handler ─────────────────────────────────────────────────────

export async function GET() {
  const dataDir = path.join(process.cwd(), 'public', 'data');
  const liveStatePath = path.join(dataDir, 'live_state.json');

  // 1. Read current live_state.json
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

  // 2. Identify open positions and fetch current prices for stocks only
  const openPositions = liveState.positions.filter(
    p => p.status === 'active' || p.status === 'exit_today'
  );

  const stockSymbols = openPositions
    .filter(p => !p.instrument.toUpperCase().includes('OPTION'))
    .map(p => p.symbol);

  // Fetch prices in parallel (only stocks — options use last known price)
  const priceResults = await Promise.all(
    stockSymbols.map(async sym => ({
      symbol: sym,
      price: await fetchCurrentPrice(sym),
    }))
  );

  const priceMap = new Map<string, number>();
  for (const r of priceResults) {
    if (r.price !== null) priceMap.set(r.symbol, r.price);
  }

  // 3. Update each position's price fields
  const updatedPositions: LivePosition[] = liveState.positions.map(pos => {
    const isOption = pos.instrument.toUpperCase().includes('OPTION');

    // For options: keep last known price from live_state (unreliable after hours)
    // For stocks: use fresh Polygon price if available
    const currentPrice = isOption
      ? pos.current_price
      : (priceMap.get(pos.symbol) ?? pos.current_price);

    const entryPrice = pos.entry_price;
    const stopPrice = pos.stop_price;

    // Recalculate P&L
    const sharesOrContracts = pos.shares_or_contracts;
    const unrealizedPnl = isOption
      ? (currentPrice - entryPrice) * sharesOrContracts * 100
      : (currentPrice - entryPrice) * sharesOrContracts;

    const unrealizedPct = entryPrice > 0
      ? ((currentPrice / entryPrice) - 1) * 100
      : 0;

    const distanceToStopPct = currentPrice > 0
      ? ((currentPrice - stopPrice) / currentPrice) * 100
      : 100;

    // 4. Determine status
    const daysHeld = tradingDaysSince(pos.entry_date, today);
    const daysRemaining = Math.max(0, bizDaysBetween(today, pos.exit_date));

    let status: 'active' | 'exit_today' | 'pending' = pos.status;

    // Check stop: current price crossed below stop
    if (currentPrice < stopPrice && status !== 'pending') {
      status = 'exit_today';
    }

    // 5. Check exit_date: today >= exit_date
    if (today >= pos.exit_date && status !== 'pending') {
      status = 'exit_today';
    }

    // If neither triggered, revert to active (in case a previously flagged
    // position recovered intraday — price is now above stop and date not due)
    if (status === 'exit_today' && currentPrice >= stopPrice && today < pos.exit_date) {
      status = 'active';
    }

    return {
      ...pos,
      current_price: Math.round(currentPrice * 100) / 100,
      unrealized_pnl: Math.round(unrealizedPnl),
      unrealized_pct: Math.round(unrealizedPct * 10) / 10,
      distance_to_stop_pct: Math.round(distanceToStopPct * 10) / 10,
      days_held: daysHeld,
      days_remaining: daysRemaining,
      status,
    };
  });

  // Recompute summary totals
  const currentMonth = today.substring(0, 7);
  const totalUnrealized = updatedPositions
    .filter(p => p.status !== 'pending')
    .reduce((s, p) => s + p.unrealized_pnl, 0);
  const monthUnrealized = updatedPositions
    .filter(p => p.status !== 'pending' && p.entry_date.startsWith(currentMonth))
    .reduce((s, p) => s + p.unrealized_pnl, 0);

  // 6. Write updated live_state.json
  const updatedState: LiveState = {
    ...liveState,
    last_refresh: new Date().toISOString(),
    market_date: today,
    positions: updatedPositions,
    // pending_signals stays untouched — set externally by Python backtest
    summary: {
      ...liveState.summary,
      total_positions: updatedPositions.filter(p => p.status !== 'pending').length,
      total_unrealized: Math.round(totalUnrealized),
      month_unrealized: Math.round(monthUnrealized),
    },
  };

  try {
    try { fs.writeFileSync(liveStatePath, JSON.stringify(updatedState, null, 2)); } catch { /* read-only filesystem on Railway — skip write */ }
  } catch (err) {
    return NextResponse.json({ error: `Failed to write live_state.json: ${err}` }, { status: 500 });
  }

  return NextResponse.json({
    success: true,
    market_date: today,
    positions_updated: updatedPositions.length,
    prices_fetched: priceMap.size,
    last_refresh: updatedState.last_refresh,
  });
}
