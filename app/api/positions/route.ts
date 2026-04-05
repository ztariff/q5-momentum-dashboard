import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

/**
 * GET /api/positions
 * Pure display layer — reads pre-computed data from live_state.json.
 * All signal computation happens in /api/refresh.
 */
export async function GET() {
  try {
    const liveStatePath = path.join(process.cwd(), 'public', 'data', 'live_state.json');

    if (!fs.existsSync(liveStatePath)) {
      return NextResponse.json(
        { error: 'live_state.json not found — run /api/refresh first' },
        { status: 404 }
      );
    }

    const liveState = JSON.parse(fs.readFileSync(liveStatePath, 'utf-8'));

    const positions = liveState.positions ?? [];
    const pending = liveState.pending_signals ?? [];
    const summary = liveState.summary ?? {};

    // Merge active + pending for the combined positions table
    // Pending signals are prepended as they represent tomorrow's entries
    const allPositions = [
      ...pending.map((s: Record<string, unknown>) => ({
        ...s,
        // Map pending_signal fields to Position shape for table compatibility
        instrument: 'OPTION',
        entry_date: liveState.market_date,
        entry_price: s.estimated_entry,
        current_price: s.estimated_entry,
        exit_date: null,
        unrealized_pnl: 0,
        unrealized_pct: 0,
        days_held: 0,
        days_remaining: null,
        is_open: false,
      })),
      ...positions,
    ];

    return NextResponse.json({
      positions: allPositions,
      count: allPositions.length,
      pending_count: pending.length,
      summary: {
        total_positions: summary.total_positions ?? 0,
        total_unrealized: summary.total_unrealized ?? 0,
        total_realized: summary.realized_pnl ?? 0,
        month_unrealized: summary.month_unrealized ?? 0,
        month_realized: summary.month_realized ?? 0,
        year_realized: summary.year_realized ?? 0,
        current_month: liveState.market_date?.substring(0, 7) ?? '',
      },
      last_refresh: liveState.last_refresh,
      market_date: liveState.market_date,
      risk_alerts: liveState.risk_alerts ?? [],
      recently_closed: liveState.recently_closed ?? [],
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error('Error in /api/positions:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
