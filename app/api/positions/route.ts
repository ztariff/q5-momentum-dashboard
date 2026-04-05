import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

/**
 * GET /api/positions
 * Pure display layer — reads pre-computed data from live_state.json.
 * Returns only real positions. Pending signals belong to /api/signals.
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
    const summary = liveState.summary ?? {};

    return NextResponse.json({
      positions: liveState.positions ?? [],
      count: (liveState.positions ?? []).length,
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
