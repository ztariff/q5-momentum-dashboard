import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

/**
 * GET /api/signals
 * Returns the Q4 watchlist from live_state.json.
 * These are symbols approaching Q5 — not yet actionable, just to monitor.
 *
 * Q5 signals (pending_signals) belong in Current Positions as PENDING entries.
 * They are served by /api/positions.
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

    // Only expose the Q4 watchlist here.
    // pending_signals (Q5) are handled exclusively by /api/positions.
    const watchlist = (liveState.watchlist ?? []).map((item: Record<string, unknown>) => ({
      symbol: item.symbol ?? null,
      quintile: item.quintile ?? 4,
      z_score: item.z_score ?? null,
      slope: item.slope ?? null,
      sma50_slope: item.sma50_slope ?? null,
      body_pct: item.body_pct ?? null,
      distance_to_q5: item.distance_to_q5 ?? null,
      approaching: item.approaching ?? false,
    }));

    return NextResponse.json({
      watchlist,
      total_scanned: liveState.total_scanned ?? 0,
      last_refresh: liveState.last_refresh ?? null,
      market_date: liveState.market_date ?? null,
    });
  } catch (err) {
    console.error('Error in /api/signals:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
