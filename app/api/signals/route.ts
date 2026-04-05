import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

/**
 * GET /api/signals
 * Pure display layer — reads pre-computed pending_signals from live_state.json.
 * All computation happens in /api/refresh (the daily engine).
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

    const pendingSignals = liveState.pending_signals ?? [];

    // Build watchlist-style data from pending signals (Q5 stocks near trigger)
    // For a pure display layer, this just surfaces what /api/refresh wrote
    const signals = pendingSignals.map((s: Record<string, unknown>) => ({
      symbol: s.symbol,
      z_score: s.z_score,
      slope: 0, // not stored in live_state — could be added in future refresh
      quintile: 5,
      body_pct: s.body_pct,
      atr_change: s.atr_change_3d,
      tier: s.tier,
      sizing_rule: s.sizing_rule,
      recommended_size: s.position_size,
      recommended_contracts: s.tier === 'C' ? null : Math.max(1, Math.floor((s.position_size as number) / ((s.estimated_entry as number) * 100))),
      is_new_signal: true,
    }));

    return NextResponse.json({
      signals,
      watchlist: [], // watchlist computation requires full quintile history — deferred to future refresh enhancement
      total_scanned: liveState.summary?.total_positions !== undefined ? 230 : 0,
      timestamp: liveState.last_refresh,
      market_date: liveState.market_date,
      last_refresh: liveState.last_refresh,
    });
  } catch (err) {
    console.error('Error in /api/signals:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
