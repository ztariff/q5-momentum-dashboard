import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

/**
 * GET /api/positions
 * Pure display layer — reads pre-computed data from live_state.json.
 * All signal computation happens in /api/refresh.
 *
 * pending_signals are merged in as status='pending' rows so they appear
 * in Current Positions with amber PENDING highlighting.
 * They do NOT appear in the Watchlist tab.
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

    const activePositions = liveState.positions ?? [];
    const pendingSignals = liveState.pending_signals ?? [];
    const summary = liveState.summary ?? {};

    // Map pending signals to the Position shape expected by PositionsTable.
    // Every field that might be undefined is given a safe fallback so the
    // table never renders "undefined".
    const pendingPositions = pendingSignals.map((s: Record<string, unknown>) => {
      const estEntry = typeof s.estimated_entry === 'number' ? s.estimated_entry : null;
      const size = typeof s.recommended_size === 'number' ? s.recommended_size : 0;

      return {
        symbol: s.symbol ?? '—',
        status: 'pending',
        instrument: 'OPTION',
        tier: s.tier ?? '?',
        option_ticker: s.option_ticker ?? null,
        // entry date is the signal date (next market open is entry)
        entry_date: liveState.market_date ?? new Date().toISOString().substring(0, 10),
        entry_price: estEntry ?? 0,
        current_price: null,
        stop_price: null,
        exit_date: null,
        scheduled_exit_date: null,
        position_size: size,
        shares_or_contracts: typeof s.recommended_contracts === 'number' ? s.recommended_contracts : 0,
        unrealized_pnl: 0,
        unrealized_pnl_pct: 0,
        days_held: 0,
        days_remaining: null,
        max_hold_days: 0,
        is_open: false,
        distance_to_stop_pct: undefined,
        // sizing info surfaced for reference
        sizing_rule: typeof s.sizing_rule === 'string' ? s.sizing_rule : null,
        recommended_contracts: typeof s.recommended_contracts === 'number' ? s.recommended_contracts : null,
        z_score: typeof s.z_score === 'number' ? s.z_score : null,
        body_pct: typeof s.body_pct === 'number' ? s.body_pct : null,
      };
    });

    // Pending entries shown first, then active positions
    const allPositions = [...pendingPositions, ...activePositions];

    return NextResponse.json({
      positions: allPositions,
      count: allPositions.length,
      pending_count: pendingPositions.length,
      summary: {
        total_positions: summary.total_positions ?? 0,
        total_unrealized: summary.total_unrealized ?? 0,
        total_realized: summary.realized_pnl ?? 0,
        month_unrealized: summary.month_unrealized ?? 0,
        month_realized: summary.month_realized ?? 0,
        year_realized: summary.year_realized ?? 0,
        current_month: liveState.market_date?.substring(0, 7) ?? '',
      },
      last_refresh: liveState.last_refresh ?? null,
      market_date: liveState.market_date ?? null,
      risk_alerts: liveState.risk_alerts ?? [],
      recently_closed: liveState.recently_closed ?? [],
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error('Error in /api/positions:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
