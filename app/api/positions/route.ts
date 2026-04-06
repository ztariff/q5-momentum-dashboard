import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

export async function GET() {
  try {
    const liveStatePath = path.join(process.cwd(), 'public', 'data', 'live_state.json');

    if (!fs.existsSync(liveStatePath)) {
      return NextResponse.json(
        { error: 'live_state.json not found' },
        { status: 404 }
      );
    }

    const liveState = JSON.parse(fs.readFileSync(liveStatePath, 'utf-8'));

    const activePositions = liveState.positions ?? [];
    const pendingSignals = liveState.pending_signals ?? [];
    const summary = liveState.summary ?? {};
    const riskAlerts = liveState.risk_alerts ?? [];
    const recentlyClosed = liveState.recently_closed ?? [];

    // Merge pending signals into positions as status='pending'
    const pendingPositions = pendingSignals.map((s: Record<string, unknown>) => ({
      symbol: s.symbol ?? '',
      status: 'pending',
      instrument: s.instrument ?? 'OPTION',
      tier: s.tier ?? '',
      option_ticker: '',
      entry_date: s.entry_date ?? '',
      entry_price: 0,
      current_price: 0,
      stop_price: null,
      exit_date: null,
      position_size: s.recommended_size ?? 0,
      shares_or_contracts: s.recommended_contracts ?? 0,
      unrealized_pnl: 0,
      unrealized_pct: 0,
      days_held: 0,
      days_remaining: null,
      body_pct: s.body_pct ?? null,
      atr_change_3d: s.atr_change ?? null,
      distance_to_stop_pct: null,
      sizing_rule: s.sizing_rule ?? '',
      is_open: true,
    }));

    const allPositions = [...pendingPositions, ...activePositions];

    return NextResponse.json({
      positions: allPositions,
      count: allPositions.length,
      summary,
      risk_alerts: riskAlerts,
      recently_closed: recentlyClosed,
      last_refresh: liveState.last_refresh,
      market_date: liveState.market_date,
      timestamp: new Date().toISOString(),
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
