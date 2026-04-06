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

    const positions = liveState.positions ?? [];
    const summary = liveState.summary ?? {};
    const riskAlerts = liveState.risk_alerts ?? [];
    const recentlyClosed = liveState.recently_closed ?? [];

    return NextResponse.json({
      positions,
      count: positions.length,
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
