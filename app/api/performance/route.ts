import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

export async function GET() {
  try {
    const liveStatePath = path.join(process.cwd(), 'public', 'data', 'live_state.json');
    const liveState = JSON.parse(fs.readFileSync(liveStatePath, 'utf-8'));

    const summary = liveState.summary ?? {};
    const yearlyPnl = liveState.yearly_pnl ?? [];
    const monthlyEquity = liveState.monthly_equity ?? [];

    // Build equity curve from monthly data
    const equityCurve = monthlyEquity.map((m: { date: string; equity: number }) => ({
      date: m.date,
      equity: m.equity,
    }));

    // Monthly data for bar chart
    const monthlyData = monthlyEquity.map((m: { date: string; pnl: number }) => ({
      month: m.date,
      net_total: m.pnl / 1000, // convert to thousands for chart
      win_rate: 0,
      days: 0,
    }));

    const totalPnl = summary.realized_pnl ?? 0;

    return NextResponse.json({
      total_realized_pnl: totalPnl,
      total_unrealized_pnl: summary.total_unrealized ?? 0,
      total_pnl: totalPnl + (summary.total_unrealized ?? 0),
      win_rate: summary.win_rate ?? 0,
      profit_factor: summary.profit_factor ?? 0,
      max_drawdown: -410000, // from backtest
      sharpe: 1.71,
      total_trades: 2082,
      avg_pnl_per_trade: Math.round(totalPnl / 2082),
      stop_rate: 22,
      monthly_data: monthlyData,
      equity_curve: equityCurve,
      yearly_pnl: yearlyPnl,
      timestamp: new Date().toISOString(),
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
