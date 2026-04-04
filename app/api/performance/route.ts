import { NextResponse } from 'next/server';
import { parseTrades, parseMonthly } from '@/lib/csvParser';
import path from 'path';
import fs from 'fs';

export async function GET() {
  try {
    const csvPath = path.join(process.cwd(), 'public', 'data', 'hybrid_all_trades_calendar.csv');
    const monthlyPath = path.join(process.cwd(), 'public', 'data', 'portfolio_monthly.csv');
    
    const csvText = fs.readFileSync(csvPath, 'utf-8');
    const monthlyText = fs.readFileSync(monthlyPath, 'utf-8');
    
    const trades = parseTrades(csvText);
    const monthlyData = parseMonthly(monthlyText);
    
    // Compute performance metrics from closed trades
    const closedTrades = trades.filter(t => t.exit_price > 0 && t.net_pnl !== 0);
    
    const winners = closedTrades.filter(t => t.net_pnl > 0);
    const losers = closedTrades.filter(t => t.net_pnl < 0);
    
    const totalPnl = closedTrades.reduce((sum, t) => sum + t.net_pnl, 0);
    const winRate = closedTrades.length > 0 ? (winners.length / closedTrades.length) * 100 : 0;
    
    const sumWins = winners.reduce((s, t) => s + t.net_pnl, 0);
    const sumLosses = Math.abs(losers.reduce((s, t) => s + t.net_pnl, 0));
    const profitFactor = sumLosses > 0 ? sumWins / sumLosses : 999;
    
    const stopTrades = closedTrades.filter(t => t.exit_type?.includes('STOP'));
    const stopRate = closedTrades.length > 0 ? (stopTrades.length / closedTrades.length) * 100 : 0;
    
    // Build equity curve from monthly data
    let cumulative = 0;
    const equityCurve = monthlyData.map(m => {
      cumulative += m.net_total * 1000; // monthly data is in thousands
      return {
        date: m.month,
        equity: cumulative,
      };
    });
    
    // Compute max drawdown from equity curve
    let peak = 0;
    let maxDrawdown = 0;
    for (const point of equityCurve) {
      if (point.equity > peak) peak = point.equity;
      const dd = point.equity - peak;
      if (dd < maxDrawdown) maxDrawdown = dd;
    }
    
    // Yearly P&L
    const yearlyMap = new Map<string, number>();
    for (const t of closedTrades) {
      const year = t.exit_date?.substring(0, 4);
      if (year) {
        yearlyMap.set(year, (yearlyMap.get(year) || 0) + t.net_pnl);
      }
    }
    
    const yearlyPnl = Array.from(yearlyMap.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([year, pnl]) => ({ year, pnl }));
    
    return NextResponse.json({
      total_realized_pnl: totalPnl,
      total_unrealized_pnl: 0, // Will be added by frontend
      total_pnl: totalPnl,
      win_rate: winRate,
      profit_factor: profitFactor,
      max_drawdown: maxDrawdown,
      sharpe: 0.61, // From strategy_parameters.csv
      total_trades: closedTrades.length,
      avg_pnl_per_trade: closedTrades.length > 0 ? totalPnl / closedTrades.length : 0,
      stop_rate: stopRate,
      monthly_data: monthlyData,
      equity_curve: equityCurve,
      yearly_pnl: yearlyPnl,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error('Error in /api/performance:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
