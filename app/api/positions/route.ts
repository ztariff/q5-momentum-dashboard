import { NextResponse } from 'next/server';
import { parseTrades } from '@/lib/csvParser';
import { deriveOpenPositions } from '@/lib/positions';
import { fetchMultiplePrices } from '@/lib/polygon';
import { Position } from '@/lib/types';
import path from 'path';
import fs from 'fs';

// Strategy date - use today but cap at last data date
const STRATEGY_DATE = '2026-04-03';

export async function GET() {
  try {
    const csvPath = path.join(process.cwd(), 'public', 'data', 'hybrid_all_trades_calendar.csv');
    const csvText = fs.readFileSync(csvPath, 'utf-8');
    const trades = parseTrades(csvText);
    
    // Find genuinely open positions (entry <= today, exit > today)
    let openPositions = deriveOpenPositions(trades);
    
    // If no truly open positions (backtest ended), show positions from the last
    // trading week as "recently active" to demo the UI, with a data note
    let isDemo = false;
    if (openPositions.length === 0) {
      isDemo = true;
      // Get positions that were recently open (exited within last 15 calendar days)
      const cutoff = '2026-03-20';
      const recentTrades = trades.filter(t => 
        t.entry_date >= '2026-01-01' && 
        t.exit_date >= cutoff
      );
      
      // Map to Position type
      openPositions = recentTrades.map(t => {
        const isOption = t.instrument?.toUpperCase().includes('OPTION');
        const maxHoldDays = isOption ? 13 : 20;
        
        const pos: Position = {
          ...t,
          days_held: t.hold_days,
          days_remaining: 0,
          scheduled_exit_date: t.exit_date,
          max_hold_days: maxHoldDays,
          is_open: false, // mark as closed/demo
          unrealized_pnl: t.net_pnl,
          unrealized_pnl_pct: t.return_pct,
          current_price: t.exit_price,
        };
        return pos;
      });
    }
    
    // Get unique symbols for price fetching
    const symbols = [...new Set(openPositions.map(p => p.symbol))];
    
    // Fetch current prices for all symbols
    const prices = await fetchMultiplePrices(symbols);
    
    // Enrich positions with live prices (if we have them)
    const enriched = openPositions.map(pos => {
      const livePrice = prices.get(pos.symbol);
      
      if (livePrice && pos.is_open) {
        const isStock = pos.instrument?.toUpperCase().startsWith('STOCK');
        const unrealizedPnl = isStock 
          ? (livePrice - pos.entry_price) * pos.shares_or_contracts
          : 0; // Options need option price, not underlying
        const unrealizedPct = ((livePrice / pos.entry_price) - 1) * 100;
        
        return {
          ...pos,
          current_price: livePrice,
          unrealized_pnl: unrealizedPnl,
          unrealized_pnl_pct: unrealizedPct,
        };
      }
      
      // For demo/closed positions, also enrich with live price
      if (livePrice) {
        return {
          ...pos,
          current_price: livePrice,
        };
      }
      
      return pos;
    });
    
    return NextResponse.json({ 
      positions: enriched, 
      count: enriched.length,
      is_demo: isDemo,
      demo_note: isDemo ? 'Backtest data ends 2026-04-02. Showing recent closed positions with live prices for demonstration.' : null,
      strategy_date: STRATEGY_DATE,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error('Error in /api/positions:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
