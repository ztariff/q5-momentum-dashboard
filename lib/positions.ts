import { Trade, Position } from './types';

// Today's date
const TODAY = new Date('2026-04-03');

// Get trading days between two dates (approximate using ~252 trading days/year)
function getTradingDaysBetween(startDate: string, endDate: string): number {
  const start = new Date(startDate);
  const end = new Date(endDate);
  let count = 0;
  const current = new Date(start);
  
  while (current <= end) {
    const day = current.getDay();
    if (day !== 0 && day !== 6) {
      count++;
    }
    current.setDate(current.getDate() + 1);
  }
  
  return count;
}

function addTradingDays(startDate: string, tradingDays: number): string {
  const start = new Date(startDate);
  let count = 0;
  const current = new Date(start);
  
  while (count < tradingDays) {
    current.setDate(current.getDate() + 1);
    const day = current.getDay();
    if (day !== 0 && day !== 6) {
      count++;
    }
  }
  
  return current.toISOString().split('T')[0];
}

export function deriveOpenPositions(trades: Trade[]): Position[] {
  const todayStr = TODAY.toISOString().split('T')[0];
  const openPositions: Position[] = [];
  
  for (const trade of trades) {
    // Check if the trade is currently open:
    // - entry_date <= today
    // - exit_date > today (not yet exited)
    const entryDate = new Date(trade.entry_date);
    const exitDate = new Date(trade.exit_date);
    
    if (entryDate <= TODAY && exitDate > TODAY) {
      const maxHoldDays = trade.instrument === 'OPTION' || trade.instrument?.includes('OPTION') ? 13 : 20;
      const daysHeld = getTradingDaysBetween(trade.entry_date, todayStr) - 1;
      const daysRemaining = Math.max(0, maxHoldDays - daysHeld);
      
      const position: Position = {
        ...trade,
        days_held: daysHeld,
        days_remaining: daysRemaining,
        scheduled_exit_date: trade.exit_date,
        max_hold_days: maxHoldDays,
        is_open: true,
      };
      
      openPositions.push(position);
    }
  }
  
  return openPositions;
}

export function enrichPositionsWithPrices(
  positions: Position[],
  prices: Map<string, number>
): Position[] {
  return positions.map(pos => {
    // For options, we use the option ticker; for stocks use the symbol
    const priceKey = pos.instrument === 'STOCK' || 
                     pos.instrument?.includes('fallback') || 
                     pos.instrument?.startsWith('STOCK') 
                     ? pos.symbol 
                     : pos.symbol; // For options we still use underlying symbol for tracking
    
    const currentPrice = prices.get(priceKey);
    
    if (!currentPrice) {
      return { ...pos, current_price: pos.entry_price };
    }
    
    let unrealizedPnl = 0;
    let unrealizedPct = 0;
    
    if (pos.instrument === 'STOCK' || pos.instrument?.includes('fallback') || pos.instrument?.startsWith('STOCK')) {
      // For stock: (current - entry) * shares
      unrealizedPnl = (currentPrice - pos.entry_price) * pos.shares_or_contracts;
      unrealizedPct = ((currentPrice / pos.entry_price) - 1) * 100;
    } else {
      // For options: use underlying price change as proxy
      // The actual option position would require option price data
      // Show underlying price and mark P&L as approximate
      unrealizedPnl = (currentPrice - pos.entry_price) * pos.shares_or_contracts;
      unrealizedPct = ((currentPrice / pos.entry_price) - 1) * 100;
    }
    
    return {
      ...pos,
      current_price: currentPrice,
      unrealized_pnl: unrealizedPnl,
      unrealized_pct: unrealizedPct,
    };
  });
}

export function computeStopPrice(
  signalDayLow: number,
  atr14: number
): number {
  return signalDayLow - 3.0 * atr14;
}

export function getPositionStatus(pos: Position): 'profitable' | 'losing' | 'near_exit' | 'deep_loss' {
  if (pos.days_remaining !== undefined && pos.days_remaining <= 2) {
    return 'near_exit';
  }
  if (pos.unrealized_pct !== undefined && pos.unrealized_pct < -50) {
    return 'deep_loss';
  }
  if (pos.unrealized_pnl !== undefined && pos.unrealized_pnl > 0) {
    return 'profitable';
  }
  return 'losing';
}
