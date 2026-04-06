import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

<<<<<<< Updated upstream
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
=======
// Strategy date - use today but cap at last data date
const STRATEGY_DATE = '2026-04-03';

const SCALE_FACTOR = 0.0666; // Target $40K daily vol = 1/15th of backtest size

// Scale sizing_rule text to reflect new dollar amounts
function scaleSizingRule(rule: string): string {
  return rule
    .replace(/\$2[,.]?000K|\$2M/g, '$133K')
    .replace(/\$1[,.]?500K|\$1\.5M/g, '$100K')
    .replace(/\$1[,.]?000K|\$1M/g, '$67K')
    .replace(/\$500K/g, '$33K')
    .replace(/\$200K/g, '$13K')
    .replace(/\$100K/g, '$7K');
}

export async function GET() {
  try {
    const csvPath = path.join(process.cwd(), 'public', 'data', 'hybrid_all_trades_calendar.csv');
    const csvText = fs.readFileSync(csvPath, 'utf-8');
    const rawTrades = parseTrades(csvText);

    // Apply scale factor to position sizes, shares, and P&L
    const trades = rawTrades.map(t => ({
      ...t,
      position_size: Math.round(t.position_size * SCALE_FACTOR),
      shares_or_contracts: Math.floor(t.shares_or_contracts * SCALE_FACTOR),
      net_pnl: t.net_pnl * SCALE_FACTOR,
      sizing_rule: scaleSizingRule(t.sizing_rule || ''),
    }));

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
>>>>>>> Stashed changes
      );
    }
<<<<<<< Updated upstream

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
=======
    
    // Get unique symbols for price fetching
    const symbols = [...new Set(openPositions.map(p => p.symbol))];
    
    // Fetch current prices for all symbols
    const prices = await fetchMultiplePrices(symbols);
    
    // Enrich positions with live prices (if we have them)
    const enriched = openPositions.map(pos => {
      const livePrice = prices.get(pos.symbol);
      
      if (livePrice && pos.is_open) {
        const isStock = pos.instrument?.toUpperCase().startsWith('STOCK');
        // shares_or_contracts is already scaled; unrealized P&L uses scaled shares
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
>>>>>>> Stashed changes
    });
  } catch (err) {
    console.error('Error in /api/positions:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
