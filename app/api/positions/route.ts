import { NextResponse } from 'next/server';
import { parseTrades, parseUniverse } from '@/lib/csvParser';
import { fetchMultiplePrices, fetchDailyBars, computeSMA50, computeSlope, computeZScore, computeATR14 } from '@/lib/polygon';
import { Position, PositionStatus } from '@/lib/types';
import path from 'path';
import fs from 'fs';

const SCALE_FACTOR = 0.0666; // Target $40K daily vol = 1/15th of backtest size

// Position size budgets (same as signal scanner)
const TIER_A_BUDGET = 13000;
const TIER_B_BUDGET = 7000;

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

/**
 * Compute the proper exit date for a trade: entry_date + holdDays business days.
 * Options hold 13 biz days; stocks hold 20 biz days.
 */
function computeProperExitDate(entryDate: string, instrument: string): string {
  const entry = new Date(entryDate);
  const isOption = instrument?.toUpperCase().includes('OPTION');
  const holdDays = isOption ? 13 : 20;
  let bizDays = 0;
  const d = new Date(entry);
  while (bizDays < holdDays) {
    d.setDate(d.getDate() + 1);
    const dow = d.getDay();
    if (dow !== 0 && dow !== 6) bizDays++;
  }
  return d.toISOString().split('T')[0];
}

/** Next business day after a given date string */
function nextBusinessDay(dateStr: string): string {
  const d = new Date(dateStr);
  d.setDate(d.getDate() + 1);
  while (d.getDay() === 0 || d.getDay() === 6) {
    d.setDate(d.getDate() + 1);
  }
  return d.toISOString().split('T')[0];
}

/** Recommended contracts for options (min 1) */
function recommendedContracts(budget: number, optionPrice: number | null): number {
  if (!optionPrice || optionPrice <= 0) return 1;
  return Math.max(1, Math.floor(budget / (optionPrice * 100)));
}

/**
 * Run the signal scanner logic inline and return pending Position objects.
 * Scans up to 60 symbols from the universe.
 */
async function buildPendingPositions(today: string, openPositions: Position[]): Promise<Position[]> {
  try {
    const universePath = path.join(process.cwd(), 'public', 'data', 'universe.csv');
    if (!fs.existsSync(universePath)) return [];
    const universeText = fs.readFileSync(universePath, 'utf-8');
    const symbols = parseUniverse(universeText);

    const fromDate = new Date(today);
    fromDate.setFullYear(fromDate.getFullYear() - 2);
    const fromStr = fromDate.toISOString().split('T')[0];

    const batchSize = 20;
    const symbolData: Map<string, {
      z: number | null;
      slope: number | null;
      body_pct: number | null;
      atr_change: number | null;
      last_close: number | null;
      atr14: number | null;
    }> = new Map();

    for (let i = 0; i < Math.min(symbols.length, 60); i += batchSize) {
      const batch = symbols.slice(i, i + batchSize);
      const results = await Promise.all(batch.map(async (symbol) => {
        try {
          const bars = await fetchDailyBars(symbol, fromStr, today);
          if (bars.length < 70) return null;

          const closes = bars.map(b => b.close);
          const sma = computeSMA50(closes);
          const slopes = computeSlope(sma);
          const zScores = computeZScore(slopes);
          const atrs = computeATR14(bars);

          const lastIdx = bars.length - 1;
          const bar = bars[lastIdx];
          const range = bar.high - bar.low;
          const body_pct = range > 0 ? (bar.close - bar.open) / range : null;

          const atr_now = atrs[lastIdx];
          const atr_3d = atrs[lastIdx - 3];
          const atr_change = atr_now !== null && atr_3d !== null && atr_3d > 0
            ? (atr_now - atr_3d) / atr_3d
            : null;

          return {
            symbol,
            z: zScores[lastIdx],
            slope: slopes[lastIdx],
            body_pct,
            atr_change,
            last_close: bar.close,
            atr14: atr_now,
          };
        } catch {
          return null;
        }
      }));

      for (const r of results) {
        if (r) {
          symbolData.set(r.symbol, {
            z: r.z ?? null,
            slope: r.slope ?? null,
            body_pct: r.body_pct ?? null,
            atr_change: r.atr_change ?? null,
            last_close: r.last_close ?? null,
            atr14: r.atr14 ?? null,
          });
        }
      }
    }

    // Cross-sectional quintile assignment
    const validSymbols = Array.from(symbolData.entries())
      .filter(([, d]) => d.z !== null)
      .sort((a, b) => (a[1].z || 0) - (b[1].z || 0));

    const n = validSymbols.length;
    const quintileMap = new Map<string, number>();
    validSymbols.forEach(([symbol], i) => {
      const q = Math.min(5, Math.ceil(((i + 1) / n) * 5));
      quintileMap.set(symbol, q);
    });

    const entryDate = nextBusinessDay(today);
    const pending: Position[] = [];

    // No-re-entry rule: skip symbols we already hold an active position in
    const activeSymbols = new Set(openPositions.map(p => p.symbol));

    for (const [symbol, data] of symbolData.entries()) {
      const q = quintileMap.get(symbol);
      if (q !== 5 || data.z === null) continue;

      // No-re-entry rule: skip if already holding this symbol
      if (activeSymbols.has(symbol)) continue;

      const bp = data.body_pct ?? 0;
      const atrChg = data.atr_change ?? 0;

      // Skip Q1 body (very bearish candle)
      if (bp <= -0.53) continue;

      let tier: string;
      let sizingRule: string;
      let recommendedSize: number;
      let recContracts: number | null = null;
      let instrument: string;

      if (bp > 0.57 && atrChg > 0.10) {
        tier = 'A';
        recommendedSize = TIER_A_BUDGET;
        recContracts = recommendedContracts(TIER_A_BUDGET, null);
        sizingRule = `$${(TIER_A_BUDGET / 1000).toFixed(0)}K options (Tier A) — Min 1 contract`;
        instrument = 'OPTION';
      } else if (bp > 0.23 && atrChg > 0.05) {
        tier = 'B';
        recommendedSize = TIER_B_BUDGET;
        recContracts = recommendedContracts(TIER_B_BUDGET, null);
        sizingRule = `$${(TIER_B_BUDGET / 1000).toFixed(0)}K options (Tier B) — Min 1 contract`;
        instrument = 'OPTION';
      } else {
        tier = 'C';
        recContracts = null;
        instrument = 'STOCK';
        if (bp <= -0.17) {
          sizingRule = '$33K position (Tier C Q2)';
          recommendedSize = 33000;
        } else if (bp <= 0.23) {
          sizingRule = '$67K position (Tier C Q3)';
          recommendedSize = 67000;
        } else if (bp <= 0.57) {
          sizingRule = '$100K position (Tier C Q4)';
          recommendedSize = 100000;
        } else {
          sizingRule = '$133K position (Tier C Q5)';
          recommendedSize = 133000;
        }
      }

      const lastClose = data.last_close ?? 0;
      // ATR-based stop: entry price - 3×ATR14
      const atr = data.atr14 ?? 0;
      const stopPrice = instrument === 'OPTION'
        ? lastClose * 0.50
        : lastClose - 3 * atr;

      const maxHoldDays = instrument === 'OPTION' ? 13 : 20;
      const scheduledExit = computeProperExitDate(entryDate, instrument);

      const pendingPos: Position = {
        // Trade fields
        symbol,
        instrument,
        tier,
        body_quintile: '',
        sizing_rule: sizingRule,
        option_ticker: '',
        strike: null,
        expiry: '',
        signal_date: today,
        entry_date: entryDate,
        entry_time: 'open',
        entry_price: lastClose,
        exit_date: scheduledExit,
        exit_time: '',
        exit_price: 0,
        exit_type: '',
        position_size: recommendedSize,
        shares_or_contracts: recContracts ?? Math.floor(recommendedSize / (lastClose || 1)),
        hold_days: 0,
        net_pnl: 0,
        return_pct: 0,
        body_pct: bp,
        atr_change_3d: atrChg,
        opt_source: '',
        // Position fields
        status: 'pending' as PositionStatus,
        current_price: lastClose,
        unrealized_pnl: undefined,
        unrealized_pnl_pct: undefined,
        stop_price: stopPrice,
        days_held: 0,
        days_remaining: maxHoldDays,
        scheduled_exit_date: scheduledExit,
        distance_to_stop_pct: lastClose > 0 ? ((lastClose - stopPrice) / lastClose) * 100 : 100,
        distance_to_stop_dollar: lastClose - stopPrice,
        max_hold_days: maxHoldDays,
        is_open: false,
      };

      pending.push(pendingPos);
    }

    return pending;
  } catch (err) {
    console.error('buildPendingPositions error:', err);
    return [];
  }
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

    const today = new Date().toISOString().split('T')[0]; // e.g. '2026-04-03'

    // Helper: count trading days from entry to today (approximate)
    function tradingDaysSinceEntry(entryDate: string): number {
      const start = new Date(entryDate);
      const end = new Date(today);
      let count = 0;
      const cur = new Date(start);
      while (cur < end) {
        cur.setDate(cur.getDate() + 1);
        const day = cur.getDay();
        if (day !== 0 && day !== 6) count++;
      }
      return count;
    }

    // Helper: count business days between two date strings (start exclusive, end inclusive)
    function bizDaysBetween(startDate: string, endDate: string): number {
      const start = new Date(startDate);
      const end = new Date(endDate);
      if (end <= start) return 0;
      let count = 0;
      const cur = new Date(start);
      while (cur < end) {
        cur.setDate(cur.getDate() + 1);
        const dow = cur.getDay();
        if (dow !== 0 && dow !== 6) count++;
      }
      return count;
    }

    type TradeWithProperExit = (typeof trades)[number] & {
      proper_exit_date: string;
      csv_exit_date: string;
      is_truncated: boolean;
    };

    const tradesWithProper: TradeWithProperExit[] = trades.map(t => {
      const proper_exit_date = computeProperExitDate(t.entry_date, t.instrument);
      const csv_exit_date = t.exit_date;
      const is_truncated = proper_exit_date > csv_exit_date;
      return { ...t, proper_exit_date, csv_exit_date, is_truncated };
    });

    // Open positions: those whose proper exit date is today or in the future
    const openTrades = tradesWithProper.filter(t => t.proper_exit_date >= today);

    let isRecentlyClosed = false;
    let positionsSource: TradeWithProperExit[] = openTrades;

    // Fallback: if zero open positions, show recently closed trades
    if (openTrades.length === 0) {
      isRecentlyClosed = true;

      const recentCutoff = new Date(today);
      let tradingDaysBack = 0;
      while (tradingDaysBack < 5) {
        recentCutoff.setDate(recentCutoff.getDate() - 1);
        const day = recentCutoff.getDay();
        if (day !== 0 && day !== 6) tradingDaysBack++;
      }
      const cutoffStr = recentCutoff.toISOString().split('T')[0];
      positionsSource = tradesWithProper.filter(t => t.proper_exit_date >= cutoffStr);
    }

    function buildPosition(t: TradeWithProperExit): Position {
      const isOption = t.instrument?.toUpperCase().includes('OPTION');
      const maxHoldDays = isOption ? 13 : 20;

      const canonicalExitDate = t.proper_exit_date;
      const daysHeld = tradingDaysSinceEntry(t.entry_date);
      const daysRemaining = isRecentlyClosed
        ? 0
        : Math.max(0, bizDaysBetween(today, canonicalExitDate));

      const lastKnownPrice = t.exit_price || t.entry_price;

      const unrealizedPnl = isOption
        ? (lastKnownPrice - t.entry_price) * t.shares_or_contracts * 100
        : (lastKnownPrice - t.entry_price) * t.shares_or_contracts;
      const unrealizedPnlPct = t.entry_price > 0
        ? ((lastKnownPrice / t.entry_price) - 1) * 100
        : 0;

      const stopPrice = isOption
        ? t.entry_price * 0.50
        : t.entry_price * 0.90;

      const currentPriceProxy = lastKnownPrice;
      const distanceToStop = currentPriceProxy > 0
        ? ((currentPriceProxy - stopPrice) / currentPriceProxy) * 100
        : 100;

      const isOpen = !isRecentlyClosed;

      // Determine status
      let status: PositionStatus = 'active';
      if (daysRemaining === 0 && isOpen) {
        status = 'exit_today';
      }

      return {
        ...t,
        exit_date: canonicalExitDate,
        exit_price: isOpen && t.is_truncated ? lastKnownPrice : t.exit_price,
        days_held: daysHeld,
        days_remaining: daysRemaining,
        scheduled_exit_date: canonicalExitDate,
        max_hold_days: maxHoldDays,
        is_open: isOpen,
        unrealized_pnl: unrealizedPnl,
        unrealized_pnl_pct: unrealizedPnlPct,
        current_price: currentPriceProxy,
        stop_price: stopPrice,
        distance_to_stop_pct: distanceToStop,
        distance_to_stop_dollar: currentPriceProxy - stopPrice,
        status,
      } as Position;
    }

    const positionsToShow: Position[] = positionsSource.map(buildPosition);

    // Fetch live stock prices for STOCK positions only
    const stockSymbols = [
      ...new Set(
        positionsToShow
          .filter(p => !p.instrument?.toUpperCase().includes('OPTION'))
          .map(p => p.symbol)
      ),
    ];
    const prices = await fetchMultiplePrices(stockSymbols);

    // Enrich STOCK positions with live prices where available
    const enriched = positionsToShow.map(pos => {
      const isOption = pos.instrument?.toUpperCase().includes('OPTION');
      if (isOption) return pos;

      const livePrice = prices.get(pos.symbol);
      if (!livePrice) return pos;

      const unrealizedPnl = (livePrice - pos.entry_price) * pos.shares_or_contracts;
      const unrealizedPct = pos.entry_price > 0
        ? ((livePrice / pos.entry_price) - 1) * 100
        : 0;

      const stopPrice = pos.stop_price ?? pos.entry_price * 0.90;
      const distanceToStop = livePrice > 0
        ? ((livePrice - stopPrice) / livePrice) * 100
        : 100;

      return {
        ...pos,
        current_price: livePrice,
        unrealized_pnl: unrealizedPnl,
        unrealized_pnl_pct: unrealizedPct,
        distance_to_stop_pct: distanceToStop,
        distance_to_stop_dollar: livePrice - stopPrice,
      };
    });

    // Build pending positions from signal scanner
    const pendingPositions = await buildPendingPositions(today, enriched);

    // Merge: pending at top, then active/exit_today positions
    const allPositions = [...pendingPositions, ...enriched];

    const totalPositions = enriched.length;
    const totalUnrealized = enriched.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);

    const closedTrades = trades.filter(t => {
      const proper = computeProperExitDate(t.entry_date, t.instrument);
      const isTruncated = proper > t.exit_date;
      return !isTruncated && t.exit_price > 0 && t.net_pnl !== 0;
    });
    const totalRealized = closedTrades.reduce((s, t) => s + t.net_pnl, 0);

    const currentMonth = today.substring(0, 7);

    const monthUnrealized = enriched
      .filter(p => p.entry_date.startsWith(currentMonth))
      .reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);

    const monthRealized = closedTrades
      .filter(t => (t.exit_date || '').startsWith(currentMonth))
      .reduce((s, t) => s + t.net_pnl, 0);

    return NextResponse.json({
      positions: allPositions,
      count: allPositions.length,
      pending_count: pendingPositions.length,
      is_demo: false,
      demo_note: isRecentlyClosed
        ? 'No positions with future exit dates found — showing recently closed positions from the last 5 trading days.'
        : null,
      summary: {
        total_positions: totalPositions,
        total_unrealized: totalUnrealized,
        total_realized: totalRealized,
        month_unrealized: monthUnrealized,
        month_realized: monthRealized,
        current_month: currentMonth,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.error('Error in /api/positions:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
