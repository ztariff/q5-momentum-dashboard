import { NextResponse } from 'next/server';
import { parseUniverse } from '@/lib/csvParser';
import {
  fetchDailyBars,
  computeSMA50,
  computeSlope,
  computeZScore,
  computeATR14,
  DailyBar,
} from '@/lib/polygon';
import { parseTrades } from '@/lib/csvParser';
import path from 'path';
import fs from 'fs';

// ─── Constants ───────────────────────────────────────────────────────────────
const SCALE_FACTOR = 0.0666;
const TIER_A_BUDGET = Math.round(13000);
const TIER_B_BUDGET = Math.round(7000);

// Rate-limit: Polygon free tier allows ~5 req/s — we batch with a small delay
const POLYGON_BATCH_SIZE = 5;
const POLYGON_BATCH_DELAY_MS = 1050; // ~1 req/s to be safe

// ─── Types ────────────────────────────────────────────────────────────────────
interface LivePosition {
  symbol: string;
  status: 'active' | 'exit_today' | 'pending';
  instrument: string;
  tier: string;
  option_ticker: string;
  entry_date: string;
  entry_price: number;
  current_price: number;
  stop_price: number;
  exit_date: string;
  position_size: number;
  shares_or_contracts: number;
  unrealized_pnl: number;
  unrealized_pct: number;
  days_held: number;
  days_remaining: number;
  body_pct: number;
  atr_change_3d: number;
}

interface PendingSignal {
  symbol: string;
  status: 'pending';
  tier: string;
  z_score: number;
  body_pct: number;
  atr_change_3d: number;
  estimated_entry: number;
  stop_price: number;
  position_size: number;
  sizing_rule: string;
}

interface RecentlyClosed {
  symbol: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  net_pnl: number;
  return_pct: number;
  exit_type: string;
}

interface RiskAlert {
  type: 'near_stop' | 'exit_soon' | 'stop_crossed' | 'deep_loss';
  symbol: string;
  days_remaining?: number;
  distance_pct?: number;
}

interface LiveState {
  last_refresh: string;
  market_date: string;
  positions: LivePosition[];
  pending_signals: PendingSignal[];
  recently_closed: RecentlyClosed[];
  summary: {
    total_positions: number;
    total_pending: number;
    total_unrealized: number;
    realized_pnl: number;
    month_realized: number;
    month_unrealized: number;
    win_rate: number;
    profit_factor: number;
  };
  risk_alerts: RiskAlert[];
  refresh_progress?: { done: number; total: number };
}

// ─── Helper functions ─────────────────────────────────────────────────────────

function addBizDays(dateStr: string, n: number): string {
  const d = new Date(dateStr);
  let count = 0;
  while (count < n) {
    d.setDate(d.getDate() + 1);
    if (d.getDay() !== 0 && d.getDay() !== 6) count++;
  }
  return d.toISOString().split('T')[0];
}

function bizDaysBetween(startStr: string, endStr: string): number {
  const start = new Date(startStr);
  const end = new Date(endStr);
  if (end <= start) return 0;
  let count = 0;
  const cur = new Date(start);
  while (cur < end) {
    cur.setDate(cur.getDate() + 1);
    if (cur.getDay() !== 0 && cur.getDay() !== 6) count++;
  }
  return count;
}

function tradingDaysSince(entryDate: string, today: string): number {
  return bizDaysBetween(entryDate, today);
}

function recommendedContracts(budget: number, optionPrice: number | null): number {
  if (!optionPrice || optionPrice <= 0) return 1;
  return Math.max(1, Math.floor(budget / (optionPrice * 100)));
}

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

// ─── Core signal computation (mirrors Python backtest exactly) ────────────────

interface SymbolMetrics {
  symbol: string;
  z: number | null;
  slope: number | null;
  prevZ: number | null;
  body_pct: number | null;
  atr_change_3d: number | null;
  last_close: number | null;
  last_low: number | null;
  atr14: number | null;
}

async function computeMetricsForSymbol(
  symbol: string,
  fromStr: string,
  toStr: string
): Promise<SymbolMetrics> {
  const empty: SymbolMetrics = {
    symbol, z: null, slope: null, prevZ: null,
    body_pct: null, atr_change_3d: null, last_close: null,
    last_low: null, atr14: null,
  };

  try {
    const bars: DailyBar[] = await fetchDailyBars(symbol, fromStr, toStr);
    if (bars.length < 70) return empty;

    const closes = bars.map(b => b.close);
    const sma = computeSMA50(closes);
    const slopes = computeSlope(sma);
    const zScores = computeZScore(slopes);
    const atrs = computeATR14(bars);

    const lastIdx = bars.length - 1;
    const prevIdx = lastIdx - 1;

    const bar = bars[lastIdx];
    const range = bar.high - bar.low;
    const body_pct = range > 0 ? (bar.close - bar.open) / range : null;

    const atrNow = atrs[lastIdx];
    const atr3d = lastIdx >= 3 ? atrs[lastIdx - 3] : null;
    const atr_change_3d =
      atrNow !== null && atr3d !== null && atr3d > 0
        ? (atrNow - atr3d) / atr3d
        : null;

    return {
      symbol,
      z: zScores[lastIdx] ?? null,
      slope: slopes[lastIdx] ?? null,
      prevZ: zScores[prevIdx] ?? null,
      body_pct,
      atr_change_3d,
      last_close: bar.close,
      last_low: bar.low,
      atr14: atrNow,
    };
  } catch {
    return empty;
  }
}

// ─── Progress tracking (written to live_state during refresh) ─────────────────

function writeLiveStateProgress(
  liveStatePath: string,
  current: LiveState,
  done: number,
  total: number
): void {
  try {
    const updated = { ...current, refresh_progress: { done, total } };
    fs.writeFileSync(liveStatePath, JSON.stringify(updated, null, 2));
  } catch {
    // best-effort
  }
}

// ─── Main refresh handler ─────────────────────────────────────────────────────

export async function GET() {
  const dataDir = path.join(process.cwd(), 'public', 'data');
  const liveStatePath = path.join(dataDir, 'live_state.json');
  const csvPath = path.join(dataDir, 'hybrid_all_trades_calendar.csv');
  const universePath = path.join(dataDir, 'universe.csv');

  // Load existing live_state as starting point (keep positions across refresh)
  let existingState: LiveState | null = null;
  try {
    if (fs.existsSync(liveStatePath)) {
      existingState = JSON.parse(fs.readFileSync(liveStatePath, 'utf-8')) as LiveState;
    }
  } catch { /* start fresh */ }

  const today = new Date().toISOString().split('T')[0];

  // ── 1. Load universe ────────────────────────────────────────────────────────
  if (!fs.existsSync(universePath)) {
    return NextResponse.json({ error: 'universe.csv not found' }, { status: 500 });
  }
  const universeText = fs.readFileSync(universePath, 'utf-8');
  const symbols = parseUniverse(universeText);

  // Date range: 600 calendar days back to get ~400 trading days
  const fromDate = new Date(today);
  fromDate.setDate(fromDate.getDate() - 600);
  const fromStr = fromDate.toISOString().split('T')[0];

  // ── 2. Fetch bars for all 230 symbols in rate-limited batches ───────────────
  const symbolMetrics: Map<string, SymbolMetrics> = new Map();
  const totalSymbols = symbols.length;

  // Seed progress into live_state so UI can poll it
  if (existingState) {
    writeLiveStateProgress(liveStatePath, existingState, 0, totalSymbols);
  }

  for (let i = 0; i < symbols.length; i += POLYGON_BATCH_SIZE) {
    const batch = symbols.slice(i, i + POLYGON_BATCH_SIZE);
    const results = await Promise.all(
      batch.map(sym => computeMetricsForSymbol(sym, fromStr, today))
    );
    for (const r of results) {
      symbolMetrics.set(r.symbol, r);
    }

    // Update progress
    const done = Math.min(i + POLYGON_BATCH_SIZE, totalSymbols);
    if (existingState) {
      writeLiveStateProgress(liveStatePath, existingState, done, totalSymbols);
    }

    // Rate-limit delay (skip on last batch)
    if (i + POLYGON_BATCH_SIZE < symbols.length) {
      await sleep(POLYGON_BATCH_DELAY_MS);
    }
  }

  // ── 3. Cross-sectional quintile sort ────────────────────────────────────────
  const validEntries = Array.from(symbolMetrics.values())
    .filter(m => m.z !== null)
    .sort((a, b) => (a.z as number) - (b.z as number));

  const n = validEntries.length;
  const quintileMap = new Map<string, number>();
  validEntries.forEach((m, i) => {
    const q = Math.min(5, Math.ceil(((i + 1) / n) * 5));
    quintileMap.set(m.symbol, q);
  });

  // Yesterday quintiles (from existing state — we don't have them unless we
  // persist them separately; default to null = treat all Q5 as new)
  // For a production system we'd store prev quintiles. Here we mark is_new = true
  // for any symbol currently in Q5.

  // ── 4. Load backtest CSV for open positions & realized stats ─────────────────
  let csvText = '';
  try { csvText = fs.readFileSync(csvPath, 'utf-8'); } catch { /* skip */ }

  const rawTrades = csvText ? parseTrades(csvText) : [];
  const scaledTrades = rawTrades.map(t => ({
    ...t,
    position_size: Math.round(t.position_size * SCALE_FACTOR),
    shares_or_contracts: Math.floor(t.shares_or_contracts * SCALE_FACTOR),
    net_pnl: t.net_pnl * SCALE_FACTOR,
  }));

  // Open positions: proper_exit_date >= today
  function getProperExit(entryDate: string, instrument: string): string {
    const isOpt = instrument.toUpperCase().includes('OPTION');
    return addBizDays(entryDate, isOpt ? 13 : 20);
  }

  const openTrades = scaledTrades.filter(t => {
    const pe = getProperExit(t.entry_date, t.instrument);
    return pe >= today;
  });

  // Build LivePosition objects from open trades
  const livePositions: LivePosition[] = openTrades.map(t => {
    const properExit = getProperExit(t.entry_date, t.instrument);
    const daysHeld = tradingDaysSince(t.entry_date, today);
    const daysRemaining = Math.max(0, bizDaysBetween(today, properExit));
    const isOption = t.instrument.toUpperCase().includes('OPTION');

    // Use last_close from Polygon if available; fall back to CSV exit_price or entry_price
    const metrics = symbolMetrics.get(t.symbol);
    const liveClose = metrics?.last_close ?? null;
    const currentPrice = liveClose ?? (t.exit_price > 0 ? t.exit_price : t.entry_price);

    const stopPrice = isOption ? t.entry_price * 0.50 : t.entry_price * 0.90;

    const unrealizedPnl = isOption
      ? (currentPrice - t.entry_price) * t.shares_or_contracts * 100
      : (currentPrice - t.entry_price) * t.shares_or_contracts;

    const unrealizedPct = t.entry_price > 0
      ? ((currentPrice / t.entry_price) - 1) * 100
      : 0;

    let status: 'active' | 'exit_today' = daysRemaining === 0 ? 'exit_today' : 'active';

    // Check stop trigger: today's low < stop price
    if (metrics?.last_low != null && metrics.last_low < stopPrice) {
      // Position would have been stopped — but we still show it as stopped-active
      // The user needs to take action; mark as exit_today
      status = 'exit_today';
    }

    return {
      symbol: t.symbol,
      status,
      instrument: t.instrument,
      tier: t.tier || 'B',
      option_ticker: t.option_ticker || '',
      entry_date: t.entry_date,
      entry_price: t.entry_price,
      current_price: currentPrice,
      stop_price: stopPrice,
      exit_date: properExit,
      position_size: t.position_size,
      shares_or_contracts: t.shares_or_contracts,
      unrealized_pnl: Math.round(unrealizedPnl),
      unrealized_pct: Math.round(unrealizedPct * 10) / 10,
      days_held: daysHeld,
      days_remaining: daysRemaining,
      body_pct: t.body_pct,
      atr_change_3d: t.atr_change_3d,
    };
  });

  // Active symbol set for no-re-entry check
  const activeSymbols = new Set(livePositions.map(p => p.symbol));

  // ── 5. Build pending signals (Q5 entrants, no-re-entry enforced) ─────────────
  const pendingSignals: PendingSignal[] = [];

  for (const [symbol, metrics] of symbolMetrics.entries()) {
    const q = quintileMap.get(symbol);
    if (q !== 5 || metrics.z === null) continue;
    if (activeSymbols.has(symbol)) continue; // no re-entry

    const bp = metrics.body_pct ?? 0;
    const atrChg = metrics.atr_change_3d ?? 0;

    // Skip Q1 body filter (very bearish candle)
    if (bp <= -0.53) continue;

    let tier: string;
    let sizingRule: string;
    let positionSize: number;

    if (bp > 0.57 && atrChg > 0.10) {
      tier = 'A';
      positionSize = TIER_A_BUDGET;
      sizingRule = `Tier A: $${(TIER_A_BUDGET / 1000).toFixed(0)}K options`;
    } else if (bp > 0.23 && atrChg > 0.05) {
      tier = 'B';
      positionSize = TIER_B_BUDGET;
      sizingRule = `Tier B: $${(TIER_B_BUDGET / 1000).toFixed(0)}K options`;
    } else {
      tier = 'C';
      if (bp <= -0.17) { positionSize = 33000; sizingRule = 'Tier C: $33K stock'; }
      else if (bp <= 0.23) { positionSize = 67000; sizingRule = 'Tier C: $67K stock'; }
      else if (bp <= 0.57) { positionSize = 100000; sizingRule = 'Tier C: $100K stock'; }
      else { positionSize = 133000; sizingRule = 'Tier C: $133K stock'; }
    }

    const lastClose = metrics.last_close ?? 0;
    const atr = metrics.atr14 ?? 0;
    const stopPrice = (tier === 'A' || tier === 'B')
      ? lastClose * 0.50
      : lastClose - 3 * atr;

    pendingSignals.push({
      symbol,
      status: 'pending',
      tier,
      z_score: Math.round(metrics.z * 1000) / 1000,
      body_pct: Math.round(bp * 10000) / 10000,
      atr_change_3d: Math.round(atrChg * 10000) / 10000,
      estimated_entry: Math.round(lastClose * 100) / 100,
      stop_price: Math.round(stopPrice * 100) / 100,
      position_size: positionSize,
      sizing_rule: sizingRule,
    });
  }

  // Sort by z_score descending
  pendingSignals.sort((a, b) => b.z_score - a.z_score);

  // ── 6. Recently closed (last 5 trading days) ──────────────────────────────────
  const recentCutoff = new Date(today);
  let tradingDaysBack = 0;
  while (tradingDaysBack < 5) {
    recentCutoff.setDate(recentCutoff.getDate() - 1);
    if (recentCutoff.getDay() !== 0 && recentCutoff.getDay() !== 6) tradingDaysBack++;
  }
  const cutoffStr = recentCutoff.toISOString().split('T')[0];

  const recentlyClosed: RecentlyClosed[] = scaledTrades
    .filter(t => {
      const pe = getProperExit(t.entry_date, t.instrument);
      return pe < today && pe >= cutoffStr;
    })
    .slice(-10)
    .map(t => ({
      symbol: t.symbol,
      exit_date: getProperExit(t.entry_date, t.instrument),
      entry_price: t.entry_price,
      exit_price: t.exit_price > 0 ? t.exit_price : t.entry_price,
      net_pnl: Math.round(t.net_pnl),
      return_pct: Math.round(t.return_pct * 10) / 10,
      exit_type: t.exit_type || 'TIME',
    }));

  // ── 7. Realized P&L stats from backtest CSV ───────────────────────────────────
  const closedTrades = scaledTrades.filter(t => {
    const pe = getProperExit(t.entry_date, t.instrument);
    return pe < today && t.exit_price > 0 && t.net_pnl !== 0;
  });

  const totalRealized = closedTrades.reduce((s, t) => s + t.net_pnl, 0);
  const wins = closedTrades.filter(t => t.net_pnl > 0);
  const losses = closedTrades.filter(t => t.net_pnl < 0);
  const winRate = closedTrades.length > 0 ? (wins.length / closedTrades.length) * 100 : 0;
  const grossProfit = wins.reduce((s, t) => s + t.net_pnl, 0);
  const grossLoss = Math.abs(losses.reduce((s, t) => s + t.net_pnl, 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;

  const currentMonth = today.substring(0, 7);
  const monthRealized = closedTrades
    .filter(t => getProperExit(t.entry_date, t.instrument).startsWith(currentMonth))
    .reduce((s, t) => s + t.net_pnl, 0);

  const totalUnrealized = livePositions.reduce((s, p) => s + p.unrealized_pnl, 0);
  const monthUnrealized = livePositions
    .filter(p => p.entry_date.startsWith(currentMonth))
    .reduce((s, p) => s + p.unrealized_pnl, 0);

  // ── 8. Risk alerts ────────────────────────────────────────────────────────────
  const riskAlerts: RiskAlert[] = [];

  for (const pos of livePositions) {
    const distPct = pos.current_price > 0
      ? ((pos.current_price - pos.stop_price) / pos.current_price) * 100
      : 100;

    if (pos.current_price < pos.stop_price) {
      riskAlerts.push({ type: 'stop_crossed', symbol: pos.symbol });
    } else if (distPct < 20) {
      riskAlerts.push({ type: 'near_stop', symbol: pos.symbol, distance_pct: Math.round(distPct * 10) / 10 });
    }

    if (pos.days_remaining <= 2 && pos.days_remaining >= 0) {
      riskAlerts.push({ type: 'exit_soon', symbol: pos.symbol, days_remaining: pos.days_remaining });
    }

    const unrealizedPct = pos.entry_price > 0
      ? ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
      : 0;
    if (unrealizedPct < -50) {
      riskAlerts.push({ type: 'deep_loss', symbol: pos.symbol });
    }
  }

  // ── 9. Write live_state.json ──────────────────────────────────────────────────
  const liveState: LiveState = {
    last_refresh: new Date().toISOString(),
    market_date: today,
    positions: livePositions,
    pending_signals: pendingSignals,
    recently_closed: recentlyClosed,
    summary: {
      total_positions: livePositions.length,
      total_pending: pendingSignals.length,
      total_unrealized: Math.round(totalUnrealized),
      realized_pnl: Math.round(totalRealized),
      month_realized: Math.round(monthRealized),
      month_unrealized: Math.round(monthUnrealized),
      win_rate: Math.round(winRate * 10) / 10,
      profit_factor: Math.round(profitFactor * 100) / 100,
    },
    risk_alerts: riskAlerts,
  };

  try {
    fs.writeFileSync(liveStatePath, JSON.stringify(liveState, null, 2));
  } catch (err) {
    return NextResponse.json({ error: `Failed to write live_state.json: ${err}` }, { status: 500 });
  }

  return NextResponse.json({
    success: true,
    market_date: today,
    symbols_processed: symbolMetrics.size,
    positions: livePositions.length,
    pending_signals: pendingSignals.length,
    last_refresh: liveState.last_refresh,
  });
}
