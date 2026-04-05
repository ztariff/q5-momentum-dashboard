import { NextResponse } from 'next/server';
import { parseUniverse } from '@/lib/csvParser';
import { fetchDailyBars, computeSMA50, computeSlope, computeZScore, computeATR14 } from '@/lib/polygon';
import path from 'path';
import fs from 'fs';

// Cache for signal results (recompute every 4 hours)
let signalsCache: { data: unknown; timestamp: number } | null = null;
const SIGNALS_CACHE_TTL = 4 * 60 * 60 * 1000;

export async function GET() {
  if (signalsCache && Date.now() - signalsCache.timestamp < SIGNALS_CACHE_TTL) {
    return NextResponse.json(signalsCache.data);
  }

  try {
    const universePath = path.join(process.cwd(), 'public', 'data', 'universe.csv');
    const universeText = fs.readFileSync(universePath, 'utf-8');
    const symbols = parseUniverse(universeText);
    
    const today = new Date('2026-04-03');
    const fromDate = new Date(today);
    fromDate.setFullYear(fromDate.getFullYear() - 2);
    const fromStr = fromDate.toISOString().split('T')[0];
    const toStr = today.toISOString().split('T')[0];
    
    // Fetch data for a subset of symbols to avoid rate limiting
    // Process in batches of 20
    const batchSize = 20;
    const symbolData: Map<string, { z: number | null; slope: number | null; quintile: number | null; prevQuintile: number | null; body_pct: number | null; atr_change: number | null }> = new Map();
    
    let processed = 0;
    for (let i = 0; i < Math.min(symbols.length, 60); i += batchSize) {
      const batch = symbols.slice(i, i + batchSize);
      
      const promises = batch.map(async (symbol) => {
        try {
          const bars = await fetchDailyBars(symbol, fromStr, toStr);
          if (bars.length < 70) return { symbol, z: null, slope: null, quintile: null, prevQuintile: null, body_pct: null, atr_change: null };
          
          const closes = bars.map(b => b.close);
          const sma = computeSMA50(closes);
          const slopes = computeSlope(sma);
          const zScores = computeZScore(slopes);
          const atrs = computeATR14(bars);
          
          const lastIdx = bars.length - 1;
          const prevIdx = lastIdx - 1;
          
          const z = zScores[lastIdx];
          const slope = slopes[lastIdx];
          
          // Body pct on last bar
          const bar = bars[lastIdx];
          const range = bar.high - bar.low;
          const body_pct = range > 0 ? (bar.close - bar.open) / range : null;
          
          // ATR change over 3 days
          const atr_now = atrs[lastIdx];
          const atr_3d = atrs[lastIdx - 3];
          const atr_change = atr_now !== null && atr_3d !== null && atr_3d > 0 
            ? (atr_now - atr_3d) / atr_3d 
            : null;
          
          return { 
            symbol, 
            z, 
            slope,
            body_pct,
            atr_change,
            lastZ: z,
            prevZ: zScores[prevIdx],
          };
        } catch {
          return { symbol, z: null, slope: null, quintile: null, prevQuintile: null, body_pct: null, atr_change: null };
        }
      });
      
      const results = await Promise.all(promises);
      for (const r of results) {
        symbolData.set(r.symbol, {
          z: r.z ?? null,
          slope: r.slope ?? null,
          quintile: null,
          prevQuintile: null,
          body_pct: r.body_pct ?? null,
          atr_change: r.atr_change ?? null,
        });
      }
      
      processed += batch.length;
    }
    
    // Cross-sectional quintile sort for today
    const validSymbols = Array.from(symbolData.entries())
      .filter(([, d]) => d.z !== null)
      .sort((a, b) => (a[1].z || 0) - (b[1].z || 0));
    
    const n = validSymbols.length;
    validSymbols.forEach(([symbol, data], i) => {
      const quintile = Math.min(5, Math.ceil(((i + 1) / n) * 5));
      symbolData.set(symbol, { ...data, quintile });
    });
    
    // Build signals and watchlist
    const signals: Array<{
      symbol: string;
      z_score: number;
      slope: number;
      quintile: number;
      body_pct: number;
      atr_change: number;
      tier: string;
      sizing_rule: string;
      recommended_size: number;
      is_new_signal: boolean;
    }> = [];
    
    const watchlist: Array<{
      symbol: string;
      quintile: number;
      z_score: number;
      slope: number;
      approaching: boolean;
    }> = [];
    
    for (const [symbol, data] of symbolData.entries()) {
      if (data.quintile === null || data.z === null) continue;
      
      const isQ5 = data.quintile === 5;
      const isQ4 = data.quintile === 4;
      
      if (isQ5) {
        // Determine tier from body_pct
        const bp = data.body_pct ?? 0;
        let tier = 'C';
        let sizingRule = '';
        let recommendedSize = 67000;
        
        if (bp <= -0.53) {
          // Q1 body - SKIP
          continue;
        } else if (bp <= -0.17) {
          sizingRule = '$33K position (Tier C Q2)';
          recommendedSize = 33000;
          tier = 'C';
        } else if (bp <= 0.23) {
          sizingRule = '$67K position (Tier C Q3)';
          recommendedSize = 67000;
          tier = 'C';
        } else if (bp <= 0.57) {
          sizingRule = '$100K position (Tier C Q4)';
          recommendedSize = 100000;
          tier = 'C';
        } else {
          sizingRule = '$133K position (Tier C Q5)';
          recommendedSize = 133000;
          tier = 'C';
        }
        
        signals.push({
          symbol,
          z_score: data.z,
          slope: data.slope ?? 0,
          quintile: data.quintile,
          body_pct: bp,
          atr_change: data.atr_change ?? 0,
          tier,
          sizing_rule: sizingRule,
          recommended_size: recommendedSize,
          is_new_signal: true,
        });
      } else if (isQ4) {
        // Q4 stocks approaching Q5
        watchlist.push({
          symbol,
          quintile: 4,
          z_score: data.z,
          slope: data.slope ?? 0,
          approaching: data.z > 1.0, // High z-score indicates likely to enter Q5 soon
        });
      }
    }
    
    // Sort signals by z_score descending
    signals.sort((a, b) => b.z_score - a.z_score);
    
    // Sort watchlist by z_score descending
    watchlist.sort((a, b) => b.z_score - a.z_score);
    
    const result = {
      signals,
      watchlist,
      total_scanned: processed,
      timestamp: new Date().toISOString(),
    };
    
    signalsCache = { data: result, timestamp: Date.now() };
    
    return NextResponse.json(result);
  } catch (err) {
    console.error('Error in /api/signals:', err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
