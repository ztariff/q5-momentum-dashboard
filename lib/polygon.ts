const POLYGON_API_KEY = process.env.POLYGON_API_KEY || process.env.NEXT_PUBLIC_POLYGON_API_KEY || 'cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF';
const BASE_URL = 'https://api.polygon.io';

// In-memory cache for daily bars
const barsCache = new Map<string, { data: DailyBar[]; timestamp: number }>();
const priceCache = new Map<string, { price: number; timestamp: number }>();

const BARS_CACHE_TTL = 4 * 60 * 60 * 1000; // 4 hours
const PRICE_CACHE_TTL = 60 * 1000; // 60 seconds

export interface DailyBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export async function fetchDailyBars(
  symbol: string,
  fromDate: string,
  toDate: string
): Promise<DailyBar[]> {
  const cacheKey = `${symbol}-${fromDate}-${toDate}`;
  const cached = barsCache.get(cacheKey);
  
  if (cached && Date.now() - cached.timestamp < BARS_CACHE_TTL) {
    return cached.data;
  }

  try {
    const url = `${BASE_URL}/v2/aggs/ticker/${symbol}/range/1/day/${fromDate}/${toDate}?adjusted=true&sort=asc&limit=5000&apiKey=${POLYGON_API_KEY}`;
    const resp = await fetch(url, { next: { revalidate: 3600 } });
    
    if (!resp.ok) {
      console.error(`Polygon error for ${symbol}: ${resp.status}`);
      return [];
    }
    
    const json = await resp.json();
    
    if (!json.results || json.results.length === 0) {
      return [];
    }
    
    const bars: DailyBar[] = json.results.map((r: Record<string, number>) => ({
      date: new Date(r.t).toISOString().split('T')[0],
      open: r.o,
      high: r.h,
      low: r.l,
      close: r.c,
      volume: r.v,
    }));
    
    barsCache.set(cacheKey, { data: bars, timestamp: Date.now() });
    return bars;
  } catch (err) {
    console.error(`Error fetching bars for ${symbol}:`, err);
    return [];
  }
}

export async function fetchCurrentPrice(symbol: string): Promise<number | null> {
  const cached = priceCache.get(symbol);
  if (cached && Date.now() - cached.timestamp < PRICE_CACHE_TTL) {
    return cached.price;
  }

  try {
    // Try snapshot endpoint first (more reliable for current price)
    const url = `${BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/${symbol}?apiKey=${POLYGON_API_KEY}`;
    const resp = await fetch(url, { cache: 'no-store' });
    
    if (!resp.ok) {
      return null;
    }
    
    const json = await resp.json();
    
    if (json.ticker && json.ticker.lastTrade) {
      const price = json.ticker.lastTrade.p;
      priceCache.set(symbol, { price, timestamp: Date.now() });
      return price;
    }
    
    // Fallback to day's close
    if (json.ticker && json.ticker.day) {
      const price = json.ticker.day.c || json.ticker.day.o;
      if (price) {
        priceCache.set(symbol, { price, timestamp: Date.now() });
        return price;
      }
    }

    // Fallback to previous day close
    if (json.ticker && json.ticker.prevDay) {
      const price = json.ticker.prevDay.c;
      if (price) {
        priceCache.set(symbol, { price, timestamp: Date.now() });
        return price;
      }
    }
    
    return null;
  } catch (err) {
    console.error(`Error fetching price for ${symbol}:`, err);
    return null;
  }
}

/**
 * Fetch current option prices via Polygon snapshot endpoint.
 * Uses ticker.any_of which accepts up to 250 option tickers per call.
 * Returns mid (bid+ask)/2 preferred, falls back to session close or last trade.
 */
export async function fetchOptionSnapshots(
  optionTickers: string[]
): Promise<Map<string, number>> {
  const result = new Map<string, number>();
  if (optionTickers.length === 0) return result;

  // Batch into groups of 250 (Polygon limit for ticker.any_of)
  for (let i = 0; i < optionTickers.length; i += 250) {
    const chunk = optionTickers.slice(i, i + 250);
    try {
      const url = `${BASE_URL}/v3/snapshot?ticker.any_of=${chunk.join(',')}&limit=250&apiKey=${POLYGON_API_KEY}`;
      const resp = await fetch(url, { cache: 'no-store' });
      if (!resp.ok) continue;
      const json = await resp.json();
      const results = json.results || [];
      for (const r of results) {
        const ticker = r.ticker;
        if (!ticker) continue;
        let price: number | null = null;
        // Prefer mid of NBBO
        const bid = r.last_quote?.bid;
        const ask = r.last_quote?.ask;
        if (bid && ask && bid > 0 && ask > 0) {
          price = (bid + ask) / 2;
        }
        // Fallback to session data
        if (!price && r.session) {
          price = r.session.close || r.session.vwap || r.session.open;
        }
        // Fallback to last trade
        if (!price && r.last_trade?.price) {
          price = r.last_trade.price;
        }
        if (price && price > 0) {
          result.set(ticker, price);
        }
      }
    } catch (err) {
      console.error('Error fetching option snapshots:', err);
    }
  }

  return result;
}

export async function fetchMultiplePrices(
  symbols: string[]
): Promise<Map<string, number>> {
  const result = new Map<string, number>();
  
  // Batch fetch using snapshot endpoint
  const chunks: string[][] = [];
  for (let i = 0; i < symbols.length; i += 50) {
    chunks.push(symbols.slice(i, i + 50));
  }
  
  for (const chunk of chunks) {
    try {
      const tickerList = chunk.join(',');
      const url = `${BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers?tickers=${tickerList}&apiKey=${POLYGON_API_KEY}`;
      const resp = await fetch(url, { cache: 'no-store' });
      
      if (!resp.ok) continue;
      
      const json = await resp.json();
      
      if (json.tickers) {
        for (const t of json.tickers) {
          let price: number | null = null;
          
          if (t.lastTrade && t.lastTrade.p) {
            price = t.lastTrade.p;
          } else if (t.day && t.day.c) {
            price = t.day.c;
          } else if (t.prevDay && t.prevDay.c) {
            price = t.prevDay.c;
          }
          
          if (price && price > 0) {
            result.set(t.ticker, price);
            priceCache.set(t.ticker, { price, timestamp: Date.now() });
          }
        }
      }
    } catch (err) {
      console.error('Error batch fetching prices:', err);
    }
  }
  
  return result;
}

// Compute SMA50
export function computeSMA50(closes: number[]): (number | null)[] {
  return closes.map((_, i) => {
    if (i < 49) return null;
    const slice = closes.slice(i - 49, i + 1);
    return slice.reduce((a, b) => a + b, 0) / 50;
  });
}

// Compute 10-day slope of SMA50
export function computeSlope(sma: (number | null)[]): (number | null)[] {
  return sma.map((v, i) => {
    if (v === null || i < 10) return null;
    const prev = sma[i - 10];
    if (prev === null || prev === 0) return null;
    return 100 * (v / prev - 1);
  });
}

// Compute rolling z-score per symbol
export function computeZScore(
  slopes: (number | null)[],
  window = 252,
  minPeriods = 60
): (number | null)[] {
  return slopes.map((s, i) => {
    if (s === null) return null;
    
    const start = Math.max(0, i - window);
    const slice = slopes.slice(start, i + 1).filter((v): v is number => v !== null);
    
    if (slice.length < minPeriods) return null;
    
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length;
    const std = Math.sqrt(variance);
    
    if (std === 0) return null;
    
    return (s - mean) / std;
  });
}

// Compute ATR(14) using Wilder's smoothing
export function computeATR14(bars: DailyBar[]): (number | null)[] {
  const trs: number[] = [];
  
  for (let i = 0; i < bars.length; i++) {
    if (i === 0) {
      trs.push(bars[i].high - bars[i].low);
      continue;
    }
    const prevClose = bars[i - 1].close;
    const tr = Math.max(
      bars[i].high - bars[i].low,
      Math.abs(bars[i].high - prevClose),
      Math.abs(bars[i].low - prevClose)
    );
    trs.push(tr);
  }
  
  const atrs: (number | null)[] = new Array(bars.length).fill(null);
  
  // First ATR is simple average of first 14 TRs
  if (trs.length >= 14) {
    const firstATR = trs.slice(0, 14).reduce((a, b) => a + b, 0) / 14;
    atrs[13] = firstATR;
    
    for (let i = 14; i < trs.length; i++) {
      const prev = atrs[i - 1];
      if (prev !== null) {
        atrs[i] = prev * (13 / 14) + trs[i] * (1 / 14);
      }
    }
  }
  
  return atrs;
}
