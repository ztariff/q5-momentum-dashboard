import { NextResponse } from 'next/server';
import { fetchMultiplePrices } from '@/lib/polygon';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { symbols } = body as { symbols: string[] };
    
    if (!symbols || !Array.isArray(symbols)) {
      return NextResponse.json({ error: 'symbols array required' }, { status: 400 });
    }
    
    const prices = await fetchMultiplePrices(symbols);
    
    const result: Record<string, number> = {};
    prices.forEach((v, k) => { result[k] = v; });
    
    return NextResponse.json({ 
      prices: result,
      timestamp: new Date().toISOString() 
    });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
