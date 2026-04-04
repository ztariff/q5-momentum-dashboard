import { Trade } from './types';

// Parse CSV respecting quoted fields
function parseCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  
  result.push(current.trim());
  return result;
}

export function parseTrades(csvText: string): Trade[] {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return [];
  
  const header = parseCsvLine(lines[0]);
  const trades: Trade[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;
    
    const values = parseCsvLine(line);

    const get = (field: string): string => {
      const idx = header.indexOf(field);
      return idx >= 0 ? (values[idx] || '') : '';
    };

    const symbol = get('symbol');
    if (!symbol) continue;

    const trade: Trade = {
      symbol,
      instrument: get('instrument'),
      tier: get('tier'),
      body_quintile: get('body_quintile'),
      sizing_rule: get('sizing_rule'),
      option_ticker: get('option_ticker'),
      strike: parseFloat(get('strike')) || null,
      expiry: get('expiry'),
      signal_date: get('signal_date'),
      entry_date: get('entry_date'),
      entry_time: get('entry_time'),
      entry_price: parseFloat(get('entry_price')) || 0,
      exit_date: get('exit_date'),
      exit_time: get('exit_time'),
      exit_price: parseFloat(get('exit_price')) || 0,
      exit_type: get('exit_type'),
      position_size: parseFloat(get('position_size')) || 0,
      shares_or_contracts: parseFloat(get('shares_or_contracts')) || 0,
      hold_days: parseInt(get('hold_days')) || 0,
      net_pnl: parseFloat(get('net_pnl')) || 0,
      return_pct: parseFloat(get('return_pct')) || 0,
      body_pct: parseFloat(get('body_pct')) || 0,
      atr_change_3d: parseFloat(get('atr_change_3d')) || 0,
      opt_source: get('opt_source'),
    };

    trades.push(trade);
  }

  return trades;
}

export function parseMonthly(csvText: string) {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return [];
  
  const header = parseCsvLine(lines[0]);
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCsvLine(lines[i]);
    const get = (field: string): string => {
      const idx = header.indexOf(field);
      return idx >= 0 ? (values[idx] || '') : '';
    };
    
    const monthStr = get('month');
    if (!monthStr) continue;
    
    data.push({
      month: monthStr,
      net_total: parseFloat(get('net_total')) || 0,
      win_rate: parseFloat(get('win_rate')) || 0,
      days: parseInt(get('days')) || 0,
    });
  }
  
  return data;
}

export function parseUniverse(csvText: string): string[] {
  const lines = csvText.trim().split('\n');
  const symbols: string[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const parts = parseCsvLine(lines[i]);
    if (parts[0] && parts[0].trim()) {
      symbols.push(parts[0].trim());
    }
  }
  
  return symbols;
}
