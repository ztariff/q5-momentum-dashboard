export interface Trade {
  symbol: string;
  instrument: string;
  tier: string;
  body_quintile: string;
  sizing_rule: string;
  option_ticker: string;
  strike: number | null;
  expiry: string;
  signal_date: string;
  entry_date: string;
  entry_time: string;
  entry_price: number;
  exit_date: string;
  exit_time: string;
  exit_price: number;
  exit_type: string;
  position_size: number;
  shares_or_contracts: number;
  hold_days: number;
  net_pnl: number;
  return_pct: number;
  body_pct: number;
  atr_change_3d: number;
  opt_source: string;
}

export interface Position extends Trade {
  current_price?: number;
  unrealized_pnl?: number;
  unrealized_pnl_pct?: number;
  stop_price?: number;
  days_held?: number;
  days_remaining?: number;
  scheduled_exit_date?: string;
  distance_to_stop_pct?: number;
  distance_to_stop_dollar?: number;
  max_hold_days: number;
  is_open: boolean;
}

export interface WatchlistItem {
  symbol: string;
  quintile: number;
  z_score: number;
  slope: number;
  distance_to_q5: number;
  approaching: boolean;
}

export interface Signal {
  symbol: string;
  z_score: number;
  slope: number;
  quintile: number;
  prev_quintile: number;
  body_pct: number;
  atr_change_3d: number;
  tier: string;
  sizing_rule: string;
  recommended_size: number;
  signal_date: string;
}

export interface PerformanceData {
  total_realized_pnl: number;
  total_unrealized_pnl: number;
  total_pnl: number;
  win_rate: number;
  profit_factor: number;
  max_drawdown: number;
  sharpe: number;
  total_trades: number;
  avg_pnl_per_trade: number;
  stop_rate: number;
  monthly_data: MonthlyData[];
  equity_curve: EquityPoint[];
  yearly_pnl: YearlyPnl[];
}

export interface MonthlyData {
  month: string;
  net_total: number;
  win_rate: number;
  days: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
}

export interface YearlyPnl {
  year: string;
  pnl: number;
}

export interface PriceData {
  symbol: string;
  price: number;
  timestamp: number;
  error?: string;
}
