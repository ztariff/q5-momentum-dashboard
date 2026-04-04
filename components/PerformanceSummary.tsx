'use client';

import { useState, useEffect } from 'react';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts';
import { TrendingUp, Award, Target, BarChart2 } from 'lucide-react';

interface PerfData {
  total_realized_pnl: number;
  total_pnl: number;
  win_rate: number;
  profit_factor: number;
  max_drawdown: number;
  sharpe: number;
  total_trades: number;
  avg_pnl_per_trade: number;
  stop_rate: number;
  monthly_data: Array<{ month: string; net_total: number; win_rate: number; days: number }>;
  equity_curve: Array<{ date: string; equity: number }>;
  yearly_pnl: Array<{ year: string; pnl: number }>;
}

function formatPnl(val: number): string {
  const abs = Math.abs(val);
  const sign = val >= 0 ? '+' : '-';
  if (abs >= 1000000) return `${sign}$${(abs / 1000000).toFixed(2)}M`;
  if (abs >= 1000) return `${sign}$${(abs / 1000).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

const CustomTooltip = ({ active, payload, label }: Record<string, unknown>) => {
  if ((active as boolean) && (payload as unknown[])?.length) {
    const data = (payload as Array<{ value: number; name: string }>)[0];
    return (
      <div
        className="rounded-lg px-3 py-2 text-xs"
        style={{ backgroundColor: '#1a2035', border: '1px solid #2d4a7a', color: '#e2e8f0' }}
      >
        <p style={{ color: '#64748b' }}>{label as string}</p>
        <p style={{ color: data.value >= 0 ? '#22c55e' : '#ef4444', fontWeight: 700 }}>
          {typeof data.value === 'number' && Math.abs(data.value) > 1000
            ? formatPnl(data.value)
            : `${data.value >= 0 ? '+' : ''}${(typeof data.value === 'number' ? data.value : 0).toFixed(1)}${data.name.includes('rate') ? '%' : ''}`}
        </p>
      </div>
    );
  }
  return null;
};

export default function PerformanceSummary() {
  const [data, setData] = useState<PerfData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeChart, setActiveChart] = useState<'equity' | 'monthly' | 'yearly'>('equity');

  useEffect(() => {
    fetch('/api/performance')
      .then(r => r.json())
      .then(d => setData(d))
      .catch(e => console.error(e))
      .finally(() => setIsLoading(false));
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="animate-spin w-8 h-8 border-2 rounded-full" style={{ borderColor: '#2d4a7a', borderTopColor: '#60a5fa' }} />
      </div>
    );
  }

  if (!data) return null;

  // Transform monthly data - values are in $K
  const monthlyChart = data.monthly_data.slice(-24).map(m => ({
    month: m.month.substring(0, 7),
    pnl: m.net_total * 1000,
    win_rate: m.win_rate,
  }));

  // Equity curve
  const equityChart = data.equity_curve.map(p => ({
    date: p.date,
    equity: p.equity,
  }));

  const yearlyChart = data.yearly_pnl.map(y => ({
    year: y.year,
    pnl: y.pnl,
  }));

  const statCards = [
    { label: 'Total P&L', value: formatPnl(data.total_realized_pnl), positive: data.total_realized_pnl >= 0, icon: <TrendingUp className="w-4 h-4" /> },
    { label: 'Win Rate', value: `${data.win_rate.toFixed(1)}%`, positive: data.win_rate >= 50, icon: <Award className="w-4 h-4" /> },
    { label: 'Profit Factor', value: data.profit_factor.toFixed(2), positive: data.profit_factor >= 1, icon: <BarChart2 className="w-4 h-4" /> },
    { label: 'Sharpe Ratio', value: data.sharpe.toFixed(2), positive: data.sharpe >= 0, icon: <Target className="w-4 h-4" /> },
    { label: 'Max Drawdown', value: formatPnl(data.max_drawdown), positive: false, icon: <TrendingUp className="w-4 h-4" style={{ transform: 'rotate(180deg)' }} /> },
    { label: 'Avg P&L/Trade', value: formatPnl(data.avg_pnl_per_trade), positive: data.avg_pnl_per_trade >= 0, icon: <BarChart2 className="w-4 h-4" /> },
    { label: 'Total Trades', value: data.total_trades.toLocaleString(), positive: true, icon: <Target className="w-4 h-4" /> },
    { label: 'Stop Rate', value: `${data.stop_rate.toFixed(1)}%`, positive: data.stop_rate < 30, icon: <Award className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-6">
      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {statCards.map((card, i) => (
          <div key={i} className="stat-card">
            <div className="flex items-center gap-2 mb-2" style={{ color: '#64748b' }}>
              {card.icon}
              <span className="text-xs uppercase tracking-wider">{card.label}</span>
            </div>
            <div
              className="text-xl font-bold font-mono"
              style={{ color: card.positive ? '#22c55e' : '#ef4444' }}
            >
              {card.value}
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="rounded-lg p-4" style={{ backgroundColor: '#1a2035', border: '1px solid #2d4a7a' }}>
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-semibold tracking-wider uppercase" style={{ color: '#64748b' }}>
            Performance Charts
          </span>
          <div className="flex gap-1">
            {(['equity', 'monthly', 'yearly'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveChart(tab)}
                className="px-3 py-1 rounded text-xs transition-all"
                style={{
                  backgroundColor: activeChart === tab ? '#1e3a5f' : 'transparent',
                  color: activeChart === tab ? '#60a5fa' : '#64748b',
                  border: `1px solid ${activeChart === tab ? '#3b82f6' : '#2d4a7a'}`,
                }}
              >
                {tab === 'equity' ? 'Equity Curve' : tab === 'monthly' ? 'Monthly P&L' : 'Yearly P&L'}
              </button>
            ))}
          </div>
        </div>

        <div style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            {activeChart === 'equity' ? (
              <AreaChart data={equityChart} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: '#475569' }}
                  tickLine={false}
                  axisLine={{ stroke: '#2d4a7a' }}
                  tickFormatter={v => v.substring(0, 7)}
                  interval={11}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: '#475569' }}
                  tickLine={false}
                  axisLine={{ stroke: '#2d4a7a' }}
                  tickFormatter={v => `$${(v / 1000000).toFixed(0)}M`}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fill="url(#equityGradient)"
                />
              </AreaChart>
            ) : activeChart === 'monthly' ? (
              <BarChart data={monthlyChart} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis
                  dataKey="month"
                  tick={{ fontSize: 9, fill: '#475569' }}
                  tickLine={false}
                  axisLine={{ stroke: '#2d4a7a' }}
                  interval={2}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: '#475569' }}
                  tickLine={false}
                  axisLine={{ stroke: '#2d4a7a' }}
                  tickFormatter={v => `$${(v / 1000).toFixed(0)}K`}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="#2d4a7a" />
                <Bar
                  dataKey="pnl"
                  fill="#3b82f6"
                  radius={[2, 2, 0, 0]}
                  // Color each bar based on positive/negative
                  // Using a custom cell approach
                />
              </BarChart>
            ) : (
              <BarChart data={yearlyChart} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis
                  dataKey="year"
                  tick={{ fontSize: 11, fill: '#475569' }}
                  tickLine={false}
                  axisLine={{ stroke: '#2d4a7a' }}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: '#475569' }}
                  tickLine={false}
                  axisLine={{ stroke: '#2d4a7a' }}
                  tickFormatter={v => `$${(v / 1000000).toFixed(0)}M`}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="#2d4a7a" />
                <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
                  {yearlyChart.map((entry, index) => (
                    <rect key={index} fill={entry.pnl >= 0 ? '#22c55e' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>

      {/* Annual P&L breakdown */}
      <div className="rounded-lg p-4" style={{ backgroundColor: '#1a2035', border: '1px solid #2d4a7a' }}>
        <span className="text-xs font-semibold tracking-wider uppercase mb-3 block" style={{ color: '#64748b' }}>
          Annual P&L Breakdown
        </span>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          {yearlyChart.map(y => (
            <div key={y.year} className="text-center">
              <div className="text-xs mb-1" style={{ color: '#64748b' }}>{y.year}</div>
              <div
                className="font-bold text-sm"
                style={{ color: y.pnl >= 0 ? '#22c55e' : '#ef4444' }}
              >
                {formatPnl(y.pnl)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
