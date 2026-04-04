'use client';

import { Position } from '@/lib/types';
import { AlertTriangle, Clock, TrendingDown, Target, Info } from 'lucide-react';

interface RiskAlertsProps {
  positions: Position[];
  isDemoMode?: boolean;
}

export default function RiskAlerts({ positions, isDemoMode = false }: RiskAlertsProps) {
  if (isDemoMode) {
    return (
      <div>
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-4 h-4" style={{ color: '#f59e0b' }} />
          <span className="text-xs font-semibold tracking-wider uppercase" style={{ color: '#64748b' }}>Risk Alerts</span>
        </div>
        <div className="rounded-lg p-6 text-center" style={{ backgroundColor: 'rgba(59, 130, 246, 0.05)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
          <Info className="w-8 h-8 mx-auto mb-3" style={{ color: '#60a5fa' }} />
          <p className="text-sm font-medium mb-1" style={{ color: '#93c5fd' }}>No Live Positions</p>
          <p className="text-xs" style={{ color: '#475569' }}>
            Risk alerts track live open positions. The backtest data ends on 2026-04-02.
            When live trading begins, alerts will appear here for:
          </p>
          <div className="grid grid-cols-2 gap-3 mt-4 max-w-md mx-auto">
            {[
              { icon: <Target className="w-4 h-4" />, label: 'Stop Price Crossings', color: '#ef4444' },
              { icon: <AlertTriangle className="w-4 h-4" />, label: 'Within 20% of Stop', color: '#f59e0b' },
              { icon: <Clock className="w-4 h-4" />, label: '2 Days to Exit', color: '#a855f7' },
              { icon: <TrendingDown className="w-4 h-4" />, label: 'Down More Than 50%', color: '#ef4444' },
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-2 rounded" style={{ backgroundColor: '#1a2035' }}>
                <span style={{ color: item.color }}>{item.icon}</span>
                <span className="text-xs" style={{ color: '#64748b' }}>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const openPositions = positions.filter(p => p.is_open);
  const nearStop = openPositions.filter(p => (p.distance_to_stop_pct ?? 100) < 20);
  const nearExit = openPositions.filter(p => (p.days_remaining ?? 99) <= 2);
  const deepLoss = openPositions.filter(p => (p.unrealized_pnl_pct ?? 0) < -50);
  const stopCrossed = openPositions.filter(p => {
    if (!p.stop_price || !p.current_price) return false;
    return p.current_price < p.stop_price;
  });

  const allAlerts = [
    ...stopCrossed.map(p => ({
      type: 'STOP CROSSED',
      symbol: p.symbol,
      detail: `Current $${p.current_price?.toFixed(2)} below stop $${p.stop_price?.toFixed(2)}`,
      severity: 'critical' as const,
      icon: <Target className="w-4 h-4" />,
    })),
    ...deepLoss.map(p => ({
      type: 'DEEP LOSS',
      symbol: p.symbol,
      detail: `Down ${(p.unrealized_pnl_pct ?? 0).toFixed(1)}% from entry`,
      severity: 'high' as const,
      icon: <TrendingDown className="w-4 h-4" />,
    })),
    ...nearStop.filter(p => !stopCrossed.includes(p)).map(p => ({
      type: 'NEAR STOP',
      symbol: p.symbol,
      detail: `${(p.distance_to_stop_pct ?? 0).toFixed(1)}% above stop price $${p.stop_price?.toFixed(2)}`,
      severity: 'medium' as const,
      icon: <AlertTriangle className="w-4 h-4" />,
    })),
    ...nearExit.map(p => ({
      type: 'EXIT SOON',
      symbol: p.symbol,
      detail: `${p.days_remaining}d remaining, scheduled exit ${p.scheduled_exit_date}`,
      severity: 'low' as const,
      icon: <Clock className="w-4 h-4" />,
    })),
  ];

  const severityStyles = {
    critical: { bg: 'rgba(220, 38, 38, 0.1)', border: '#dc2626', label: '#ef4444', icon: '#ef4444' },
    high: { bg: 'rgba(239, 68, 68, 0.1)', border: '#ef4444', label: '#f87171', icon: '#f87171' },
    medium: { bg: 'rgba(245, 158, 11, 0.1)', border: '#f59e0b', label: '#fbbf24', icon: '#fbbf24' },
    low: { bg: 'rgba(168, 85, 247, 0.1)', border: '#a855f7', label: '#c084fc', icon: '#c084fc' },
  };

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <AlertTriangle className="w-4 h-4" style={{ color: '#f59e0b' }} />
        <span className="text-xs font-semibold tracking-wider uppercase" style={{ color: '#64748b' }}>Risk Alerts</span>
        {allAlerts.length > 0 && (
          <span className="text-xs px-2 py-0.5 rounded-full font-bold" style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}>
            {allAlerts.length}
          </span>
        )}
      </div>

      {allAlerts.length === 0 ? (
        <div className="rounded-lg p-4 text-center text-sm" style={{ backgroundColor: 'rgba(34, 197, 94, 0.05)', border: '1px solid rgba(34, 197, 94, 0.2)' }}>
          <span style={{ color: '#4ade80' }}>No active risk alerts — all positions within normal parameters</span>
        </div>
      ) : (
        <div className="space-y-2">
          {allAlerts.map((alert, i) => {
            const style = severityStyles[alert.severity];
            return (
              <div key={i} className="flex items-center gap-3 rounded-lg px-4 py-3" style={{ backgroundColor: style.bg, border: `1px solid ${style.border}20` }}>
                <div style={{ color: style.icon }}>{alert.icon}</div>
                <div className="flex items-center gap-2 flex-1">
                  <span className="badge text-xs font-bold" style={{ backgroundColor: `${style.border}20`, color: style.label, border: `1px solid ${style.border}40` }}>
                    {alert.type}
                  </span>
                  <span className="font-bold text-sm" style={{ color: '#f1f5f9' }}>{alert.symbol}</span>
                  <span className="text-xs" style={{ color: '#94a3b8' }}>{alert.detail}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
