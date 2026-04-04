'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import Header from '@/components/Header';
import PositionsTable from '@/components/PositionsTable';
import RiskAlerts from '@/components/RiskAlerts';
import SignalScanner from '@/components/SignalScanner';
import PerformanceSummary from '@/components/PerformanceSummary';
import { Position } from '@/lib/types';
import { Activity, TrendingUp, AlertTriangle, Zap, BarChart2, Info } from 'lucide-react';

type TabId = 'positions' | 'signals' | 'risk' | 'performance';

const TABS = [
  { id: 'positions' as const, label: 'Current Positions', icon: Activity },
  { id: 'risk' as const, label: 'Risk Alerts', icon: AlertTriangle },
  { id: 'signals' as const, label: 'Signal Scanner', icon: Zap },
  { id: 'performance' as const, label: 'Performance', icon: BarChart2 },
];

const REFRESH_INTERVAL = 60 * 1000;

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>('positions');
  const [positions, setPositions] = useState<Position[]>([]);
  const [positionsLoading, setPositionsLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [demoNote, setDemoNote] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadPositions = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const resp = await fetch('/api/positions');
      if (!resp.ok) throw new Error(`API error: ${resp.status}`);
      const data = await resp.json();
      setPositions(data.positions || []);
      setIsDemoMode(data.is_demo || false);
      setDemoNote(data.demo_note || null);
      setLastRefresh(new Date().toLocaleTimeString());
    } catch (err) {
      console.error('Error loading positions:', err);
    } finally {
      setPositionsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadPositions();
  }, [loadPositions]);

  useEffect(() => {
    intervalRef.current = setInterval(loadPositions, REFRESH_INTERVAL);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [loadPositions]);

  const nearStop = positions.filter(p => p.is_open && (p.distance_to_stop_pct ?? 100) < 20).length;
  const nearExit = positions.filter(p => p.is_open && (p.days_remaining ?? 99) <= 2).length;
  const riskCount = nearStop + nearExit;

  const totalUnrealized = positions.filter(p => p.is_open).reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);
  const totalRealizedRecent = positions.filter(p => !p.is_open).reduce((s, p) => s + (p.net_pnl ?? 0), 0);

  const displayPnl = isDemoMode ? totalRealizedRecent : totalUnrealized;
  const pnlLabel = isDemoMode ? 'Recent Realized P&L' : 'Unrealized P&L';

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#0a0e1a' }}>
      <Header lastRefresh={lastRefresh} isRefreshing={isRefreshing} onRefresh={loadPositions} />

      <main className="max-w-screen-2xl mx-auto px-4 py-6">
        {/* Demo mode notice */}
        {isDemoMode && demoNote && (
          <div
            className="flex items-start gap-3 rounded-lg px-4 py-3 mb-5 text-sm"
            style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)', color: '#93c5fd' }}
          >
            <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{demoNote}</span>
          </div>
        )}

        {/* Top-level stats bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <div className="stat-card flex items-center gap-3">
            <Activity className="w-5 h-5 flex-shrink-0" style={{ color: '#60a5fa' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>
                {isDemoMode ? 'Recent Positions' : 'Open Positions'}
              </div>
              <div className="text-2xl font-bold" style={{ color: '#f1f5f9' }}>{positions.length}</div>
            </div>
          </div>
          <div className="stat-card flex items-center gap-3">
            <TrendingUp className="w-5 h-5 flex-shrink-0" style={{ color: displayPnl >= 0 ? '#22c55e' : '#ef4444' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>{pnlLabel}</div>
              <div className="text-2xl font-bold" style={{ color: displayPnl >= 0 ? '#22c55e' : '#ef4444' }}>
                {displayPnl >= 0 ? '+' : ''}
                {Math.abs(displayPnl) >= 1000000
                  ? `$${(displayPnl / 1000000).toFixed(2)}M`
                  : `$${(displayPnl / 1000).toFixed(1)}K`}
              </div>
            </div>
          </div>
          <div className="stat-card flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" style={{ color: riskCount > 0 ? '#f59e0b' : '#22c55e' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>Risk Alerts</div>
              <div className="text-2xl font-bold" style={{ color: riskCount > 0 ? '#f59e0b' : '#22c55e' }}>
                {isDemoMode ? '—' : riskCount}
              </div>
            </div>
          </div>
          <div className="stat-card flex items-center gap-3">
            <BarChart2 className="w-5 h-5 flex-shrink-0" style={{ color: '#a855f7' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>Backtest P&L (2020–26)</div>
              <div className="text-2xl font-bold" style={{ color: '#22c55e' }}>+$88.2M</div>
            </div>
          </div>
        </div>

        {/* Navigation tabs */}
        <div className="flex gap-1 mb-5 overflow-x-auto pb-1">
          {TABS.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const badge = tab.id === 'risk' && riskCount > 0 && !isDemoMode ? riskCount : null;

            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold whitespace-nowrap transition-all"
                style={{
                  backgroundColor: isActive ? '#1e3a5f' : '#1a2035',
                  color: isActive ? '#60a5fa' : '#64748b',
                  border: `1px solid ${isActive ? '#3b82f6' : '#2d4a7a'}`,
                }}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
                {badge !== null && (
                  <span className="text-xs px-1.5 py-0.5 rounded-full font-bold" style={{ backgroundColor: 'rgba(245, 158, 11, 0.2)', color: '#fbbf24' }}>
                    {badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <div className="rounded-xl p-5" style={{ backgroundColor: '#0d1425', border: '1px solid #1a2035' }}>
          {activeTab === 'positions' && (
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4" style={{ color: '#60a5fa' }} />
                  <h2 className="text-sm font-semibold tracking-wider uppercase" style={{ color: '#64748b' }}>
                    {isDemoMode ? 'Recent Trade History (Q1–Q2 2026)' : 'Current Open Positions'}
                  </h2>
                </div>
                {isDemoMode && (
                  <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                    HISTORICAL DATA
                  </span>
                )}
              </div>
              <PositionsTable positions={positions} isLoading={positionsLoading} isDemoMode={isDemoMode} />
            </div>
          )}

          {activeTab === 'risk' && (
            <div>
              <RiskAlerts positions={positions} isDemoMode={isDemoMode} />
            </div>
          )}

          {activeTab === 'signals' && (
            <div>
              <SignalScanner />
            </div>
          )}

          {activeTab === 'performance' && (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <BarChart2 className="w-4 h-4" style={{ color: '#a855f7' }} />
                <h2 className="text-sm font-semibold tracking-wider uppercase" style={{ color: '#64748b' }}>
                  Strategy Performance Summary (2020–2026)
                </h2>
              </div>
              <PerformanceSummary />
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-4 flex items-center justify-between px-1">
          <span className="text-xs" style={{ color: '#374151' }}>
            Q5 Momentum Dashboard — Data via Polygon.io — 230 symbols
          </span>
          <span className="text-xs" style={{ color: '#374151' }}>
            Prices refresh every 60s during market hours
          </span>
        </div>
      </main>
    </div>
  );
}
