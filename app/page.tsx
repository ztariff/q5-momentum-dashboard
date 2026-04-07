'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import Header from '@/components/Header';
import PositionsTable from '@/components/PositionsTable';
import RiskAlerts from '@/components/RiskAlerts';
import SignalScanner from '@/components/SignalScanner';
import PerformanceSummary from '@/components/PerformanceSummary';
import { Position } from '@/lib/types';
import { Activity, TrendingUp, AlertTriangle, Eye, BarChart2, DollarSign, Calendar } from 'lucide-react';

type TabId = 'positions' | 'signals' | 'risk' | 'performance';

const TABS = [
  { id: 'positions' as const, label: 'Current Positions', icon: Activity },
  { id: 'risk' as const, label: 'Risk Alerts', icon: AlertTriangle },
  { id: 'signals' as const, label: 'Watchlist', icon: Eye },
  { id: 'performance' as const, label: 'Performance', icon: BarChart2 },
];

// Display refresh: re-read live_state.json every 60s
const DISPLAY_REFRESH_INTERVAL = 60 * 1000;

// Data staleness threshold: if last_refresh > 6 hours ago, auto-trigger /api/refresh
const STALE_THRESHOLD_MS = 6 * 60 * 60 * 1000;

interface Summary {
  total_positions: number;
  total_unrealized: number;
  total_realized: number;
  month_unrealized: number;
  month_realized: number;
  year_realized: number;
  year_total: number;
  current_month: string;
}

function formatPnlShort(val: number): string {
  const abs = Math.abs(val);
  const sign = val >= 0 ? '+' : '-';
  if (abs >= 1000000) return `${sign}$${(abs / 1000000).toFixed(2)}M`;
  if (abs >= 1000) return `${sign}$${(abs / 1000).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

function formatMonthLabel(ym: string): string {
  if (!ym) return 'Current Month';
  const [year, month] = ym.split('-');
  const d = new Date(parseInt(year), parseInt(month) - 1, 1);
  return d.toLocaleString('default', { month: 'long', year: 'numeric' });
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>('positions');
  const [positions, setPositions] = useState<Position[]>([]);
  const [positionsLoading, setPositionsLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [summary, setSummary] = useState<Summary | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoRefreshTriggered = useRef(false);

  // ── Display refresh: reads live_state.json via /api/positions ─────────────
  const loadPositions = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const resp = await fetch('/api/positions');
      if (!resp.ok) throw new Error(`API error: ${resp.status}`);
      const data = await resp.json();
      setPositions(data.positions || []);
      setSummary(data.summary || null);
      setLastRefresh(new Date().toLocaleTimeString());
    } catch (err) {
      console.error('Error loading positions:', err);
    } finally {
      setPositionsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  // ── Auto data refresh: if live_state.json is stale, call /api/refresh once ─
  const triggerDataRefreshIfStale = useCallback(async () => {
    if (autoRefreshTriggered.current) return;

    try {
      const resp = await fetch('/api/positions');
      if (!resp.ok) return;
      const data = await resp.json();
      const lastRefreshStr: string | null = data.last_refresh;

      if (!lastRefreshStr) {
        // No data at all — trigger refresh
        autoRefreshTriggered.current = true;
        triggerBackgroundRefresh();
        return;
      }

      const lastRefreshTime = new Date(lastRefreshStr).getTime();
      const ageMs = Date.now() - lastRefreshTime;

      if (ageMs > STALE_THRESHOLD_MS) {
        autoRefreshTriggered.current = true;
        triggerBackgroundRefresh();
      }
    } catch {
      // silently fail — don't block page load
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const triggerBackgroundRefresh = useCallback(async () => {
    try {
      const refreshResp = await fetch('/api/refresh');
      if (refreshResp.ok) {
        await loadPositions();
      }
    } catch {
      // silently fail — don't block page load
    }
  }, [loadPositions]);

  useEffect(() => {
    loadPositions();
    // triggerDataRefreshIfStale(); // disabled — causes Railway issues
  }, [loadPositions, triggerDataRefreshIfStale]);

  useEffect(() => {
    intervalRef.current = setInterval(loadPositions, DISPLAY_REFRESH_INTERVAL);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [loadPositions]);

  // ── Risk badge counts ──────────────────────────────────────────────────────
  const stopCrossed = positions.filter(p => p.is_open && p.stop_price != null && p.current_price != null && p.current_price < p.stop_price).length;
  const overdue = positions.filter(p => p.is_open && (p.days_remaining ?? 99) === 0).length;
  const deepLoss = positions.filter(p => p.is_open && (p.unrealized_pnl_pct ?? 0) < -50).length;
  const nearStop = positions.filter(p => p.is_open && (p.distance_to_stop_pct ?? 100) < 20 && (p.distance_to_stop_pct ?? 100) >= 0).length;
  const nearExit = positions.filter(p => { const rem = p.days_remaining ?? 99; return p.is_open && rem >= 1 && rem <= 2; }).length;
  const riskCount = stopCrossed + overdue + deepLoss + nearStop + nearExit;

  const totalPositions = summary?.total_positions ?? 0;
  const totalUnrealized = summary?.total_unrealized ?? 0;
  const totalRealized = summary?.total_realized ?? 0;
  const monthUnrealized = summary?.month_unrealized ?? 0;
  const monthRealized = summary?.month_realized ?? 0;
  const yearRealized = summary?.year_realized ?? 0;
  const yearTotal = summary?.year_total ?? yearRealized;
  const currentMonth = summary?.current_month ?? '';
  const monthLabel = formatMonthLabel(currentMonth);

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#0a0e1a' }}>
      <Header
        lastRefresh={lastRefresh}
        isRefreshing={isRefreshing}
        onRefresh={loadPositions}
      />

      <main className="max-w-screen-2xl mx-auto px-4 py-6">
        {/* Top-level stats bar */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-6">

          <div className="stat-card flex items-center gap-3">
            <Activity className="w-5 h-5 flex-shrink-0" style={{ color: '#60a5fa' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>
                Total Positions
              </div>
              <div className="text-2xl font-bold" style={{ color: '#f1f5f9' }}>
                {(totalPositions || 0).toLocaleString()}
              </div>
            </div>
          </div>

          <div className="stat-card flex items-center gap-3">
            <TrendingUp className="w-5 h-5 flex-shrink-0" style={{ color: totalUnrealized >= 0 ? '#22c55e' : '#ef4444' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>Total Unrealized</div>
              <div className="text-2xl font-bold" style={{ color: totalUnrealized >= 0 ? '#22c55e' : '#ef4444' }}>
                {formatPnlShort(totalUnrealized)}
              </div>
            </div>
          </div>

          <div className="stat-card flex items-center gap-3">
            <BarChart2 className="w-5 h-5 flex-shrink-0" style={{ color: '#a855f7' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>Realized P&L</div>
              <div className="text-2xl font-bold" style={{ color: totalRealized >= 0 ? '#22c55e' : '#ef4444' }}>
                {formatPnlShort(totalRealized)}
              </div>
            </div>
          </div>

          <div className="stat-card flex items-center gap-3">
            <Calendar className="w-5 h-5 flex-shrink-0" style={{ color: monthUnrealized >= 0 ? '#22c55e' : '#ef4444' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>
                {monthLabel} Unrealized
              </div>
              <div className="text-2xl font-bold" style={{ color: monthUnrealized >= 0 ? '#22c55e' : '#ef4444' }}>
                {formatPnlShort(monthUnrealized)}
              </div>
            </div>
          </div>

          <div className="stat-card flex items-center gap-3">
            <DollarSign className="w-5 h-5 flex-shrink-0" style={{ color: monthRealized >= 0 ? '#22c55e' : '#ef4444' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>
                {monthLabel} Realized
              </div>
              <div className="text-2xl font-bold" style={{ color: monthRealized >= 0 ? '#22c55e' : '#ef4444' }}>
                {formatPnlShort(monthRealized)}
              </div>
            </div>
          </div>

          <div className="stat-card flex items-center gap-3">
            <TrendingUp className="w-5 h-5 flex-shrink-0" style={{ color: yearTotal >= 0 ? '#22c55e' : '#ef4444' }} />
            <div>
              <div className="text-xs uppercase tracking-wider mb-0.5" style={{ color: '#64748b' }}>
                Total 2026 P&L
              </div>
              <div className="text-2xl font-bold" style={{ color: yearTotal >= 0 ? '#22c55e' : '#ef4444' }}>
                {formatPnlShort(yearTotal)}
              </div>
            </div>
          </div>

        </div>

        {/* Navigation tabs */}
        <div className="flex gap-1 mb-5 overflow-x-auto pb-1">
          {TABS.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const badge = tab.id === 'risk' && riskCount > 0 ? riskCount : null;

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
              <div className="flex items-center gap-2 mb-4">
                <Activity className="w-4 h-4" style={{ color: '#60a5fa' }} />
                <h2 className="text-sm font-semibold tracking-wider uppercase" style={{ color: '#64748b' }}>
                  Current Positions
                </h2>
              </div>
              <PositionsTable positions={positions} isLoading={positionsLoading} />
            </div>
          )}

          {activeTab === 'risk' && (
            <div>
              <RiskAlerts positions={positions} />
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
            Q5 Momentum — Dual Ranking (R1: 230 options, R2: 366 all) — Polygon.io
          </span>
          <div className="flex items-center gap-3">
            <button
              onClick={async () => {
                const btn = document.getElementById('engine-btn');
                if (btn) {
                  btn.textContent = 'Running...';
                  (btn as HTMLButtonElement).disabled = true;
                }
                try {
                  const resp = await fetch('/api/engine', { method: 'POST' });
                  const data = await resp.json();
                  if (data.success) {
                    if (btn) btn.textContent = 'Done ✓';
                    loadPositions();
                    setTimeout(() => { if (btn) { btn.textContent = 'Run Engine'; (btn as HTMLButtonElement).disabled = false; } }, 3000);
                  } else {
                    if (btn) { btn.textContent = 'Error'; (btn as HTMLButtonElement).disabled = false; }
                  }
                } catch {
                  if (btn) { btn.textContent = 'Error'; (btn as HTMLButtonElement).disabled = false; }
                }
              }}
              id="engine-btn"
              className="px-3 py-1 rounded text-xs font-semibold transition-all"
              style={{ backgroundColor: '#1e3a5f', color: '#60a5fa', border: '1px solid #3b82f6' }}
            >
              Run Engine
            </button>
            <span className="text-xs" style={{ color: '#374151' }}>
              Auto-runs at 4:17 PM ET
            </span>
          </div>
        </div>
      </main>
    </div>
  );
}
