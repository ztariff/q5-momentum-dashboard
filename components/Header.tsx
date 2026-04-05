'use client';

import { Activity, TrendingUp, RefreshCw, Database } from 'lucide-react';
import { useState } from 'react';

interface HeaderProps {
  lastRefresh: string | null;
  isRefreshing: boolean;
  onRefresh: () => void;
  onDataRefresh?: () => Promise<void>;
  dataRefreshProgress?: { done: number; total: number } | null;
}

export default function Header({
  lastRefresh,
  isRefreshing,
  onRefresh,
  onDataRefresh,
  dataRefreshProgress,
}: HeaderProps) {
  const [isDataRefreshing, setIsDataRefreshing] = useState(false);
  const [dataRefreshStatus, setDataRefreshStatus] = useState<string | null>(null);

  const now = new Date();
  // Adjust for ET: UTC-5 (EST) or UTC-4 (EDT)
  const isDST = now.getMonth() >= 2 && now.getMonth() <= 10;
  const etOffset = isDST ? 4 : 5;
  const etHour = (now.getUTCHours() - etOffset + 24) % 24;
  const isMarketHours =
    etHour >= 9 && etHour < 16 && now.getUTCDay() >= 1 && now.getUTCDay() <= 5;

  const handleDataRefresh = async () => {
    if (isDataRefreshing) return;
    setIsDataRefreshing(true);
    setDataRefreshStatus('Fetching 230 symbols from Polygon...');

    try {
      if (onDataRefresh) {
        await onDataRefresh();
        setDataRefreshStatus('Done');
      } else {
        // Direct call to /api/refresh
        const resp = await fetch('/api/refresh');
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ error: resp.statusText }));
          setDataRefreshStatus(`Error: ${err.error ?? resp.statusText}`);
          return;
        }
        const result = await resp.json();
        setDataRefreshStatus(
          `Done — ${result.symbols_processed ?? 0} symbols, ${result.positions ?? 0} positions, ${result.pending_signals ?? 0} signals`
        );
        // Reload display data
        onRefresh();
      }
    } catch (err) {
      setDataRefreshStatus(`Error: ${String(err)}`);
    } finally {
      setIsDataRefreshing(false);
      // Clear status after 8 seconds
      setTimeout(() => setDataRefreshStatus(null), 8000);
    }
  };

  const progress = dataRefreshProgress;

  return (
    <header className="border-b" style={{ borderColor: '#2d4a7a', backgroundColor: '#0d1425' }}>
      <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center justify-between gap-3">
        {/* Left: Logo */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5" style={{ color: '#60a5fa' }} />
            <span className="font-bold text-lg tracking-wider" style={{ color: '#e2e8f0' }}>
              Q5 MOMENTUM
            </span>
          </div>
          <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: '#1a2035', color: '#64748b' }}>
            DASHBOARD
          </span>
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-3 flex-wrap justify-end">
          {/* Data refresh progress / status */}
          {(isDataRefreshing || dataRefreshStatus) && (
            <div className="text-xs px-3 py-1 rounded" style={{ backgroundColor: 'rgba(96,165,250,0.1)', color: '#60a5fa', border: '1px solid rgba(96,165,250,0.25)' }}>
              {isDataRefreshing && progress ? (
                <span>
                  Refreshing data... {progress.done}/{progress.total} symbols
                </span>
              ) : isDataRefreshing ? (
                <span>Refreshing data...</span>
              ) : (
                <span>{dataRefreshStatus}</span>
              )}
            </div>
          )}

          {/* Market status */}
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full live-indicator"
              style={{ backgroundColor: isMarketHours ? '#22c55e' : '#ef4444' }}
            />
            <span className="text-xs" style={{ color: '#64748b' }}>
              {isMarketHours ? 'MARKET OPEN' : 'MARKET CLOSED'}
            </span>
          </div>

          {/* Live indicator */}
          <div className="flex items-center gap-1">
            <Activity className="w-3 h-3" style={{ color: '#22c55e' }} />
            <span className="text-xs live-indicator" style={{ color: '#22c55e' }}>LIVE</span>
          </div>

          {/* Last display refresh */}
          {lastRefresh && (
            <span className="text-xs" style={{ color: '#475569' }}>
              Updated {lastRefresh}
            </span>
          )}

          {/* Display refresh (reads live_state.json) */}
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1 px-3 py-1.5 rounded text-xs transition-all"
            style={{
              backgroundColor: '#1a2035',
              color: '#64748b',
              border: '1px solid #2d4a7a',
            }}
            title="Reload display from live_state.json"
          >
            <RefreshCw
              className={`w-3 h-3 ${isRefreshing ? 'animate-spin' : ''}`}
              style={{ color: '#60a5fa' }}
            />
            {isRefreshing ? 'Loading...' : 'Reload'}
          </button>

          {/* Data refresh (runs signal engine, calls /api/refresh) */}
          <button
            onClick={handleDataRefresh}
            disabled={isDataRefreshing}
            className="flex items-center gap-1 px-3 py-1.5 rounded text-xs font-semibold transition-all"
            style={{
              backgroundColor: isDataRefreshing ? 'rgba(245,158,11,0.15)' : 'rgba(245,158,11,0.1)',
              color: '#fbbf24',
              border: `1px solid ${isDataRefreshing ? 'rgba(245,158,11,0.5)' : 'rgba(245,158,11,0.25)'}`,
              opacity: isDataRefreshing ? 0.8 : 1,
            }}
            title="Re-run signal computation for all 230 symbols (takes ~2 min)"
          >
            <Database className={`w-3 h-3 ${isDataRefreshing ? 'animate-pulse' : ''}`} />
            {isDataRefreshing ? 'Running Engine...' : 'Refresh Data'}
          </button>
        </div>
      </div>
    </header>
  );
}
