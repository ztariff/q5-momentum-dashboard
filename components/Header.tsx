'use client';

import { Activity, TrendingUp, RefreshCw } from 'lucide-react';

interface HeaderProps {
  lastRefresh: string | null;
  isRefreshing: boolean;
  onRefresh: () => void;
}

export default function Header({ lastRefresh, isRefreshing, onRefresh }: HeaderProps) {
  const now = new Date();
  const hour = now.getUTCHours() - 5; // EST approximate
  const isMarketHours = hour >= 9 && hour < 16 && now.getDay() >= 1 && now.getDay() <= 5;

  return (
    <header className="border-b" style={{ borderColor: '#2d4a7a', backgroundColor: '#0d1425' }}>
      <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
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

        <div className="flex items-center gap-4">
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

          {/* Last refresh */}
          {lastRefresh && (
            <span className="text-xs" style={{ color: '#475569' }}>
              Updated {lastRefresh}
            </span>
          )}

          {/* Refresh button */}
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1 px-3 py-1.5 rounded text-xs transition-all"
            style={{
              backgroundColor: '#1a2035',
              color: '#64748b',
              border: '1px solid #2d4a7a',
            }}
          >
            <RefreshCw
              className={`w-3 h-3 ${isRefreshing ? 'animate-spin' : ''}`}
              style={{ color: '#60a5fa' }}
            />
            {isRefreshing ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>
    </header>
  );
}
