'use client';

import { useState, useEffect } from 'react';
import { Zap, RefreshCw, Info } from 'lucide-react';

interface Signal {
  symbol: string;
  z_score: number;
  slope: number;
  quintile: number;
  body_pct: number;
  atr_change: number;
  tier: string;
  sizing_rule: string;
  recommended_size: number;
  recommended_contracts: number | null;
  is_new_signal: boolean;
}

interface WatchlistItem {
  symbol: string;
  quintile: number;
  z_score: number;
  slope: number;
  approaching: boolean;
}

interface ScannerData {
  signals: Signal[];
  watchlist: WatchlistItem[];
  total_scanned: number;
  timestamp: string;
}

function formatSize(val: number): string {
  if (val >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
  if (val >= 1000) return `$${(val / 1000).toFixed(0)}K`;
  return `$${val}`;
}

export default function SignalScanner() {
  const [data, setData] = useState<ScannerData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'signals' | 'watchlist'>('signals');
  const [error, setError] = useState<string | null>(null);

  const runScan = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const resp = await fetch('/api/signals');
      if (!resp.ok) throw new Error(`API error: ${resp.status}`);
      const json = await resp.json();
      if (json.error) throw new Error(json.error);
      setData(json);
    } catch (err) {
      setError(String(err));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    runScan();
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Zap className="w-4 h-4" style={{ color: '#fbbf24' }} />
          <span className="text-xs font-semibold tracking-wider uppercase" style={{ color: '#64748b' }}>
            Signal Scanner
          </span>
          {data && (
            <span className="text-xs" style={{ color: '#475569' }}>
              {data.total_scanned} symbols scanned
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {data?.timestamp && (
            <span className="text-xs" style={{ color: '#475569' }}>
              {new Date(data.timestamp).toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={runScan}
            disabled={isLoading}
            className="flex items-center gap-1 px-3 py-1.5 rounded text-xs"
            style={{ backgroundColor: '#1a2035', color: '#60a5fa', border: '1px solid #2d4a7a' }}
          >
            <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
            {isLoading ? 'Scanning...' : 'Run Scan'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4">
        {(['signals', 'watchlist'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className="px-4 py-1.5 rounded text-xs font-semibold transition-all"
            style={{
              backgroundColor: activeTab === tab ? '#1e3a5f' : '#1a2035',
              color: activeTab === tab ? '#60a5fa' : '#64748b',
              border: `1px solid ${activeTab === tab ? '#3b82f6' : '#2d4a7a'}`,
            }}
          >
            {tab === 'signals' ? (
              <>New Signals {data ? `(${data.signals.length})` : ''}</>
            ) : (
              <>Q4 Watchlist {data ? `(${data.watchlist.length})` : ''}</>
            )}
          </button>
        ))}
      </div>

      {/* Option contract sizing note */}
      <div
        className="flex items-start gap-2 rounded-lg p-2 mb-4 text-xs"
        style={{ backgroundColor: 'rgba(96, 165, 250, 0.07)', border: '1px solid rgba(96, 165, 250, 0.2)', color: '#94a3b8' }}
      >
        <Info className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: '#60a5fa' }} />
        <span>
          <span style={{ color: '#60a5fa' }} className="font-semibold">Sizing:</span>{' '}
          Tier A options = $13K budget &nbsp;|&nbsp; Tier B options = $7K budget &nbsp;|&nbsp; Tier C stocks scaled $33K–$133K by body quality.{' '}
          <span style={{ color: '#fbbf24' }} className="font-semibold">
            Minimum 1 contract per option trade regardless of premium.
          </span>{' '}
          Contracts = max(1, floor(budget / (price × 100))).
        </span>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12" style={{ color: '#475569' }}>
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-2 rounded-full mx-auto mb-3" style={{ borderColor: '#2d4a7a', borderTopColor: '#fbbf24' }} />
            <p className="text-sm">Scanning 230 symbols via Polygon.io...</p>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg p-4 text-sm" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#f87171' }}>
          Error: {error}
        </div>
      )}

      {!isLoading && !error && data && (
        <>
          {activeTab === 'signals' && (
            <div>
              {data.signals.length === 0 ? (
                <div className="text-center py-8 text-sm" style={{ color: '#475569' }}>
                  No new Q5 entry signals detected in scanned symbols
                </div>
              ) : (
                <div className="overflow-x-auto rounded-lg border" style={{ borderColor: '#2d4a7a' }}>
                  <table className="trading-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Quintile</th>
                        <th>Z-Score</th>
                        <th>SMA50 Slope</th>
                        <th>Body %</th>
                        <th>ATR Chg</th>
                        <th>Tier</th>
                        <th>Sizing Rule</th>
                        <th>Rec Size</th>
                        <th title="Minimum 1 contract per option trade regardless of premium. Formula: max(1, floor(budget / (price × 100)))">
                          Contracts
                        </th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.signals.map((sig, i) => (
                        <tr key={i}>
                          <td>
                            <span className="font-bold text-sm" style={{ color: '#f1f5f9' }}>{sig.symbol}</span>
                          </td>
                          <td>
                            <span
                              className="badge"
                              style={{
                                backgroundColor: 'rgba(34, 197, 94, 0.2)',
                                color: '#4ade80',
                                border: '1px solid rgba(34, 197, 94, 0.4)',
                              }}
                            >
                              Q{sig.quintile}
                            </span>
                          </td>
                          <td>
                            <span style={{ color: sig.z_score > 1.5 ? '#22c55e' : '#94a3b8' }}>
                              {sig.z_score.toFixed(3)}
                            </span>
                          </td>
                          <td>
                            <span style={{ color: sig.slope > 0 ? '#22c55e' : '#ef4444' }}>
                              {sig.slope >= 0 ? '+' : ''}{sig.slope.toFixed(3)}%
                            </span>
                          </td>
                          <td style={{ color: '#94a3b8' }}>
                            {sig.body_pct >= 0 ? '+' : ''}{(sig.body_pct * 100).toFixed(1)}%
                          </td>
                          <td>
                            <span style={{ color: sig.atr_change > 0 ? '#f59e0b' : '#64748b' }}>
                              {sig.atr_change >= 0 ? '+' : ''}{(sig.atr_change * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td>
                            <span className={`badge ${sig.tier === 'A' ? 'badge-tier-a' : sig.tier === 'B' ? 'badge-tier-b' : 'badge-tier-c'}`}>
                              {sig.tier}
                            </span>
                          </td>
                          <td style={{ color: '#94a3b8', fontSize: '0.7rem' }}>{sig.sizing_rule}</td>
                          <td className="font-bold" style={{ color: '#60a5fa' }}>
                            {formatSize(sig.recommended_size)}
                          </td>
                          <td>
                            {sig.recommended_contracts !== null ? (
                              <span
                                className="font-bold"
                                style={{ color: '#c084fc' }}
                                title="Minimum 1 contract per option trade regardless of premium"
                              >
                                {sig.recommended_contracts}x
                              </span>
                            ) : (
                              <span style={{ color: '#475569' }}>—</span>
                            )}
                          </td>
                          <td>
                            <span
                              className="badge"
                              style={{ backgroundColor: 'rgba(34, 197, 94, 0.15)', color: '#4ade80', border: '1px solid rgba(34, 197, 94, 0.3)' }}
                            >
                              NEW SIGNAL
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {activeTab === 'watchlist' && (
            <div>
              <p className="text-xs mb-3" style={{ color: '#64748b' }}>
                Q4 stocks with high z-scores — likely to enter Q5 within 1-2 days
              </p>
              {data.watchlist.length === 0 ? (
                <div className="text-center py-8 text-sm" style={{ color: '#475569' }}>
                  No Q4 stocks approaching Q5 in scanned symbols
                </div>
              ) : (
                <div className="overflow-x-auto rounded-lg border" style={{ borderColor: '#2d4a7a' }}>
                  <table className="trading-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Current Quintile</th>
                        <th>Z-Score</th>
                        <th>SMA50 Slope</th>
                        <th>Momentum</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.watchlist.slice(0, 25).map((item, i) => (
                        <tr key={i}>
                          <td>
                            <span className="font-bold text-sm" style={{ color: '#f1f5f9' }}>{item.symbol}</span>
                          </td>
                          <td>
                            <span
                              className="badge"
                              style={{
                                backgroundColor: 'rgba(245, 158, 11, 0.15)',
                                color: '#fbbf24',
                                border: '1px solid rgba(245, 158, 11, 0.3)',
                              }}
                            >
                              Q{item.quintile}
                            </span>
                          </td>
                          <td>
                            <span style={{ color: item.z_score > 1.5 ? '#22c55e' : '#94a3b8' }}>
                              {item.z_score.toFixed(3)}
                            </span>
                          </td>
                          <td>
                            <span style={{ color: item.slope > 0 ? '#22c55e' : '#ef4444' }}>
                              {item.slope >= 0 ? '+' : ''}{item.slope.toFixed(3)}%
                            </span>
                          </td>
                          <td>
                            {item.approaching ? (
                              <span
                                className="badge"
                                style={{ backgroundColor: 'rgba(234, 179, 8, 0.15)', color: '#eab308', border: '1px solid rgba(234, 179, 8, 0.3)' }}
                              >
                                APPROACHING Q5
                              </span>
                            ) : (
                              <span className="text-xs" style={{ color: '#475569' }}>Monitor</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
