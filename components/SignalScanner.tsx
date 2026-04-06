'use client';

import { useState, useEffect } from 'react';
import { Eye, Info, Clock } from 'lucide-react';

interface WatchlistItem {
  symbol: string;
  quintile: number;
  z_score: number;
  slope: number;
  sma50_slope: number;
  body_pct: number;
  distance_to_q5: string;
  approaching: boolean;
}

interface WatchlistData {
  watchlist: WatchlistItem[];
  total_scanned: number;
  last_refresh: string;
  market_date: string;
}

function formatRefreshTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString('en-US', {
      month: 'short', day: 'numeric',
      hour: 'numeric', minute: '2-digit',
      timeZoneName: 'short',
    });
  } catch {
    return iso;
  }
}

function safeFixed(val: unknown, decimals: number): string {
  if (val === undefined || val === null || isNaN(Number(val))) return '—';
  return Number(val).toFixed(decimals);
}

function safePct(val: unknown, decimals: number): string {
  if (val === undefined || val === null || isNaN(Number(val))) return '—';
  const n = Number(val);
  return `${n >= 0 ? '+' : ''}${(n * 100).toFixed(decimals)}%`;
}

export default function Watchlist() {
  const [data, setData] = useState<WatchlistData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadWatchlist = async () => {
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
    loadWatchlist();
  }, []);

  return (
    <div>
      {/* Header row */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Eye className="w-4 h-4" style={{ color: '#fbbf24' }} />
          <span className="text-xs font-semibold tracking-wider uppercase" style={{ color: '#64748b' }}>
            Q4 Watchlist
          </span>
          {data && (
            <span className="text-xs" style={{ color: '#475569' }}>
              {data.watchlist.length} symbol{data.watchlist.length !== 1 ? 's' : ''} in Q4
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {data?.last_refresh && (
            <div className="flex items-center gap-1">
              <Clock className="w-3 h-3" style={{ color: '#475569' }} />
              <span className="text-xs" style={{ color: '#475569' }}>
                Last computed: {formatRefreshTime(data.last_refresh)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Context banner */}
      <div
        className="flex items-start gap-2 rounded-lg p-3 mb-3 text-xs"
        style={{ backgroundColor: 'rgba(245, 158, 11, 0.07)', border: '1px solid rgba(245, 158, 11, 0.2)', color: '#92400e' }}
      >
        <Eye className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: '#fbbf24' }} />
        <span style={{ color: '#d97706' }}>
          These symbols are in <span style={{ color: '#fbbf24' }} className="font-semibold">Q4</span> — one quintile away from a Q5 entry signal.
          Monitor for potential entries.{' '}
          <span style={{ color: '#92400e' }}>No action required today.</span>
        </span>
      </div>

      {/* Info note */}
      <div
        className="flex items-start gap-2 rounded-lg p-2 mb-4 text-xs"
        style={{ backgroundColor: 'rgba(96,165,250,0.07)', border: '1px solid rgba(96,165,250,0.2)', color: '#94a3b8' }}
      >
        <Info className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: '#60a5fa' }} />
        <span>
          <span style={{ color: '#60a5fa' }} className="font-semibold">When a symbol reaches Q5</span>{' '}
          it moves to <span style={{ color: '#fbbf24' }} className="font-semibold">Current Positions</span> as a{' '}
          <span style={{ color: '#fbbf24' }} className="font-semibold">PENDING</span> entry.
          This list is computed daily at 4:05 PM ET.
        </span>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12" style={{ color: '#475569' }}>
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-2 rounded-full mx-auto mb-3" style={{ borderColor: '#2d4a7a', borderTopColor: '#fbbf24' }} />
            <p className="text-sm">Loading watchlist...</p>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg p-4 text-sm" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#f87171' }}>
          Error: {error}
        </div>
      )}

      {!isLoading && !error && data && (
        <div>
          {data.watchlist.length === 0 ? (
            <div className="text-center py-8 text-sm" style={{ color: '#475569' }}>
              No Q4 symbols in current watchlist.
              <br />
              <span className="text-xs mt-1 block" style={{ color: '#374151' }}>
                Click &quot;Refresh Data&quot; in the header to run the signal engine.
              </span>
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
                    <th>Proximity</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.watchlist.map((item, i) => (
                    <tr key={i}>
                      {/* Symbol */}
                      <td>
                        <span className="font-bold text-sm" style={{ color: '#f1f5f9' }}>{item.symbol ?? '—'}</span>
                      </td>

                      {/* Quintile */}
                      <td>
                        <span
                          className="badge"
                          style={{
                            backgroundColor: 'rgba(245, 158, 11, 0.15)',
                            color: '#fbbf24',
                            border: '1px solid rgba(245, 158, 11, 0.35)',
                          }}
                        >
                          Q{item.quintile ?? '?'}
                        </span>
                      </td>

                      {/* Z-Score */}
                      <td>
                        <span style={{ color: (item.z_score ?? 0) > 0.5 ? '#fbbf24' : '#94a3b8' }}>
                          {safeFixed(item.z_score, 3)}
                        </span>
                      </td>

                      {/* SMA50 Slope */}
                      <td style={{ color: '#94a3b8' }}>
                        {safeFixed(item.sma50_slope, 4)}
                      </td>

                      {/* Body % */}
                      <td style={{ color: '#94a3b8' }}>
                        {safePct(item.body_pct, 1)}
                      </td>

                      {/* Proximity to Q5 */}
                      <td>
                        <span
                          style={{
                            color: item.distance_to_q5 === 'close' ? '#fbbf24' : '#64748b',
                            fontSize: '0.7rem',
                          }}
                        >
                          {item.distance_to_q5 === 'close' ? 'Close to Q5' : item.distance_to_q5 === 'moderate' ? 'Moderate' : (item.distance_to_q5 ?? '—')}
                        </span>
                      </td>

                      {/* Status */}
                      <td>
                        <span
                          className="badge"
                          style={{
                            backgroundColor: 'rgba(245, 158, 11, 0.12)',
                            color: '#fbbf24',
                            border: '1px solid rgba(245, 158, 11, 0.3)',
                          }}
                        >
                          WATCHING
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
    </div>
  );
}
