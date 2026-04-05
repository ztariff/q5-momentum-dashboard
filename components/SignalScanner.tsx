'use client';

import { useState, useEffect } from 'react';
import { Zap, Info, Clock } from 'lucide-react';

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

interface ScannerData {
  signals: Signal[];
  watchlist: unknown[];
  total_scanned: number;
  timestamp: string;
  market_date: string;
  last_refresh: string;
}

function formatSize(val: number): string {
  if (val >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
  if (val >= 1000) return `$${(val / 1000).toFixed(0)}K`;
  return `$${val}`;
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

export default function SignalScanner() {
  const [data, setData] = useState<ScannerData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSignals = async () => {
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
    loadSignals();
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

      {/* Architecture note */}
      <div
        className="flex items-start gap-2 rounded-lg p-2 mb-3 text-xs"
        style={{ backgroundColor: 'rgba(96,165,250,0.07)', border: '1px solid rgba(96,165,250,0.2)', color: '#94a3b8' }}
      >
        <Info className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: '#60a5fa' }} />
        <span>
          <span style={{ color: '#60a5fa' }} className="font-semibold">Pre-computed signals</span>{' '}
          — data is computed daily at 4:05 PM ET by the signal engine (230 symbols, Polygon.io).
          Use the <span style={{ color: '#fbbf24' }} className="font-semibold">Refresh Data</span> button in the header to re-run manually.
        </span>
      </div>

      {/* Note: signals are connected to Current Positions */}
      <div
        className="flex items-start gap-2 rounded-lg p-2 mb-3 text-xs"
        style={{ backgroundColor: 'rgba(245, 158, 11, 0.07)', border: '1px solid rgba(245, 158, 11, 0.25)', color: '#92400e' }}
      >
        <span style={{ color: '#fbbf24' }} className="font-semibold">Note:</span>
        <span style={{ color: '#d97706' }}>
          Signals detected here appear in <span style={{ color: '#fbbf24' }} className="font-semibold">Current Positions</span> as{' '}
          <span style={{ color: '#fbbf24' }} className="font-semibold">PENDING</span> entries until the next market open.
        </span>
      </div>

      {/* Sizing note */}
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
            <p className="text-sm">Loading signals from live_state.json...</p>
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
          {data.signals.length === 0 ? (
            <div className="text-center py-8 text-sm" style={{ color: '#475569' }}>
              No Q5 entry signals in current live_state.json.
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
                    <th>Body %</th>
                    <th>ATR Chg</th>
                    <th>Tier</th>
                    <th>Sizing Rule</th>
                    <th>Rec Size</th>
                    <th title="Minimum 1 contract per option trade regardless of premium. Formula: max(1, floor(budget / (price × 100)))">
                      Contracts
                    </th>
                    <th>Est. Entry</th>
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
                      <td style={{ color: '#94a3b8' }}>
                        {'estimated_entry' in sig
                          ? `$${(sig as Record<string, unknown>).estimated_entry}`
                          : '—'}
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
    </div>
  );
}
