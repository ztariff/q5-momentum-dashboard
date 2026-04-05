'use client';

import { useState } from 'react';
import { Position } from '@/lib/types';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface PositionsTableProps {
  positions: Position[];
  isLoading: boolean;
  isDemoMode?: boolean;
}

type SortKey = 'symbol' | 'unrealized_pnl' | 'net_pnl' | 'return_pct' | 'days_remaining' | 'entry_date' | 'position_size' | 'exit_date';

function getRowClass(pos: Position, isDemo: boolean): string {
  if (isDemo) {
    if (pos.net_pnl > 0) return 'row-profitable';
    return 'row-losing';
  }
  const unrealPnl = pos.unrealized_pnl ?? 0;
  const stopDist = pos.distance_to_stop_pct ?? 100;
  const daysRem = pos.days_remaining ?? 99;
  const unrealPct = pos.unrealized_pnl_pct ?? 0;
  if (unrealPct < -50) return 'row-deep-loss';
  if (stopDist < 20) return 'row-near-stop';
  if (daysRem <= 2) return 'row-near-exit';
  if (unrealPnl > 0) return 'row-profitable';
  return 'row-losing';
}

function formatPnl(val: number | undefined): string {
  if (val === undefined || val === 0) return '$0';
  const abs = Math.abs(val);
  const sign = val >= 0 ? '+' : '-';
  if (abs >= 1000000) return `${sign}$${(abs / 1000000).toFixed(2)}M`;
  if (abs >= 1000) return `${sign}$${(abs / 1000).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

function formatSize(val: number): string {
  if (val >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
  if (val >= 1000) return `$${(val / 1000).toFixed(0)}K`;
  return `$${val}`;
}

function getInstrumentBadge(instrument: string): string {
  const upper = (instrument || '').toUpperCase();
  if (upper.includes('OPTION')) return 'badge badge-option';
  return 'badge badge-stock';
}

function getInstrumentLabel(instrument: string): string {
  const upper = (instrument || '').toUpperCase();
  if (upper.includes('OPTION')) return 'OPTION';
  if (upper.includes('FALLBACK')) return 'STOCK*';
  return 'STOCK';
}

function getTierBadge(tier: string): string {
  if (tier === 'A') return 'badge badge-tier-a';
  if (tier === 'B') return 'badge badge-tier-b';
  return 'badge badge-tier-c';
}

export default function PositionsTable({ positions, isLoading, isDemoMode = false }: PositionsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>(isDemoMode ? 'exit_date' : 'days_remaining');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>(isDemoMode ? 'desc' : 'asc');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sorted = [...positions].sort((a, b) => {
    let av: number | string = 0;
    let bv: number | string = 0;

    switch (sortKey) {
      case 'symbol': av = a.symbol; bv = b.symbol; break;
      case 'unrealized_pnl': av = a.unrealized_pnl ?? -999999; bv = b.unrealized_pnl ?? -999999; break;
      case 'net_pnl': av = a.net_pnl ?? -999999; bv = b.net_pnl ?? -999999; break;
      case 'return_pct': av = a.return_pct ?? -999; bv = b.return_pct ?? -999; break;
      case 'days_remaining': av = a.days_remaining ?? 99; bv = b.days_remaining ?? 99; break;
      case 'entry_date': av = a.entry_date; bv = b.entry_date; break;
      case 'exit_date': av = a.exit_date; bv = b.exit_date; break;
      case 'position_size': av = a.position_size; bv = b.position_size; break;
    }

    if (typeof av === 'string') {
      return sortDir === 'asc' ? av.localeCompare(bv as string) : (bv as string).localeCompare(av);
    }
    return sortDir === 'asc' ? av - (bv as number) : (bv as number) - av;
  });

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return <span className="ml-1 opacity-30">↕</span>;
    return sortDir === 'asc' ? <ChevronUp className="inline w-3 h-3 ml-0.5" /> : <ChevronDown className="inline w-3 h-3 ml-0.5" />;
  };

  const totalPnl = isDemoMode
    ? positions.reduce((s, p) => s + (p.net_pnl ?? 0), 0)
    : positions.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);

  const winners = positions.filter(p => (isDemoMode ? p.net_pnl : (p.unrealized_pnl ?? 0)) > 0).length;
  const winRate = positions.length > 0 ? (winners / positions.length * 100).toFixed(0) : '0';

  return (
    <div>
      {/* Summary bar */}
      <div className="flex flex-wrap items-center gap-5 mb-3 px-1">
        <span className="text-xs" style={{ color: '#64748b' }}>
          {positions.length} position{positions.length !== 1 ? 's' : ''}
        </span>
        <span className="text-xs" style={{ color: '#64748b' }}>
          {isDemoMode ? 'Realized P&L' : 'Unrealized P&L'}:{' '}
          <span className={totalPnl >= 0 ? 'pnl-positive font-bold' : 'pnl-negative font-bold'}>
            {formatPnl(totalPnl)}
          </span>
        </span>
        <span className="text-xs" style={{ color: '#64748b' }}>
          Win rate: <span style={{ color: '#94a3b8' }}>{winRate}%</span>
        </span>
        <span className="text-xs" style={{ color: '#475569' }}>
          Click column headers to sort
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border" style={{ borderColor: '#2d4a7a' }}>
        {isLoading ? (
          <div className="flex items-center justify-center py-16" style={{ color: '#475569' }}>
            <div className="text-center">
              <div className="animate-spin w-8 h-8 border-2 rounded-full mx-auto mb-3" style={{ borderColor: '#2d4a7a', borderTopColor: '#60a5fa' }} />
              <p className="text-sm">Loading positions...</p>
            </div>
          </div>
        ) : positions.length === 0 ? (
          <div className="flex items-center justify-center py-16" style={{ color: '#475569' }}>
            <p className="text-sm">No positions to display</p>
          </div>
        ) : (
          <table className="trading-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('symbol')}>Symbol <SortIcon col="symbol" /></th>
                <th>Type</th>
                <th>Tier</th>
                <th onClick={() => handleSort('entry_date')}>Entry Date <SortIcon col="entry_date" /></th>
                <th>Entry Price</th>
                <th>Current Price</th>
                {isDemoMode ? (
                  <>
                    <th onClick={() => handleSort('net_pnl')}>Net P&L <SortIcon col="net_pnl" /></th>
                    <th onClick={() => handleSort('return_pct')}>Return % <SortIcon col="return_pct" /></th>
                  </>
                ) : (
                  <>
                    <th onClick={() => handleSort('unrealized_pnl')}>Unreal P&L <SortIcon col="unrealized_pnl" /></th>
                    <th>Unreal %</th>
                  </>
                )}
                <th onClick={() => handleSort('position_size')}>Size <SortIcon col="position_size" /></th>
                {isDemoMode ? (
                  <>
                    <th onClick={() => handleSort('exit_date')}>Exit Date <SortIcon col="exit_date" /></th>
                    <th>Exit Type</th>
                    <th>Hold Days</th>
                  </>
                ) : (
                  <>
                    <th>Stop Price</th>
                    <th>Dist to Stop</th>
                    <th onClick={() => handleSort('days_remaining')}>Days Rem <SortIcon col="days_remaining" /></th>
                    <th>Exit Date</th>
                  </>
                )}
                <th>Option Ticker</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((pos, i) => {
                const pnlValue = isDemoMode ? pos.net_pnl : (pos.unrealized_pnl ?? 0);
                const pnlPct = isDemoMode ? pos.return_pct : (pos.unrealized_pnl_pct ?? 0);
                const daysRem = pos.days_remaining ?? 0;
                const daysHeld = pos.days_held ?? pos.hold_days ?? 0;
                const maxDays = pos.max_hold_days;
                const progressPct = maxDays > 0 ? Math.min(100, (daysHeld / maxDays) * 100) : 0;

                return (
                  <tr key={`${pos.symbol}-${pos.entry_date}-${i}`} className={getRowClass(pos, isDemoMode)}>
                    <td>
                      <span className="font-bold text-sm" style={{ color: '#f1f5f9' }}>{pos.symbol}</span>
                    </td>
                    <td>
                      <span className={getInstrumentBadge(pos.instrument)}>
                        {getInstrumentLabel(pos.instrument)}
                      </span>
                    </td>
                    <td>
                      <span className={getTierBadge(pos.tier)}>{pos.tier || '?'}</span>
                    </td>
                    <td style={{ color: '#94a3b8' }}>{pos.entry_date}</td>
                    <td style={{ color: '#cbd5e1' }}>${pos.entry_price.toFixed(2)}</td>
                    <td>
                      {(pos.current_price || pos.exit_price) ? (
                        <span style={{ color: (pos.current_price || pos.exit_price)! >= pos.entry_price ? '#22c55e' : '#ef4444' }}>
                          ${(pos.current_price || pos.exit_price)!.toFixed(2)}
                        </span>
                      ) : (
                        <span style={{ color: '#475569' }}>—</span>
                      )}
                    </td>
                    <td className={pnlValue >= 0 ? 'pnl-positive font-bold' : 'pnl-negative font-bold'}>
                      {formatPnl(pnlValue)}
                    </td>
                    <td className={pnlPct >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                      {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
                    </td>
                    <td style={{ color: '#94a3b8' }}>{formatSize(pos.position_size)}</td>
                    {isDemoMode ? (
                      <>
                        <td style={{ color: '#64748b' }}>{pos.exit_date}</td>
                        <td>
                          <span className="badge" style={{
                            backgroundColor: (pos.exit_type || '').includes('STOP') ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.15)',
                            color: (pos.exit_type || '').includes('STOP') ? '#f87171' : '#4ade80',
                            border: `1px solid ${(pos.exit_type || '').includes('STOP') ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.3)'}`,
                          }}>
                            {(pos.exit_type || '').includes('STOP') ? 'STOP' : 'TIME'}
                          </span>
                        </td>
                        <td>
                          <div className="flex items-center gap-2">
                            <span style={{ color: '#94a3b8' }}>{daysHeld}d / {maxDays}d</span>
                            <div className="w-12 rounded-full overflow-hidden" style={{ height: '4px', backgroundColor: '#1a2035' }}>
                              <div className="rounded-full h-full" style={{ width: `${progressPct}%`, backgroundColor: '#3b82f6' }} />
                            </div>
                          </div>
                        </td>
                      </>
                    ) : (
                      <>
                        <td style={{ color: '#f59e0b' }}>
                          {pos.stop_price ? `$${pos.stop_price.toFixed(2)}` : '—'}
                        </td>
                        <td>
                          {pos.distance_to_stop_pct !== undefined ? (
                            <span style={{ color: pos.distance_to_stop_pct < 20 ? '#f59e0b' : '#64748b' }}>
                              {pos.distance_to_stop_pct.toFixed(1)}%
                            </span>
                          ) : '—'}
                        </td>
                        <td>
                          <div className="flex items-center gap-2">
                            <span style={{ color: daysRem <= 2 ? '#a855f7' : '#94a3b8' }}>
                              {daysRem}d / {maxDays}d
                            </span>
                            <div className="w-16 rounded-full overflow-hidden" style={{ height: '4px', backgroundColor: '#1a2035' }}>
                              <div className="rounded-full h-full" style={{ width: `${progressPct}%`, backgroundColor: progressPct > 75 ? '#a855f7' : '#3b82f6' }} />
                            </div>
                          </div>
                        </td>
                        <td style={{ color: '#64748b' }}>{pos.scheduled_exit_date || pos.exit_date}</td>
                      </>
                    )}
                    <td>
                      {pos.option_ticker ? (
                        <span className="text-xs font-mono" style={{ color: '#c084fc' }}>{pos.option_ticker}</span>
                      ) : (
                        <span style={{ color: '#374151' }}>—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
