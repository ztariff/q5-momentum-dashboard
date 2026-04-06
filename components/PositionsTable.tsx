'use client';

import { useState } from 'react';
import { Position } from '@/lib/types';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface PositionsTableProps {
  positions: Position[];
  isLoading: boolean;
}

type SortKey = 'symbol' | 'unrealized_pnl' | 'net_pnl' | 'return_pct' | 'days_remaining' | 'entry_date' | 'position_size' | 'exit_date' | 'status';

function getRowClass(pos: Position): string {
  // Pending: signal fired, not yet entered
  if (pos.status === 'pending') return 'row-pending';

  // Exit today: hold period expires today
  if (pos.status === 'exit_today') return 'row-exit-today';

  // Active positions: standard color coding
  const unrealPct = pos.unrealized_pnl_pct ?? 0;
  const stopDist = pos.distance_to_stop_pct ?? 100;
  const daysRem = pos.days_remaining ?? 99;

  if (unrealPct < -50) return 'row-deep-loss';
  if (stopDist < 20) return 'row-near-stop';
  if (daysRem <= 2) return 'row-near-exit';
  if ((pos.unrealized_pnl ?? 0) > 0) return 'row-profitable';
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

/**
 * Compute recommended contracts for an option trade.
 * Rule: always buy at least 1 contract regardless of premium.
 * Formula: max(1, floor(budget / (optionPrice * 100)))
 */
function recommendedContracts(budget: number, optionPrice: number): number {
  if (!optionPrice || optionPrice <= 0) return 1;
  return Math.max(1, Math.floor(budget / (optionPrice * 100)));
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


function formatDate(dateStr: string): string {
  const d = new Date(dateStr + 'T00:00:00');
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const year = d.getFullYear();
  return `${month}/${day}/${year}`;
}

const MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/**
 * Parse a Polygon option ticker like "O:OXY200221C00047000"
 * Format: O:{SYMBOL}{YYMMDD}{C|P}{strike*1000 padded to 8 digits}
 * Output: "OXY Feb 21 2020 $47 Call"
 */
function formatOptionTicker(ticker: string): string {
  if (!ticker) return '';
  const raw = ticker.startsWith('O:') ? ticker.slice(2) : ticker;
  const match = raw.match(/^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/);
  if (!match) return ticker;
  const [, symbol, yy, mm, dd, cpFlag, strikePadded] = match;
  const year = 2000 + parseInt(yy, 10);
  const monthIdx = parseInt(mm, 10) - 1;
  const day = parseInt(dd, 10);
  const strike = parseInt(strikePadded, 10) / 1000;
  const monthName = MONTH_ABBR[monthIdx] ?? mm;
  const callPut = cpFlag === 'C' ? 'Call' : 'Put';
  const strikeStr = strike % 1 === 0 ? (strike || 0).toFixed(0) : (strike || 0).toFixed(2);
  return `${symbol} ${monthName} ${day} ${year} $${strikeStr} ${callPut}`;
}

/** Render the status badge cell for a position */
function StatusBadge({ pos }: { pos: Position }) {
  if (pos.status === 'pending') {
    return (
      <span className="badge badge-pending">
        PENDING
      </span>
    );
  }
  if (pos.status === 'exit_today') {
    return (
      <span className="badge badge-exit-today">
        EXIT AT CLOSE
      </span>
    );
  }
  return (
    <span className="badge badge-active">
      ACTIVE
    </span>
  );
}

export default function PositionsTable({ positions, isLoading }: PositionsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('status');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  // Status sort order: pending first, then exit_today, then active
  function statusOrder(pos: Position): number {
    if (pos.status === 'pending') return 0;
    if (pos.status === 'exit_today') return 1;
    return 2;
  }

  const sorted = [...positions].sort((a, b) => {
    // Always sort pending to top regardless of sort key
    const statusDiff = statusOrder(a) - statusOrder(b);
    if (sortKey === 'status') {
      return sortDir === 'asc' ? statusDiff : -statusDiff;
    }

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

    let cmp: number;
    if (typeof av === 'string') {
      cmp = sortDir === 'asc' ? av.localeCompare(bv as string) : (bv as string).localeCompare(av);
    } else {
      cmp = sortDir === 'asc' ? av - (bv as number) : (bv as number) - av;
    }

    // Secondary sort: keep pending before others within same primary sort value
    return cmp !== 0 ? cmp : statusDiff;
  });

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return <span className="ml-1 opacity-30">↕</span>;
    return sortDir === 'asc' ? <ChevronUp className="inline w-3 h-3 ml-0.5" /> : <ChevronDown className="inline w-3 h-3 ml-0.5" />;
  };

  const activePositions = positions.filter(p => p.status !== 'pending');
  const pendingPositions = positions.filter(p => p.status === 'pending');
  const totalPnl = activePositions.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0);
  const winners = activePositions.filter(p => (p.net_pnl ?? 0) > 0).length;
  const winRate = activePositions.length > 0 ? (winners / activePositions.length * 100).toFixed(0) : '0';

  return (
    <div>
      {/* Summary bar */}
      <div className="flex flex-wrap items-center gap-5 mb-3 px-1">
        <span className="text-xs" style={{ color: '#64748b' }}>
          {activePositions.length} active position{activePositions.length !== 1 ? 's' : ''}
        </span>
        {pendingPositions.length > 0 && (
          <span className="text-xs font-semibold" style={{ color: '#fbbf24' }}>
            {pendingPositions.length} pending (entry at next open)
          </span>
        )}
        <span className="text-xs" style={{ color: '#64748b' }}>
          Unrealized P&L:{' '}
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

      {/* Pending positions legend */}
      {pendingPositions.length > 0 && (
        <div
          className="flex items-center gap-2 rounded-lg px-3 py-2 mb-3 text-xs"
          style={{ backgroundColor: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', color: '#fbbf24' }}
        >
          <span className="font-semibold">PENDING</span>
          <span style={{ color: '#d97706' }}>—</span>
          <span style={{ color: '#92400e' }}>
            Signal fired today. These positions enter at tomorrow&apos;s open (9:30 AM). No P&amp;L yet.
          </span>
        </div>
      )}

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
                <th onClick={() => handleSort('status')}>Status <SortIcon col="status" /></th>
                <th onClick={() => handleSort('symbol')}>Symbol <SortIcon col="symbol" /></th>
                <th>Type</th>
                <th>Tier</th>
                <th onClick={() => handleSort('entry_date')}>Entry Date <SortIcon col="entry_date" /></th>
                <th>Entry Price</th>
                <th>Current Price</th>
                <th onClick={() => handleSort('unrealized_pnl')}>Unreal P&L <SortIcon col="unrealized_pnl" /></th>
                <th>Unreal %</th>
                <th onClick={() => handleSort('position_size')}>Size / Shares <SortIcon col="position_size" /></th>
                <th>Stop Price</th>
                <th>Dist to Stop</th>
                <th onClick={() => handleSort('days_remaining')}>Days Rem <SortIcon col="days_remaining" /></th>
                <th>Exit Date</th>
                <th title="Minimum 1 contract per option trade regardless of premium. Formula: max(1, floor(budget / (price × 100)))">Option / Contracts</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((pos, i) => {
                const isPending = pos.status === 'pending';
                const isExitToday = pos.status === 'exit_today';

                const pnlValue = pos.unrealized_pnl;
                const pnlPct = pos.unrealized_pnl_pct ?? 0;
                const daysRem = pos.days_remaining ?? 0;
                const daysHeld = pos.days_held ?? pos.hold_days ?? 0;
                const maxDays = pos.instrument === 'STOCK' ? 20 : 13;
                const progressPct = maxDays > 0 ? Math.min(100, (daysHeld / maxDays) * 100) : 0;

                return (
                  <tr key={`${pos.symbol}-${pos.entry_date}-${pos.status}-${i}`} className={getRowClass(pos)}>
                    {/* Status */}
                    <td>
                      <StatusBadge pos={pos} />
                    </td>

                    {/* Symbol */}
                    <td>
                      <span className="font-bold text-sm" style={{ color: '#f1f5f9' }}>{pos.symbol}</span>
                    </td>

                    {/* Type */}
                    <td>
                      <span className={getInstrumentBadge(pos.instrument)}>
                        {getInstrumentLabel(pos.instrument)}
                      </span>
                    </td>

                    {/* Tier */}
                    <td>
                      <span className={getTierBadge(pos.tier)}>{pos.tier || '?'}</span>
                    </td>

                    {/* Entry Date */}
                    <td style={{ color: '#94a3b8' }}>
                      {isPending ? (
                        <span style={{ color: '#fbbf24' }}>{formatDate(pos.entry_date)}</span>
                      ) : formatDate(pos.entry_date)}
                    </td>

                    {/* Entry Price */}
                    <td style={{ color: '#cbd5e1' }}>
                      {isPending ? (
                        <span style={{ color: '#d97706' }}>
                          Est. ${((pos.entry_price || 0).toFixed(2))}
                        </span>
                      ) : (
                        `$${((pos.entry_price || 0).toFixed(2))}`
                      )}
                    </td>

                    {/* Current Price */}
                    <td>
                      {isPending ? (
                        <span style={{ color: '#92400e' }} className="text-xs">—</span>
                      ) : pos.current_price ? (
                        <span style={{ color: ((pos.current_price || 0) >= (pos.entry_price || 0)) ? '#22c55e' : '#ef4444' }}>
                          ${((pos.current_price || 0).toFixed(2))}
                        </span>
                      ) : (
                        <span style={{ color: '#475569' }}>—</span>
                      )}
                    </td>

                    {/* Unrealized P&L */}
                    <td>
                      {isPending ? (
                        <span style={{ color: '#475569' }}>—</span>
                      ) : isExitToday ? (
                        <span className={`font-bold ${(pnlValue ?? 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}`}>
                          {formatPnl(pnlValue)}
                          <span className="text-xs ml-1" style={{ color: '#c084fc' }}>(final)</span>
                        </span>
                      ) : (
                        <span className={(pnlValue ?? 0) >= 0 ? 'pnl-positive font-bold' : 'pnl-negative font-bold'}>
                          {formatPnl(pnlValue)}
                        </span>
                      )}
                    </td>

                    {/* Unrealized % */}
                    <td>
                      {isPending ? (
                        <span style={{ color: '#475569' }}>—</span>
                      ) : (
                        <span className={pnlPct >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                          {pnlPct >= 0 ? '+' : ''}{(pnlPct || 0).toFixed(2)}%
                        </span>
                      )}
                    </td>

                    {/* Size */}
                    <td style={{ color: '#94a3b8' }}>
                      {pos.instrument?.toUpperCase().includes('STOCK')
                        ? `${Math.round(pos.shares_or_contracts || 0).toLocaleString()} shares`
                        : formatSize(pos.position_size)}
                    </td>

                    {/* Stop Price */}
                    <td style={{ color: '#f59e0b' }}>
                      {pos.stop_price ? `$${(pos.stop_price || 0).toFixed(2)}` : '—'}
                    </td>

                    {/* Distance to Stop */}
                    <td>
                      {pos.distance_to_stop_pct != null ? (
                        <span style={{ color: pos.distance_to_stop_pct < 20 ? '#f59e0b' : '#64748b' }}>
                          {(pos.distance_to_stop_pct ?? 0).toFixed(1)}%
                        </span>
                      ) : '—'}
                    </td>

                    {/* Days Remaining */}
                    <td>
                      {isPending ? (
                        <span style={{ color: '#fbbf24' }} className="text-xs font-semibold">
                          Entry at open
                        </span>
                      ) : isExitToday ? (
                        <span style={{ color: '#c084fc' }} className="text-xs font-semibold">
                          Exit at close
                        </span>
                      ) : (
                        <div className="flex items-center gap-2">
                          <span style={{ color: daysRem <= 2 ? '#a855f7' : '#94a3b8' }}>
                            {daysRem}d / {maxDays}d
                          </span>
                          <div className="w-16 rounded-full overflow-hidden" style={{ height: '4px', backgroundColor: '#1a2035' }}>
                            <div className="rounded-full h-full" style={{ width: `${progressPct}%`, backgroundColor: progressPct > 75 ? '#a855f7' : '#3b82f6' }} />
                          </div>
                        </div>
                      )}
                    </td>

                    {/* Exit Date */}
                    <td style={{ color: isExitToday ? '#c084fc' : '#64748b' }}>
                      {pos.scheduled_exit_date ? formatDate(pos.scheduled_exit_date) : pos.exit_date ? formatDate(pos.exit_date) : '—'}
                    </td>

                    {/* Option / Contracts */}
                    <td>
                      {pos.option_ticker ? (
                        <div>
                          <span className="text-xs" style={{ color: '#c084fc' }} title={pos.option_ticker}>
                            {formatOptionTicker(pos.option_ticker)}
                          </span>
                          {pos.entry_price > 0 && (
                            <div
                              className="text-xs mt-0.5"
                              style={{ color: '#a78bfa' }}
                              title="Minimum 1 contract per option trade regardless of premium"
                            >
                              {(() => {
                                const budget = pos.tier === 'A' ? 13000 : 7000;
                                const contracts = recommendedContracts(budget, pos.entry_price);
                                return `${contracts} contract${contracts !== 1 ? 's' : ''}`;
                              })()}
                            </div>
                          )}
                        </div>
                      ) : isPending && pos.instrument?.toUpperCase().includes('OPTION') ? (
                        <span className="text-xs" style={{ color: '#92400e' }}>
                          30-delta call, ~1mo expiry
                        </span>
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
// Build cache buster: 1775470967
