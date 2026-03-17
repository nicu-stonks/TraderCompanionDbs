import React, { useState, useMemo } from 'react';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import type { CustomColumn, ColumnOrder, CustomColumnValue } from '@/TradeHistoryPage/types/CustomTradeData';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { format, parseISO } from 'date-fns';

interface TopPercentTradesTableProps {
  trades: Trade[];
  sortOrder?: 'return' | 'chronological';
  sortBy?: 'amount' | 'percent';
  mode?: 'winners' | 'losers';
  customColumns?: CustomColumn[];
  columnOrder?: ColumnOrder[];
  customColumnValues?: CustomColumnValue[];
  onPin?: (ids: Set<number>) => void;
}

type ColKey = 'month' | 'ticker' | 'status' | 'entryDate' | 'exitDate' | 'entryPrice' | 'exitPrice' | 'pattern' | 'return' | 'returnPct' | 'pctOfEquity' | 'marketCond' | 'category' | 'C' | 'A' | 'N' | 'S' | 'L' | 'I' | 'M' | (string & {});

export const TopPercentTradesTable: React.FC<TopPercentTradesTableProps> = ({ trades, sortOrder = 'return', sortBy = 'percent', mode = 'winners', customColumns = [], columnOrder = [], customColumnValues = [], onPin }) => {
  const [expanded, setExpanded] = useState(false);
  const [colSort, setColSort] = useState<{ key: ColKey | null; dir: 'asc' | 'desc' }>({ key: null, dir: 'asc' });
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [lastClickedIdx, setLastClickedIdx] = useState<number | null>(null);

  // Build ordered list of custom columns using ColumnOrder positions
  const orderedCustomColumns = useMemo(() => {
    if (!customColumns.length) return [];
    const orderMap = new Map(columnOrder.map(o => [o.column_key, o.position]));
    return [...customColumns].sort((a, b) => {
      const posA = orderMap.get(`custom_${a.id}`) ?? 9999;
      const posB = orderMap.get(`custom_${b.id}`) ?? 9999;
      return posA - posB;
    });
  }, [customColumns, columnOrder]);

  // Quick lookup: trade ID -> column ID -> value
  const customValueMap = useMemo(() => {
    const map = new Map<number, Map<number, string>>();
    for (const v of customColumnValues) {
      if (!map.has(v.trade_id)) map.set(v.trade_id, new Map());
      map.get(v.trade_id)!.set(v.column, v.value);
    }
    return map;
  }, [customColumnValues]);

  const handleColClick = (key: ColKey) => {
    setColSort(prev =>
      prev.key === key
        ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: 'asc' }
    );
  };

  const rows = useMemo(() => {
    const processed = [...trades]
      .filter(t => t.Status === 'Exited' && t.Exit_Price != null)
      .map(t => {
        const returnPct = t.Entry_Price > 0
          ? ((t.Exit_Price! - t.Entry_Price) / t.Entry_Price) * 100
          : 0;
        return { trade: t, returnPct };
      });

    if (sortOrder === 'return') {
      const valueOf = (x: { trade: Trade; returnPct: number }) =>
        sortBy === 'amount' ? (x.trade.Return ?? 0) : x.returnPct;
      // winners: highest first; losers: lowest first (worst loss at top)
      processed.sort((a, b) =>
        mode === 'winners' ? valueOf(b) - valueOf(a) : valueOf(a) - valueOf(b)
      );
    } else if (sortOrder === 'chronological') {
      processed.sort((a, b) => {
        if (a.trade.streakId != null && b.trade.streakId != null && a.trade.streakId !== b.trade.streakId) {
          return a.trade.streakId - b.trade.streakId;
        }
        return new Date(a.trade.Entry_Date).getTime() - new Date(b.trade.Entry_Date).getTime();
      });
    }

    if (colSort.key) {
      const dir = colSort.dir === 'asc' ? 1 : -1;
      const boolVal = (v: boolean | undefined | null) => (v ? 1 : 0);
      processed.sort((a, b) => {
        const t1 = a.trade, t2 = b.trade;
        switch (colSort.key) {
          case 'month':
          case 'entryDate': return dir * (new Date(t1.Entry_Date).getTime() - new Date(t2.Entry_Date).getTime());
          case 'exitDate':  return dir * ((t1.Exit_Date ? new Date(t1.Exit_Date).getTime() : 0) - (t2.Exit_Date ? new Date(t2.Exit_Date).getTime() : 0));
          case 'ticker':    return dir * t1.Ticker.localeCompare(t2.Ticker);
          case 'status':    return dir * (t1.Status ?? '').localeCompare(t2.Status ?? '');
          case 'entryPrice':return dir * (t1.Entry_Price - t2.Entry_Price);
          case 'exitPrice': return dir * ((t1.Exit_Price ?? 0) - (t2.Exit_Price ?? 0));
          case 'pattern':   return dir * (t1.Pattern ?? '').localeCompare(t2.Pattern ?? '');
          case 'return':    return dir * ((t1.Return ?? 0) - (t2.Return ?? 0));
          case 'returnPct': return dir * (a.returnPct - b.returnPct);
          case 'pctOfEquity':return dir * ((t1.Pct_Of_Equity ?? 0) - (t2.Pct_Of_Equity ?? 0));
          case 'marketCond':return dir * (t1.Market_Condition ?? '').localeCompare(t2.Market_Condition ?? '');
          case 'category':  return dir * (t1.Category ?? '').localeCompare(t2.Category ?? '');
          case 'C': return dir * (boolVal(t1.C) - boolVal(t2.C));
          case 'A': return dir * (boolVal(t1.A) - boolVal(t2.A));
          case 'N': return dir * (boolVal(t1.N) - boolVal(t2.N));
          case 'S': return dir * (boolVal(t1.S) - boolVal(t2.S));
          case 'L': return dir * (boolVal(t1.L) - boolVal(t2.L));
          case 'I': return dir * (boolVal(t1.I) - boolVal(t2.I));
          case 'M': return dir * (boolVal(t1.M) - boolVal(t2.M));
          default: {
            // custom column sort: key is `custom_<id>`
            const match = (colSort.key as string).match(/^custom_(\d+)$/);
            if (match) {
              const colId = parseInt(match[1]);
              const va = customValueMap.get(t1.ID)?.get(colId) ?? '';
              const vb = customValueMap.get(t2.ID)?.get(colId) ?? '';
              return dir * va.localeCompare(vb);
            }
            return 0;
          }
        }
      });
    }

    return processed;
  }, [trades, sortOrder, sortBy, mode, colSort, customValueMap]);

  const boolLabel = (v: boolean | undefined | null) =>
    v ? '✓' : '✗';

  const fmtDate = (d: string | null) => {
    if (!d) return '—';
    try { return format(parseISO(d), 'MMM dd, yyyy'); } catch { return d; }
  };

  const hasStreaks = useMemo(() => rows.some(r => r.trade.streakId != null), [rows]);

  const handleRowClick = (e: React.MouseEvent, idx: number, tradeId: number) => {
    if (e.shiftKey && lastClickedIdx !== null) {
      const start = Math.min(lastClickedIdx, idx);
      const end = Math.max(lastClickedIdx, idx);
      const rangeIds = rows.slice(start, end + 1).map(r => r.trade.ID);
      setSelectedIds(prev => { const next = new Set(prev); rangeIds.forEach(id => next.add(id)); return next; });
    } else {
      setSelectedIds(prev => { const next = new Set(prev); if (next.has(tradeId)) next.delete(tradeId); else next.add(tradeId); return next; });
      setLastClickedIdx(idx);
    }
  };

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setExpanded(prev => !prev)}
        className="flex items-center gap-1 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
      >
        {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        {expanded ? 'Hide' : 'Show'} matched trades ({rows.length})
      </button>

      {expanded && selectedIds.size > 0 && (
        <div className="mt-2 flex items-center gap-2">
          <button
            type="button"
            onClick={() => { onPin?.(new Set(selectedIds)); setSelectedIds(new Set()); setLastClickedIdx(null); }}
            className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground hover:bg-primary/90 font-medium"
          >
            Use selected {selectedIds.size} trade{selectedIds.size !== 1 ? 's' : ''} in statistics
          </button>
          <button
            type="button"
            onClick={() => { setSelectedIds(new Set()); setLastClickedIdx(null); }}
            className="text-xs text-muted-foreground hover:text-foreground underline"
          >
            Clear
          </button>
        </div>
      )}

      {expanded && (
        <>
          <div className="mt-2 text-xs text-muted-foreground">
            Tip: Shift+Click another row to select a full range.
          </div>
          <div className="mt-2 max-h-[400px] overflow-auto border border-border rounded-md">
          <table className="w-full text-xs">
            <thead className="bg-muted/50 sticky top-0">
              <tr>
                {hasStreaks && <th className="px-2 py-1.5 text-left font-medium">Streak</th>}
                {(['month', 'ticker', 'status', 'entryDate', 'exitDate'] as ColKey[]).map((key) => {
                  const labels: Record<string, string> = { month: 'Month', ticker: 'Ticker', status: 'Status', entryDate: 'Entry Date', exitDate: 'Exit Date' };
                  return (
                    <th key={key} onClick={() => handleColClick(key)} className="px-2 py-1.5 text-left font-medium cursor-pointer select-none hover:bg-muted/80 whitespace-nowrap">
                      {labels[key]}{colSort.key === key ? (colSort.dir === 'asc' ? ' ▲' : ' ▼') : ''}
                    </th>
                  );
                })}
                {(['entryPrice', 'exitPrice'] as ColKey[]).map(key => {
                  const labels: Record<string, string> = { entryPrice: 'Entry Price', exitPrice: 'Exit Price' };
                  return (
                    <th key={key} onClick={() => handleColClick(key)} className="px-2 py-1.5 text-right font-medium cursor-pointer select-none hover:bg-muted/80 whitespace-nowrap">
                      {labels[key]}{colSort.key === key ? (colSort.dir === 'asc' ? ' ▲' : ' ▼') : ''}
                    </th>
                  );
                })}
                <th onClick={() => handleColClick('pattern')} className="px-2 py-1.5 text-left font-medium cursor-pointer select-none hover:bg-muted/80 whitespace-nowrap">
                  Pattern{colSort.key === 'pattern' ? (colSort.dir === 'asc' ? ' ▲' : ' ▼') : ''}
                </th>
                {(['return', 'returnPct', 'pctOfEquity'] as ColKey[]).map(key => {
                  const labels: Record<string, string> = { return: 'Return', returnPct: 'Return %', pctOfEquity: 'Pct Of Equity' };
                  return (
                    <th key={key} onClick={() => handleColClick(key)} className="px-2 py-1.5 text-right font-medium cursor-pointer select-none hover:bg-muted/80 whitespace-nowrap">
                      {labels[key]}{colSort.key === key ? (colSort.dir === 'asc' ? ' ▲' : ' ▼') : ''}
                    </th>
                  );
                })}
                {(['marketCond', 'category'] as ColKey[]).map(key => {
                  const labels: Record<string, string> = { marketCond: 'Market Cond.', category: 'Category' };
                  return (
                    <th key={key} onClick={() => handleColClick(key)} className="px-2 py-1.5 text-left font-medium cursor-pointer select-none hover:bg-muted/80 whitespace-nowrap">
                      {labels[key]}{colSort.key === key ? (colSort.dir === 'asc' ? ' ▲' : ' ▼') : ''}
                    </th>
                  );
                })}
                {(['C', 'A', 'N', 'S', 'L', 'I', 'M'] as ColKey[]).map(key => (
                  <th key={key} onClick={() => handleColClick(key)} className="px-2 py-1.5 text-center font-medium cursor-pointer select-none hover:bg-muted/80">
                    {key}{colSort.key === key ? (colSort.dir === 'asc' ? '▲' : '▼') : ''}
                  </th>
                ))}
                {orderedCustomColumns.map(col => {
                  const ckey = `custom_${col.id}` as ColKey;
                  return (
                    <th key={ckey} onClick={() => handleColClick(ckey)} className="px-2 py-1.5 text-left font-medium cursor-pointer select-none hover:bg-muted/80 whitespace-nowrap">
                      {col.name}{colSort.key === ckey ? (colSort.dir === 'asc' ? ' ▲' : ' ▼') : ''}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map(({ trade: t, returnPct }) => (
                <tr
                  key={t.ID}
                  onClick={e => handleRowClick(e, rows.findIndex(r => r.trade.ID === t.ID), t.ID)}
                  className={`transition-colors select-none ${
                    selectedIds.has(t.ID) ? 'bg-primary/15 hover:bg-primary/20 cursor-pointer' : 'hover:bg-muted/30 cursor-pointer'
                  }`}
                >
                  {hasStreaks && (
                    <td className="px-2 py-1.5 font-medium whitespace-nowrap">
                      {t.streakId != null ? `Streak #${t.streakId}` : '—'}
                    </td>
                  )}
                  <td className="px-2 py-1.5">{fmtDate(t.Entry_Date).slice(0, 8)}</td>
                  <td className="px-2 py-1.5 font-medium">{t.Ticker}</td>
                  <td className="px-2 py-1.5">{t.Status}</td>
                  <td className="px-2 py-1.5">{fmtDate(t.Entry_Date)}</td>
                  <td className="px-2 py-1.5">{fmtDate(t.Exit_Date)}</td>
                  <td className="px-2 py-1.5 text-right">${t.Entry_Price.toFixed(2)}</td>
                  <td className="px-2 py-1.5 text-right">{t.Exit_Price != null ? `$${t.Exit_Price.toFixed(2)}` : '—'}</td>
                  <td className="px-2 py-1.5">{t.Pattern || '—'}</td>
                  <td className={`px-2 py-1.5 text-right ${(t.Return ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {t.Return != null ? `$${t.Return.toFixed(2)}` : '—'}
                  </td>
                  <td className={`px-2 py-1.5 text-right font-medium ${returnPct >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}%
                  </td>
                  <td className="px-2 py-1.5 text-right">
                    {t.Pct_Of_Equity != null ? `${(t.Pct_Of_Equity * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="px-2 py-1.5">{t.Market_Condition || '—'}</td>
                  <td className="px-2 py-1.5">{t.Category || '—'}</td>
                  <td className="px-2 py-1.5 text-center">{boolLabel(t.C)}</td>
                  <td className="px-2 py-1.5 text-center">{boolLabel(t.A)}</td>
                  <td className="px-2 py-1.5 text-center">{boolLabel(t.N)}</td>
                  <td className="px-2 py-1.5 text-center">{boolLabel(t.S)}</td>
                  <td className="px-2 py-1.5 text-center">{boolLabel(t.L)}</td>
                  <td className="px-2 py-1.5 text-center">{boolLabel(t.I)}</td>
                  <td className="px-2 py-1.5 text-center">{boolLabel(t.M)}</td>
                  {orderedCustomColumns.map(col => (
                    <td key={`custom_${col.id}`} className="px-2 py-1.5">
                      {customValueMap.get(t.ID)?.get(col.id) ?? '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </>
      )}
    </div>
  );
};
