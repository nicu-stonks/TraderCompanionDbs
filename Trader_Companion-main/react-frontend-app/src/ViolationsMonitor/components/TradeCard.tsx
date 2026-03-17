import { useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, X, RefreshCw, LineChart } from 'lucide-react';
import type { MonAlertTrade, TradeViolationsResult, ViolationItem } from '../types';
import { updateTrade, deleteTrade } from '../api';
import { RiskAnalysisModal } from './RiskAnalysisModal';
import { TradeDailyChartModal } from './TradeDailyChartModal';

interface Props {
  trade: MonAlertTrade;
  result: TradeViolationsResult | null;
  loading: boolean;
  onTradeUpdated: () => void;
  onRefresh: () => void;
  showAllGroupedDates: boolean;
  tickerColRef?: (el: HTMLDivElement | null) => void;
}

export function TradeCard({ trade, result, loading, onTradeUpdated, onRefresh, showAllGroupedDates, tickerColRef }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [editingStartDate, setEditingStartDate] = useState(false);
  const [editingEndDate, setEditingEndDate] = useState(false);
  const [useLatest, setUseLatest] = useState(trade.use_latest_end_date);
  const [startDateLocal, setStartDateLocal] = useState(trade.start_date);
  const [endDateLocal, setEndDateLocal] = useState(trade.end_date || '');
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [alsoRemoveFromTickerMonitoring, setAlsoRemoveFromTickerMonitoring] = useState(true);
  const [isDeletingTrade, setIsDeletingTrade] = useState(false);
  const [showRiskModal, setShowRiskModal] = useState(false);
  const [showDailyChartModal, setShowDailyChartModal] = useState(false);
  const updateQueueRef = useRef<Promise<void>>(Promise.resolve());
  const removeConfirmBtnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    setUseLatest(trade.use_latest_end_date);
    setStartDateLocal(trade.start_date);
    setEndDateLocal(trade.end_date || '');
  }, [trade.use_latest_end_date, trade.start_date, trade.end_date]);

  useEffect(() => {
    if (showDeleteModal && !isDeletingTrade) {
      removeConfirmBtnRef.current?.focus();
    }
  }, [showDeleteModal, isDeletingTrade]);

  const queueTradeUpdate = (payload: Partial<MonAlertTrade>) => {
    updateQueueRef.current = updateQueueRef.current
      .then(() => updateTrade(trade.id, payload))
      .then(() => {
        onTradeUpdated();
        onRefresh();
      })
      .catch((e) => {
        console.error(e);
      });
  };

  const shiftDate = (field: 'start_date' | 'end_date', days: number) => {
    const currentDate = field === 'start_date' ? startDateLocal : endDateLocal;
    if (!currentDate) return;
    const d = new Date(currentDate);
    d.setDate(d.getDate() + days);
    const newDate = d.toISOString().split('T')[0];

    if (field === 'start_date') {
      setStartDateLocal(newDate);
    } else {
      setEndDateLocal(newDate);
    }
    queueTradeUpdate({ [field]: newDate });
  };

  const handleDateChange = async (field: 'start_date' | 'end_date', value: string) => {
    if (field === 'start_date') {
      setStartDateLocal(value);
    } else {
      setEndDateLocal(value);
    }
    queueTradeUpdate({ [field]: value });
    setEditingStartDate(false);
    setEditingEndDate(false);
  };

  const handleToggleLatest = async () => {
    const newVal = !useLatest;
    setUseLatest(newVal);
    const nextEndDate = newVal ? null : (endDateLocal || new Date().toISOString().split('T')[0]);
    if (!newVal && !endDateLocal) {
      setEndDateLocal(nextEndDate || '');
    }
    try {
      await updateTrade(trade.id, {
        use_latest_end_date: newVal,
        end_date: nextEndDate,
      });
      onTradeUpdated();
    } catch (e) {
      console.error(e);
      setUseLatest(!newVal);
    }
  };

  const handleDelete = async () => {
    setAlsoRemoveFromTickerMonitoring(true);
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = async () => {
    setIsDeletingTrade(true);
    try {
      await deleteTrade(trade.id);
      if (alsoRemoveFromTickerMonitoring) {
        try {
          await fetch(`http://localhost:8000/ticker_data/api/ticker_data/tickers/${trade.ticker}`, { method: 'DELETE' });
        } catch (e) {
          console.warn('Could not remove ticker from ticker monitoring:', e);
        }
      }
      setShowDeleteModal(false);
      onTradeUpdated();
    } catch (e) {
      console.error(e);
    } finally {
      setIsDeletingTrade(false);
    }
  };

  const totalV = result?.total_violations ?? 0;
  const totalC = result?.total_confirmations ?? 0;
  const violationItems = result?.violations ?? [];
  const confirmationItems = result?.confirmations ?? [];
  const weeklyConfirmationCount = confirmationItems.filter((item) => item.type.toLowerCase().startsWith('weekly_')).length;
  const dailyConfirmationCount = totalC - weeklyConfirmationCount;
  const weeklyViolationCount = violationItems.filter((item) => item.type.toLowerCase().startsWith('weekly_')).length;
  const dailyViolationCount = totalV - weeklyViolationCount;

  const groupByType = (items: ViolationItem[]) => {
    const grouped: Record<string, ViolationItem[]> = {};
    items.forEach((item) => {
      if (!grouped[item.type]) grouped[item.type] = [];
      grouped[item.type].push(item);
    });

    const dateValue = (value?: string) => {
      if (!value) return Number.MAX_SAFE_INTEGER;
      const ts = Date.parse(value);
      return Number.isNaN(ts) ? Number.MAX_SAFE_INTEGER : ts;
    };

    return Object.entries(grouped)
      .map(([type, groupItems]) => {
        const sortedItems = [...groupItems].sort((a, b) => dateValue(a.date) - dateValue(b.date));
        return { type, count: sortedItems.length, items: sortedItems };
      })
      .sort((a, b) => {
        const aFirst = dateValue(a.items[0]?.date);
        const bFirst = dateValue(b.items[0]?.date);
        if (aFirst !== bFirst) return aFirst - bFirst;
        return a.type.localeCompare(b.type);
      });
  };

  // For preview chips: show latest occurrence description for big up/down day types
  const previewLabel = (group: { type: string; count: number; items: ViolationItem[] }, panelType: 'confirmation' | 'violation') => {
    const streakTypes = new Set(['daily_higher_highs', 'weekly_higher_highs', 'daily_lower_lows', 'weekly_lower_lows']);
    const latest = group.items[group.items.length - 1];

    if (group.type === 'above_20day_20pct' && group.count === 1 && latest?.description) {
      return latest.description.split(':')[0];
    }

    if ((group.type === 'up_30pct_vol_increase' || group.type === 'down_50pct_vol_increase') && group.count === 1 && latest?.description) {
      return latest.description.split(':')[0];
    }

    if ((group.type === 'big_up_day' || group.type === 'big_down_day') && group.count === 1 && group.items.length > 0) {
      // Latest item's description is e.g. "Up +15.2%" or "Down -4.1%"
      return latest.description || humanizeType(group.type, panelType);
    }

    const label = humanizeType(group.type, panelType);
    if (streakTypes.has(group.type)) {
      const streakMatch = latest?.description?.match(/^(\d+)\s+consecutive\b/i);
      const streakLength = streakMatch ? Number(streakMatch[1]) : group.count;
      return `${streakLength} ${label}`;
    }
    if (group.count > 1) {
      return `${label} (${group.count}x)`;
    }
    return label;
  };

  const previewChipClass = (group: { type: string; items: ViolationItem[] }, panelType: 'confirmation' | 'violation') => {
    const severity = group.items[0]?.severity;
    if (severity === 'green') {
      return 'bg-green-700/75 text-green-50';
    }
    if (severity === 'red') {
      return panelType === 'violation'
        ? 'bg-red-800/70 text-red-50'
        : 'bg-red-700/75 text-red-50';
    }

    return panelType === 'violation'
      ? 'bg-red-800/70 text-red-50'
      : 'bg-green-700/75 text-green-50';
  };

  const humanizeType = (type: string, panelType: 'confirmation' | 'violation' = 'violation') =>
    type
      .replace(/_/g, ' ')
      .replace(/(\d+)\s*ma\b/gi, '$1MA')
      .replace(/\brs\b/gi, 'RS')
      .replace(/\bma\b/gi, 'MA')
      .replace(/pct/gi, '%')
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .replace(/^Squat Reversal$/i, 'Squat')
      .replace(/Large Squat Reversal/g, 'Large Reversal')
      .replace(/^Up 30% Vol Increase$/i, 'Up On 30%+ Volume')
      .replace(/^Down 50% Volume Increase$/i, 'Down On 50%+ Volume')
      .replace(/^Above 20day 20%$/i, '20%+ Above 20MA')
      .replace(/^Down Up Largest Vol$/i, panelType === 'confirmation' ? 'Up Largest Vol' : 'Down Largest Vol')
      .replace(/^Largest Pct Down High Vol$/i, 'Largest % Down Day On High Volume')
      .replace(/^Largest % Down High Vol$/i, 'Largest % Down Day On High Volume')
      .replace(/^Good Bad Close$/i, panelType === 'confirmation' ? '>70% Good closes' : '<30% Good closes')
      .replace(/^Days Up Down$/i, panelType === 'confirmation' ? '>70% Days Up' : '<30% Days Up');

  const humanizeLabel = (label: string, panelType: 'confirmation' | 'violation' = 'violation') =>
    label
      .replace(/(\d+)\s*ma\b/gi, '$1MA')
      .replace(/\brs\b/gi, 'RS')
      .replace(/\bma\b/gi, 'MA')
      .replace(/\bants\b/gi, 'ANTS')
      .replace(/pct/gi, '%')
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .replace(/Down Up Largest Vol/g, panelType === 'confirmation' ? 'Up Largest Vol' : 'Down Largest Vol')
      .replace(/Squat Reversal/g, 'Squat')
      .replace(/Large Squat Reversal/g, 'Large Reversal')
      .replace(/Largest % Down High Vol/g, 'Largest % Down Day On High Volume')
      .replace(/^Good\/Bad Closes$/i, panelType === 'confirmation' ? '>70% Good closes' : '<30% Good closes')
      .replace(/^Good Bad Close$/i, panelType === 'confirmation' ? '>70% Good closes' : '<30% Good closes')
      .replace(/^Days Up Down$/i, panelType === 'confirmation' ? '>70% Days Up' : '<30% Days Up');

  const confirmationGroups = groupByType(confirmationItems);
  const violationGroups = groupByType(violationItems);

  const daysUp = result?.info?.days_up ?? 0;
  const daysDown = result?.info?.days_down ?? 0;
  const goodCloses = result?.info?.good_closes ?? 0;
  const badCloses = result?.info?.bad_closes ?? 0;

  const daysTotal = daysUp + daysDown;
  const goodCloseTotal = goodCloses + badCloses;
  const daysUpRatio = daysTotal > 0 ? daysUp / daysTotal : 0.5;
  const goodCloseRatio = goodCloseTotal > 0 ? goodCloses / goodCloseTotal : 0.5;

  const daysColorClass =
    daysUpRatio > 0.5
      ? 'text-green-50 bg-green-700/75 font-semibold'
      : daysUpRatio < 0.5
        ? 'text-red-50 bg-red-800/70 font-semibold'
        : 'text-muted-foreground bg-muted';

  const goodCloseColorClass =
    goodCloseRatio > 0.5
      ? 'text-green-50 bg-green-700/75 font-semibold'
      : goodCloseRatio < 0.5
        ? 'text-red-50 bg-red-800/70 font-semibold'
        : 'text-muted-foreground bg-muted';

  const renderViolationList = (items: ViolationItem[], panelType: 'violation' | 'confirmation') => {
    const grouped = groupByType(items);

    return (
      <div className="space-y-1">
        {grouped.map((group) => (
          <div key={group.type} className="text-xs">
            {(() => {
              const defaultLabel = humanizeType(group.type, panelType);
              const singleItemLabel = humanizeLabel(group.items[0]?.description?.split(':')[0] || defaultLabel, panelType);
              const groupLabel = group.items.length > 1 ? defaultLabel : singleItemLabel;
              return (
                <span
                  className={`font-medium ${group.items[0].severity === 'red'
                    ? (panelType === 'violation' ? 'text-red-400' : 'text-red-500')
                    : group.items[0].severity === 'green'
                        ? 'text-green-500'
                        : 'text-muted-foreground'
                    }`}
                >
                  {groupLabel}
                </span>
              );
            })()}
            <span className="text-muted-foreground ml-1">
              {' (' + group.items.length + 'x)'}
              {(showAllGroupedDates || group.items.length <= 3) &&
                ' — ' + group.items.map((g) => g.date).join(', ')}
            </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div data-trade-id={trade.id} className="border border-border rounded-md bg-background relative">
      {/* Compact header */}
      <div
        className="flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-muted/50"
        onClick={() => setExpanded(!expanded)}
      >
        <div ref={tickerColRef} className="shrink-0 flex flex-col justify-center mr-1">
          <div className="flex items-baseline gap-1.5 whitespace-nowrap">
            <span className="font-bold text-sm leading-tight">{trade.ticker}</span>
            {result?.info?.current_price !== undefined && (
              <>
                <span className="text-sm font-semibold">{result.info.current_price}</span>
                <span className={`text-[11px] font-bold ${(result.info.current_change_pct ?? 0) > 0
                  ? 'text-[#22c55e]'
                  : ((result.info.current_change_pct ?? 0) < 0 ? 'text-[#f87171]' : 'text-white')
                  }`}>
                  {(result.info.current_change_pct ?? 0) > 0 ? '+' : ''}{result.info.current_change_pct}%
                </span>
              </>
            )}
          </div>
        </div>

        <div className="flex-1 min-w-0 grid grid-cols-2 gap-1">
          {result ? (
            <>
              <div className="min-w-0 text-[11px] flex items-start gap-1 font-semibold">
                <div className="shrink-0 flex flex-col items-start gap-0.5">
                  <span className="font-bold px-1.5 py-0.5 rounded bg-green-700/75 text-green-50">
                    D:{dailyConfirmationCount}
                  </span>
                  <span className="font-bold px-1.5 py-0.5 rounded bg-green-700/75 text-green-50">
                    W:{weeklyConfirmationCount}
                  </span>
                </div>
                <div className="min-w-0 flex flex-wrap gap-1 content-start">
                  {confirmationGroups.length === 0 ? (
                    <span className="text-[11px] text-green-50 bg-green-700/75 px-1.5 py-0.5 rounded font-semibold">None</span>
                  ) : (
                    confirmationGroups.map((group) => (
                      <span
                        key={`c-${group.type}`}
                        className={`px-1.5 py-0.5 rounded font-semibold ${previewChipClass(group, 'confirmation')}`}
                      >
                        {previewLabel(group, 'confirmation')}
                      </span>
                    ))
                  )}
                </div>
              </div>
              <div className="min-w-0 text-[11px] flex items-start gap-1 font-semibold">
                <div className="shrink-0 flex flex-col items-start gap-0.5">
                  <span className="font-bold px-1.5 py-0.5 rounded bg-red-800/70 text-red-50">
                    D:{dailyViolationCount}
                  </span>
                  <span className="font-bold px-1.5 py-0.5 rounded bg-red-800/70 text-red-50">
                    W:{weeklyViolationCount}
                  </span>
                </div>
                <div className="min-w-0 flex flex-wrap gap-1 content-start">
                  {violationGroups.length === 0 ? (
                    <span className="text-[11px] text-red-50 bg-red-800/70 px-1.5 py-0.5 rounded font-semibold">None</span>
                  ) : (
                    violationGroups.map((group) => (
                      <span
                        key={`v-${group.type}`}
                        className={`px-1.5 py-0.5 rounded font-semibold ${previewChipClass(group, 'violation')}`}
                      >
                        {previewLabel(group, 'violation')}
                      </span>
                    ))
                  )}
                </div>
              </div>
            </>
          ) : loading ? (
            <div className="col-span-2 flex items-center justify-center">
              <RefreshCw className="h-3 w-3 animate-spin text-muted-foreground" />
            </div>
          ) : null}
        </div>

        {/* D/GC badges */}
        <div className="flex flex-col items-end gap-0.5">
          {result?.info && (
            <>
              <span
                className={`text-xs px-1.5 py-0.5 rounded ${daysColorClass}`}
              >
                UD:{daysUp}/{daysDown}
              </span>
              <span
                className={`text-xs px-1.5 py-0.5 rounded ${goodCloseColorClass}`}
              >
                GC:{goodCloses}/{badCloses}
              </span>
            </>
          )}
        </div>

        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowDailyChartModal(true);
          }}
          className="p-1 hover:bg-muted rounded text-muted-foreground"
          title="Open daily chart"
        >
          <LineChart className="h-4 w-4" />
        </button>

        <button
          onClick={(e) => {
            e.stopPropagation();
            handleDelete();
          }}
          className="p-1 hover:bg-red-500/20 rounded text-red-500"
          title="Remove"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="border-t border-border px-3 py-3 space-y-3">
          {/* Date controls */}
          <div className="flex flex-wrap items-center gap-3 text-sm">
            {/* Start date */}
            <div className="flex items-center gap-1">
              <span className="text-muted-foreground">Start:</span>
              <button
                onClick={() => shiftDate('start_date', -1)}
                className="p-0.5 hover:bg-muted rounded"
              >
                <ChevronLeft className="h-3 w-3" />
              </button>
              {editingStartDate ? (
                <input
                  type="date"
                  defaultValue={startDateLocal}
                  onBlur={(e) => handleDateChange('start_date', e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleDateChange('start_date', (e.target as HTMLInputElement).value);
                  }}
                  autoFocus
                  className="px-1 py-0.5 border border-border rounded text-xs bg-background w-32"
                />
              ) : (
                <span
                  className="cursor-pointer hover:underline font-medium"
                  onClick={(e) => {
                    e.stopPropagation();
                    setEditingStartDate(true);
                  }}
                >
                  {startDateLocal}
                </span>
              )}
              <button
                onClick={() => shiftDate('start_date', 1)}
                className="p-0.5 hover:bg-muted rounded"
              >
                <ChevronRight className="h-3 w-3" />
              </button>
            </div>

            {/* End date */}
            <div className="flex items-center gap-1">
              <span className="text-muted-foreground">End:</span>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useLatest}
                  onChange={handleToggleLatest}
                  className="rounded h-3 w-3"
                />
                <span className="text-xs font-medium">Latest</span>
              </label>
              {!useLatest && (
                <>
                  <button
                    onClick={() => shiftDate('end_date', -1)}
                    className="p-0.5 hover:bg-muted rounded"
                  >
                    <ChevronLeft className="h-3 w-3" />
                  </button>
                  {editingEndDate ? (
                    <input
                      type="date"
                      defaultValue={endDateLocal}
                      onBlur={(e) => handleDateChange('end_date', e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleDateChange('end_date', (e.target as HTMLInputElement).value);
                      }}
                      autoFocus
                      className="px-1 py-0.5 border border-border rounded text-xs bg-background w-32"
                    />
                  ) : (
                    <span
                      className="cursor-pointer hover:underline font-medium"
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingEndDate(true);
                      }}
                    >
                      {endDateLocal || '—'}
                    </span>
                  )}
                  <button
                    onClick={() => shiftDate('end_date', 1)}
                    className="p-0.5 hover:bg-muted rounded"
                  >
                    <ChevronRight className="h-3 w-3" />
                  </button>
                </>
              )}
            </div>

            {/* Trend-up start info */}
            {result?.trend_up_start && (
              <span className="text-xs text-muted-foreground">
                Trend up start: {result.trend_up_start}
              </span>
            )}

            <div className="ml-auto flex gap-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowDailyChartModal(true);
                }}
                className="px-2 py-1 hover:bg-muted rounded text-xs font-medium text-muted-foreground"
                title="Open Daily Chart"
              >
                Daily Chart
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowRiskModal(true);
                }}
                className="px-2 py-1 hover:bg-muted rounded text-xs font-medium text-muted-foreground"
                title="Risk Analysis"
                disabled={!result?.risk_analysis}
              >
                Risk Analysis
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRefresh();
                }}
                className="p-1 hover:bg-muted rounded text-muted-foreground"
                title="Refresh data"
              >
                <RefreshCw className="h-3 w-3" />
              </button>
            </div>
          </div>

          {/* Violations & Confirmations */}
          {result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Confirmations column */}
              <div className="bg-green-500/5 border border-green-500/20 rounded-md p-2">
                <h5 className="text-xs font-bold text-green-500 mb-1 flex items-center gap-1">
                  Confirmations
                  <span className="text-green-500 text-sm px-1.5 py-0.5 ml-auto">
                    {totalC}
                  </span>
                </h5>
                {confirmationItems.length > 0 ? (
                  renderViolationList(confirmationItems, 'confirmation')
                ) : (
                  <p className="text-xs text-muted-foreground">None</p>
                )}
              </div>

              {/* Violations column */}
              <div className="bg-red-500/5 border border-red-500/20 rounded-md p-2">
                <h5 className="text-xs font-bold text-red-400 mb-1 flex items-center gap-1">
                  Violations
                  <span className="text-red-400 text-sm px-1.5 py-0.5 ml-auto">
                    {totalV}
                  </span>
                </h5>
                {violationItems.length > 0 ? (
                  renderViolationList(violationItems, 'violation')
                ) : (
                  <p className="text-xs text-muted-foreground">None</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {showDeleteModal && (
        <div
          className="absolute inset-0 z-20 flex items-center justify-center bg-black/50"
          onClick={() => !isDeletingTrade && setShowDeleteModal(false)}
        >
          <div
            className="w-full max-w-sm rounded-md border border-border bg-background p-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !isDeletingTrade) {
                e.preventDefault();
                handleConfirmDelete();
              }
            }}
          >
            <h4 className="text-sm font-semibold mb-2 text-red-400">Remove Monitored Position</h4>
            <p className="text-xs text-muted-foreground mb-3">
              Remove <span className="font-semibold text-foreground">{trade.ticker}</span> from monitored positions?
            </p>

            <label className="flex items-center gap-2 text-xs cursor-pointer mb-3">
              <input
                type="checkbox"
                checked={alsoRemoveFromTickerMonitoring}
                onChange={(e) => setAlsoRemoveFromTickerMonitoring(e.target.checked)}
                className="rounded h-3 w-3"
                disabled={isDeletingTrade}
              />
              Also remove from Ticker Monitoring
            </label>

            <div className="flex gap-2 justify-end">
              <button
                type="button"
                className="px-2 py-1 text-xs rounded border border-border hover:bg-muted disabled:opacity-50"
                onClick={() => setShowDeleteModal(false)}
                disabled={isDeletingTrade}
              >
                Cancel
              </button>
              <button
                type="button"
                className="px-2 py-1 text-xs rounded bg-red-600 text-white hover:bg-red-500 disabled:opacity-50"
                onClick={handleConfirmDelete}
                disabled={isDeletingTrade}
                ref={removeConfirmBtnRef}
              >
                {isDeletingTrade ? 'Removing...' : 'Remove'}
              </button>
            </div>
          </div>
        </div>
      )}

      {showRiskModal && result?.risk_analysis && (
        <RiskAnalysisModal
          ticker={trade.ticker}
          risk={result.risk_analysis}
          onClose={() => setShowRiskModal(false)}
        />
      )}

      {showDailyChartModal && (
        <TradeDailyChartModal
          trade={trade}
          initialResult={result}
          onClose={() => setShowDailyChartModal(false)}
        />
      )}
    </div>
  );
}
