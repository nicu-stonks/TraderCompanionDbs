import { Trade } from '@/TradeHistoryPage/types/Trade';
import React from 'react';
import { addMonths, format, parseISO } from 'date-fns';
import { Metric, TradeGrade, TradeGradeDeletion, MetricGradeCheckSetting } from '../types/types';
import { Loader, Save } from 'lucide-react';
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { gradeService, analysisService, metricCheckSettingService } from '../services/postAnalysis';
import TradeCaseDetails from './TradeCaseDetailsProps';
import { Checkbox } from '@/components/ui/checkbox';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';


const TradeGrader: React.FC<{
  trades: Trade[];
  metrics: Metric[];
  tradeGrades: TradeGrade[];
  checkSettings: MetricGradeCheckSetting[];
  onRefetchCheckSettings: () => void;
}> = ({ trades, metrics, tradeGrades, checkSettings, onRefetchCheckSettings }) => {
  const [localGrades, setLocalGrades] = useState<TradeGrade[]>([]);
  const [saving, setSaving] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false); // internal flag before auto-save completes
  const [expandedTradeId, setExpandedTradeId] = useState<number | null>(null);
  // Keep a stable positional index for navigation to avoid relying solely on ID lookups (which can cause skips if the array mutates)
  const [currentTradeIndex, setCurrentTradeIndex] = useState<number>(-1);
  const currentTradeIndexRef = useRef<number>(-1);
  useEffect(() => { currentTradeIndexRef.current = currentTradeIndex; }, [currentTradeIndex]);
  const [pendingDeletions, setPendingDeletions] = useState<TradeGradeDeletion[]>([]);
  const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightSave = useRef<Promise<void> | null>(null);
  const latestGradesRef = useRef<TradeGrade[]>([]);
  const latestDeletionsRef = useRef<TradeGradeDeletion[]>([]);

  const defaultRequiredMetrics = ['Exit at loss', 'Exit at profit'];
  const defaultExcludeMetric = 'Entry Point';
  const [requiredMetricsSelected, setRequiredMetricsSelected] = useState<string[]>(defaultRequiredMetrics);
  const [excludeMetricSelected, setExcludeMetricSelected] = useState<string>(defaultExcludeMetric);
  const [savingCheckSettings, setSavingCheckSettings] = useState(false);

  const [displayCount, setDisplayCount] = useState<number>(50);
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [useDateFilter, setUseDateFilter] = useState<boolean>(false);

  // Keep refs synced
  useEffect(() => {
    latestGradesRef.current = localGrades;
  }, [localGrades]);
  useEffect(() => {
    latestDeletionsRef.current = pendingDeletions;
  }, [pendingDeletions]);

  useEffect(() => {
    const existing = checkSettings[0];
    if (existing) {
      const required = existing.required_metrics
        ? existing.required_metrics.split(',').map(v => v.trim()).filter(Boolean)
        : defaultRequiredMetrics;
      setRequiredMetricsSelected(required);
      setExcludeMetricSelected(existing.exclude_metric || defaultExcludeMetric);
    }
  }, [checkSettings]);

  const tradeDropZoneRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const expandedTradeIdRef = useRef<number | null>(null);
  useEffect(() => { expandedTradeIdRef.current = expandedTradeId; }, [expandedTradeId]);

  // Prefetch analyses for all trades so navigation never skips just because a trade wasn't expanded yet
  useEffect(() => {
    let cancelled = false;
    if (!trades.length) return;
    (async () => {
      for (const t of trades) {
        if (cancelled) break;
        if (!analysesByTradeRef.current[t.ID]) {
          try {
            const data = await analysisService.listByTrade(t.ID);
            const images = data.filter(a => !!a.image).map(a => a.image as string);
            analysesByTradeRef.current[t.ID] = { ids: data.map(a => a.id), images };
            if (!(t.ID in currentImageIndexRef.current)) {
              currentImageIndexRef.current[t.ID] = 0;
            } else {
              const cur = currentImageIndexRef.current[t.ID];
              if (cur >= images.length) currentImageIndexRef.current[t.ID] = Math.max(0, images.length - 1);
            }
          } catch {
            // Ignore prefetch errors
          }
        }
      }
    })();
    return () => { cancelled = true; };
  }, [trades]);

  // Fullscreen image viewer state (declared after toggleTradeDetails so dependencies order is valid)
  const [fullscreenTradeId, setFullscreenTradeId] = useState<number | null>(null);
  const [fullscreenImageUrl, setFullscreenImageUrl] = useState<string | null>(null);
  // Track analyses per trade and current image index per trade for keyboard navigation
  const analysesByTradeRef = useRef<Record<number, { ids: number[]; images: string[] }>>({});
  const currentImageIndexRef = useRef<Record<number, number>>({});

  const centerImageWithRetry = useCallback((tradeId: number, attempt = 0) => {
    if (expandedTradeIdRef.current !== tradeId) return; // trade no longer expanded
    const container = tradeDropZoneRefs.current[tradeId];
    if (!container) {
      if (attempt < 10) setTimeout(() => centerImageWithRetry(tradeId, attempt + 1), 80);
      return;
    }
    const img = container.querySelector('img');
    if (!img || !img.complete || (img as HTMLImageElement).naturalHeight === 0) {
      if (attempt < 15) setTimeout(() => centerImageWithRetry(tradeId, attempt + 1), 120);
      return;
    }
    const rect = img.getBoundingClientRect();
    const docTop = window.scrollY + rect.top;
    const targetScrollTop = docTop + rect.height / 2 - window.innerHeight / 2;
    window.scrollTo({ top: Math.max(0, targetScrollTop), behavior: 'smooth' });
  }, []);

  const toggleTradeDetails = useCallback((tradeId: number, focusAfter = false) => {
    setExpandedTradeId(prev => {
      const newId = prev === tradeId ? null : tradeId;
      if (focusAfter && newId !== null) {
        // Initial slight delay for render, then run retry-based centering
        setTimeout(() => {
          const container = tradeDropZoneRefs.current[newId];
          if (container) {
            container.focus();
          }
          centerImageWithRetry(newId, 0);
        }, 50);
      }
      // Sync positional index
      if (newId !== null) {
        const idx = trades.findIndex(t => t.ID === newId);
        setCurrentTradeIndex(idx);
      } else {
        setCurrentTradeIndex(-1);
      }
      return newId;
    });
  }, [centerImageWithRetry, trades]);

  const getTradeIndexById = useCallback((id: number | null) => {
    if (id == null) return -1;
    return trades.findIndex(t => t.ID === id);
  }, [trades]);

  const openFullscreenForTrade = useCallback((tradeId: number, imageIdx?: number) => {
    const data = analysesByTradeRef.current[tradeId];
    if (!data || !data.images.length) return;
    const idx = imageIdx != null ? imageIdx : (currentImageIndexRef.current[tradeId] || 0);
    const bounded = ((idx % data.images.length) + data.images.length) % data.images.length;
    currentImageIndexRef.current[tradeId] = bounded;
    setFullscreenImageUrl(data.images[bounded]);
    setFullscreenTradeId(tradeId);
  }, []);

  const closeFullscreen = useCallback(() => {
    setFullscreenTradeId(null);
    setFullscreenImageUrl(null);
  }, []);

  const navigateFullscreen = useCallback((direction: 1 | -1) => {
    if (fullscreenTradeId == null || !trades.length) return;
    const tradeId = fullscreenTradeId;
    const data = analysesByTradeRef.current[tradeId];
    if (data && data.images.length > 1) {
      // Move within same trade's images first
      const cur = currentImageIndexRef.current[tradeId] || 0;
      const nextIdx = cur + direction;
      if (nextIdx >= 0 && nextIdx < data.images.length) {
        openFullscreenForTrade(tradeId, nextIdx);
        return;
      }
    }
    // Move to next/prev trade with images
    const tradeIndex = getTradeIndexById(tradeId);
    if (tradeIndex === -1) return;
    for (let i = 1; i <= trades.length; i++) { // at most one full loop
      const candidateIndex = (tradeIndex + direction * i + trades.length) % trades.length;
      const candidate = trades[candidateIndex];
      const candData = analysesByTradeRef.current[candidate.ID];
      if (candData && candData.images.length) {
        currentImageIndexRef.current[candidate.ID] = direction === 1 ? 0 : candData.images.length - 1;
        if (expandedTradeId !== candidate.ID) {
          toggleTradeDetails(candidate.ID);
          setTimeout(() => openFullscreenForTrade(candidate.ID), 80);
        } else {
          openFullscreenForTrade(candidate.ID);
        }
        setCurrentTradeIndex(candidateIndex);
        return;
      }
      // No images: skip while staying in fullscreen
    }
  }, [fullscreenTradeId, trades, expandedTradeId, toggleTradeDetails, openFullscreenForTrade, getTradeIndexById]);

  // Keyboard navigation: left/right arrows move between trades and auto-focus image drop zone
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (fullscreenTradeId !== null) {
      if (e.key === 'Escape') { e.preventDefault(); closeFullscreen(); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); navigateFullscreen(1); return; }
      if (e.key === 'ArrowLeft') { e.preventDefault(); navigateFullscreen(-1); return; }
    }
    if (['ArrowLeft', 'ArrowRight'].includes(e.key)) {
      // Enhanced navigation: cycle through images of current trade using left/right without fullscreen
      if (expandedTradeId !== null) {
        const data = analysesByTradeRef.current[expandedTradeId];
        if (data && data.images.length) {
          const dir = e.key === 'ArrowRight' ? 1 : -1;
          let cur = currentImageIndexRef.current[expandedTradeId] || 0;
          cur += dir;
          if (cur >= 0 && cur < data.images.length) {
            currentImageIndexRef.current[expandedTradeId] = cur;
            // Open fullscreen for a focused viewing experience
            openFullscreenForTrade(expandedTradeId, cur);
            e.preventDefault();
            return;
          }
          // If we've moved past ends, fall through to change trade
        }
      }
      if (!trades.length) return;
      e.preventDefault();
      const delta = e.key === 'ArrowRight' ? 1 : -1;
      let baseIndex = currentTradeIndexRef.current;
      if (expandedTradeId === null || baseIndex === -1) {
        // Start navigation from before first / after last depending on direction
        baseIndex = (e.key === 'ArrowRight') ? -1 : trades.length;
      }
      let nextIndex = baseIndex + delta;
      if (nextIndex < 0) nextIndex = trades.length - 1;
      if (nextIndex >= trades.length) nextIndex = 0;
      const nextTrade = trades[nextIndex];
      setCurrentTradeIndex(nextIndex);
      toggleTradeDetails(nextTrade.ID, true);
      const nextData = analysesByTradeRef.current[nextTrade.ID];
      if (nextData && nextData.images.length) {
        currentImageIndexRef.current[nextTrade.ID] = e.key === 'ArrowRight' ? 0 : nextData.images.length - 1;
      } else {
        currentImageIndexRef.current[nextTrade.ID] = 0; // reset
      }
    }
  }, [expandedTradeId, trades, toggleTradeDetails, fullscreenTradeId, closeFullscreen, navigateFullscreen, openFullscreenForTrade]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Initialize local grades from props
  useEffect(() => {
    setLocalGrades([...tradeGrades]);
    setHasUnsavedChanges(false);
  }, [tradeGrades]);

  const sortedTradesOldestFirst = [...trades].sort(
    (a, b) => new Date(a.Entry_Date).getTime() - new Date(b.Entry_Date).getTime()
  );
  const tradeIndexMap = new Map(sortedTradesOldestFirst.map((t, idx) => [t.ID, idx + 1]));

  // Initialize default dates based on "Latest 50 trades" rule
  useEffect(() => {
    if (trades.length > 0 && !startDate && !endDate) {
      const sorted = [...trades].sort((a, b) => {
        const dateA = a.Entry_Date ? new Date(a.Entry_Date).getTime() : 0;
        const dateB = b.Entry_Date ? new Date(b.Entry_Date).getTime() : 0;
        return dateB - dateA;
      });

      const latestTrades = sorted.slice(0, 50);
      if (latestTrades.length > 0) {
        const oldestEntry = latestTrades[latestTrades.length - 1];
        const newestEntry = latestTrades[0];

        if (oldestEntry?.Entry_Date && newestEntry?.Entry_Date) {
          const minDate = new Date(oldestEntry.Entry_Date);
          const maxDate = new Date(newestEntry.Entry_Date);

          const start = addMonths(minDate, -1);
          const end = addMonths(maxDate, 1);

          setStartDate(format(start, 'yyyy-MM'));
          setEndDate(format(end, 'yyyy-MM'));
        }
      } else {
        const now = new Date();
        setStartDate(format(addMonths(now, -1), 'yyyy-MM'));
        setEndDate(format(addMonths(now, 1), 'yyyy-MM'));
      }
    }
  }, [trades, startDate, endDate]);

  const sortedTrades = useMemo(() => {
    let filtered = [...trades];

    if (useDateFilter) {
      let filterStart = '';
      let filterEnd = '';

      if (startDate) {
        filterStart = `${startDate}-01`;
      }

      if (endDate) {
        const endMonthStart = parseISO(`${endDate}-01`);
        const nextMonthStart = addMonths(endMonthStart, 1);
        filterEnd = format(nextMonthStart, 'yyyy-MM-dd');
      }

      filtered = filtered.filter((t: Trade) => {
        if (!t.Entry_Date) return false;
        if (filterStart && t.Entry_Date < filterStart) return false;
        if (filterEnd && t.Entry_Date >= filterEnd) return false;
        return true;
      });
    }

    return filtered.sort((a, b) => {
      const dateA = a.Entry_Date ? new Date(a.Entry_Date).getTime() : 0;
      const dateB = b.Entry_Date ? new Date(b.Entry_Date).getTime() : 0;
      return dateB - dateA;
    });
  }, [trades, useDateFilter, startDate, endDate]);

  const tradesToRender = useMemo(() => {
    if (useDateFilter) return sortedTrades;
    const count = Math.min(displayCount, trades.length);
    return sortedTrades.slice(0, count);
  }, [useDateFilter, sortedTrades, displayCount, trades.length]);

  const gradeLookup = useCallback(() => {
    const map = new Map<number, Map<number, string>>();
    for (const grade of localGrades) {
      const metricMap = map.get(grade.tradeId) ?? new Map<number, string>();
      metricMap.set(Number(grade.metricId), grade.selectedOptionId);
      map.set(grade.tradeId, metricMap);
    }
    return map;
  }, [localGrades]);

  const normalizeMetricName = (value: string) => value.trim().toLowerCase();

  const missingRequiredTrades = useCallback(() => {
    const requiredNames = requiredMetricsSelected.map(n => normalizeMetricName(n)).filter(Boolean);
    const excludeName = normalizeMetricName(excludeMetricSelected);

    const metricIdByName = new Map(metrics.map(m => [normalizeMetricName(m.name), m.id]));
    const requiredMetricIds = requiredNames
      .map(name => metricIdByName.get(name))
      .filter((id): id is number => id !== undefined);
    const excludeMetricId = excludeName ? metricIdByName.get(excludeName) : undefined;

    const gradesByTrade = gradeLookup();

    if (requiredMetricIds.length === 0) return [];

    return tradesToRender
      .filter((trade: Trade) => {
        if (!excludeMetricId) return true;
        const metricMap = gradesByTrade.get(trade.ID);
        return metricMap ? metricMap.has(excludeMetricId) : false;
      })
      .filter((trade: Trade) => {
        const metricMap = gradesByTrade.get(trade.ID);
        return requiredMetricIds.every(metricId => !metricMap || !metricMap.has(metricId));
      })
      .map((trade: Trade) => ({
        trade,
        tradeIndex: tradeIndexMap.get(trade.ID) ?? 0
      }))
        .sort((a: { tradeIndex: number }, b: { tradeIndex: number }) => a.tradeIndex - b.tradeIndex);
  }, [requiredMetricsSelected, excludeMetricSelected, metrics, tradesToRender, tradeIndexMap, gradeLookup]);

  const handleSaveCheckSettings = async () => {
    setSavingCheckSettings(true);
    try {
      await metricCheckSettingService.upsertSettings(requiredMetricsSelected.join(', '), excludeMetricSelected);
      onRefetchCheckSettings();
    } catch (error) {
      console.error('Failed to save check settings:', error);
    } finally {
      setSavingCheckSettings(false);
    }
  };

  const getGradeForTrade = (tradeId: number, metricId: number): string | null => {
    const grade = localGrades.find(g => g.tradeId === tradeId && parseInt(g.metricId) === metricId);
    return grade?.selectedOptionId || null;
  };

  const scheduleAutoSave = useCallback(() => {
    setHasUnsavedChanges(true);
    if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current);
    autoSaveTimer.current = setTimeout(() => {
      const run = async () => {
        setSaving(true);
        const gradesSnapshot = [...latestGradesRef.current];
        const deletionsSnapshot = [...latestDeletionsRef.current];
        try {
          const saved = await gradeService.bulkUpdateGrades(gradesSnapshot, deletionsSnapshot);
          // Replace local grades with authoritative response if provided
          if (Array.isArray(saved)) {
            setLocalGrades(saved);
            latestGradesRef.current = saved;
          }
          // Clear only deletions we sent (simple approach: clear all)
          setPendingDeletions([]);
          latestDeletionsRef.current = [];
          setHasUnsavedChanges(false);
          // Intentionally NOT calling onGradesUpdate to avoid page refresh
        } catch (err) {
          console.error('Auto-save failed:', err);
        } finally {
          setSaving(false);
        }
      };
      const p = inFlightSave.current ? inFlightSave.current.then(run) : run();
      inFlightSave.current = p;
    }, 600);
  }, []);

  const updateLocalGrade = (tradeId: number, metricId: number, optionId: number) => {
    // Remove any previous grade for this trade/metric
    const newGrades = localGrades.filter(
      g => !(g.tradeId === tradeId && parseInt(g.metricId) === metricId)
    );

    // Add new one
    newGrades.push({
      tradeId,
      metricId: metricId.toString(),
      selectedOptionId: optionId.toString()
    });

    setLocalGrades(newGrades);
    scheduleAutoSave();
  };

  const clearLocalGrade = (tradeId: number, metricId: number) => {
    // Remove the grade for this trade/metric entirely (no selection)
    const newGrades = localGrades.filter(
      g => !(g.tradeId === tradeId && parseInt(g.metricId) === metricId)
    );
    setLocalGrades(newGrades);
    // Record deletion so backend can delete persisted grade
    setPendingDeletions(prev => {
      const exists = prev.find(d => d.tradeId === tradeId && parseInt(d.metricId) === metricId);
      if (exists) return prev; // avoid duplicates
      return [...prev, { tradeId, metricId: metricId.toString() }];
    });
    scheduleAutoSave();
  };

  const toggleGrade = (tradeId: number, metricId: number, optionId: number) => {
    const current = getGradeForTrade(tradeId, metricId);
    if (current === optionId.toString()) {
      // Clicking the already selected option -> unselect (clear)
      clearLocalGrade(tradeId, metricId);
    } else {
      updateLocalGrade(tradeId, metricId, optionId);
    }
  };

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current);
    };
  }, []);

  if (metrics.length === 0) {
    return (
      <div className="bg-background rounded-lg shadow p-6 mb-6">
        <h2 className="text-2xl font-bold mb-4 flex items-center">
          <Save className="mr-2" />
          Trade Grader
        </h2>
        <p className="text-muted-foreground">Please create some metrics first to start grading trades.</p>
      </div>
    );
  }

  return (
    <div className="bg-background rounded-lg shadow p-6 mb-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold flex items-center">
          <Save className="mr-2" />
          Trade Grader
        </h2>
        <div className="text-sm text-muted-foreground h-5 flex items-center">
          {saving && (
            <span className="flex items-center"><Loader className="w-4 h-4 mr-1 animate-spin" /> Saving</span>
          )}
          {!saving && hasUnsavedChanges && <span>Pending…</span>}
        </div>
      </div>

      <div className="mb-4 p-3 border border-border rounded-lg bg-card">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold">Missing Grade Checker</h3>
            <p className="text-xs text-muted-foreground">
              Finds trades that are missing any required metric grades, but only after the exclude metric is already graded.
            </p>
            <p className="text-xs text-muted-foreground">
              Example: once a trade has Entry Point graded, flag it if Exit at loss or Exit at profit is still missing.
            </p>
          </div>
          <button
            onClick={handleSaveCheckSettings}
            disabled={savingCheckSettings}
            className="px-3 py-1.5 bg-primary text-primary-foreground text-sm rounded-md hover:bg-primary/90 disabled:opacity-50 flex items-center"
          >
            {savingCheckSettings ? <Loader className="w-3 h-3 animate-spin" /> : 'Save'}
          </button>
        </div>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div>
            <label className="block text-xs text-muted-foreground mb-2">Required metrics (missing check)</label>
            <div className="grid grid-cols-1 gap-1">
              {metrics.map(metric => (
                <label key={metric.id} className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={requiredMetricsSelected.includes(metric.name)}
                    onChange={(e) => {
                      setRequiredMetricsSelected(prev => {
                        if (e.target.checked) return [...prev, metric.name];
                        return prev.filter(name => name !== metric.name);
                      });
                    }}
                  />
                  <span>{metric.name}</span>
                </label>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-2">Exclude metric (must be graded to check)</label>
            <div className="grid grid-cols-1 gap-1">
              {metrics.map(metric => (
                <label key={metric.id} className="flex items-center gap-2 text-xs">
                  <input
                    type="radio"
                    name="exclude-metric"
                    checked={excludeMetricSelected === metric.name}
                    onChange={() => setExcludeMetricSelected(metric.name)}
                  />
                  <span>{metric.name}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-3 text-xs text-muted-foreground">
          Trades missing both {requiredMetricsSelected.join(', ')}: {missingRequiredTrades().length}
        </div>
        {missingRequiredTrades().length > 0 && (
          <div className="mt-2 max-h-40 overflow-auto border border-border rounded-md p-2 bg-background text-xs space-y-1">
            {missingRequiredTrades().map(item => (
              <div key={item.trade.ID} className="flex items-center justify-between">
                <span>#{item.tradeIndex} {item.trade.Ticker} ({item.trade.Entry_Date})</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mb-4 p-2 border border-border rounded-lg bg-card">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Checkbox
              id="trade-grader-use-date-filter"
              checked={useDateFilter}
              onCheckedChange={(checked) => setUseDateFilter(checked === true)}
            />
            <label htmlFor="trade-grader-use-date-filter" className="text-sm font-medium leading-none">
              Filter by Date Range
            </label>
          </div>

          {!useDateFilter && (
            <>
              <div className="w-px h-4 bg-border mx-2" />
              <span className="text-xs font-medium">Show latest trades:</span>
              <Slider
                min={1}
                max={trades.length}
                value={[displayCount]}
                onValueChange={([value]) => setDisplayCount(Math.max(1, value))}
                className="w-48 py-0"
              />
              <Input
                type="number"
                value={displayCount}
                onChange={(e) => setDisplayCount(Math.max(1, Number(e.target.value)))}
                className="w-16 h-6 text-xs"
                min={1}
                max={trades.length}
              />
              <span className="text-xs text-muted-foreground">of {trades.length} trades</span>
            </>
          )}
        </div>

        {useDateFilter && (
          <div className="mt-2 flex items-center space-x-4">
            <div className="flex items-center gap-2">
              <label htmlFor="trade-grader-start-date" className="text-sm font-medium">From:</label>
              <input
                id="trade-grader-start-date"
                type="month"
                className="flex h-8 w-40 rounded-md border border-input bg-background px-2 py-1 text-sm"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div className="flex items-center gap-2">
              <label htmlFor="trade-grader-end-date" className="text-sm font-medium">To:</label>
              <input
                id="trade-grader-end-date"
                type="month"
                className="flex h-8 w-40 rounded-md border border-input bg-background px-2 py-1 text-sm"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
            <span className="text-xs text-muted-foreground">Showing {sortedTrades.length} trades</span>
          </div>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-muted">
              <th className="border border-border px-4 py-2 text-left">#</th>
              <th className="border border-border px-4 py-2 text-left">Ticker</th>
              <th className="border border-border px-4 py-2 text-left">Entry Date</th>
              <th className="border border-border px-4 py-2 text-left">Exit Date</th>
              {metrics.map(metric => (
                <th key={metric.id} className="border border-border px-4 py-2 text-left">
                  {metric.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tradesToRender.map(trade => (
              <React.Fragment key={trade.ID}>
                <tr className="hover:bg-muted">
                  <td className="border border-border px-4 py-2 text-muted-foreground">
                    {tradeIndexMap.get(trade.ID) ?? ''}
                  </td>
                  {/* Clickable cells */}
                  <td
                    className="border border-border px-4 py-2 font-medium cursor-pointer"
                    onClick={() => toggleTradeDetails(trade.ID)}
                  >
                    {trade.Ticker}
                  </td>
                  <td
                    className="border border-border px-4 py-2 cursor-pointer"
                    onClick={() => toggleTradeDetails(trade.ID)}
                  >
                    {trade.Entry_Date}
                  </td>
                  <td
                    className="border border-border px-4 py-2 cursor-pointer"
                    onClick={() => toggleTradeDetails(trade.ID)}
                  >
                    {trade.Exit_Date}
                  </td>

                  {/* Non-clickable grading cells */}
                  {metrics.map(metric => (
                    <td key={metric.id} className="border border-border px-4 py-2">
                      <div className="space-y-1">
                        {metric.options.map(option => (
                          <label key={option.id} className="flex items-center">
                            <input
                              type="checkbox" /* Using checkbox to allow unchecking by clicking again */
                              value={option.id}
                              checked={getGradeForTrade(trade.ID, metric.id) === option.id.toString()}
                              onChange={() => toggleGrade(trade.ID, metric.id, option.id)}
                              className="mr-2"
                            />
                            <span className="text-sm">{option.name}</span>
                          </label>
                        ))}
                      </div>
                    </td>
                  ))}
                </tr>

                {/* Dropdown row */}
                {expandedTradeId === trade.ID && (
                  <tr>
                    <td colSpan={4 + metrics.length} className="p-2 bg-muted/30">
                      <TradeCaseDetails
                        trade={trade}
                        ref={(el) => { tradeDropZoneRefs.current[trade.ID] = el; }}
                        onAnalysesChanged={(tId, analyses) => {
                          analysesByTradeRef.current[tId] = {
                            ids: analyses.map(a => a.id),
                            images: analyses.filter(a => !!a.image).map(a => a.image as string)
                          };
                          if (!(tId in currentImageIndexRef.current)) {
                            currentImageIndexRef.current[tId] = 0;
                          } else {
                            const cur = currentImageIndexRef.current[tId];
                            const data = analysesByTradeRef.current[tId];
                            if (cur >= data.images.length) currentImageIndexRef.current[tId] = Math.max(0, data.images.length - 1);
                          }
                        }}
                        onRequestFullscreen={(imgIdx) => {
                          currentImageIndexRef.current[trade.ID] = imgIdx;
                          openFullscreenForTrade(trade.ID, imgIdx);
                        }}
                      />
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {/* Quiet auto-save: no manual save warning banner */}
      {fullscreenTradeId !== null && fullscreenImageUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/95 flex flex-col items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
        >
          <img
            src={fullscreenImageUrl}
            alt="analysis fullscreen"
            className="max-w-[100vw] max-h-[100vh] object-contain select-none"
            draggable={false}
            onClick={closeFullscreen}
          />
          <div className="absolute top-3 left-4 text-xs text-white/70 space-x-4">
            <span className="hidden sm:inline">Esc / Click: Close</span>
            <span>← → Navigate</span>
          </div>
          <button
            onClick={() => navigateFullscreen(-1)}
            className="absolute left-2 md:left-4 top-1/2 -translate-y-1/2 bg-white/10 hover:bg-white/20 text-white rounded-full p-3"
            aria-label="Previous image"
          >
            ‹
          </button>
          <button
            onClick={() => navigateFullscreen(1)}
            className="absolute right-2 md:right-4 top-1/2 -translate-y-1/2 bg-white/10 hover:bg-white/20 text-white rounded-full p-3"
            aria-label="Next image"
          >
            ›
          </button>
          <button
            onClick={closeFullscreen}
            className="absolute top-2 right-2 md:top-4 md:right-4 bg-white/10 hover:bg-white/20 text-white rounded-full px-3 py-1 text-sm"
            aria-label="Close fullscreen"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
};


export default TradeGrader;