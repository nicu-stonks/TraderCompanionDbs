import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import {
  createChart,
  createSeriesMarkers,
  BarSeries,
  HistogramSeries,
  LineSeries,
  ColorType,
  CrosshairMode,
  PriceScaleMode,
  type IChartApi,
  type LogicalRange,
  type Time,
  type MouseEventParams,
  type BarData,
  type HistogramData,
  type LineData,
  type SeriesMarker,
} from 'lightweight-charts';
import { Loader2, Settings, X } from 'lucide-react';
import { ohlcvAPI, type OHLCVBar, type OhlcvTimeframe } from '@/TradeStatisticsPage/services/ohlcvAPI';
import type { MonAlertTrade, TradeViolationsResult, ViolationItem } from '../types';
import {
  createTrade,
  computeViolations,
  deleteTrade,
  computeSessionViolations,
  fetchTrades,
  fetchChartSettings,
  updateChartSettings,
  type SmaChartSetting,
  type SmaSource,
} from '../api';

interface Props {
  trade: MonAlertTrade;
  initialResult: TradeViolationsResult | null;
  onClose: () => void;
}

interface LegendData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  change: number;
  changePct: number;
}

type TradeMarker = SeriesMarker<Time>;

type MarkerSeriesHandle = {
  setMarkers?: (markers: TradeMarker[]) => void;
};

type MarkerController = {
  setMarkers: (markers: TradeMarker[]) => void;
};

type UiSmaSetting = SmaChartSetting & {
  id: string;
};

type UiSmaSettingsByTimeframe = Record<OhlcvTimeframe, UiSmaSetting[]>;
type SmaHexDraftsByTimeframe = Record<OhlcvTimeframe, Record<string, string>>;

type HoveredSignalTarget = {
  bucket: 'violations' | 'confirmations' | 'stats';
  type?: string;
};

type DragMeasurementState = {
  visible: boolean;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  pct: number;
};

const CHART_BG = '#131722';
const GRID_COLOR = '#1e222d';
const TEXT_COLOR = '#787b86';
const UP_COLOR = '#26a69a';
const DOWN_COLOR = '#ef5350';
const VOLUME_UP = 'rgba(38,166,154,0.3)';
const VOLUME_DOWN = 'rgba(239,83,80,0.3)';
const CROSSHAIR_COLOR = '#555';
const HIGHLIGHT_SPACER_MARKER_COLOR = 'rgba(0, 0, 0, 0)';

function makeSmaSettingId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function rgbToHex(r: number, g: number, b: number) {
  const toHex = (value: number) => Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function parseHexColor(value: string): { r: number; g: number; b: number } | null {
  const trimmed = value.trim();
  const hex = trimmed.startsWith('#') ? trimmed.slice(1) : trimmed;
  if (!/^[0-9a-fA-F]{6}$/.test(hex)) return null;
  return {
    r: parseInt(hex.slice(0, 2), 16),
    g: parseInt(hex.slice(2, 4), 16),
    b: parseInt(hex.slice(4, 6), 16),
  };
}

export function TradeDailyChartModal({ trade, initialResult, onClose }: Props) {
  const [activeTrade, setActiveTrade] = useState<MonAlertTrade>(trade);
  const [chartTimeframe, setChartTimeframe] = useState<OhlcvTimeframe>('daily');
  const [ohlcvData, setOhlcvData] = useState<OHLCVBar[]>([]);
  const [loadingData, setLoadingData] = useState(true);
  const [dataError, setDataError] = useState<string | null>(null);

  const [sessionStartDate, setSessionStartDate] = useState(trade.start_date);
  const [manualEndDate, setManualEndDate] = useState(trade.end_date || '');
  const [useLatestEndDate, setUseLatestEndDate] = useState(trade.use_latest_end_date);
  const [hoverDate, setHoverDate] = useState<string | null>(null);

  const [sessionResult, setSessionResult] = useState<TradeViolationsResult | null>(initialResult);
  const [computing, setComputing] = useState(false);
  const [computeError, setComputeError] = useState<string | null>(null);
  const [sessionComputeBlocked, setSessionComputeBlocked] = useState(false);

  const [quickTickerInput, setQuickTickerInput] = useState('');
  const [quickTickerBusy, setQuickTickerBusy] = useState(false);
  const [quickTickerError, setQuickTickerError] = useState<string | null>(null);

  const [legend, setLegend] = useState<LegendData | null>(null);

  const [contextMenu, setContextMenu] = useState<{ visible: boolean; x: number; y: number; date: string | null }>({
    visible: false,
    x: 0,
    y: 0,
    date: null,
  });
  const [hoveredSignalTarget, setHoveredSignalTarget] = useState<HoveredSignalTarget | null>(null);
  const [pinnedSignalBucket, setPinnedSignalBucket] = useState<'violations' | 'confirmations' | null>(null);
  const [highlightMarkerGapSlots, setHighlightMarkerGapSlots] = useState(0);
  const [openOnBars, setOpenOnBars] = useState(false);
  const [showChartSettings, setShowChartSettings] = useState(false);
  const [smaSettingsByTimeframe, setSmaSettingsByTimeframe] = useState<UiSmaSettingsByTimeframe>({ daily: [], weekly: [] });
  const [smaHexDraftsByTimeframe, setSmaHexDraftsByTimeframe] = useState<SmaHexDraftsByTimeframe>({ daily: {}, weekly: {} });
  const [savingSmaSettings, setSavingSmaSettings] = useState(false);
  const [hasUnsavedSettings, setHasUnsavedSettings] = useState(false);
  const [dragMeasurement, setDragMeasurement] = useState<DragMeasurementState>({
    visible: false,
    startX: 0,
    startY: 0,
    endX: 0,
    endY: 0,
    pct: 0,
  });

  const chartHostRef = useRef<HTMLDivElement | null>(null);
  const chartApiRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<Parameters<IChartApi['removeSeries']>[0] | null>(null);
  const smaSeriesRefs = useRef<Array<Parameters<IChartApi['removeSeries']>[0]>>([]);
  const markerControllerRef = useRef<MarkerController | null>(null);
  const latestChartMarkersRef = useRef<TradeMarker[]>([]);
  const visibleRangeRef = useRef<LogicalRange | null>(null);
  const pendingViewportDatesRef = useRef<{ from: string; to: string } | null>(null);
  const viewportAnchorDatesRef = useRef<{ from: string; to: string } | null>(null);
  const suppressViewportAnchorUpdateRef = useRef(false);
  const ohlcvRequestSeqRef = useRef(0);
  const ohlcvForegroundSeqRef = useRef(0);
  const ohlcvCacheRef = useRef<Record<string, OHLCVBar[]>>({});
  const hoveredDateRef = useRef<string | null>(null);
  const pointerDownRef = useRef(false);
  const resizeRafRef = useRef<number | null>(null);
  const requestSeqRef = useRef(0);
  const sessionComputeAbortRef = useRef<AbortController | null>(null);
  const lastPayloadRef = useRef('');
  const quickInputRef = useRef<HTMLInputElement | null>(null);
  const temporaryTradeIdsRef = useRef<Set<number>>(new Set());
  const dragStartRef = useRef<{ x: number; y: number; price: number } | null>(null);
  const clickStartRef = useRef<{ x: number; y: number } | null>(null);
  const measuringRef = useRef(false);
  const forceResetViewportRef = useRef(false);

  const latestBarDate = ohlcvData.length > 0 ? ohlcvData[ohlcvData.length - 1].date : '';
  const showHeaderComputing = computing && !sessionResult?.trend_up_start;
  const smaSettings = useMemo(() => smaSettingsByTimeframe[chartTimeframe] || [], [smaSettingsByTimeframe, chartTimeframe]);
  const smaHexDrafts = useMemo(() => smaHexDraftsByTimeframe[chartTimeframe] || {}, [smaHexDraftsByTimeframe, chartTimeframe]);

  const effectiveEndDate = useMemo(() => {
    if (hoverDate) return hoverDate;
    if (useLatestEndDate) return latestBarDate || manualEndDate || activeTrade.end_date || activeTrade.start_date;
    return manualEndDate || latestBarDate || activeTrade.end_date || activeTrade.start_date;
  }, [hoverDate, useLatestEndDate, latestBarDate, manualEndDate, activeTrade.end_date, activeTrade.start_date]);

  const hoverSwapActive = Boolean(hoverDate && hoverDate < sessionStartDate);
  const computeStartDate = hoverSwapActive && hoverDate ? hoverDate : sessionStartDate;
  const computeEndDate = hoverSwapActive && hoverDate ? sessionStartDate : effectiveEndDate;

  const startMarkerDate = useMemo(
    () => nearestTradingDay(computeStartDate, ohlcvData) || computeStartDate,
    [computeStartDate, ohlcvData]
  );

  const endMarkerDate = useMemo(
    () => nearestTradingDay(computeEndDate, ohlcvData) || computeEndDate,
    [computeEndDate, ohlcvData]
  );

  const tradeMarkers = useMemo<TradeMarker[]>(() => {
    if (!startMarkerDate || !endMarkerDate) return [];

    return [
      {
        time: startMarkerDate as Time,
        position: 'belowBar',
        color: UP_COLOR,
        shape: 'arrowUp',
        text: 'START',
      },
      {
        time: endMarkerDate as Time,
        position: 'aboveBar',
        color: DOWN_COLOR,
        shape: 'arrowDown',
        text: 'END',
      },
    ];
  }, [startMarkerDate, endMarkerDate]);

  const effectiveSignalTarget = useMemo<HoveredSignalTarget | null>(() => {
    return hoveredSignalTarget ?? (pinnedSignalBucket ? { bucket: pinnedSignalBucket } : null);
  }, [hoveredSignalTarget, pinnedSignalBucket]);

  const hoveredSignalMarkers = useMemo<TradeMarker[]>(() => {
    if (!effectiveSignalTarget || !sessionResult) return [];

    let dates: string[] = [];

    if (effectiveSignalTarget.bucket === 'stats') {
      const normalizedStart = nearestTradingDay(computeStartDate, ohlcvData) || computeStartDate;
      const normalizedEnd = nearestTradingDay(computeEndDate, ohlcvData) || computeEndDate;
      const from = String(normalizedStart) <= String(normalizedEnd) ? String(normalizedStart) : String(normalizedEnd);
      const to = String(normalizedStart) <= String(normalizedEnd) ? String(normalizedEnd) : String(normalizedStart);

      const statDates: string[] = [];
      for (let i = 0; i < ohlcvData.length; i += 1) {
        const bar = ohlcvData[i];
        if (bar.date < from || bar.date > to) continue;

        const prevClose = i > 0 ? ohlcvData[i - 1].close : bar.close;
        const isUpDay = bar.close > prevClose;
        const isDownDay = bar.close < prevClose;

        const range = bar.high - bar.low;
        const isGoodClose = range === 0 ? true : bar.close >= bar.high - 0.5 * range;
        const isBadClose = !isGoodClose;

        if (
          (effectiveSignalTarget.type === 'days_up' && isUpDay) ||
          (effectiveSignalTarget.type === 'days_down' && isDownDay) ||
          (effectiveSignalTarget.type === 'days_up_down' && (isUpDay || isDownDay)) ||
          (effectiveSignalTarget.type === 'good_closes' && isGoodClose) ||
          (effectiveSignalTarget.type === 'bad_closes' && isBadClose) ||
          (effectiveSignalTarget.type === 'good_bad_close' && (isGoodClose || isBadClose))
        ) {
          statDates.push(bar.date);
        }
      }

      dates = statDates.sort((a, b) => String(a).localeCompare(String(b)));
    } else {
      const sourceItems = effectiveSignalTarget.bucket === 'violations'
        ? sessionResult.violations
        : sessionResult.confirmations;

      const filteredItems = effectiveSignalTarget.type
        ? sourceItems.filter((item) => item.type === effectiveSignalTarget.type)
        : sourceItems;

      dates = filteredItems
        .map((item) => nearestTradingDay(item.date, ohlcvData) || item.date)
        .filter(Boolean)
        .sort((a, b) => String(a).localeCompare(String(b)));
    }

    const markers: TradeMarker[] = dates.map((d) => ({
      time: d as Time,
      position: 'belowBar' as const,
      color: '#ffffff',
      shape: 'arrowUp' as const,
      text: '',
    }));

    return addMarkerSpacers(markers, highlightMarkerGapSlots, HIGHLIGHT_SPACER_MARKER_COLOR);
  }, [effectiveSignalTarget, sessionResult, ohlcvData, highlightMarkerGapSlots, computeStartDate, computeEndDate]);

  const chartMarkers = useMemo<TradeMarker[]>(() => {
    return [...tradeMarkers, ...hoveredSignalMarkers];
  }, [tradeMarkers, hoveredSignalMarkers]);

  const ohlcvSignature = useMemo(() => buildOhlcvSignature(ohlcvData), [ohlcvData]);

  const switchChartTimeframe = useCallback((nextTimeframe: OhlcvTimeframe) => {
    if (nextTimeframe === chartTimeframe) return;

    const chart = chartApiRef.current;
    const currentRange = chart?.timeScale().getVisibleLogicalRange() ?? null;
    const viewportDates = viewportAnchorDatesRef.current ?? logicalRangeToDateWindow(currentRange, ohlcvData);

    pendingViewportDatesRef.current = viewportDates;
    forceResetViewportRef.current = !viewportDates;
    setChartTimeframe(nextTimeframe);
  }, [chartTimeframe, ohlcvData]);

  useEffect(() => {
    latestChartMarkersRef.current = chartMarkers;
  }, [chartMarkers]);

  const fetchOhlcv = useCallback(async (options?: { silent?: boolean }) => {
    const silent = options?.silent ?? false;
    const seq = ++ohlcvRequestSeqRef.current;
    const fetchTicker = activeTrade.ticker;
    const fetchTimeframe = chartTimeframe;
    const hasFixedEndDate = Boolean(activeTrade.end_date);
    const cacheKey = `${fetchTicker}:${fetchTimeframe}`;
    const cachedRows = ohlcvCacheRef.current[cacheKey];
    const applyRows = (rowsToApply: OHLCVBar[]) => {
      setOhlcvData((prev) => (ohlcvRowsEqual(prev, rowsToApply) ? prev : rowsToApply));
      if (rowsToApply.length > 0) {
        const last = rowsToApply[rowsToApply.length - 1];
        const hoveredIndex = hoveredDateRef.current
          ? rowsToApply.findIndex((row) => row.date === hoveredDateRef.current)
          : -1;
        setLegend(buildLegendFromIndex(rowsToApply, hoveredIndex >= 0 ? hoveredIndex : rowsToApply.length - 1));
        if (!hasFixedEndDate) {
          setManualEndDate((prev) => prev || last.date);
        }
      } else {
        setLegend(null);
      }
    };

    if (!silent && cachedRows && cachedRows.length > 0) {
      applyRows(cachedRows);
      setLoadingData(false);
    } else if (!silent) {
      ohlcvForegroundSeqRef.current = seq;
      setLoadingData(true);
    }
    setDataError(null);
    try {
      let rows: OHLCVBar[] = [];

      if (fetchTimeframe === 'weekly') {
        try {
          const weeklyResp = await ohlcvAPI.getHistoricalWeeklyData(fetchTicker);
          rows = weeklyResp.data.data || [];
        } catch (weeklyErr) {
          // Weekly bars are not persisted for all symbols yet; derive weekly candles from daily as fallback.
          if (extractHttpStatus(weeklyErr) === 404) {
            const dailyResp = await ohlcvAPI.getHistoricalData(fetchTicker);
            rows = aggregateDailyBarsToWeekly(dailyResp.data.data || []);
          } else {
            throw weeklyErr;
          }
        }
      } else {
        const resp = await ohlcvAPI.getHistoricalData(fetchTicker);
        rows = resp.data.data || [];
      }

      if (seq !== ohlcvRequestSeqRef.current) {
        return;
      }

      ohlcvCacheRef.current[cacheKey] = rows;
      applyRows(rows);
    } catch (err) {
      if (seq !== ohlcvRequestSeqRef.current) {
        return;
      }
      if (cachedRows && cachedRows.length > 0) {
        return;
      }
      console.error(err);
      setDataError(`Failed to load ${fetchTicker} ${fetchTimeframe} chart data.`);
    } finally {
      if (!silent && seq === ohlcvForegroundSeqRef.current) {
        setLoadingData(false);
      }
    }
  }, [activeTrade.ticker, activeTrade.end_date, chartTimeframe]);

  useEffect(() => {
    fetchOhlcv();

    const intervalId = window.setInterval(() => {
      fetchOhlcv({ silent: true });
    }, 5000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [fetchOhlcv]);

  useEffect(() => {
    let cancelled = false;

    const prefetchOtherTimeframe = async () => {
      const otherTimeframe: OhlcvTimeframe = chartTimeframe === 'daily' ? 'weekly' : 'daily';
      const cacheKey = `${activeTrade.ticker}:${otherTimeframe}`;
      if (ohlcvCacheRef.current[cacheKey]?.length) {
        return;
      }

      try {
        let rows: OHLCVBar[] = [];
        if (otherTimeframe === 'weekly') {
          try {
            const weeklyResp = await ohlcvAPI.getHistoricalWeeklyData(activeTrade.ticker);
            rows = weeklyResp.data.data || [];
          } catch (weeklyErr) {
            if (extractHttpStatus(weeklyErr) === 404) {
              const dailyResp = await ohlcvAPI.getHistoricalData(activeTrade.ticker);
              rows = aggregateDailyBarsToWeekly(dailyResp.data.data || []);
            } else {
              throw weeklyErr;
            }
          }
        } else {
          const resp = await ohlcvAPI.getHistoricalData(activeTrade.ticker);
          rows = resp.data.data || [];
        }

        if (!cancelled) {
          ohlcvCacheRef.current[cacheKey] = rows;
        }
      } catch (err) {
        if (!cancelled) {
          console.error(`Failed to prefetch ${otherTimeframe} chart data for ${activeTrade.ticker}:`, err);
        }
      }
    };

    void prefetchOtherTimeframe();

    return () => {
      cancelled = true;
    };
  }, [activeTrade.ticker, chartTimeframe]);

  useEffect(() => {
    lastPayloadRef.current = '';
  }, [ohlcvSignature]);

  useEffect(() => {
    visibleRangeRef.current = null;
    pendingViewportDatesRef.current = null;
    viewportAnchorDatesRef.current = null;
    suppressViewportAnchorUpdateRef.current = false;
    forceResetViewportRef.current = true;
    // Invalidate any in-flight compute response from previous ticker to avoid transient stale errors.
    requestSeqRef.current += 1;
    hoveredDateRef.current = null;
    setHoverDate(null);
    setComputeError(null);
    setSessionComputeBlocked(false);
    setDataError(null);
    setDragMeasurement((prev) => (prev.visible ? { ...prev, visible: false } : prev));
  }, [activeTrade.id]);

  useEffect(() => {
    hoveredDateRef.current = null;
    setHoverDate(null);
  }, [chartTimeframe]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const resp = await fetchChartSettings();
        if (cancelled) return;
        setHighlightMarkerGapSlots(Math.max(0, Math.min(15, Number(resp.highlight_marker_gap ?? 0))));
        setOpenOnBars(Boolean(resp.open_on_bars));
        const incoming = (resp.sma_settings || []).map((s) => ({
          id: makeSmaSettingId(),
          length: s.length,
          r: s.r,
          g: s.g,
          b: s.b,
          opacity: s.opacity,
          thickness: s.thickness,
          enabled: Boolean(s.enabled),
          source: (s.source || 'close') as SmaSource,
        }));
        const incomingDaily = (resp.daily_sma_settings || resp.sma_settings || []).map((s) => ({
          id: makeSmaSettingId(),
          length: s.length,
          r: s.r,
          g: s.g,
          b: s.b,
          opacity: s.opacity,
          thickness: s.thickness,
          enabled: Boolean(s.enabled),
          source: (s.source || 'close') as SmaSource,
        }));
        const incomingWeekly = (resp.weekly_sma_settings || []).map((s) => ({
          id: makeSmaSettingId(),
          length: s.length,
          r: s.r,
          g: s.g,
          b: s.b,
          opacity: s.opacity,
          thickness: s.thickness,
          enabled: Boolean(s.enabled),
          source: (s.source || 'close') as SmaSource,
        }));
        setSmaSettingsByTimeframe({
          daily: incomingDaily.length > 0 ? incomingDaily : incoming,
          weekly: incomingWeekly.length > 0 ? incomingWeekly : incoming,
        });
        setHasUnsavedSettings(false);
      } catch (err) {
        console.error('Failed to load chart SMA settings:', err);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setSmaHexDraftsByTimeframe((prev) => {
      const next: SmaHexDraftsByTimeframe = {
        daily: {},
        weekly: {},
      };
      (['daily', 'weekly'] as const).forEach((timeframe) => {
        smaSettingsByTimeframe[timeframe].forEach((s) => {
          next[timeframe][s.id] = rgbToHex(s.r, s.g, s.b);
        });
      });
      const unchanged =
        Object.keys(next.daily).length === Object.keys(prev.daily).length &&
        Object.keys(next.weekly).length === Object.keys(prev.weekly).length &&
        Object.keys(next.daily).every((k) => prev.daily[k] === next.daily[k]) &&
        Object.keys(next.weekly).every((k) => prev.weekly[k] === next.weekly[k]);
      return unchanged ? prev : next;
    });
  }, [smaSettingsByTimeframe]);

  const persistChartSettings = useCallback(async (nextByTimeframe: UiSmaSettingsByTimeframe, nextHighlightGap?: number, nextOpenOnBars?: boolean) => {
    setSavingSmaSettings(true);
    let ok = true;
    try {
      await updateChartSettings(
        nextByTimeframe.daily.map((item) => ({
          length: item.length,
          r: item.r,
          g: item.g,
          b: item.b,
          opacity: item.opacity,
          thickness: item.thickness,
          enabled: item.enabled,
          source: item.source,
        })),
        nextByTimeframe.weekly.map((item) => ({
          length: item.length,
          r: item.r,
          g: item.g,
          b: item.b,
          opacity: item.opacity,
          thickness: item.thickness,
          enabled: item.enabled,
          source: item.source,
        })),
        Math.max(0, Math.min(15, Number(nextHighlightGap ?? highlightMarkerGapSlots))),
        Boolean(nextOpenOnBars ?? openOnBars)
      );
    } catch (err) {
      ok = false;
      console.error('Failed to save chart SMA settings:', err);
    } finally {
      setSavingSmaSettings(false);
    }
    return ok;
  }, [highlightMarkerGapSlots, openOnBars]);

  const setHighlightGap = useCallback((nextGap: number) => {
    const normalized = Math.max(0, Math.min(15, Number(nextGap)));
    setHighlightMarkerGapSlots(normalized);
    setHasUnsavedSettings(true);
  }, []);

  const setSmaSetting = useCallback((id: string, patch: Partial<SmaChartSetting>) => {
    setSmaSettingsByTimeframe((prev) => {
      const current = prev[chartTimeframe] || [];
      const next = current.map((item) => {
        if (item.id !== id) return item;
        return {
          ...item,
          ...patch,
          length: Math.max(2, Math.min(400, Number(patch.length ?? item.length))),
          r: Math.max(0, Math.min(255, Number(patch.r ?? item.r))),
          g: Math.max(0, Math.min(255, Number(patch.g ?? item.g))),
          b: Math.max(0, Math.min(255, Number(patch.b ?? item.b))),
          opacity: Math.max(0.05, Math.min(1, Number(patch.opacity ?? item.opacity))),
          thickness: Math.max(0.1, Math.min(8, Math.round(Number(patch.thickness ?? item.thickness) * 10) / 10)),
          enabled: patch.enabled ?? item.enabled,
          source: (patch.source ?? item.source) as SmaSource,
        };
      });
      setHasUnsavedSettings(true);
      return {
        ...prev,
        [chartTimeframe]: next,
      };
    });
  }, [chartTimeframe]);

  const commitSmaHex = useCallback((id: string) => {
    const draft = smaHexDrafts[id] ?? '';
    const parsed = parseHexColor(draft);
    if (!parsed) {
      const existing = smaSettings.find((item) => item.id === id);
      if (!existing) return;
      setSmaHexDraftsByTimeframe((prev) => ({
        ...prev,
        [chartTimeframe]: {
          ...prev[chartTimeframe],
          [id]: rgbToHex(existing.r, existing.g, existing.b),
        },
      }));
      return;
    }
    setSmaSetting(id, { r: parsed.r, g: parsed.g, b: parsed.b });
  }, [chartTimeframe, smaHexDrafts, smaSettings, setSmaSetting]);

  const addSmaSetting = useCallback(() => {
    setSmaSettingsByTimeframe((prev) => {
      const current = prev[chartTimeframe] || [];
      const next: UiSmaSetting[] = [
        ...current,
        {
          id: makeSmaSettingId(),
          length: 50,
          r: 255,
          g: 255,
          b: 255,
          opacity: 0.8,
          thickness: 1,
          enabled: true,
          source: 'close',
        },
      ];
      setHasUnsavedSettings(true);
      return {
        ...prev,
        [chartTimeframe]: next,
      };
    });
  }, [chartTimeframe]);

  const removeSmaSetting = useCallback((id: string) => {
    setSmaSettingsByTimeframe((prev) => {
      const next = (prev[chartTimeframe] || []).filter((item) => item.id !== id);
      setHasUnsavedSettings(true);
      return {
        ...prev,
        [chartTimeframe]: next,
      };
    });
  }, [chartTimeframe]);

  const saveChartSettings = useCallback(async () => {
    const nextByTimeframe = (['daily', 'weekly'] as const).reduce((acc, timeframe) => {
      acc[timeframe] = (smaSettingsByTimeframe[timeframe] || []).map((item) => {
        const draft = smaHexDraftsByTimeframe[timeframe]?.[item.id];
        if (!draft) return item;
        const parsed = parseHexColor(draft);
        if (!parsed) return item;
        return {
          ...item,
          r: parsed.r,
          g: parsed.g,
          b: parsed.b,
        };
      });
      return acc;
    }, { daily: [] as UiSmaSetting[], weekly: [] as UiSmaSetting[] });

    setSmaSettingsByTimeframe(nextByTimeframe);
    const ok = await persistChartSettings(nextByTimeframe, highlightMarkerGapSlots, openOnBars);
    if (ok) {
      setSmaHexDraftsByTimeframe((prev) => {
        const next: SmaHexDraftsByTimeframe = {
          daily: { ...prev.daily },
          weekly: { ...prev.weekly },
        };
        (['daily', 'weekly'] as const).forEach((timeframe) => {
          nextByTimeframe[timeframe].forEach((s) => {
            next[timeframe][s.id] = rgbToHex(s.r, s.g, s.b);
          });
        });
        return next;
      });
      setHasUnsavedSettings(false);
    }
  }, [smaSettingsByTimeframe, smaHexDraftsByTimeframe, persistChartSettings, highlightMarkerGapSlots, openOnBars]);

  const runSessionCompute = useCallback(
    async (
      payload: { start_date: string; end_date: string; use_latest_end_date: boolean },
      options?: { silent?: boolean }
    ) => {
      if (sessionComputeBlocked) {
        return;
      }
      const silent = options?.silent ?? false;
      const payloadKey = JSON.stringify(payload);
      if (payloadKey === lastPayloadRef.current) {
        return;
      }
      lastPayloadRef.current = payloadKey;
      sessionComputeAbortRef.current?.abort();
      const controller = new AbortController();
      sessionComputeAbortRef.current = controller;
      if (!silent) {
        setComputing(true);
      }
      setComputeError(null);
      const seq = ++requestSeqRef.current;
      try {
        const next = await computeSessionViolations(activeTrade.id, payload, { signal: controller.signal });
        if (seq !== requestSeqRef.current) return;
        setSessionResult(next);
      } catch (err) {
        if ((err as { name?: string } | null)?.name === 'AbortError') {
          return;
        }
        if (seq !== requestSeqRef.current) return;
        const status = (err as { status?: number } | null)?.status;
        const message = (err as { message?: string } | null)?.message || '';
        if (status === 404 && /trade not found/i.test(message)) {
          setSessionComputeBlocked(true);
          setComputeError('This trade no longer exists in monitor. Close this chart and reopen an existing trade card.');
          return;
        }
        console.error(err);
        setComputeError('Could not compute violations for current chart range.');
      } finally {
        if (sessionComputeAbortRef.current === controller) {
          sessionComputeAbortRef.current = null;
        }
        if (!silent && seq === requestSeqRef.current) {
          setComputing(false);
        }
      }
    },
    [activeTrade.id, sessionComputeBlocked]
  );

  useEffect(() => {
    if (!computeEndDate || !computeStartDate || ohlcvData.length === 0) return;

    const timeout = window.setTimeout(() => {
      runSessionCompute({
        start_date: computeStartDate,
        end_date: computeEndDate,
        use_latest_end_date: Boolean(useLatestEndDate && !hoverDate && !hoverSwapActive),
      }, { silent: Boolean(hoverDate) });
    }, hoverDate ? 250 : 0);

    return () => window.clearTimeout(timeout);
  }, [computeEndDate, computeStartDate, hoverDate, hoverSwapActive, useLatestEndDate, ohlcvData.length, ohlcvSignature, runSessionCompute]);

  useEffect(() => {
    return () => {
      sessionComputeAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    const onGlobalClick = () => {
      setContextMenu((prev) => (prev.visible ? { ...prev, visible: false } : prev));
      setShowChartSettings(false);
    };
    window.addEventListener('click', onGlobalClick);
    const onGlobalMouseUp = () => {
      pointerDownRef.current = false;
    };
    window.addEventListener('mouseup', onGlobalMouseUp);
    return () => {
      window.removeEventListener('click', onGlobalClick);
      window.removeEventListener('mouseup', onGlobalMouseUp);
    };
  }, []);

  useEffect(() => {
    if (!chartHostRef.current || ohlcvData.length === 0) return;

    if (chartApiRef.current) {
      chartApiRef.current.remove();
      chartApiRef.current = null;
    }

    const chart = createChart(chartHostRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: CHART_BG },
        textColor: TEXT_COLOR,
      },
      grid: {
        vertLines: { color: GRID_COLOR },
        horzLines: { color: GRID_COLOR },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: CROSSHAIR_COLOR,
          width: 1,
          style: 3,
          labelBackgroundColor: '#2B2B43',
        },
        horzLine: {
          color: CROSSHAIR_COLOR,
          width: 1,
          style: 3,
          labelBackgroundColor: '#2B2B43',
        },
      },
      rightPriceScale: {
        borderColor: GRID_COLOR,
        mode: PriceScaleMode.Logarithmic,
        scaleMargins: { top: 0.06, bottom: 0.2 },
      },
      timeScale: {
        borderColor: GRID_COLOR,
        timeVisible: false,
        rightOffset: 6,
        barSpacing: 7,
      },
    });
    chartApiRef.current = chart;

    const candleSeries = chart.addSeries(BarSeries, {
      upColor: UP_COLOR,
      downColor: DOWN_COLOR,
      openVisible: openOnBars,
      thinBars: false,
    });
    candleSeriesRef.current = candleSeries;

    const candleData: BarData<Time>[] = ohlcvData.map((bar) => ({
      time: bar.date as Time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));
    candleSeries.setData(candleData);

    const markerSeries = candleSeries as typeof candleSeries & MarkerSeriesHandle;
    if (typeof markerSeries.setMarkers === 'function') {
      markerControllerRef.current = {
        setMarkers: (markers: TradeMarker[]) => markerSeries.setMarkers?.(markers),
      };
    } else {
      const markersPrimitive = createSeriesMarkers(candleSeries, []);
      const primitiveWithSetter = markersPrimitive as { setMarkers?: (markers: TradeMarker[]) => void };
      if (typeof primitiveWithSetter.setMarkers === 'function') {
        markerControllerRef.current = {
          setMarkers: (markers: TradeMarker[]) => primitiveWithSetter.setMarkers?.(markers),
        };
      } else {
        // Last-resort fallback for older marker APIs.
        markerControllerRef.current = {
          setMarkers: (markers: TradeMarker[]) => {
            createSeriesMarkers(candleSeries, markers);
          },
        };
      }
    }

    // Always paint current arrows immediately on chart init/re-init.
    markerControllerRef.current?.setMarkers(latestChartMarkersRef.current);
    requestAnimationFrame(() => {
      markerControllerRef.current?.setMarkers(latestChartMarkersRef.current);
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });

    const volumeData: HistogramData<Time>[] = ohlcvData.map((bar) => {
      return {
        time: bar.date as Time,
        value: bar.volume,
        color: bar.close >= bar.open ? VOLUME_UP : VOLUME_DOWN,
      };
    });
    volumeSeries.setData(volumeData);

    const onCrosshairMove = (param: MouseEventParams<Time>) => {
      if (
        !param.time ||
        !param.seriesData ||
        !param.point ||
        param.point.x < 0 ||
        param.point.x > (chartHostRef.current?.clientWidth || 0) ||
        param.point.y < 0 ||
        param.point.y > (chartHostRef.current?.clientHeight || 0)
      ) {
        hoveredDateRef.current = null;
        setHoverDate(null);
        setLegend(buildLegendFromIndex(ohlcvData, ohlcvData.length - 1));
        return;
      }

      const candle = param.seriesData.get(candleSeries) as BarData<Time> | undefined;
      const vol = param.seriesData.get(volumeSeries) as HistogramData<Time> | undefined;
      if (!candle) return;

      const dateStr = timeToString(param.time);

      // While the user is dragging/panning, avoid hover-driven compute churn.
      if (pointerDownRef.current) {
        return;
      }

      hoveredDateRef.current = dateStr;
      setHoverDate(dateStr);

      const idx = ohlcvData.findIndex((b) => b.date === dateStr);
      const prevClose = idx > 0 ? ohlcvData[idx - 1].close : candle.open;
      const change = candle.close - prevClose;
      const changePct = prevClose !== 0 ? (change / prevClose) * 100 : 0;

      setLegend({
        date: dateStr,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: vol?.value ?? 0,
        change,
        changePct,
      });
    };

    chart.subscribeCrosshairMove(onCrosshairMove);

    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      visibleRangeRef.current = range;
      if (suppressViewportAnchorUpdateRef.current) return;
      const nextAnchor = logicalRangeToDateWindow(range ?? null, ohlcvData);
      if (nextAnchor) {
        viewportAnchorDatesRef.current = nextAnchor;
      }
    });

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.target !== chartHostRef.current || !chartApiRef.current) continue;
        const { width, height } = entry.contentRect;
        if (resizeRafRef.current) {
          cancelAnimationFrame(resizeRafRef.current);
        }
        resizeRafRef.current = requestAnimationFrame(() => {
          chartApiRef.current?.applyOptions({ width, height });
        });
      }
    });

    resizeObserver.observe(chartHostRef.current);

    const pendingViewportDates = pendingViewportDatesRef.current;
    const pendingLogicalRange = pendingViewportDates
      ? dateWindowToLogicalRange(pendingViewportDates.from, pendingViewportDates.to, ohlcvData)
      : null;

    if (pendingLogicalRange) {
      suppressViewportAnchorUpdateRef.current = true;
      chart.timeScale().setVisibleLogicalRange(pendingLogicalRange);
      visibleRangeRef.current = pendingLogicalRange;
      viewportAnchorDatesRef.current = pendingViewportDates;
      pendingViewportDatesRef.current = null;
      forceResetViewportRef.current = false;
      requestAnimationFrame(() => {
        suppressViewportAnchorUpdateRef.current = false;
      });
    } else if (visibleRangeRef.current && !forceResetViewportRef.current) {
      suppressViewportAnchorUpdateRef.current = true;
      chart.timeScale().setVisibleLogicalRange(visibleRangeRef.current);
      requestAnimationFrame(() => {
        suppressViewportAnchorUpdateRef.current = false;
      });
    } else {
      const tradingBarsInTenMonths = 210;
      const rightPaddingBars = 3;
      const right = ohlcvData.length - 0.5 + rightPaddingBars;
      const left = Math.max(-0.5, right - tradingBarsInTenMonths);
      if (ohlcvData.length > 0) {
        suppressViewportAnchorUpdateRef.current = true;
        chart.timeScale().setVisibleLogicalRange({ from: left, to: right });
        const defaultAnchor = logicalRangeToDateWindow({ from: left as LogicalRange['from'], to: right as LogicalRange['to'] }, ohlcvData);
        if (defaultAnchor) {
          viewportAnchorDatesRef.current = defaultAnchor;
        }
        requestAnimationFrame(() => {
          suppressViewportAnchorUpdateRef.current = false;
        });
      } else {
        chart.timeScale().fitContent();
        suppressViewportAnchorUpdateRef.current = false;
      }
      forceResetViewportRef.current = false;
    }

    return () => {
      chart.unsubscribeCrosshairMove(onCrosshairMove);
      if (!forceResetViewportRef.current) {
        visibleRangeRef.current = chart.timeScale().getVisibleLogicalRange();
      }
      resizeObserver.disconnect();
      markerControllerRef.current = null;
      candleSeriesRef.current = null;
      if (resizeRafRef.current) {
        cancelAnimationFrame(resizeRafRef.current);
        resizeRafRef.current = null;
      }
      chart.remove();
      chartApiRef.current = null;
    };
  }, [ohlcvData, openOnBars]);

  useEffect(() => {
    const chart = chartApiRef.current;
    if (!chart || ohlcvData.length === 0) return;

    // Remove previous SMA overlays before re-adding with updated settings.
    smaSeriesRefs.current.forEach((series) => {
      try {
        chart.removeSeries(series);
      } catch {
        // noop
      }
    });
    smaSeriesRefs.current = [];

    smaSettings
      .filter((s) => s.enabled)
      .forEach((s) => {
        const sourceSeries: Array<{ time: Time; close: number }> = ohlcvData.map((row) => ({
          time: row.date as Time,
          close: row[s.source],
        }));
        const smaData = computeSma(sourceSeries, s.length);
        const line = chart.addSeries(LineSeries, {
          color: `rgba(${s.r}, ${s.g}, ${s.b}, ${s.opacity})`,
          lineWidth: (Math.max(0.1, Math.min(8, s.thickness)) as unknown as 1 | 2 | 3 | 4),
          crosshairMarkerVisible: false,
          lastValueVisible: false,
          priceLineVisible: false,
          autoscaleInfoProvider: () => null,
        });
        line.setData(smaData);
        smaSeriesRefs.current.push(line);
      });

    return () => {
      const currentChart = chartApiRef.current;
      if (!currentChart) return;
      smaSeriesRefs.current.forEach((series) => {
        try {
          currentChart.removeSeries(series);
        } catch {
          // noop
        }
      });
      smaSeriesRefs.current = [];
    };
  }, [ohlcvData, smaSettings]);

  useEffect(() => {
    const markerController = markerControllerRef.current;
    if (!markerController) return;
    const chart = chartApiRef.current;
    if (!chart) return;

    const currentRange = chart.timeScale().getVisibleLogicalRange();
    if (currentRange) {
      visibleRangeRef.current = currentRange;
    }

    markerController.setMarkers(chartMarkers);

    // Some lightweight-charts builds may delay marker repaint until next interaction.
    // Reapply on next frame to force immediate visual update after date changes.
    requestAnimationFrame(() => {
      const nextController = markerControllerRef.current;
      nextController?.setMarkers(chartMarkers);
    });

    if (visibleRangeRef.current) {
      requestAnimationFrame(() => {
        chartApiRef.current?.timeScale().setVisibleLogicalRange(visibleRangeRef.current!);
      });
    }
  }, [chartMarkers]);

  useEffect(() => {
    if (computing) return;
    const markerController = markerControllerRef.current;
    if (!markerController) return;
    markerController.setMarkers(chartMarkers);
  }, [computing, chartMarkers]);

  const groupedViolations = useMemo(() => groupByType(sessionResult?.violations || []), [sessionResult]);
  const groupedConfirmations = useMemo(() => groupByType(sessionResult?.confirmations || []), [sessionResult]);
  const daysUp = sessionResult?.info?.days_up ?? 0;
  const daysDown = sessionResult?.info?.days_down ?? 0;
  const goodCloses = sessionResult?.info?.good_closes ?? 0;
  const badCloses = sessionResult?.info?.bad_closes ?? 0;
  const daysTotal = daysUp + daysDown;
  const goodCloseTotal = goodCloses + badCloses;
  const daysUpRatio = daysTotal > 0 ? daysUp / daysTotal : 0.5;
  const goodCloseRatio = goodCloseTotal > 0 ? goodCloses / goodCloseTotal : 0.5;
  const daysColorClass =
    daysUpRatio > 0.5
      ? 'text-green-50 bg-green-700/75'
      : daysUpRatio < 0.5
        ? 'text-red-50 bg-red-800/70'
        : 'text-muted-foreground bg-muted';
  const goodCloseColorClass =
    goodCloseRatio > 0.5
      ? 'text-green-50 bg-green-700/75'
      : goodCloseRatio < 0.5
        ? 'text-red-50 bg-red-800/70'
        : 'text-muted-foreground bg-muted';
  const enabledSmaLegendItems = useMemo(() => {
    return smaSettings
      .filter((s) => s.enabled)
      .map((s) => ({
        id: s.id,
        label: `${s.length} SMA`,
        color: `rgba(${s.r}, ${s.g}, ${s.b}, ${Math.max(0.35, Math.min(1, s.opacity))})`,
      }));
  }, [smaSettings]);

  const applyContextAction = (action: 'start' | 'end' | 'latest') => {
    const selected = contextMenu.date;
    if (action === 'latest') {
      setUseLatestEndDate(true);
      setHoverDate(null);
      setContextMenu((prev) => ({ ...prev, visible: false }));
      return;
    }

    if (!selected) {
      setContextMenu((prev) => ({ ...prev, visible: false }));
      return;
    }

    if (action === 'start') {
      setSessionStartDate(selected);
      if (effectiveEndDate && selected > effectiveEndDate) {
        setUseLatestEndDate(false);
        setManualEndDate(selected);
      }
    }

    if (action === 'end') {
      setUseLatestEndDate(false);
      setManualEndDate(selected);
      setHoverDate(null);
    }

    setContextMenu((prev) => ({ ...prev, visible: false }));
  };

  const commitHoveredDateSelection = useCallback((dateStr: string | null) => {
    if (!dateStr) return;

    if (dateStr < sessionStartDate) {
      setSessionStartDate(dateStr);
      if (effectiveEndDate && dateStr > effectiveEndDate) {
        setUseLatestEndDate(false);
        setManualEndDate(dateStr);
      }
    } else {
      setUseLatestEndDate(false);
      setManualEndDate(dateStr);
    }
    setHoverDate(null);
    hoveredDateRef.current = null;
  }, [sessionStartDate, effectiveEndDate]);

  const updateDragMeasurement = useCallback((x: number, y: number) => {
    const start = dragStartRef.current;
    const candleSeries = candleSeriesRef.current as (typeof candleSeriesRef.current & { coordinateToPrice?: (coord: number) => number | null }) | null;
    if (!start || !candleSeries?.coordinateToPrice) return;

    const endPrice = candleSeries.coordinateToPrice(y);
    if (endPrice == null || !Number.isFinite(endPrice) || start.price === 0) return;

    const pct = ((endPrice - start.price) / start.price) * 100;
    setDragMeasurement({
      visible: true,
      startX: start.x,
      startY: start.y,
      endX: x,
      endY: y,
      pct,
    });
  }, []);

  const handleChartMouseDown = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    pointerDownRef.current = true;
    setContextMenu((prev) => (prev.visible ? { ...prev, visible: false } : prev));

    const host = chartHostRef.current;
    const candleSeries = candleSeriesRef.current as (typeof candleSeriesRef.current & { coordinateToPrice?: (coord: number) => number | null }) | null;
    if (!host) {
      dragStartRef.current = null;
      clickStartRef.current = null;
      measuringRef.current = false;
      return;
    }

    const rect = host.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    clickStartRef.current = { x, y };

    // Measure only when Ctrl is held; otherwise keep default chart drag/pan.
    if (!event.ctrlKey || !candleSeries?.coordinateToPrice) {
      dragStartRef.current = null;
      measuringRef.current = false;
      setDragMeasurement((prev) => (prev.visible ? { ...prev, visible: false } : prev));
      return;
    }

    const price = candleSeries.coordinateToPrice(y);
    if (price == null || !Number.isFinite(price)) {
      dragStartRef.current = null;
      measuringRef.current = false;
      return;
    }

    measuringRef.current = true;
    chartApiRef.current?.applyOptions({
      handleScroll: false,
      handleScale: false,
    });

    dragStartRef.current = { x, y, price };
    setDragMeasurement({
      visible: false,
      startX: x,
      startY: y,
      endX: x,
      endY: y,
      pct: 0,
    });
  }, []);

  const handleChartMouseMove = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    if (!pointerDownRef.current || !measuringRef.current) return;
    const start = dragStartRef.current;
    const host = chartHostRef.current;
    if (!start || !host) return;

    const rect = host.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    updateDragMeasurement(x, y);
  }, [updateDragMeasurement]);

  const handleChartMouseUp = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;

    const clickStart = clickStartRef.current;
    const wasMeasuring = measuringRef.current;
    pointerDownRef.current = false;
    dragStartRef.current = null;
    clickStartRef.current = null;
    measuringRef.current = false;

    if (wasMeasuring) {
      chartApiRef.current?.applyOptions({
        handleScroll: true,
        handleScale: true,
      });
    }

    if (!clickStart) {
      return;
    }

    const host = chartHostRef.current;
    if (!host) return;

    const rect = host.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const dx = x - clickStart.x;
    const dy = y - clickStart.y;
    const dragDistance = Math.sqrt(dx * dx + dy * dy);

    // Treat small movement as a click-to-commit date action.
    if (dragDistance < 6) {
      const chart = chartApiRef.current;
      const hovered = hoveredDateRef.current;
      let clickedDate: string | null = hovered;

      if (!clickedDate && chart) {
        const t = chart.timeScale().coordinateToTime(x);
        if (t != null) {
          clickedDate = nearestTradingDay(timeToString(t), ohlcvData);
        }
      }
      commitHoveredDateSelection(clickedDate);
      setDragMeasurement((prev) => (prev.visible ? { ...prev, visible: false } : prev));
      return;
    }

    // Drag ruler is preview-only while mouse is held.
    setDragMeasurement((prev) => (prev.visible ? { ...prev, visible: false } : prev));
  }, [commitHoveredDateSelection, ohlcvData]);

  const measurementStyle = useMemo(() => {
    if (!dragMeasurement.visible) return null;
    const dx = dragMeasurement.endX - dragMeasurement.startX;
    const dy = dragMeasurement.endY - dragMeasurement.startY;
    const length = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);
    return {
      left: dragMeasurement.startX,
      top: dragMeasurement.startY,
      width: length,
      transform: `rotate(${angle}deg)`,
    };
  }, [dragMeasurement]);

  const closeQuickTicker = useCallback(() => {
    setQuickTickerInput('');
    setQuickTickerError(null);
    quickInputRef.current?.focus();
  }, []);

  const removeTemporaryTradeIfNeeded = useCallback(async (tradeId: number) => {
    if (!temporaryTradeIdsRef.current.has(tradeId)) return;
    try {
      await deleteTrade(tradeId);
      temporaryTradeIdsRef.current.delete(tradeId);
    } catch (err) {
      console.error(`Failed to delete temporary trade ${tradeId}:`, err);
    }
  }, []);

  const handleModalClose = useCallback(async () => {
    await removeTemporaryTradeIfNeeded(activeTrade.id);
    onClose();
  }, [activeTrade.id, onClose, removeTemporaryTradeIfNeeded]);

  const handleQuickTickerSubmit = useCallback(async () => {
    const ticker = quickTickerInput.trim().toUpperCase();
    if (!ticker) return;
    if (!/^[A-Z0-9.-]{1,10}$/.test(ticker)) {
      setQuickTickerError('Ticker must be 1-10 chars (A-Z, 0-9, ., -).');
      return;
    }

    const anchorDate = latestBarDate || new Date().toISOString().slice(0, 10);
    setQuickTickerBusy(true);
    setQuickTickerError(null);
    let createdTradeId: number | null = null;

    try {
      const allTrades = await fetchTrades();
      const matchingActiveTrades = allTrades
        .filter((item) => item.is_active && String(item.ticker || '').toUpperCase() === ticker)
        .sort((a, b) => b.id - a.id);
      const existingTrade = matchingActiveTrades[0];

      let targetTrade: MonAlertTrade;
      let newResult: TradeViolationsResult;

      if (existingTrade) {
        targetTrade = existingTrade;
        newResult = await computeViolations(existingTrade.id);
      } else {
        const newTrade = await createTrade({
          ticker,
          start_date: anchorDate,
          end_date: anchorDate,
          use_latest_end_date: false,
        });
        createdTradeId = newTrade.id;
        temporaryTradeIdsRef.current.add(newTrade.id);
        targetTrade = newTrade;
        newResult = newTrade.initial_result ?? await computeViolations(newTrade.id);
      }

      const prevTradeId = activeTrade.id;
      if (prevTradeId !== targetTrade.id) {
        await removeTemporaryTradeIfNeeded(prevTradeId);
      }

      setActiveTrade(targetTrade);
      setSessionResult(newResult);
      setSessionStartDate(anchorDate);
      setManualEndDate(anchorDate);
      setUseLatestEndDate(false);
      setHoverDate(null);
      hoveredDateRef.current = null;
      lastPayloadRef.current = '';
      closeQuickTicker();
    } catch (err) {
      console.error('Failed quick ticker switch:', err);
      if (createdTradeId != null && temporaryTradeIdsRef.current.has(createdTradeId)) {
        try {
          await deleteTrade(createdTradeId);
          temporaryTradeIdsRef.current.delete(createdTradeId);
        } catch (cleanupErr) {
          console.error(`Failed to cleanup temporary trade ${createdTradeId}:`, cleanupErr);
        }
      }
      setQuickTickerError(`Could not switch to ${ticker}.`);
    } finally {
      setQuickTickerBusy(false);
    }
  }, [quickTickerInput, latestBarDate, activeTrade.id, removeTemporaryTradeIfNeeded, closeQuickTicker]);

  useEffect(() => {
    const input = quickInputRef.current;
    if (!input) return;
    const end = input.value.length;
    input.setSelectionRange(end, end);
  }, []);

  useEffect(() => {
    const onGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      const tickerInputEl = quickInputRef.current;
      const isTickerInputTarget = Boolean(target && tickerInputEl && target === tickerInputEl);
      const isTypingTarget = Boolean(
        target?.isContentEditable ||
        tag === 'input' ||
        tag === 'textarea' ||
        tag === 'select'
      );
      if (isTypingTarget && !isTickerInputTarget) return;
      if (showChartSettings) return;

      if (event.key.length === 1 && /[a-zA-Z0-9.-]/.test(event.key)) {
        event.preventDefault();
        const nextChar = event.key.toUpperCase();
        setQuickTickerInput((prev) => `${prev}${nextChar}`);
        setQuickTickerError(null);
        const input = quickInputRef.current;
        if (input) {
          input.focus();
          requestAnimationFrame(() => {
            const end = input.value.length;
            input.setSelectionRange(end, end);
          });
        }
      }
    };

    window.addEventListener('keydown', onGlobalKeyDown);
    return () => window.removeEventListener('keydown', onGlobalKeyDown);
  }, [showChartSettings]);

  return (
    <div className="fixed inset-0 z-[100] bg-black/75 flex items-center justify-center p-2" onClick={handleModalClose}>
      <div
        className="relative rounded-lg border border-border bg-background shadow-2xl overflow-hidden"
        style={{ width: 'calc(100vw - 16px)', height: 'calc(100vh - 16px)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="h-full flex flex-col min-h-0">
          <div className="flex items-center gap-3 border-b border-border px-4 py-2">
            <div className="text-sm font-bold">{activeTrade.ticker} {chartTimeframe === 'weekly' ? 'Weekly' : 'Daily'} Chart Session</div>
            <div className="text-xs text-muted-foreground">
              Start {computeStartDate} | End {computeEndDate || '—'}
            </div>
            <div className="text-xs text-muted-foreground">
              Trend up start {sessionResult?.trend_up_start || '—'}
            </div>
            <div className="ml-auto flex items-center gap-1.5">
              <input
                ref={quickInputRef}
                type="text"
                value={quickTickerInput}
                onChange={(e) => {
                  setQuickTickerInput(e.target.value.toUpperCase());
                  setQuickTickerError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    if (!quickTickerBusy) {
                      void handleQuickTickerSubmit();
                    }
                  }
                  if (e.key === 'Escape') {
                    e.preventDefault();
                    closeQuickTicker();
                  }
                }}
                placeholder="Ticker"
                className="h-7 w-28 rounded border border-border bg-background px-2 text-xs uppercase"
                maxLength={10}
              />
              <button
                className="h-7 rounded border border-border px-2 text-xs hover:bg-muted disabled:opacity-50"
                onClick={() => {
                  if (!quickTickerBusy) {
                    void handleQuickTickerSubmit();
                  }
                }}
                disabled={quickTickerBusy}
              >
                {quickTickerBusy ? 'Switching…' : 'Go'}
              </button>
              <div className="ml-1 inline-flex h-7 overflow-hidden rounded border border-border">
                <button
                  className={`px-2 text-xs ${chartTimeframe === 'daily' ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-muted/60'}`}
                  onClick={() => switchChartTimeframe('daily')}
                  disabled={chartTimeframe === 'daily'}
                  title="Show daily chart"
                >
                  1D
                </button>
                <button
                  className={`border-l border-border px-2 text-xs ${chartTimeframe === 'weekly' ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-muted/60'}`}
                  onClick={() => switchChartTimeframe('weekly')}
                  disabled={chartTimeframe === 'weekly'}
                  title="Show weekly chart"
                >
                  1W
                </button>
              </div>
            </div>
            {showHeaderComputing && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />}
            <button className="p-1 rounded hover:bg-muted" onClick={() => void handleModalClose()} title="Close chart">
              <X className="h-4 w-4" />
            </button>
          </div>

          {quickTickerError && (
            <div className="px-4 py-1.5 text-xs text-red-400 border-b border-border">{quickTickerError}</div>
          )}

          <div className="relative flex items-center gap-2 border-b border-border px-4 py-2 text-xs">
            <span className="text-muted-foreground">Start:</span>
            <input
              type="date"
              value={sessionStartDate}
              onChange={(e) => setSessionStartDate(e.target.value)}
              className="h-7 rounded border border-border bg-background px-2"
            />
            <label className="ml-2 inline-flex items-center gap-1">
              <input
                type="checkbox"
                checked={useLatestEndDate}
                onChange={(e) => {
                  setUseLatestEndDate(e.target.checked);
                  if (e.target.checked) {
                    setHoverDate(null);
                  }
                }}
                className="h-3 w-3"
              />
              Use latest end date
            </label>
            {!useLatestEndDate && (
              <>
                <span className="text-muted-foreground ml-2">End:</span>
                <input
                  type="date"
                  value={manualEndDate}
                  onChange={(e) => {
                    setManualEndDate(e.target.value);
                    setHoverDate(null);
                  }}
                  className="h-7 rounded border border-border bg-background px-2"
                />
              </>
            )}
            <button
              className="ml-2 inline-flex items-center gap-1 rounded border border-border/60 bg-muted/20 px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
              onClick={(e) => {
                e.stopPropagation();
                setShowChartSettings((v) => !v);
              }}
            >
              <Settings className="h-3.5 w-3.5" />
              Settings
            </button>
            <span className="text-xs text-muted-foreground">
              Tip: Hover violations/confirmations to highlight on chart. Click Violations or Confirmations text to keep them on.
            </span>
            <span className="text-xs text-muted-foreground">
              Tip: Hold Ctrl and left-drag to measure % move.
            </span>

            {showChartSettings && (
              <div className="fixed inset-0 z-40 pointer-events-none flex items-center justify-center p-4">
                <div
                  className="pointer-events-auto w-[min(680px,95vw)] max-h-[80vh] overflow-auto rounded-md border border-border bg-background p-3 shadow-2xl"
                  onClick={(e) => e.stopPropagation()}
                >
                <div className="mb-2 flex items-center justify-between text-xs font-semibold">
                  <span>Chart Settings</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-muted-foreground">
                      {savingSmaSettings ? 'Saving…' : hasUnsavedSettings ? 'Unsaved changes' : ''}
                    </span>
                    <button
                      className="h-6 rounded border border-border px-2 text-[11px] hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
                      onClick={saveChartSettings}
                      disabled={savingSmaSettings || !hasUnsavedSettings}
                      title="Save chart settings"
                    >
                      Save
                    </button>
                    <button
                      className="inline-flex h-6 w-6 items-center justify-center rounded border border-border hover:bg-muted"
                      onClick={() => setShowChartSettings(false)}
                      title="Close settings"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
                <div className="mb-2 inline-flex items-center gap-2 rounded border border-border/60 bg-muted/20 px-2 py-1 w-full">
                  <label className="inline-flex items-center gap-2 text-xs text-muted-foreground whitespace-nowrap">
                    <input
                      type="checkbox"
                      checked={openOnBars}
                      onChange={(e) => {
                        setOpenOnBars(e.target.checked);
                        setHasUnsavedSettings(true);
                      }}
                      className="h-3 w-3"
                    />
                    Open on Bars (OHLC)
                  </label>
                </div>
                <div className="mb-2 inline-flex items-center gap-2 rounded border border-border/60 bg-muted/20 px-2 py-1 w-full">
                  <label htmlFor="highlight-marker-gap" className="text-xs text-muted-foreground whitespace-nowrap">
                    Highlight Marker Gap
                  </label>
                  <input
                    id="highlight-marker-gap"
                    type="range"
                    min={0}
                    max={15}
                    step={1}
                    value={highlightMarkerGapSlots}
                    onChange={(e) => setHighlightGap(Number(e.target.value))}
                    className="w-full accent-primary"
                  />
                  <span className="text-xs font-mono w-4 text-right text-muted-foreground">{highlightMarkerGapSlots}</span>
                </div>
                <div className="mt-3 border-t border-border/60 pt-2">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="text-xs font-semibold">SMA Overlays ({chartTimeframe === 'weekly' ? '1W' : '1D'})</div>
                    <button
                      className="px-2 py-1 text-xs rounded border border-border hover:bg-muted"
                      onClick={addSmaSetting}
                    >
                      Add SMA
                    </button>
                  </div>

                  <div className="space-y-2">
                    {smaSettings.map((s, idx) => (
                      <div key={s.id} className="rounded border border-border/70 bg-muted/20 p-1.5">
                        <div className="mb-1 flex items-center justify-between">
                          <span className="text-xs font-semibold">SMA {idx + 1}</span>
                          <button
                            className="px-2 py-0.5 text-xs rounded border border-border hover:bg-muted"
                            onClick={() => removeSmaSetting(s.id)}
                            disabled={smaSettings.length <= 1}
                            title={smaSettings.length <= 1 ? 'At least one SMA is required' : 'Remove SMA'}
                          >
                            Remove
                          </button>
                        </div>

                        <div className="grid grid-cols-[auto_28px_auto_1fr_auto_78px_auto_78px] gap-1.5 items-center">
                          <label className="text-[11px] text-muted-foreground">On</label>
                          <input
                            type="checkbox"
                            checked={s.enabled}
                            onChange={(e) => setSmaSetting(s.id, { enabled: e.target.checked })}
                            className="h-3 w-3"
                          />

                          <label className="text-[11px] text-muted-foreground">Source</label>
                          <select
                            value={s.source}
                            onChange={(e) => setSmaSetting(s.id, { source: e.target.value as SmaSource })}
                            className="h-6 rounded border border-border bg-background px-1.5 text-xs"
                          >
                            <option value="close">Close</option>
                            <option value="open">Open</option>
                            <option value="high">High</option>
                            <option value="low">Low</option>
                          </select>

                          <label className="text-[11px] text-muted-foreground">Length</label>
                          <input
                            type="number"
                            min={2}
                            max={400}
                            value={s.length}
                            onChange={(e) => setSmaSetting(s.id, { length: Number(e.target.value) || 2 })}
                            className="h-6 rounded border border-border bg-background px-1.5 text-xs"
                          />

                          <label className="text-[11px] text-muted-foreground">Thickness</label>
                          <input
                            type="number"
                            min={0.1}
                            max={8}
                            step={0.1}
                            value={s.thickness}
                            onChange={(e) => setSmaSetting(s.id, { thickness: Number(e.target.value) || 0.1 })}
                            className="h-6 rounded border border-border bg-background px-1.5 text-xs"
                          />
                        </div>

                        <div className="mt-1.5 grid grid-cols-[auto_54px_auto_54px_auto_54px_auto_54px_auto_1fr] gap-1.5 items-center">
                          <label className="text-[11px] text-muted-foreground">R</label>
                          <input
                            type="number"
                            min={0}
                            max={255}
                            value={s.r}
                            onChange={(e) => setSmaSetting(s.id, { r: Number(e.target.value) || 0 })}
                            className="h-6 rounded border border-border bg-background px-1.5 text-xs"
                          />
                          <label className="text-[11px] text-muted-foreground">G</label>
                          <input
                            type="number"
                            min={0}
                            max={255}
                            value={s.g}
                            onChange={(e) => setSmaSetting(s.id, { g: Number(e.target.value) || 0 })}
                            className="h-6 rounded border border-border bg-background px-1.5 text-xs"
                          />
                          <label className="text-[11px] text-muted-foreground">B</label>
                          <input
                            type="number"
                            min={0}
                            max={255}
                            value={s.b}
                            onChange={(e) => setSmaSetting(s.id, { b: Number(e.target.value) || 0 })}
                            className="h-6 rounded border border-border bg-background px-1.5 text-xs"
                          />
                          <label className="text-[11px] text-muted-foreground">Opacity</label>
                          <input
                            type="number"
                            min={0.05}
                            max={1}
                            step={0.05}
                            value={s.opacity}
                            onChange={(e) => setSmaSetting(s.id, { opacity: Number(e.target.value) || 0.05 })}
                            className="h-6 rounded border border-border bg-background px-1.5 text-xs"
                          />
                          <label className="text-[11px] text-muted-foreground">Hex</label>
                          <input
                            type="text"
                            value={smaHexDrafts[s.id] ?? rgbToHex(s.r, s.g, s.b)}
                            onChange={(e) => setSmaHexDraftsByTimeframe((prev) => ({
                              ...prev,
                              [chartTimeframe]: {
                                ...prev[chartTimeframe],
                                [s.id]: e.target.value,
                              },
                            }))}
                            onBlur={() => commitSmaHex(s.id)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') {
                                e.preventDefault();
                                commitSmaHex(s.id);
                              }
                            }}
                            placeholder="#5c2213"
                            className="h-6 rounded border border-border bg-background px-1.5 text-xs font-mono"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              </div>
            )}
          </div>

          {dataError && <div className="px-4 py-2 text-xs text-red-400">{dataError}</div>}
          {computeError && <div className="px-4 py-2 text-xs text-red-400">{computeError}</div>}

          <div className="px-4 pt-2 pb-1 text-xs font-mono" style={{ backgroundColor: CHART_BG, color: TEXT_COLOR }}>
            <span className="font-semibold text-white mr-3">{activeTrade.ticker} {chartTimeframe === 'weekly' ? '1W' : '1D'}</span>
            {legend ? (
              <>
                <span className="mr-2">O <span className="text-white">{legend.open.toFixed(2)}</span></span>
                <span className="mr-2">H <span className="text-white">{legend.high.toFixed(2)}</span></span>
                <span className="mr-2">L <span className="text-white">{legend.low.toFixed(2)}</span></span>
                <span className="mr-2">C <span className="text-white">{legend.close.toFixed(2)}</span></span>
                <span className="mr-2" style={{ color: legend.change >= 0 ? UP_COLOR : DOWN_COLOR }}>
                  {legend.change >= 0 ? '+' : ''}{legend.change.toFixed(2)} ({legend.changePct >= 0 ? '+' : ''}{legend.changePct.toFixed(2)}%)
                </span>
                <span>Vol <span className="text-gray-300">{formatVolume(legend.volume)}</span></span>
                <span className="ml-2 text-gray-400">{legend.date}</span>
                {enabledSmaLegendItems.length > 0 && (
                  <span className="ml-3 inline-flex items-center gap-3 align-middle">
                    {enabledSmaLegendItems.map((item) => (
                      <span key={item.id} className="inline-flex items-center gap-1.5 text-[11px] text-gray-200">
                        <span
                          className="inline-block h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: item.color }}
                        />
                        <span>{item.label}</span>
                      </span>
                    ))}
                  </span>
                )}
              </>
            ) : (
              <span className="text-gray-500">Hover over chart for OHLCV</span>
            )}
          </div>

          <div className="relative flex-1 min-h-0" style={{ backgroundColor: CHART_BG }}>
            {loadingData ? (
              <div className="absolute inset-0 flex items-center justify-center text-muted-foreground gap-2">
                <Loader2 className="h-5 w-5 animate-spin" />
                <span className="text-sm">Loading chart…</span>
              </div>
            ) : (
              <>
                <div
                  ref={chartHostRef}
                  className="h-full w-full"
                  style={{ cursor: 'crosshair' }}
                  onMouseDown={handleChartMouseDown}
                  onMouseMove={handleChartMouseMove}
                  onMouseUp={handleChartMouseUp}
                  onMouseLeave={() => {
                    pointerDownRef.current = false;
                    dragStartRef.current = null;
                    clickStartRef.current = null;
                    if (measuringRef.current) {
                      chartApiRef.current?.applyOptions({
                        handleScroll: true,
                        handleScale: true,
                      });
                    }
                    measuringRef.current = false;
                    setDragMeasurement((prev) => (prev.visible ? { ...prev, visible: false } : prev));
                  }}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    setContextMenu({
                      visible: true,
                      x: e.clientX,
                      y: e.clientY,
                      date: hoveredDateRef.current || effectiveEndDate || null,
                    });
                  }}
                />

                {measurementStyle && (
                  <div className="pointer-events-none absolute inset-0 z-20">
                    <div
                      className="absolute h-[2px] origin-left rounded"
                      style={{
                        ...measurementStyle,
                        backgroundColor: dragMeasurement.pct >= 0 ? 'rgba(34,197,94,0.95)' : 'rgba(248,113,113,0.95)',
                      }}
                    />
                    <div
                      className="absolute -translate-x-1/2 -translate-y-1/2 rounded px-2 py-1 text-sm font-semibold shadow"
                      style={{
                        left: dragMeasurement.endX + 38,
                        top: dragMeasurement.endY - 20,
                        backgroundColor: dragMeasurement.pct >= 0 ? 'rgba(22,101,52,0.9)' : 'rgba(127,29,29,0.9)',
                        color: '#fff',
                      }}
                    >
                      {dragMeasurement.pct >= 0 ? '+' : ''}{dragMeasurement.pct.toFixed(2)}%
                    </div>
                  </div>
                )}

                <div
                  className="absolute left-2 top-2 z-20 max-w-[340px] rounded-md border p-2 text-xs shadow-xl"
                  style={{
                    backgroundColor: 'rgba(9, 13, 24, 0.55)',
                    borderColor: 'rgba(88, 104, 140, 0.25)',
                    backdropFilter: 'blur(6px)',
                  }}
                >
                  <div className="mb-1 font-semibold">Violations / Confirmations</div>
                  <div className="mb-1 text-[11px] text-muted-foreground">
                    Range: {sessionResult?.start_date || computeStartDate} → {sessionResult?.end_date || computeEndDate || '—'}
                  </div>
                  <div className="flex gap-2 mb-1">
                    <span className="rounded bg-red-800/70 text-red-50 px-1.5 py-0.5 font-semibold">
                      V {sessionResult?.total_violations ?? 0}
                    </span>
                    <span className="rounded bg-green-700/75 text-green-50 px-1.5 py-0.5 font-semibold">
                      C {sessionResult?.total_confirmations ?? 0}
                    </span>
                    <span className={`rounded px-1.5 py-0.5 font-semibold ${daysColorClass}`}>
                      UD{' '}
                      <span
                        className={`cursor-pointer rounded px-0.5 ${hoveredSignalTarget?.bucket === 'stats' && hoveredSignalTarget.type === 'days_up' ? 'bg-white/15 ring-1 ring-white/50' : ''}`}
                        onMouseEnter={() => setHoveredSignalTarget({ bucket: 'stats', type: 'days_up' })}
                        onMouseLeave={() => setHoveredSignalTarget((prev) => (
                          prev?.bucket === 'stats' && prev.type === 'days_up' ? null : prev
                        ))}
                      >
                        {daysUp}
                      </span>
                      /
                      <span
                        className={`cursor-pointer rounded px-0.5 ${hoveredSignalTarget?.bucket === 'stats' && hoveredSignalTarget.type === 'days_down' ? 'bg-white/15 ring-1 ring-white/50' : ''}`}
                        onMouseEnter={() => setHoveredSignalTarget({ bucket: 'stats', type: 'days_down' })}
                        onMouseLeave={() => setHoveredSignalTarget((prev) => (
                          prev?.bucket === 'stats' && prev.type === 'days_down' ? null : prev
                        ))}
                      >
                        {daysDown}
                      </span>
                    </span>
                    <span className={`rounded px-1.5 py-0.5 font-semibold ${goodCloseColorClass}`}>
                      GC{' '}
                      <span
                        className={`cursor-pointer rounded px-0.5 ${hoveredSignalTarget?.bucket === 'stats' && hoveredSignalTarget.type === 'good_closes' ? 'bg-white/15 ring-1 ring-white/50' : ''}`}
                        onMouseEnter={() => setHoveredSignalTarget({ bucket: 'stats', type: 'good_closes' })}
                        onMouseLeave={() => setHoveredSignalTarget((prev) => (
                          prev?.bucket === 'stats' && prev.type === 'good_closes' ? null : prev
                        ))}
                      >
                        {goodCloses}
                      </span>
                      /
                      <span
                        className={`cursor-pointer rounded px-0.5 ${hoveredSignalTarget?.bucket === 'stats' && hoveredSignalTarget.type === 'bad_closes' ? 'bg-white/15 ring-1 ring-white/50' : ''}`}
                        onMouseEnter={() => setHoveredSignalTarget({ bucket: 'stats', type: 'bad_closes' })}
                        onMouseLeave={() => setHoveredSignalTarget((prev) => (
                          prev?.bucket === 'stats' && prev.type === 'bad_closes' ? null : prev
                        ))}
                      >
                        {badCloses}
                      </span>
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <div
                        className={`mb-1 rounded px-1 py-0.5 text-[11px] font-semibold text-red-200 bg-red-900/30 border border-red-700/40 ${(effectiveSignalTarget?.bucket === 'violations' && !effectiveSignalTarget.type) ? 'ring-1 ring-white/40' : ''}`}
                        onMouseEnter={() => setHoveredSignalTarget({ bucket: 'violations' })}
                        onMouseLeave={() => setHoveredSignalTarget((prev) => (
                          prev?.bucket === 'violations' && !prev.type ? null : prev
                        ))}
                        onClick={() => {
                          setPinnedSignalBucket((prev) => (prev === 'violations' ? null : 'violations'));
                          setHoveredSignalTarget(null);
                        }}
                      >
                        Violations
                      </div>
                      <div className="space-y-0.5">
                        {groupedViolations.map((v) => (
                          <div
                            key={`v-${v.type}`}
                            className={`text-red-300 rounded px-1 ${hoveredSignalTarget?.bucket === 'violations' && hoveredSignalTarget.type === v.type ? 'bg-white/10' : ''}`}
                            onMouseEnter={() => setHoveredSignalTarget({ bucket: 'violations', type: v.type })}
                            onMouseLeave={() => setHoveredSignalTarget((prev) => (
                              prev?.bucket === 'violations' && prev.type === v.type ? null : prev
                            ))}
                          >
                            {prettyType(v.type, 'violations', v.latestDescription, v.count)} ({v.count})
                          </div>
                        ))}
                        {groupedViolations.length === 0 && (
                          <div className="text-muted-foreground">None</div>
                        )}
                      </div>
                    </div>
                    <div>
                      <div
                        className={`mb-1 rounded px-1 py-0.5 text-[11px] font-semibold text-green-200 bg-green-900/30 border border-green-700/40 ${(effectiveSignalTarget?.bucket === 'confirmations' && !effectiveSignalTarget.type) ? 'ring-1 ring-white/40' : ''}`}
                        onMouseEnter={() => setHoveredSignalTarget({ bucket: 'confirmations' })}
                        onMouseLeave={() => setHoveredSignalTarget((prev) => (
                          prev?.bucket === 'confirmations' && !prev.type ? null : prev
                        ))}
                        onClick={() => {
                          setPinnedSignalBucket((prev) => (prev === 'confirmations' ? null : 'confirmations'));
                          setHoveredSignalTarget(null);
                        }}
                      >
                        Confirmations
                      </div>
                      <div className="space-y-0.5">
                        {groupedConfirmations.map((c) => (
                          <div
                            key={`c-${c.type}`}
                            className={`text-green-300 rounded px-1 ${hoveredSignalTarget?.bucket === 'confirmations' && hoveredSignalTarget.type === c.type ? 'bg-white/10' : ''}`}
                            onMouseEnter={() => setHoveredSignalTarget({ bucket: 'confirmations', type: c.type })}
                            onMouseLeave={() => setHoveredSignalTarget((prev) => (
                              prev?.bucket === 'confirmations' && prev.type === c.type ? null : prev
                            ))}
                          >
                            {prettyType(c.type, 'confirmations', c.latestDescription, c.count)} ({c.count})
                          </div>
                        ))}
                        {groupedConfirmations.length === 0 && (
                          <div className="text-muted-foreground">None</div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {contextMenu.visible && (
          <div
            className="fixed z-[120] min-w-[210px] rounded border border-border bg-background shadow-xl p-1"
            style={{ left: contextMenu.x, top: contextMenu.y }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="block w-full text-left px-2 py-1 text-xs rounded hover:bg-muted"
              onClick={() => applyContextAction('start')}
            >
              Set start date to {contextMenu.date || 'cursor'}
            </button>
            <button
              className="block w-full text-left px-2 py-1 text-xs rounded hover:bg-muted"
              onClick={() => applyContextAction('end')}
            >
              Set end date to {contextMenu.date || 'cursor'}
            </button>
            <button
              className="block w-full text-left px-2 py-1 text-xs rounded hover:bg-muted"
              onClick={() => applyContextAction('latest')}
            >
              Use latest end date
            </button>
          </div>
        )}

      </div>
    </div>
  );
}

function buildLegendFromIndex(data: OHLCVBar[], index: number): LegendData | null {
  if (index < 0 || index >= data.length) return null;
  const bar = data[index];
  const prevClose = index > 0 ? data[index - 1].close : bar.open;
  const change = bar.close - prevClose;
  const changePct = prevClose !== 0 ? (change / prevClose) * 100 : 0;
  return {
    date: bar.date,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume,
    change,
    changePct,
  };
}

function nearestTradingDay(targetDate: string, data: OHLCVBar[]): string | null {
  if (!targetDate || data.length === 0) return null;
  const available = new Set(data.map((row) => row.date));
  if (available.has(targetDate)) return targetDate;

  const isWeeklySeries = data.length >= 2 && data.some((row, index) => {
    if (index === 0) return false;
    const prev = new Date(`${data[index - 1].date}T00:00:00Z`).getTime();
    const curr = new Date(`${row.date}T00:00:00Z`).getTime();
    return Number.isFinite(prev) && Number.isFinite(curr) && (curr - prev) > 86400000;
  });

  if (isWeeklySeries) {
    // Weekly bars in this app are labeled by the week's anchor date.
    // A daily date that falls inside that week should snap to that same bar,
    // not to the next weekly bar.
    let best: string | null = null;
    for (const row of data) {
      if (row.date > targetDate) break;
      best = row.date;
    }
    return best ?? data[0].date;
  }

  // Daily series: snap forward to the next available trading day.
  for (const row of data) {
    if (row.date >= targetDate) return row.date;
  }

  return data[data.length - 1].date;
}

function logicalRangeToDateWindow(range: LogicalRange | null, data: OHLCVBar[]): { from: string; to: string } | null {
  if (!range || data.length === 0) return null;

  // Use symmetric nearest-index mapping for both edges so toggling 1D<->1W
  // does not drift the viewport forward over repeated switches.
  const fromIndex = Math.max(0, Math.min(data.length - 1, Math.round(range.from)));
  const toIndex = Math.max(0, Math.min(data.length - 1, Math.round(range.to)));
  const left = Math.min(fromIndex, toIndex);
  const right = Math.max(fromIndex, toIndex);

  return {
    from: data[left].date,
    to: data[right].date,
  };
}

function dateWindowToLogicalRange(fromDate: string, toDate: string, data: OHLCVBar[]): LogicalRange | null {
  if (!fromDate || !toDate || data.length === 0) return null;

  const fromIdx = findClosestIndexByDate(data, fromDate);
  const toIdx = findClosestIndexByDate(data, toDate);
  if (fromIdx < 0 || toIdx < 0) return null;

  const left = Math.min(fromIdx, toIdx) - 0.5;
  const right = Math.max(fromIdx, toIdx) + 0.5;
  return {
    from: left as LogicalRange['from'],
    to: right as LogicalRange['to'],
  };
}

function findClosestIndexByDate(data: OHLCVBar[], targetDate: string): number {
  if (data.length === 0) return -1;

  const targetTs = Date.parse(`${targetDate}T00:00:00Z`);
  if (Number.isNaN(targetTs)) return data.length - 1;

  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (let i = 0; i < data.length; i += 1) {
    const ts = Date.parse(`${data[i].date}T00:00:00Z`);
    if (Number.isNaN(ts)) continue;

    const distance = Math.abs(ts - targetTs);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;

      if (distance === 0) break;
    }
  }

  return bestIndex;
}

function timeToString(time: Time): string {
  if (typeof time === 'string') return time;
  if (typeof time === 'number') return String(time);
  const maybe = time as { year?: number; month?: number; day?: number };
  if (typeof maybe.year === 'number' && typeof maybe.month === 'number' && typeof maybe.day === 'number') {
    return `${maybe.year}-${String(maybe.month).padStart(2, '0')}-${String(maybe.day).padStart(2, '0')}`;
  }
  return String(time);
}

function groupByType(items: ViolationItem[]): Array<{ type: string; count: number; latestDescription?: string; severity?: ViolationItem['severity'] }> {
  const grouped = new Map<string, { count: number; latestDescription?: string; severity?: ViolationItem['severity'] }>();
  for (const item of items) {
    const existing = grouped.get(item.type);
    if (existing) {
      existing.count += 1;
      if (item.description) existing.latestDescription = item.description;
      if (item.severity) existing.severity = item.severity;
    } else {
      grouped.set(item.type, {
        count: 1,
        latestDescription: item.description,
        severity: item.severity,
      });
    }
  }
  return Array.from(grouped.entries())
    .map(([type, data]) => ({ type, count: data.count, latestDescription: data.latestDescription, severity: data.severity }))
    .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
}

function prettyType(
  type: string,
  bucket: 'violations' | 'confirmations',
  latestDescription?: string,
  count?: number
): string {
  const base = type
    .replace(/_/g, ' ')
    .replace(/\brs\b/gi, 'RS')
    .replace(/\bma\b/gi, 'MA')
    .replace(/\b\w/g, (c) => c.toUpperCase());

  if (/^down up largest vol$/i.test(base)) {
    return bucket === 'confirmations' ? 'Up Largest Vol' : 'Down Largest Vol';
  }
  if (/^days up down$/i.test(base)) {
    return bucket === 'confirmations' ? '>70% Days Up' : '<30% Days Up';
  }
  if (/^good bad close$/i.test(base)) {
    return bucket === 'confirmations' ? '>70% Good closes' : '<30% Good closes';
  }
  if (/^squat reversal$/i.test(base)) {
    return 'Squat';
  }
  if (/^large squat reversal$/i.test(base)) {
    return 'Large Reversal';
  }
  if (/^largest pct down high vol$/i.test(base)) {
    return 'Largest % Down Day High Vol';
  }
  if (/^daily lower lows$/i.test(base) && latestDescription) {
    const streakMatch = latestDescription.match(/^(\d+)\s+consecutive\s+daily\s+lower\s+lows/i);
    if (streakMatch) {
      return `${streakMatch[1]} Daily Lower Lows`;
    }
  }
  if (/^daily higher highs$/i.test(base) && latestDescription) {
    const streakMatch = latestDescription.match(/^(\d+)\s+consecutive\s+daily\s+higher\s+highs/i);
    if (streakMatch) {
      return `${streakMatch[1]} Daily Higher Highs`;
    }
  }
  if (/^weekly lower lows$/i.test(base) && latestDescription) {
    const streakMatch = latestDescription.match(/^(\d+)\s+consecutive\s+weekly\s+lower\s+lows/i);
    if (streakMatch) {
      return `${streakMatch[1]} Weekly Lower Lows`;
    }
  }
  if (/^weekly higher highs$/i.test(base) && latestDescription) {
    const streakMatch = latestDescription.match(/^(\d+)\s+consecutive\s+weekly\s+higher\s+highs/i);
    if (streakMatch) {
      return `${streakMatch[1]} Weekly Higher Highs`;
    }
  }
  if (/^up\s+30(?:%|pct)\s*\+?\s*vol\s+increase$/i.test(base)) {
    if ((count || 0) > 1 || !latestDescription) {
      return 'Up 30%+ Vol Increase';
    }
    const volMatch = latestDescription.match(/up\s+on\s+([\d.]+)%\s+vol\s+increase/i);
    if (volMatch) {
      return `Up ${volMatch[1]}% Vol Increase`;
    }
    return 'Up 30%+ Vol Increase';
  }
  if (/^down\s+50(?:%|pct)\s*(?:volume|vol)\s+increase$/i.test(base)) {
    if ((count || 0) > 1 || !latestDescription) {
      return 'Down 50%+ Vol Increase';
    }
    const volMatch = latestDescription.match(/down\s+on\s+([\d.]+)%\s+vol\s+increase/i);
    if (volMatch) {
      return `Down ${volMatch[1]}% Vol Increase`;
    }
    return 'Down 50%+ Vol Increase';
  }
  if (/^above\s+20day\s+20(?:%|pct)$/i.test(base)) {
    if ((count || 0) > 1 || !latestDescription) {
      return '20%+ Above 20MA';
    }
    const aboveMatch = latestDescription.match(/^([\d.]+)%\s+above\s+20-day\s+ma/i);
    if (aboveMatch) {
      return `${aboveMatch[1]}% Above 20MA`;
    }
    return '20%+ Above 20MA';
  }

  return base;
}

function ohlcvRowsEqual(left: OHLCVBar[], right: OHLCVBar[]): boolean {
  if (left === right) return true;
  if (left.length !== right.length) return false;

  for (let i = 0; i < left.length; i += 1) {
    const a = left[i];
    const b = right[i];
    if (
      a.date !== b.date ||
      a.open !== b.open ||
      a.high !== b.high ||
      a.low !== b.low ||
      a.close !== b.close ||
      a.volume !== b.volume
    ) {
      return false;
    }
  }

  return true;
}

function buildOhlcvSignature(data: OHLCVBar[]): string {
  if (data.length === 0) return 'empty';

  const last = data[data.length - 1];
  return [data.length, last.date, last.open, last.high, last.low, last.close, last.volume].join('|');
}

function extractHttpStatus(err: unknown): number | null {
  if (!err || typeof err !== 'object') return null;
  const maybeResponse = (err as { response?: { status?: unknown } }).response;
  if (!maybeResponse) return null;
  const status = maybeResponse.status;
  return typeof status === 'number' ? status : null;
}

function aggregateDailyBarsToWeekly(dailyBars: OHLCVBar[]): OHLCVBar[] {
  if (dailyBars.length === 0) return [];

  const sorted = [...dailyBars].sort((a, b) => a.date.localeCompare(b.date));
  const weekly: OHLCVBar[] = [];
  let currentWeekKey: string | null = null;
  let bucket: OHLCVBar | null = null;

  for (const bar of sorted) {
    const weekKey = isoWeekKey(bar.date);
    if (weekKey !== currentWeekKey || !bucket) {
      if (bucket) weekly.push(bucket);
      currentWeekKey = weekKey;
      bucket = { ...bar };
      continue;
    }

    bucket.high = Math.max(bucket.high, bar.high);
    bucket.low = Math.min(bucket.low, bar.low);
    bucket.close = bar.close;
    bucket.volume += bar.volume;
    bucket.date = bar.date;
  }

  if (bucket) weekly.push(bucket);
  return weekly;
}

function isoWeekKey(dateStr: string): string {
  const d = new Date(`${dateStr}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return dateStr;

  const day = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  const weekNo = Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  return `${d.getUTCFullYear()}-W${String(weekNo).padStart(2, '0')}`;
}

function formatVolume(vol: number): string {
  if (vol >= 1_000_000_000) return `${(vol / 1_000_000_000).toFixed(2)}B`;
  if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(2)}M`;
  if (vol >= 1_000) return `${(vol / 1_000).toFixed(1)}K`;
  return String(vol);
}

function addMarkerSpacers(markers: TradeMarker[], gapSlots: number, spacerColor: string): TradeMarker[] {
  if (gapSlots <= 0 || markers.length === 0) return markers;

  const withSpacers: TradeMarker[] = [];
  let currentTime: Time | null = null;

  for (const marker of markers) {
    if (marker.time !== currentTime) {
      currentTime = marker.time;
      for (let i = 0; i < gapSlots; i += 1) {
        withSpacers.push({
          time: marker.time,
          position: 'belowBar' as const,
          color: spacerColor,
          shape: 'arrowDown' as const,
          text: '',
        });
      }
    }
    withSpacers.push(marker);
  }

  return withSpacers;
}

function computeSma(data: Array<{ time: Time; close: number }>, length: number): LineData<Time>[] {
  if (length <= 1 || data.length === 0) return [];

  const out: LineData<Time>[] = [];
  let rolling = 0;

  for (let i = 0; i < data.length; i += 1) {
    rolling += data[i].close;
    if (i >= length) {
      rolling -= data[i - length].close;
    }
    if (i >= length - 1) {
      out.push({
        time: data[i].time,
        value: rolling / length,
      });
    }
  }

  return out;
}
