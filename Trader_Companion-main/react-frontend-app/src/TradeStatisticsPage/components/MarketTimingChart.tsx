import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import {
  createChart,
  createSeriesMarkers,
  BarSeries,
  HistogramSeries,
  LineSeries,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type LogicalRange,
  type Time,
  type SeriesMarker,
  type MouseEventParams,
  type BarData,
  type HistogramData,
  type LineData,
} from 'lightweight-charts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ohlcvAPI, type OHLCVBar, type OhlcvTimeframe } from '../services/ohlcvAPI';
import type { Trade } from '@/TradeHistoryPage/types/Trade';
import { Loader2, RefreshCw } from 'lucide-react';

interface MarketTimingChartProps {
  filteredTrades: Trade[];
  currentBalance: number;
  useMonthFilter: boolean;
  onUseMonthFilterChange: (checked: boolean) => void;
}

interface OHLCVLegend {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  change: number;
  changePct: number;
  date: string;
}

type TradeMarker = SeriesMarker<Time> & {
  fullText?: string;
  isSpacer?: boolean;
};

type MarkerSeriesHandle = {
  setMarkers?: (markers: TradeMarker[]) => void;
};

type CrosshairSyncChartHandle = IChartApi & {
  setCrosshairPosition?: (price: number, time: Time, seriesApi: unknown) => void;
  clearCrosshairPosition?: () => void;
};

// ── Style constants ──
const CHART_BG = '#131722';
const GRID_COLOR = '#1e222d';
const TEXT_COLOR = '#787b86';
const UP_COLOR = '#26a69a';
const DOWN_COLOR = '#ef5350';
const CROSSHAIR_COLOR = '#555';
const VOLUME_UP = 'rgba(38,166,154,0.3)';
const VOLUME_DOWN = 'rgba(239,83,80,0.3)';
const EXPOSURE_COLOR = '#2962FF';
const SPACER_MARKER_COLOR = 'rgba(0, 0, 0, 0)';
const DEFAULT_TOP_MARKER_GAP_SLOTS = 4;
const DEFAULT_DAILY_BARS_IN_TWO_YEARS = 252 * 2;
const DEFAULT_WEEKLY_BARS_IN_TWO_YEARS = 52 * 2;
const DEFAULT_RIGHT_PADDING_BARS = 6;
const BASE_PRICE_BOTTOM_MARGIN = 0.12;
const EXTRA_BOTTOM_MARGIN_PER_GAP_SLOT = 0.01;
const MAX_PRICE_BOTTOM_MARGIN = 0.22;
const RIGHT_PRICE_SCALE_WIDTH = 72;

export const MarketTimingChart: React.FC<MarketTimingChartProps> = ({
  filteredTrades,
  // currentBalance reserved for future equity-at-entry calculations
  useMonthFilter,
  onUseMonthFilterChange,
}) => {
  const mainChartRef = useRef<HTMLDivElement>(null);
  const exposureChartRef = useRef<HTMLDivElement>(null);
  const mainChartApiRef = useRef<IChartApi | null>(null);
  const exposureChartApiRef = useRef<IChartApi | null>(null);
  const visibleRangeRef = useRef<LogicalRange | null>(null);

  // Ref to hold the main price series so we can update markers on hover without recreating chart
  const candleSeriesRef = useRef<MarkerSeriesHandle | null>(null);
  // Ref to hold the original markers data
  const baseMarkersRef = useRef<TradeMarker[]>([]);

  const [ohlcvData, setOhlcvData] = useState<OHLCVBar[]>([]);
  const [timeframe, setTimeframe] = useState<OhlcvTimeframe>('daily');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [legend, setLegend] = useState<OHLCVLegend | null>(null);
  const [topMarkerGapSlots, setTopMarkerGapSlots] = useState(DEFAULT_TOP_MARKER_GAP_SLOTS);
  const [tradeTooltip, setTradeTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    date: string;
    trades: string[];
  }>({ visible: false, x: 0, y: 0, date: '', trades: [] });

  // ── Fetch SPY OHLCV data ──
  const fetchData = useCallback(async (options?: { silent?: boolean }) => {
    const silent = options?.silent ?? false;

    if (!silent) {
      setLoading(true);
      setError(null);
    }

    try {
      const resp = timeframe === 'weekly'
        ? await ohlcvAPI.getHistoricalWeeklyData('SPY')
        : await ohlcvAPI.getHistoricalData('SPY');

      if (resp.data.data && resp.data.data.length > 0) {
        setOhlcvData(prev => (isSameOhlcvData(prev, resp.data.data) ? prev : resp.data.data));
      } else {
        if (!silent) {
          setError(timeframe === 'weekly' ? 'No SPY weekly data available yet.' : 'No SPY data available.');
        }
      }
    } catch (err) {
      if (!silent) {
        setError(
          timeframe === 'weekly'
            ? 'Failed to load SPY weekly data. Make sure initial full fetch has run for SPY.'
            : 'Failed to load SPY data. Make sure the backend is running and SPY is being tracked.'
        );
      }
      console.error('OHLCV fetch error:', err);
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [timeframe]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    const pollId = window.setInterval(() => {
      fetchData({ silent: true });
    }, 5000);

    return () => {
      window.clearInterval(pollId);
    };
  }, [fetchData]);

  useEffect(() => {
    setLegend(buildLegendFromIndex(ohlcvData, ohlcvData.length - 1));
  }, [ohlcvData]);

  // ── Build trade markers ──
  const tradeMarkers = useMemo((): TradeMarker[] => {
    if (ohlcvData.length === 0) return [];

    const ohlcvDates = new Set(ohlcvData.map(b => b.date));
    const markers: TradeMarker[] = [];

    const exitedTrades = filteredTrades
      .filter(t => t.Status === 'Exited' && t.Entry_Date && t.Exit_Date)
      .sort((a, b) => new Date(a.Entry_Date).getTime() - new Date(b.Entry_Date).getTime());

    exitedTrades.forEach((trade, idx) => {
      const tradeNum = idx + 1;
      const entryDate = trade.Entry_Date.split('T')[0];
      const exitDate = trade.Exit_Date!.split('T')[0];

      const nearestEntry = findNearestTradingDay(entryDate, ohlcvDates, ohlcvData);
      const nearestExit = findNearestTradingDay(exitDate, ohlcvDates, ohlcvData);

      if (nearestEntry) {
        markers.push({
          time: nearestEntry as Time,
          position: 'belowBar',
          color: UP_COLOR,
          shape: 'arrowUp',
          text: '',
          fullText: `B${tradeNum} ${trade.Ticker}`,
        });
      }

      if (nearestExit) {
        markers.push({
          time: nearestExit as Time,
          position: 'belowBar',
          color: DOWN_COLOR,
          shape: 'arrowUp',
          text: '',
          fullText: `S${tradeNum} ${trade.Ticker}`,
        });
      }
    });

    // Sort markers by time (required by lightweight-charts)
    markers.sort((a, b) => {
      if (a.time < b.time) return -1;
      if (a.time > b.time) return 1;
      return 0;
    });

    return addMarkerSpacers(markers, topMarkerGapSlots);
  }, [filteredTrades, ohlcvData, topMarkerGapSlots]);

  // ── Build exposure data (Pct_Of_Equity is 0-1, chart shows 0-100%) ──
  const exposureData = useMemo((): LineData<Time>[] => {
    if (ohlcvData.length === 0) return [];

    const exitedTrades = filteredTrades
      .filter(t => t.Status === 'Exited' && t.Entry_Date && t.Exit_Date);
    const activeTrades = filteredTrades
      .filter(t => t.Status !== 'Exited' && t.Entry_Date);

    const data: LineData<Time>[] = [];

    for (const bar of ohlcvData) {
      const date = bar.date;
      let totalExposure = 0;

      for (const trade of exitedTrades) {
        const entryDate = trade.Entry_Date.split('T')[0];
        const exitDate = trade.Exit_Date!.split('T')[0];
        if (date >= entryDate && date <= exitDate) {
          totalExposure += trade.Pct_Of_Equity || 0;
        }
      }

      for (const trade of activeTrades) {
        const entryDate = trade.Entry_Date.split('T')[0];
        if (date >= entryDate) {
          totalExposure += trade.Pct_Of_Equity || 0;
        }
      }

      // Values are 0-1, display as 0-100%. Cap at 1.0 (100%).
      const displayValue = Math.min(1, Math.max(0, totalExposure)) * 100;

      data.push({
        time: date as Time,
        value: displayValue,
      });
    }

    return data;
  }, [filteredTrades, ohlcvData]);

  // ── Create & manage charts ──
  useEffect(() => {
    if (!mainChartRef.current || ohlcvData.length === 0) return;

    const priceBottomMargin = Math.min(
      MAX_PRICE_BOTTOM_MARGIN,
      BASE_PRICE_BOTTOM_MARGIN + topMarkerGapSlots * EXTRA_BOTTOM_MARGIN_PER_GAP_SLOT
    );

    if (mainChartApiRef.current) {
      visibleRangeRef.current = mainChartApiRef.current.timeScale().getVisibleLogicalRange();
    }

    // Clean up existing charts
    if (mainChartApiRef.current) {
      mainChartApiRef.current.remove();
      mainChartApiRef.current = null;
    }
    if (exposureChartApiRef.current) {
      exposureChartApiRef.current.remove();
      exposureChartApiRef.current = null;
    }

    // ═══════════════════════════════════════
    //  MAIN S&P CHART (candlestick + volume)
    // ═══════════════════════════════════════
    const mainChart = createChart(mainChartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: CHART_BG },
        textColor: TEXT_COLOR,
        fontFamily: "'Inter', -apple-system, sans-serif",
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
        minimumWidth: RIGHT_PRICE_SCALE_WIDTH,
        scaleMargins: {
          top: 0.05,
          bottom: priceBottomMargin, // keep space for markers while prioritizing price area
        },
      },
      timeScale: {
        borderColor: GRID_COLOR,
        timeVisible: false,
        rightOffset: 6,
        barSpacing: 7,
        minBarSpacing: 0.5,
      },
    });

    mainChartApiRef.current = mainChart;

    // HLC bar series (open tick hidden)
    const candleSeries = mainChart.addSeries(BarSeries, {
      upColor: UP_COLOR,
      downColor: DOWN_COLOR,
      openVisible: false,
      thinBars: false,
    });

    const candleData: BarData<Time>[] = ohlcvData.map(bar => ({
      time: bar.date as Time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));

    const candleSeriesWithMarkers = candleSeries as typeof candleSeries & MarkerSeriesHandle;

    candleSeries.setData(candleData);
    candleSeriesRef.current = candleSeriesWithMarkers;
    baseMarkersRef.current = tradeMarkers;

    // Trade markers (v4/v5 API: setMarkers)
    if (tradeMarkers.length > 0) {
      if (typeof candleSeriesWithMarkers.setMarkers === 'function') {
        candleSeriesWithMarkers.setMarkers(tradeMarkers);
      } else {
        createSeriesMarkers(candleSeries, tradeMarkers);
      }
    }

    // Volume series (histogram overlay on separate price scale)
    const volumeSeries = mainChart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });

    mainChart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.9,
        bottom: 0,
      },
    });

    const volumeData: HistogramData<Time>[] = ohlcvData.map((bar, i) => {
      const prevClose = i > 0 ? ohlcvData[i - 1].close : bar.open;
      return {
        time: bar.date as Time,
        value: bar.volume,
        color: bar.close >= prevClose ? VOLUME_UP : VOLUME_DOWN,
      };
    });

    volumeSeries.setData(volumeData);

    // ═══════════════════════════════════════
    //  EXPOSURE INDICATOR CHART
    // ═══════════════════════════════════════
    let exposureChart: IChartApi | null = null;
    let exposureSeries: unknown = null;

    if (exposureChartRef.current) {
      exposureChart = createChart(exposureChartRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: CHART_BG },
          textColor: TEXT_COLOR,
          fontFamily: "'Inter', -apple-system, sans-serif",
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
          minimumWidth: RIGHT_PRICE_SCALE_WIDTH,
          scaleMargins: { top: 0.05, bottom: 0.05 },
        },
        timeScale: {
          borderColor: GRID_COLOR,
          timeVisible: false,
          rightOffset: 6,
          barSpacing: 7,
          minBarSpacing: 0.5,
        },
      });

      exposureChartApiRef.current = exposureChart;

      exposureSeries = exposureChart.addSeries(LineSeries, {
        color: EXPOSURE_COLOR,
        lineWidth: 2,
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => `${price.toFixed(0)}%`,
        },
      });

      if (exposureData.length > 0) {
        (exposureSeries as { setData: (data: LineData<Time>[]) => void }).setData(exposureData);
      }

      // Sync time scales
      mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        visibleRangeRef.current = range;
        if (range && exposureChart) {
          exposureChart.timeScale().setVisibleLogicalRange(range);
        }
      });

      exposureChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range && mainChartApiRef.current) {
          mainChartApiRef.current.timeScale().setVisibleLogicalRange(range);
        }
      });
    }

    const syncExposureCrosshair = (time?: Time) => {
      if (!time || !exposureChart || !exposureSeries) return;

      const target = exposureData.find(d => timeToString(d.time) === timeToString(time));
      const syncChart = exposureChart as CrosshairSyncChartHandle;

      if (!target) {
        if (typeof syncChart.clearCrosshairPosition === 'function') {
          syncChart.clearCrosshairPosition();
        }
        return;
      }

      if (typeof syncChart.setCrosshairPosition === 'function') {
        syncChart.setCrosshairPosition(target.value, target.time, exposureSeries);
      }
    };

    const clearExposureCrosshair = () => {
      if (!exposureChart) return;
      const syncChart = exposureChart as CrosshairSyncChartHandle;
      if (typeof syncChart.clearCrosshairPosition === 'function') {
        syncChart.clearCrosshairPosition();
      }
    };

    // ═══════════════════════════════════════
    //  CROSSHAIR LEGEND (OHLCV info on hover)
    // ═══════════════════════════════════════
    const handleMainCrosshairMove = (param: MouseEventParams<Time>) => {
      if (
        !param.time ||
        !param.seriesData ||
        !param.point ||
        param.point.x < 0 ||
        param.point.x > mainChartRef.current!.clientWidth ||
        param.point.y < 0 ||
        param.point.y > mainChartRef.current!.clientHeight
      ) {
        clearExposureCrosshair();
        setLegend(buildLegendFromIndex(ohlcvData, ohlcvData.length - 1));
        setTradeTooltip(prev => (prev.visible ? { ...prev, visible: false } : prev));
        return;
      }

      syncExposureCrosshair(param.time);

      const candle = param.seriesData.get(candleSeries) as BarData<Time> | undefined;
      const vol = param.seriesData.get(volumeSeries) as HistogramData<Time> | undefined;

      if (!candle) {
        clearExposureCrosshair();
        setLegend(null);
        setTradeTooltip(prev => (prev.visible ? { ...prev, visible: false } : prev));
        return;
      }

      // Find previous bar for change calculation
      const currentIdx = ohlcvData.findIndex(b => b.date === (param.time as string));
      const prevClose = currentIdx > 0 ? ohlcvData[currentIdx - 1].close : candle.open;
      const change = candle.close - prevClose;
      const changePct = prevClose !== 0 ? (change / prevClose) * 100 : 0;

      setLegend({
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: vol?.value ?? 0,
        change,
        changePct,
        date: param.time as string,
      });

      // Update Tooltip
      const hoveredTimeStr = timeToString(param.time);
      const tradesForTime = tradeMarkers
        .filter(m => !m.isSpacer && timeToString(m.time) === hoveredTimeStr && m.fullText)
        .map(m => m.fullText as string);

      if (tradesForTime.length > 0) {
        setTradeTooltip({
          visible: true,
          x: param.point.x,
          y: param.point.y,
          date: hoveredTimeStr,
          trades: tradesForTime,
        });
      } else {
        setTradeTooltip(prev => (prev.visible ? { ...prev, visible: false } : prev));
      }
    };

    mainChart.subscribeCrosshairMove(handleMainCrosshairMove);

    const mainEl = mainChartRef.current;
    const exposureEl = exposureChartRef.current;

    // ═══════════════════════════════════════
    //  RESIZE OBSERVER
    // ═══════════════════════════════════════
    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (entry.target === mainEl && mainChartApiRef.current) {
          mainChartApiRef.current.applyOptions({ width, height });
        }
        if (entry.target === exposureEl && exposureChartApiRef.current) {
          exposureChartApiRef.current.applyOptions({ width, height });
        }
      }
    });

    resizeObserver.observe(mainEl);
    if (exposureEl) resizeObserver.observe(exposureEl);

    // Restore zoom/pan across data refreshes; only fit on first load.
    if (visibleRangeRef.current) {
      mainChart.timeScale().setVisibleLogicalRange(visibleRangeRef.current);
      if (exposureChartApiRef.current) {
        exposureChartApiRef.current.timeScale().setVisibleLogicalRange(visibleRangeRef.current);
      }
    } else {
      const visibleBars = timeframe === 'weekly'
        ? DEFAULT_WEEKLY_BARS_IN_TWO_YEARS
        : DEFAULT_DAILY_BARS_IN_TWO_YEARS;
      const right = ohlcvData.length - 0.5 + DEFAULT_RIGHT_PADDING_BARS;
      const left = Math.max(-0.5, right - visibleBars);

      if (ohlcvData.length > 0) {
        const defaultRange = { from: left, to: right };
        mainChart.timeScale().setVisibleLogicalRange(defaultRange);
        if (exposureChartApiRef.current) {
          exposureChartApiRef.current.timeScale().setVisibleLogicalRange(defaultRange);
        }
      } else {
        mainChart.timeScale().fitContent();
      }
    }

    return () => {
      mainChart.unsubscribeCrosshairMove(handleMainCrosshairMove);
      resizeObserver.disconnect();
      if (mainChartApiRef.current) {
        mainChartApiRef.current.remove();
        mainChartApiRef.current = null;
      }
      if (exposureChartApiRef.current) {
        exposureChartApiRef.current.remove();
        exposureChartApiRef.current = null;
      }
    };
  }, [ohlcvData, tradeMarkers, exposureData, topMarkerGapSlots, timeframe]);

  // ═══════════════════════════════════════
  //  RENDER
  // ═══════════════════════════════════════
  return (
    <Card className="col-span-1 md:col-span-2 lg:col-span-3">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="space-y-1.5">
            <CardTitle>Market Timing — S&P 500</CardTitle>
            <p className="text-sm text-muted-foreground">
              Your trade entries/exits overlaid on S&P 500 ({timeframe === 'weekly' ? 'weekly' : 'daily'}) with portfolio exposure
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 px-1 py-1 rounded border border-border/60 bg-muted/20">
              <Button
                variant={timeframe === 'daily' ? 'default' : 'ghost'}
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() => setTimeframe('daily')}
                disabled={loading}
              >
                Daily
              </Button>
              <Button
                variant={timeframe === 'weekly' ? 'default' : 'ghost'}
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() => setTimeframe('weekly')}
                disabled={loading}
              >
                Weekly
              </Button>
            </div>
            <div className="flex items-center gap-2 px-2 py-1 rounded border border-border/60 bg-muted/20">
              <input
                type="checkbox"
                id="use-month-filter-market"
                checked={useMonthFilter}
                onChange={e => onUseMonthFilterChange(e.target.checked)}
                className="accent-primary h-3.5 w-3.5 cursor-pointer"
              />
              <label htmlFor="use-month-filter-market" className="text-xs text-muted-foreground cursor-pointer whitespace-nowrap">
                Use selected months
              </label>
            </div>
            <div className="flex items-center gap-2 px-2 py-1 rounded border border-border/60 bg-muted/20">
              <label htmlFor="marker-gap-slider" className="text-xs text-muted-foreground whitespace-nowrap">
                Marker Gap
              </label>
              <input
                id="marker-gap-slider"
                type="range"
                min={0}
                max={15}
                step={1}
                value={topMarkerGapSlots}
                onChange={e => setTopMarkerGapSlots(Number(e.target.value))}
                className="w-24 accent-primary"
              />
              <span className="text-xs font-mono w-4 text-right">{topMarkerGapSlots}</span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchData()}
              disabled={loading}
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin mr-1" />
              ) : (
                <RefreshCw className="h-4 w-4 mr-1" />
              )}
              {loading ? 'Loading…' : 'Refresh Data'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-0">
        {error && (
          <div className="mb-4 p-3 rounded-md bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}

        {loading && ohlcvData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[500px] text-muted-foreground gap-3">
            <Loader2 className="h-8 w-8 animate-spin" />
            <p>Loading S&P 500 data…</p>
          </div>
        ) : (
          <>
            {/* OHLCV Legend bar */}
            <div
              className="flex items-center gap-3 text-xs font-mono px-2 py-1.5 rounded-t-md"
              style={{ backgroundColor: CHART_BG, color: TEXT_COLOR }}
            >
              <span className="font-semibold text-white">SPY {timeframe === 'weekly' ? '1W' : '1D'}</span>
              {legend ? (
                <>
                  <LegendItem label="O" value={legend.open} up={legend.close >= legend.open} />
                  <LegendItem label="H" value={legend.high} up={legend.close >= legend.open} />
                  <LegendItem label="L" value={legend.low} up={legend.close >= legend.open} />
                  <LegendItem label="C" value={legend.close} up={legend.close >= legend.open} />
                  <span style={{ color: legend.change >= 0 ? UP_COLOR : DOWN_COLOR }}>
                    {legend.change >= 0 ? '+' : ''}{legend.change.toFixed(2)} ({legend.changePct >= 0 ? '+' : ''}{legend.changePct.toFixed(2)}%)
                  </span>
                  <span>Vol <span className="text-gray-400">{formatVolume(legend.volume)}</span></span>
                </>
              ) : (
                <span className="text-gray-500">Hover over chart for details</span>
              )}
            </div>

            {/* Main S&P Chart */}
            <div className="relative w-full" style={{ height: '73vh', minHeight: '350px' }}>
              <div
                ref={mainChartRef}
                style={{ width: '100%', height: '100%', cursor: 'crosshair' }}
              />
              {tradeTooltip.visible && (
                <div
                  className="absolute z-50 pointer-events-none rounded shadow-lg px-3 py-2 text-xs font-medium border"
                  style={{
                    backgroundColor: 'rgba(30, 34, 45, 0.95)',
                    color: '#d1d4dc',
                    borderColor: '#2B2B43',
                    backdropFilter: 'blur(4px)',
                    left: Math.min(tradeTooltip.x + 15, (mainChartRef.current?.clientWidth || 500) - 150),
                    top: Math.min(tradeTooltip.y + 15, (mainChartRef.current?.clientHeight || 400) - 50),
                  }}
                >
                  <div className="mb-1 text-[10px] uppercase text-gray-400 font-bold border-b border-gray-600/50 pb-1">
                    {tradeTooltip.date}
                  </div>
                  <div className="flex flex-col gap-1 mt-1">
                    {tradeTooltip.trades.map((t, i) => {
                      const isBuy = t.startsWith('B');
                      return (
                        <div key={i} className="flex items-center gap-1.5 whitespace-nowrap">
                          <span
                            className="inline-block w-2 h-2 rounded-full"
                            style={{ backgroundColor: isBuy ? UP_COLOR : DOWN_COLOR }}
                          />
                          <span>{t}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Exposure Indicator label */}
            <div
              className="flex items-center gap-2 text-xs font-mono px-2 py-1"
              style={{ backgroundColor: CHART_BG, color: TEXT_COLOR }}
            >
              <span style={{ color: EXPOSURE_COLOR }}>●</span>
              <span>Portfolio Exposure (%)</span>
              <span className="text-gray-600 ml-auto">0% = no positions · 100% = fully invested</span>
            </div>

            {/* Exposure chart */}
            <div
              ref={exposureChartRef}
              style={{ width: '100%', height: '15vh', minHeight: '100px', cursor: 'crosshair' }}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
};

// ── Sub-components ──

function LegendItem({ label, value, up }: { label: string; value: number; up: boolean }) {
  return (
    <span>
      {label}{' '}
      <span style={{ color: up ? UP_COLOR : DOWN_COLOR }}>{value.toFixed(2)}</span>
    </span>
  );
}

// ── Helpers ──

function addMarkerSpacers(markers: TradeMarker[], topMarkerGapSlots: number): TradeMarker[] {
  const markersWithSpacers: TradeMarker[] = [];
  let currentTime: Time | null = null;

  for (const marker of markers) {
    if (marker.time !== currentTime) {
      currentTime = marker.time;
      for (let i = 0; i < topMarkerGapSlots; i += 1) {
        markersWithSpacers.push({
          time: marker.time,
          position: 'belowBar',
          color: SPACER_MARKER_COLOR,
          shape: 'arrowDown',
          text: '',
          isSpacer: true,
        });
      }
    }

    markersWithSpacers.push(marker);
  }

  return markersWithSpacers;
}

function buildLegendFromIndex(data: OHLCVBar[], index: number): OHLCVLegend | null {
  if (index < 0 || index >= data.length) return null;

  const bar = data[index];
  const prevClose = index > 0 ? data[index - 1].close : bar.open;
  const change = bar.close - prevClose;
  const changePct = prevClose !== 0 ? (change / prevClose) * 100 : 0;

  return {
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume,
    change,
    changePct,
    date: bar.date,
  };
}

function isSameOhlcvData(a: OHLCVBar[], b: OHLCVBar[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;

  for (let i = 0; i < a.length; i += 1) {
    const left = a[i];
    const right = b[i];
    if (
      left.date !== right.date ||
      left.open !== right.open ||
      left.high !== right.high ||
      left.low !== right.low ||
      left.close !== right.close ||
      left.volume !== right.volume
    ) {
      return false;
    }
  }

  return true;
}

function findNearestTradingDay(
  targetDate: string,
  availableDates: Set<string>,
  ohlcvData: OHLCVBar[]
): string | null {
  if (availableDates.has(targetDate)) return targetDate;

  const sorted = ohlcvData.map(b => b.date);
  let bestDist = Infinity;
  let best: string | null = null;
  const targetTime = new Date(targetDate).getTime();

  for (const d of sorted) {
    const dist = Math.abs(new Date(d).getTime() - targetTime);
    if (dist < bestDist) {
      bestDist = dist;
      best = d;
    }
  }

  // Only snap if within 5 calendar days
  if (best && bestDist <= 5 * 24 * 60 * 60 * 1000) return best;
  return null;
}

function formatVolume(vol: number): string {
  if (vol >= 1_000_000_000) return `${(vol / 1_000_000_000).toFixed(2)}B`;
  if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(2)}M`;
  if (vol >= 1_000) return `${(vol / 1_000).toFixed(1)}K`;
  return vol.toString();
}

function timeToString(time: unknown): string {
  if (!time) return '';
  if (typeof time === 'string') return time;
  if (typeof time === 'number') return time.toString();
  if (typeof time === 'object' && 'year' in time && 'month' in time && 'day' in time) {
    return `${time.year}-${String(time.month).padStart(2, '0')}-${String(time.day).padStart(2, '0')}`;
  }
  return String(time);
}
