import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { MonthlyStats } from '../types';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import { AlertTriangle, TrendingUp, Clock, Calendar, BarChart3, Maximize2, X } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { subDays, subMonths, format, parseISO, isAfter, differenceInDays, startOfMonth, endOfMonth } from 'date-fns';

type DisplayMode = 'percent' | 'dollar';

interface RiskOfRuinStatsProps {
  monthlyStats: MonthlyStats[];
  targetReturnPercent: number;
  onTargetReturnChange: (value: number) => void;
  filteredTrades: (Trade & { Exit_Price: number; Exit_Date: string })[];
  currentBalance: number;
  useMonthFilter: boolean;
  onUseMonthFilterChange: (checked: boolean) => void;
}

const computeTimeToLevel = (
  monthlyRatePct: number,
  levelPct: number // e.g. -50 for ruin, +100 for return
): { text: string; color: string } => {
  const r = monthlyRatePct / 100;
  if (Math.abs(r) < 0.0001) return { text: 'N/A', color: '' };

  // levelPct is the target equity change, e.g. -50 means 50% of starting equity remains
  const targetMultiplier = 1 + levelPct / 100; // -50% → 0.5, +100% → 2.0
  if (targetMultiplier <= 0) return { text: 'N/A', color: '' };

  const months = Math.log(targetMultiplier) / Math.log(1 + r);

  if (months <= 0) return { text: 'Never', color: levelPct < 0 ? '#34d399' : '#f87171' };

  const years = months / 12;
  const sign = levelPct >= 0 ? '+' : '−';
  const absLevel = Math.abs(levelPct);
  const color = levelPct < 0 ? '#f87171' : '#34d399';
  return { text: `${years.toFixed(1)}y → ${sign}${absLevel}%`, color };
};

const ruinLevels = [-5, -10, -15, -25, -50];
const GAIN_COLOR = '#34d399';
const LOSS_COLOR = '#f87171';

interface TimelineEvent {
  date: Date;
  dollarAmount: number;
  pctAmount: number;
}

interface PeriodTimeline {
  start: Date;
  end: Date;
  events: TimelineEvent[];
}

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const buildTimelineFromTrades = (
  trades: (Trade & { Exit_Price: number; Exit_Date: string })[],
  start: Date,
  end: Date
): PeriodTimeline => {
  const events = trades
    .map((trade) => {
      const tradeDate = parseISO(trade.Entry_Date);
      const dollarAmount = trade.Return ?? 0;
      const pctAmount = trade.Entry_Price > 0
        ? ((trade.Exit_Price - trade.Entry_Price) / trade.Entry_Price) * 100
        : 0;
      return {
        date: tradeDate,
        dollarAmount,
        pctAmount,
      };
    })
    .filter((event) => !Number.isNaN(event.date.getTime()) && (Math.abs(event.dollarAmount) > 0.00001 || Math.abs(event.pctAmount) > 0.00001))
    .sort((a, b) => a.date.getTime() - b.date.getTime());

  return {
    start,
    end,
    events,
  };
};

const RiskEventSparkline: React.FC<{ timeline: PeriodTimeline | null; displayMode?: DisplayMode; fillHeight?: boolean }> = ({ timeline, displayMode = 'dollar', fillHeight = false }) => {
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [tooltipState, setTooltipState] = useState<{ visible: boolean; svgRect: DOMRect | null }>({
    visible: false,
    svgRect: null,
  });
  if (!timeline) return null;

  const width = 220;
  const height = 120;
  const baselineY = 60;
  const maxBarHeight = 54;
  const minBarHeight = 2;
  const barWidth = 2.4;
  const durationMs = Math.max(timeline.end.getTime() - timeline.start.getTime(), 1);

  // For % mode, use log scale so geometrically equal moves have equal bar heights
  // e.g. +100% (2x) and -50% (0.5x) are inverses and should be equal height
  const getBarMagnitude = (event: TimelineEvent): number => {
    if (displayMode === 'percent') {
      const multiplier = 1 + event.pctAmount / 100;
      if (multiplier <= 0) return 0;
      return Math.abs(Math.log(multiplier));
    }
    return Math.abs(event.dollarAmount);
  };

  const isPositive = (event: TimelineEvent): boolean => {
    return displayMode === 'percent' ? event.pctAmount >= 0 : event.dollarAmount >= 0;
  };

  const maxMagnitude = Math.max(...timeline.events.map(getBarMagnitude), 0);

  const formatTooltip = (event: TimelineEvent): string => {
    const dollarSign = event.dollarAmount >= 0 ? '+' : '−';
    const dollarStr = `${dollarSign}$${Math.abs(event.dollarAmount).toFixed(2)}`;
    const pctSign = event.pctAmount >= 0 ? '+' : '−';
    const pctStr = `${pctSign}${Math.abs(event.pctAmount).toFixed(2)}%`;
    return `${format(event.date, 'MMM d, yyyy')} • ${dollarStr} (${pctStr})`;
  };

  const bars = timeline.events.map((event, index) => {
    const ratio = clamp((event.date.getTime() - timeline.start.getTime()) / durationMs, 0, 1);
    const xCenter = 4 + ratio * (width - 8);
    const x = xCenter - barWidth / 2;
    const magnitude = getBarMagnitude(event);
    const strength = maxMagnitude > 0 ? magnitude / maxMagnitude : 0;
    const barHeight = minBarHeight + (maxBarHeight - minBarHeight) * strength;
    const positive = isPositive(event);
    const y = positive ? baselineY - barHeight : baselineY;

    return {
      key: `${event.date.getTime()}-${event.dollarAmount}-${index}`,
      x,
      xCenter,
      y,
      barHeight,
      color: event.dollarAmount >= 0 ? GAIN_COLOR : LOSS_COLOR,
      tooltip: formatTooltip(event),
    };
  });

  const safeIndex = clamp(selectedIndex, 0, Math.max(bars.length - 1, 0));
  const selectedBar = bars[safeIndex] ?? null;

  const updateSelectionFromMouse = (clientX: number, clientY: number, rect: DOMRect) => {
    if (!bars.length || rect.width <= 0 || rect.height <= 0) return;
    const mouseX = clamp(clientX - rect.left, 0, rect.width);
    const mouseY = clamp(clientY - rect.top, 0, rect.height);
    // Convert to SVG viewBox coordinates, accounting for preserveAspectRatio
    let viewX: number;
    let viewY: number;
    if (fillHeight) {
      // xMidYMid meet: uniform scale with pillarbox/letterbox offsets
      const uniformScale = Math.min(rect.width / width, rect.height / height);
      const offsetX = (rect.width - width * uniformScale) / 2;
      const offsetY = (rect.height - height * uniformScale) / 2;
      viewX = (mouseX - offsetX) / uniformScale;
      viewY = (mouseY - offsetY) / uniformScale;
    } else {
      // preserveAspectRatio="none": simple stretch mapping
      viewX = (mouseX / rect.width) * width;
      viewY = (mouseY / rect.height) * height;
    }

    // Find the nearest bar using 2D distance to bar center
    // This ensures same-day trades (same x) are distinguished by y position
    let nearestIndex = 0;
    let nearestDist = Infinity;

    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      const barCenterY = bar.y + bar.barHeight / 2;
      const dx = bars[i].xCenter - viewX;
      const dy = barCenterY - viewY;
      const dist = dx * dx + dy * dy; // squared distance (no need for sqrt)
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestIndex = i;
      }
    }

    setSelectedIndex(nearestIndex);
    setTooltipState({ visible: true, svgRect: rect });
  };

  // Convert the selected bar's SVG position to pixel coordinates relative to the container div.
  // Handles both preserveAspectRatio="none" (card) and "xMidYMid meet" (fullscreen).
  const getTooltipPixelPos = (): { left: number; top: number } | null => {
    if (!tooltipState.svgRect || !selectedBar) return null;
    const rect = tooltipState.svgRect;
    let scaleX: number;
    let scaleY: number;
    let offsetX = 0;
    let offsetY = 0;
    if (fillHeight) {
      // xMidYMid meet: uniform scale so the viewBox fits inside the element keeping aspect ratio
      const uniformScale = Math.min(rect.width / width, rect.height / height);
      scaleX = uniformScale;
      scaleY = uniformScale;
      // Content is centered; compute pillarbox / letterbox offsets
      offsetX = (rect.width - width * uniformScale) / 2;
      offsetY = (rect.height - height * uniformScale) / 2;
    } else {
      // preserveAspectRatio="none": SVG is stretched to fill exactly
      scaleX = rect.width / width;
      scaleY = rect.height / height;
    }
    // Bar tooltip anchors at the tip of the bar (top edge of bar rect)
    const barTipSvgY = selectedBar.y;
    const left = offsetX + selectedBar.xCenter * scaleX;
    const top = offsetY + barTipSvgY * scaleY;
    return { left, top };
  };

  const tooltipPixelPos = tooltipState.visible && selectedBar ? getTooltipPixelPos() : null;

  return (
    <div className={fillHeight ? 'relative h-full flex-1' : 'mb-2 relative'}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className={fillHeight ? 'w-full h-full' : 'w-full h-32'}
        preserveAspectRatio={fillHeight ? 'xMidYMid meet' : 'none'}
        aria-hidden="true"
        onMouseMove={(e) => updateSelectionFromMouse(e.clientX, e.clientY, e.currentTarget.getBoundingClientRect())}
        onMouseLeave={() => setTooltipState((prev) => ({ ...prev, visible: false }))}
      >
        <line
          x1={0}
          y1={baselineY}
          x2={width}
          y2={baselineY}
          stroke="hsl(var(--muted-foreground))"
          strokeOpacity={0.4}
          strokeWidth={1.2}
        />
        {bars.map((bar) => (
          <rect
            key={bar.key}
            x={bar.x}
            y={bar.y}
            width={barWidth}
            height={bar.barHeight}
            fill={bar.color}
            rx={0.8}
            opacity={selectedBar?.key === bar.key ? 1 : 0.85}
          />
        ))}
      </svg>
      {tooltipPixelPos && selectedBar ? (
        <div
          className="absolute z-10 pointer-events-none rounded-md border border-border bg-popover px-2 py-1 text-xs font-medium text-popover-foreground shadow-sm"
          style={{
            left: tooltipPixelPos.left,
            top: tooltipPixelPos.top,
            transform: 'translate(-50%, calc(-100% - 6px))',
            whiteSpace: 'nowrap',
          }}
        >
          {selectedBar.tooltip}
        </div>
      ) : null}
      <div className="flex items-center justify-between text-xs font-medium text-muted-foreground mt-0.5">
        <span>Start: {format(timeline.start, 'MMM d, yyyy')}</span>
        <span>End: {format(timeline.end, 'MMM d, yyyy')}</span>
      </div>
    </div>
  );
};

export const RiskOfRuinStats: React.FC<RiskOfRuinStatsProps> = ({
  targetReturnPercent,
  onTargetReturnChange,
  filteredTrades,
  currentBalance,
  useMonthFilter,
  onUseMonthFilterChange,
}) => {
  const [lastNTrades, setLastNTrades] = useState<number>(10);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('dollar');
  const [maximizedPeriod, setMaximizedPeriod] = useState<string | null>(null);

  // Lock body scroll when fullscreen overlay is open
  useEffect(() => {
    if (maximizedPeriod) {
      const prev = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      return () => { document.body.style.overflow = prev; };
    }
  }, [maximizedPeriod]);

  // currentBalance from props is actually the INITIAL balance (from balanceAPI).
  // Compute the real current balance by adding all trade dollar returns.
  const actualCurrentBalance = useMemo(() => {
    const totalReturn = filteredTrades.reduce((sum, t) => sum + (t.Return ?? 0), 0);
    return currentBalance + totalReturn;
  }, [filteredTrades, currentBalance]);

  // Trailing 30 days: find trades with Entry_Date in the last 30 days
  const trailing30 = useMemo(() => {
    const now = new Date();
    const thirtyDaysAgo = subDays(now, 30);
    const trades30 = filteredTrades.filter(t => {
      const entryDate = parseISO(t.Entry_Date);
      return isAfter(entryDate, thirtyDaysAgo);
    });
    const dollarReturn = trades30.reduce((sum, t) => sum + (t.Return ?? 0), 0);
    // Equity at start of 30-day window = actual current balance - dollar return from these trades
    const equityAtStart = actualCurrentBalance - dollarReturn;
    const equityChangePct = equityAtStart > 0 ? (dollarReturn / equityAtStart) * 100 : 0;
    const dateLabel = `${format(thirtyDaysAgo, 'MMM d')} - ${format(now, 'MMM d')}`;
    const timeline = buildTimelineFromTrades(trades30, thirtyDaysAgo, now);
    return { equityChangePct, dateLabel, hasData: trades30.length > 0, timeline };
  }, [filteredTrades, actualCurrentBalance]);

  const computeWindowPeriod = useCallback((
    start: Date,
    end: Date
  ) => {
    const tradesInWindow = filteredTrades.filter((t) => {
      const tradeDate = parseISO(t.Entry_Date);
      return tradeDate >= start && tradeDate <= end;
    });

    const dollarReturn = tradesInWindow.reduce((sum, t) => sum + (t.Return ?? 0), 0);
    const equityAtStart = actualCurrentBalance - dollarReturn;
    const equityChangePct = equityAtStart > 0 ? (dollarReturn / equityAtStart) * 100 : 0;

    const daySpan = Math.max(differenceInDays(end, start), 1);
    const monthsSpanned = daySpan / 30;
    const growthBase = 1 + equityChangePct / 100;
    const avgMonthly = monthsSpanned > 0 && growthBase > 0
      ? (Math.pow(growthBase, 1 / monthsSpanned) - 1) * 100
      : equityChangePct;

    return {
      equityChangePct,
      avgMonthly,
      hasData: tradesInWindow.length > 0,
      timeline: buildTimelineFromTrades(tradesInWindow, start, end),
      dateLabel: `${format(start, 'MMM d, yyyy')} - ${format(end, 'MMM d, yyyy')}`,
    };
  }, [filteredTrades, actualCurrentBalance]);

  // Last N trades
  const lastNTradesData = useMemo(() => {
    if (filteredTrades.length === 0) return { equityChangePct: 0, avgMonthly: 0, hasData: false, tradeCount: 0, dateLabel: '', timeline: null as PeriodTimeline | null };

    // Sort by Entry_Date descending and take last N
    const sorted = [...filteredTrades].sort(
      (a, b) => new Date(b.Entry_Date).getTime() - new Date(a.Entry_Date).getTime()
    );
    const nTrades = sorted.slice(0, lastNTrades);
    const tradeCount = nTrades.length;

    const dollarReturn = nTrades.reduce((sum, t) => sum + (t.Return ?? 0), 0);
    const equityAtStart = actualCurrentBalance - dollarReturn;
    const equityChangePct = equityAtStart > 0 ? (dollarReturn / equityAtStart) * 100 : 0;

    // Compute time span to extrapolate to monthly rate (geometric)
    const dates = nTrades.map(t => parseISO(t.Entry_Date));
    const earliest = new Date(Math.min(...dates.map(d => d.getTime())));
    const latest = new Date(Math.max(...dates.map(d => d.getTime())));
    const daySpan = Math.max(differenceInDays(latest, earliest), 1);
    const monthsSpanned = daySpan / 30;
    // Geometric: (1 + totalChange)^(1/months) - 1
    const avgMonthly = monthsSpanned > 0
      ? (Math.pow(1 + equityChangePct / 100, 1 / monthsSpanned) - 1) * 100
      : equityChangePct;

    const dateLabel = `${format(earliest, 'MMM d, yyyy')} - ${format(latest, 'MMM d, yyyy')}`;
    const timeline = buildTimelineFromTrades(nTrades, earliest, latest);

    return { equityChangePct, avgMonthly, hasData: tradeCount > 0, tradeCount, dateLabel, timeline };
  }, [filteredTrades, actualCurrentBalance, lastNTrades]);

  const last6MonthsData = useMemo(() => {
    const now = new Date();
    const start = startOfMonth(subMonths(now, 5));
    const end = endOfMonth(now);
    return computeWindowPeriod(start, end);
  }, [computeWindowPeriod]);

  const last12MonthsData = useMemo(() => {
    const now = new Date();
    const start = startOfMonth(subMonths(now, 11));
    const end = endOfMonth(now);
    return computeWindowPeriod(start, end);
  }, [computeWindowPeriod]);

  const allTimeData = useMemo(() => {
    if (filteredTrades.length === 0) {
      return { equityChangePct: 0, avgMonthly: 0, hasData: false, dateLabel: '', timeline: null as PeriodTimeline | null };
    }

    const dates = filteredTrades.map((t) => parseISO(t.Entry_Date));
    const earliest = new Date(Math.min(...dates.map((d) => d.getTime())));
    const latest = new Date(Math.max(...dates.map((d) => d.getTime())));
    return computeWindowPeriod(earliest, latest);
  }, [filteredTrades, computeWindowPeriod]);

  const periods = [
    {
      label: `Trailing 30 Days (${trailing30.dateLabel})`,
      equityChange: trailing30.equityChangePct,
      avgMonthly: trailing30.equityChangePct,
      hasData: trailing30.hasData,
      timeline: trailing30.timeline,
      icon: Clock,
    },
    {
      label: `Last ${lastNTrades} Trades${lastNTradesData.hasData ? ` (${lastNTradesData.dateLabel})` : ''}`,
      equityChange: lastNTradesData.equityChangePct,
      avgMonthly: lastNTradesData.avgMonthly,
      hasData: lastNTradesData.hasData,
      timeline: lastNTradesData.timeline,
      icon: BarChart3,
    },
    {
      label: `Last 6 Months (${last6MonthsData.dateLabel})`,
      equityChange: last6MonthsData.equityChangePct,
      avgMonthly: last6MonthsData.avgMonthly,
      hasData: last6MonthsData.hasData,
      timeline: last6MonthsData.timeline,
      icon: Calendar,
    },
    {
      label: `Last 12 Months (${last12MonthsData.dateLabel})`,
      equityChange: last12MonthsData.equityChangePct,
      avgMonthly: last12MonthsData.avgMonthly,
      hasData: last12MonthsData.hasData,
      timeline: last12MonthsData.timeline,
      icon: TrendingUp,
    },
    {
      label: `All Time${allTimeData.hasData ? ` (${allTimeData.dateLabel})` : ''}`,
      equityChange: allTimeData.equityChangePct,
      avgMonthly: allTimeData.avgMonthly,
      hasData: allTimeData.hasData,
      timeline: allTimeData.timeline,
      icon: TrendingUp,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Risk of Ruin / Return Analysis
            </CardTitle>
            <div className="flex items-center gap-2">
              <Checkbox
                id="use-month-filter-risk"
                checked={useMonthFilter}
                onCheckedChange={(checked) => onUseMonthFilterChange(checked === true)}
              />
              <label htmlFor="use-month-filter-risk" className="text-sm text-muted-foreground cursor-pointer whitespace-nowrap">
                Use selected months
              </label>
            </div>
          </div>
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground whitespace-nowrap">Target Return:</label>
              <Input
                type="number"
                value={targetReturnPercent}
                onChange={(e) => onTargetReturnChange(Number(e.target.value) || 100)}
                className="w-20 h-8 text-sm"
                min={1}
              />
              <span className="text-sm text-muted-foreground">%</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground whitespace-nowrap">Last N Trades:</label>
              <Input
                type="number"
                value={lastNTrades}
                onChange={(e) => setLastNTrades(Math.max(1, Number(e.target.value) || 10))}
                className="w-20 h-8 text-sm"
                min={1}
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-muted-foreground whitespace-nowrap">Bar Height:</label>
              <Select value={displayMode} onValueChange={(v) => setDisplayMode(v as DisplayMode)}>
                <SelectTrigger className="w-[140px] h-8 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="dollar">$ Amount</SelectItem>
                  <SelectItem value="percent">% Gain/Loss</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {periods.map((period) => (
            <Card key={period.label} className="bg-card border border-border">
              <CardContent className="py-3 px-4">
                <div className="flex items-center gap-2 mb-2">
                  <period.icon className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground font-medium">{period.label}</span>
                </div>
                {period.hasData ? (
                  <>
                    <div className="flex items-center justify-between mb-1">
                      <div className="text-sm text-muted-foreground">Equity Change</div>
                      <button
                        onClick={() => setMaximizedPeriod(maximizedPeriod === period.label ? null : period.label)}
                        className="p-1 rounded hover:bg-muted text-muted-foreground"
                        title="Maximize"
                      >
                        <Maximize2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <div className={`text-lg font-semibold mb-3 ${period.equityChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {period.equityChange >= 0 ? '+' : ''}{period.equityChange.toFixed(2)}%
                    </div>
                    <RiskEventSparkline timeline={period.timeline} displayMode={displayMode} />
                    <div className="space-y-1">
                      {ruinLevels.map((level) => {
                        const result = computeTimeToLevel(period.avgMonthly, level);
                        return (
                          <div key={level} className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground whitespace-nowrap">To {level}%:</span>
                            <span style={{ color: result.color }} className="font-medium whitespace-nowrap">
                              {result.text}
                            </span>
                          </div>
                        );
                      })}
                      <div className="border-t border-border my-1" />
                      {(() => {
                        const result = computeTimeToLevel(period.avgMonthly, targetReturnPercent);
                        return (
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground whitespace-nowrap">To +{targetReturnPercent}%:</span>
                            <span style={{ color: result.color }} className="font-medium whitespace-nowrap">
                              {result.text}
                            </span>
                          </div>
                        );
                      })()}
                    </div>
                  </>
                ) : (
                  <div className="text-sm text-muted-foreground">No data</div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </CardContent>

      {/* Full-screen maximize overlay via portal */}
      {maximizedPeriod && (() => {
        const period = periods.find(p => p.label === maximizedPeriod);
        if (!period || !period.hasData) return null;
        return createPortal(
          <div className="fixed inset-0 z-[9999] bg-background flex flex-col" style={{ isolation: 'isolate' }}>
            <div className="flex items-center justify-between p-4 border-b border-border">
              <div className="flex items-center gap-2">
                <period.icon className="w-5 h-5 text-muted-foreground" />
                <span className="text-lg font-semibold">{period.label}</span>
                <span className={`text-lg font-semibold ml-4 ${period.equityChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {period.equityChange >= 0 ? '+' : ''}{period.equityChange.toFixed(2)}%
                </span>
              </div>
              <button
                onClick={() => setMaximizedPeriod(null)}
                className="p-2 rounded-md hover:bg-muted text-muted-foreground"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 flex flex-col overflow-hidden">
              <div className="flex-1 min-h-0 p-6 pb-2">
                <RiskEventSparkline timeline={period.timeline} displayMode={displayMode} fillHeight />
              </div>
              <div className="shrink-0 border-t border-border px-6 py-4">
                <div className="flex flex-wrap items-center gap-x-8 gap-y-2 justify-center">
                  {ruinLevels.map((level) => {
                    const result = computeTimeToLevel(period.avgMonthly, level);
                    return (
                      <div key={level} className="flex items-center gap-2 text-sm">
                        <span className="text-muted-foreground">To {level}%:</span>
                        <span style={{ color: result.color }} className="font-semibold">
                          {result.text}
                        </span>
                      </div>
                    );
                  })}
                  <div className="w-px h-5 bg-border" />
                  {(() => {
                    const result = computeTimeToLevel(period.avgMonthly, targetReturnPercent);
                    return (
                      <div className="flex items-center gap-2 text-sm">
                        <span className="text-muted-foreground">To +{targetReturnPercent}%:</span>
                        <span style={{ color: result.color }} className="font-semibold">
                          {result.text}
                        </span>
                      </div>
                    );
                  })()}
                </div>
              </div>
            </div>
          </div>,
          document.body
        );
      })()}
    </Card>
  );
};
