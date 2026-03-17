import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { TrendingUp, BarChart3, Filter, ShieldCheck, ChevronDown, ChevronUp } from 'lucide-react';
import { Metric, TradeGrade } from '@/PostAnalysisPage/types/types';
import { gradeService, metricService } from '@/PostAnalysisPage/services/postAnalysis';
import type { Trade } from '@/TradeHistoryPage/types/Trade';
import { customTradeDataAPI } from '@/TradeHistoryPage/services/customTradeDataAPI';
import type { CustomColumn, CustomColumnValue } from '@/TradeHistoryPage/types/CustomTradeData';

// ── Color palette ──
const vibrantColors = [
  '#3b82f6', '#10b981', '#f59e42', '#a21caf', '#ef4444',
  '#14b8a6', '#eab308', '#6366f1', '#f472b6', '#22d3ee',
  '#84cc16', '#fb923c',
];

const optionColorMap: Record<string, string> = {
  'Bought perfect': '#10b981',
  'Bought too soon': '#3b82f6',
  'Bought too late': '#f59e42',
  'Faulty set-up': '#a21caf',
  'Bad buy': '#ef4444',
  'Cut loss perfect': '#10b981',
  'Cut loss too late': '#ef4444',
  'Cut loss too soon': '#f59e42',
  'Sold perfect': '#10b981',
  'Sold too late': '#f59e42',
  'Sold too soon': '#3b82f6',
  'Excellent': '#10b981',
  'Mediocre': '#3b82f6',
  'Poor': '#a21caf',
};

// ── Filter fields to chart ──
const FILTER_FIELDS: { key: keyof Trade; label: string }[] = [
  { key: 'Pattern', label: 'Pattern' },
  { key: 'Status', label: 'Status' },
  { key: 'Category', label: 'Category' },
  { key: 'Market_Condition', label: 'Market Condition' },
  { key: 'C', label: 'C (Current Earnings)' },
  { key: 'A', label: 'A (Annual Earnings)' },
  { key: 'N', label: 'N (New Products/Highs)' },
  { key: 'S', label: 'S (Supply/Demand)' },
  { key: 'L', label: 'L (Leader/Laggard)' },
  { key: 'I', label: 'I (Institutional)' },
  { key: 'M', label: 'M (Market Direction)' },
];

const CAN_SLIM_KEYS: (keyof Trade)[] = ['C', 'A', 'N', 'S', 'L', 'I', 'M'];

// ── Helper to get color ──
function getOptionColor(name: string, index: number) {
  return optionColorMap[name] || vibrantColors[index % vibrantColors.length];
}

// ── Types ──
interface TradeTrendChartsProps {
  filteredTrades: Trade[];
  useMonthFilter: boolean;
  onUseMonthFilterChange: (checked: boolean) => void;
}

export const TradeTrendCharts: React.FC<TradeTrendChartsProps> = ({
  filteredTrades,
  useMonthFilter,
  onUseMonthFilterChange,
}) => {
  // ── State ──
  const [trailingWindow, setTrailingWindow] = useState(50);
  const [canSlimWindow, setCanSlimWindow] = useState(5);
  const [layoutMode, setLayoutMode] = useState<'stacked' | 'grid'>('grid');
  const [isExpanded, setIsExpanded] = useState(false);

  // ── Fetch metrics + grades ──
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [tradeGrades, setTradeGrades] = useState<TradeGrade[]>([]);
  const [customColumns, setCustomColumns] = useState<CustomColumn[]>([]);
  const [customValues, setCustomValues] = useState<CustomColumnValue[]>([]);

  useEffect(() => {
    Promise.all([metricService.getMetrics(), gradeService.getGrades()])
      .then(([m, g]) => { setMetrics(m); setTradeGrades(g); })
      .catch(err => console.error('Error loading post-analysis data:', err));
  }, []);

  useEffect(() => {
    customTradeDataAPI.getColumns()
      .then(resp => setCustomColumns(resp.data))
      .catch(err => console.error('Error loading custom columns:', err));
  }, []);

  useEffect(() => {
    const tradeIds = filteredTrades.map(t => t.ID);
    if (tradeIds.length === 0) {
      setCustomValues([]);
      return;
    }

    customTradeDataAPI.getColumnValues(tradeIds)
      .then(resp => setCustomValues(resp.data))
      .catch(err => console.error('Error loading custom column values:', err));
  }, [filteredTrades]);

  // ── Sort trades by entry date ──
  const sortedTrades = useMemo(() =>
    [...filteredTrades].sort((a, b) => new Date(a.Entry_Date).getTime() - new Date(b.Entry_Date).getTime()),
    [filteredTrades]
  );

  const customValuesByTrade = useMemo(() => {
    const map = new Map<number, Map<number, string>>();

    for (const val of customValues) {
      const inner = map.get(val.trade_id) ?? new Map<number, string>();
      inner.set(val.column, val.value);
      map.set(val.trade_id, inner);
    }

    return map;
  }, [customValues]);

  const characteristicFields = useMemo(() => {
    const builtInFields = FILTER_FIELDS.map(field => ({
      id: `builtin:${String(field.key)}`,
      label: field.label,
      getValue: (trade: Trade) => {
        const val = trade[field.key];
        if (val === undefined || val === null) return null;
        return String(val);
      },
    }));

    const customFieldDefs = customColumns.map(column => ({
      id: `custom:${column.id}`,
      label: column.name,
      getValue: (trade: Trade) => {
        const tradeMap = customValuesByTrade.get(trade.ID);
        const val = tradeMap?.get(column.id);
        if (val === undefined || val === null || val === '') return null;
        return String(val);
      },
    }));

    return [...builtInFields, ...customFieldDefs];
  }, [customColumns, customValuesByTrade]);

  // ═══════════════════════════════════════
  // 1) POST ANALYSIS METRIC TREND CHARTS
  // ═══════════════════════════════════════
  const metricChartData = useMemo(() => {
    const gradedTradeIds = new Set(tradeGrades.map(g => g.tradeId));
    let lastGradedIndex = -1;
    for (let i = 0; i < sortedTrades.length; i++) {
      if (gradedTradeIds.has(sortedTrades[i].ID)) lastGradedIndex = i;
    }
    const effectiveTrades = lastGradedIndex >= 0 ? sortedTrades.slice(0, lastGradedIndex + 1) : [];

    const data: Record<number, { tradeIndex: number; ticker: string; date: string;[k: string]: string | number }[]> = {};

    for (const metric of metrics) {
      const metricData: typeof data[number] = [];

      for (let i = 0; i < effectiveTrades.length; i++) {
        const windowStart = Math.max(0, i - (trailingWindow - 1));
        const windowTrades = effectiveTrades.slice(windowStart, i + 1);
        const windowTotal = windowTrades.length;

        const point: typeof metricData[number] = {
          tradeIndex: i + 1,
          ticker: effectiveTrades[i].Ticker,
          date: effectiveTrades[i].Entry_Date,
          __windowTotal: windowTotal,
        };

        for (const option of metric.options) {
          const count = windowTrades.reduce((acc, trade) => {
            const grade = tradeGrades.find(
              g => g.tradeId === trade.ID && parseInt(g.metricId) === metric.id
            );
            return acc + (grade?.selectedOptionId === option.id.toString() ? 1 : 0);
          }, 0);
          point[option.name] = count;
        }

        metricData.push(point);
      }

      data[metric.id] = metricData;
    }

    return data;
  }, [sortedTrades, metrics, tradeGrades, trailingWindow]);

  // ═══════════════════════════════════════
  // 2) TRADE FILTER FIELD TREND CHARTS
  // ═══════════════════════════════════════
  const filterFieldChartData = useMemo(() => {
    const data: Record<string, { tradeIndex: number; ticker: string; date: string;[k: string]: string | number }[]> = {};

    for (const field of characteristicFields) {
      const fieldKey = field.id;

      // Collect all unique values for this field
      const uniqueValues = new Set<string>();
      for (const trade of sortedTrades) {
        const val = field.getValue(trade);
        if (val !== undefined && val !== null) {
          uniqueValues.add(val);
        }
      }

      const valueList = Array.from(uniqueValues).sort();
      if (valueList.length === 0) continue;

      const fieldData: typeof data[string] = [];

      for (let i = 0; i < sortedTrades.length; i++) {
        const windowStart = Math.max(0, i - (trailingWindow - 1));
        const windowTrades = sortedTrades.slice(windowStart, i + 1);
        const windowTotal = windowTrades.length;

        const point: typeof fieldData[number] = {
          tradeIndex: i + 1,
          ticker: sortedTrades[i].Ticker,
          date: sortedTrades[i].Entry_Date,
          __windowTotal: windowTotal,
        };

        for (const val of valueList) {
          const count = windowTrades.filter(t => field.getValue(t) === val).length;
          point[val] = count;
        }

        fieldData.push(point);
      }

      data[fieldKey] = fieldData;
    }

    return data;
  }, [sortedTrades, trailingWindow, characteristicFields]);

  // Collect unique values per field for legend colors
  const filterFieldValues = useMemo(() => {
    const result: Record<string, string[]> = {};
    for (const field of characteristicFields) {
      const vals = new Set<string>();
      for (const trade of sortedTrades) {
        const val = field.getValue(trade);
        if (val !== undefined && val !== null) vals.add(val);
      }
      result[field.id] = Array.from(vals).sort();
    }
    return result;
  }, [sortedTrades, characteristicFields]);

  // ═══════════════════════════════════════
  // 3) CAN SLIM TRAILING COVERAGE CHART
  // ═══════════════════════════════════════
  const canSlimData = useMemo(() => {
    if (sortedTrades.length === 0) return [];

    const data: { tradeIndex: number; ticker: string; date: string; coverage: number }[] = [];

    for (let i = 0; i < sortedTrades.length; i++) {
      const windowStart = Math.max(0, i - (canSlimWindow - 1));
      const windowTrades = sortedTrades.slice(windowStart, i + 1);

      // For each trade in window, compute CAN SLIM completion %
      let totalCompletion = 0;
      for (const trade of windowTrades) {
        let checked = 0;
        for (const k of CAN_SLIM_KEYS) {
          if (trade[k]) checked++;
        }
        totalCompletion += (checked / CAN_SLIM_KEYS.length) * 100;
      }

      const avgCoverage = totalCompletion / windowTrades.length;

      data.push({
        tradeIndex: i + 1,
        ticker: sortedTrades[i].Ticker,
        date: sortedTrades[i].Entry_Date,
        coverage: Math.round(avgCoverage * 100) / 100,
      });
    }

    return data;
  }, [sortedTrades, canSlimWindow]);

  // ═══════════════════════════════════════
  //  RENDER
  // ═══════════════════════════════════════
  const chartHeight = layoutMode === 'grid' ? 'h-64' : 'h-80';

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
        <div className="space-y-1.5">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Trade Trend Analytics
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Trailing window analysis of your trading patterns, post-analysis metrics, and CAN SLIM coverage
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-muted/20">
          <input
            type="checkbox"
            id="use-month-filter-trends-header"
            checked={useMonthFilter}
            onChange={e => onUseMonthFilterChange(e.target.checked)}
            className="accent-primary h-4 w-4 cursor-pointer"
          />
          <label htmlFor="use-month-filter-trends-header" className="text-sm font-medium text-muted-foreground cursor-pointer whitespace-nowrap">
            Use selected months
          </label>
        </div>
      </CardHeader>
      <CardContent className="space-y-8">
        {/* ═══════════════════════════════════════ */}
        {/*  CAN SLIM TRAILING COVERAGE CHART      */}
        {/* ═══════════════════════════════════════ */}
        {sortedTrades.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <ShieldCheck className="h-4 w-4" />
              CAN SLIM Trailing Coverage
            </h3>
            <div className="flex items-center gap-2 mb-3">
              <label className="text-sm font-medium whitespace-nowrap">Trailing Window:</label>
              <input
                type="number"
                value={canSlimWindow}
                onChange={e => setCanSlimWindow(Math.max(1, Number(e.target.value)))}
                className="border border-input rounded-md px-2 py-1 w-20 text-center bg-background text-foreground text-sm"
              />
              <span className="text-xs text-muted-foreground">
                trades (average CAN SLIM completion % over last {canSlimWindow} trades)
              </span>
            </div>
            <div className="border border-border rounded-lg p-4 bg-card">
              <h4 className="text-base font-semibold mb-3 flex items-center">
                <ShieldCheck className="mr-2 w-4 h-4" />
                CAN SLIM Coverage % (Trailing {canSlimWindow} Trades)
              </h4>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={canSlimData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
                    <XAxis dataKey="tradeIndex" label={{ value: 'Trade Number', position: 'insideBottom', offset: -5 }} stroke="hsl(var(--foreground))" tick={{ fill: 'hsl(var(--foreground))' }} />
                    <YAxis
                      domain={[0, 100]}
                      label={{ value: 'Coverage %', angle: -90, position: 'insideLeft' }}
                      stroke="hsl(var(--foreground))"
                      tick={{ fill: 'hsl(var(--foreground))' }}
                      tickFormatter={v => `${v}%`}
                    />
                    <Tooltip
                      labelFormatter={v => `Trade #${v}`}
                      formatter={(value: number) => [`${value.toFixed(1)}%`, 'CAN SLIM Coverage']}
                      contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', color: 'hsl(var(--foreground))' }}
                    />
                    <Legend wrapperStyle={{ color: 'hsl(var(--foreground))' }} />
                    <Line type="monotone" dataKey="coverage" name="CAN SLIM Coverage %" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Each point shows the average CAN SLIM checkbox completion (C+A+N+S+L+I+M = 7 checks) across the trailing {canSlimWindow} trades. 100% means all {canSlimWindow} trades had all 7 CAN SLIM criteria checked.
              </p>
            </div>
          </div>
        )}

        {sortedTrades.length > 0 && (
          <div className="pt-2 flex justify-center">
            <button
              onClick={() => setIsExpanded(p => !p)}
              className="flex items-center gap-3 bg-primary text-primary-foreground px-8 py-3 rounded-full font-bold shadow-lg hover:bg-primary/90 transition-all active:scale-95"
            >
              <span className="text-sm uppercase tracking-widest">{isExpanded ? 'Hide Detailed Trends' : 'View Detailed Trends'}</span>
              {isExpanded ? <ChevronUp className="h-5 w-5 stroke-[3]" /> : <ChevronDown className="h-5 w-5 stroke-[3]" />}
            </button>
          </div>
        )}

        {isExpanded && sortedTrades.length > 0 && (
          <div className="space-y-6 pt-4 border-t border-border border-dashed animate-in fade-in duration-500">
            {/* ── Controls ── */}
            <div className="flex flex-wrap items-center gap-4">
              {/* Trailing window */}
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium whitespace-nowrap">Trailing Window:</label>
                <input
                  type="number"
                  value={trailingWindow}
                  onChange={e => setTrailingWindow(Math.max(1, Number(e.target.value)))}
                  className="border border-input rounded-md px-2 py-1 w-20 text-center bg-background text-foreground text-sm"
                />
              </div>

              {/* Layout toggle */}
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Layout:</span>
                <div className="inline-flex rounded-md overflow-hidden border border-border">
                  <button
                    type="button"
                    onClick={() => setLayoutMode('stacked')}
                    className={`px-3 py-1 text-xs font-medium transition-colors ${layoutMode === 'stacked' ? 'bg-primary text-primary-foreground' : 'bg-card text-foreground hover:bg-muted'}`}
                  >
                    Stacked
                  </button>
                  <button
                    type="button"
                    onClick={() => setLayoutMode('grid')}
                    className={`px-3 py-1 text-xs font-medium transition-colors border-l border-border ${layoutMode === 'grid' ? 'bg-primary text-primary-foreground' : 'bg-card text-foreground hover:bg-muted'}`}
                  >
                    2×2
                  </button>
                </div>
              </div>
            </div>

            {/* ═══════════════════════════════════════ */}
            {/*  POST ANALYSIS METRIC TREND CHARTS     */}
            {/* ═══════════════════════════════════════ */}
            {metrics.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" />
                  Post Analysis Trends
                </h3>
                <div className={layoutMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 gap-6' : 'space-y-6'}>
                  {metrics.map(metric => {
                    const data = metricChartData[metric.id];
                    if (!data || data.length === 0) return null;
                    return (
                      <div key={metric.id} className="border border-border rounded-lg p-4 bg-card">
                        <h4 className="text-base font-semibold mb-3 flex items-center">
                          <TrendingUp className="mr-2 w-4 h-4" />
                          {metric.name} Trends (Trailing {trailingWindow} Trades)
                        </h4>
                        <div className={chartHeight}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data}>
                              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
                              <XAxis dataKey="tradeIndex" label={{ value: 'Trade Number', position: 'insideBottom', offset: -5 }} stroke="hsl(var(--foreground))" tick={{ fill: 'hsl(var(--foreground))' }} />
                              <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} stroke="hsl(var(--foreground))" tick={{ fill: 'hsl(var(--foreground))' }} />
                              <Tooltip
                                labelFormatter={v => `Trade #${v}`}
                                formatter={(value: number, name: string, entry: { payload?: Record<string, unknown> }) => {
                                  const windowTotalRaw = entry?.payload?.__windowTotal;
                                  const windowTotal = typeof windowTotalRaw === 'number' ? windowTotalRaw : 0;
                                  const pct = windowTotal > 0 ? (value / windowTotal) * 100 : 0;
                                  return [`${value} trades (${pct.toFixed(1)}%)`, name];
                                }}
                                contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', color: 'hsl(var(--foreground))' }}
                              />
                              <Legend wrapperStyle={{ color: 'hsl(var(--foreground))' }} />
                              {metric.options.map((option, idx) => (
                                <Line key={option.id} type="monotone" dataKey={option.name} stroke={getOptionColor(option.name, idx)} strokeWidth={2} dot={false} activeDot={{ r: 5 }} connectNulls />
                              ))}
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* ═══════════════════════════════════════ */}
            {/*  TRADE FILTER FIELD TREND CHARTS       */}
            {/* ═══════════════════════════════════════ */}
            {sortedTrades.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Filter className="h-4 w-4" />
                  Trade Characteristic Trends
                </h3>
                <div className={layoutMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 gap-6' : 'space-y-6'}>
                  {characteristicFields.map(field => {
                    const data = filterFieldChartData[field.id];
                    if (!data || data.length === 0) return null;
                    const values = filterFieldValues[field.id] || [];
                    if (values.length === 0) return null;

                    return (
                      <div key={field.id} className="border border-border rounded-lg p-4 bg-card">
                        <h4 className="text-base font-semibold mb-3 flex items-center">
                          <TrendingUp className="mr-2 w-4 h-4" />
                          {field.label} Trends (Trailing {trailingWindow} Trades)
                        </h4>
                        <div className={chartHeight}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data}>
                              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
                              <XAxis dataKey="tradeIndex" label={{ value: 'Trade Number', position: 'insideBottom', offset: -5 }} stroke="hsl(var(--foreground))" tick={{ fill: 'hsl(var(--foreground))' }} />
                              <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} stroke="hsl(var(--foreground))" tick={{ fill: 'hsl(var(--foreground))' }} />
                              <Tooltip
                                labelFormatter={v => `Trade #${v}`}
                                formatter={(value: number, name: string, entry: { payload?: Record<string, unknown> }) => {
                                  const windowTotalRaw = entry?.payload?.__windowTotal;
                                  const windowTotal = typeof windowTotalRaw === 'number' ? windowTotalRaw : 0;
                                  const pct = windowTotal > 0 ? (value / windowTotal) * 100 : 0;
                                  return [`${value} trades (${pct.toFixed(1)}%)`, name];
                                }}
                                contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', color: 'hsl(var(--foreground))' }}
                              />
                              <Legend wrapperStyle={{ color: 'hsl(var(--foreground))' }} />
                              {values.map((val, idx) => (
                                <Line key={val} type="monotone" dataKey={val} stroke={vibrantColors[idx % vibrantColors.length]} strokeWidth={2} dot={false} activeDot={{ r: 5 }} connectNulls />
                              ))}
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

          </div>
        )}

        {sortedTrades.length === 0 && (
          <div className="text-center py-12 text-muted-foreground">
            No trades match the current filters. Adjust your filters or date range to see trend analytics.
          </div>
        )}
      </CardContent>
    </Card>
  );
};
