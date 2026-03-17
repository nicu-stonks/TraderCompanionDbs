import React, { useMemo } from 'react';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import { Metric, TradeGrade } from '../types/types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { BarChart3, TrendingUp } from 'lucide-react';

// Vibrant color palette for chart lines (works in both dark and light mode)
const vibrantColors = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e42', // orange
  '#a21caf', // purple
  '#ef4444', // red
  '#14b8a6', // teal
  '#eab308', // yellow
  '#6366f1', // indigo
];

// Hard-coded mapping of specific metric option names (or IDs) to colors.
// Edit / extend this object to force particular options to always have the same color
// irrespective of ordering. Keys should match option.name exactly (case-sensitive) unless
// you normalize below.
const optionColorMap: Record<string, string> = {
  // EXAMPLES (replace with your real option names):
  'Bought perfect': '#10b981',
  'Bought too soon': '#3b82f6',
  'Bought too late': '#f59e42',
  'Faulty set-up': '#a21caf',
  'Cut loss perfect': '#10b981',
  'Cut loss too late': '#ef4444',
  'Cut loss too soon': '#f59e42',
  'Sold perfect': '#10b981',
  'Sold too late': '#f59e42',
  'Excellent': '#10b981',
  'Mediocre': '#3b82f6',
  'Poor': '#a21caf'
};

const MetricAnalytics: React.FC<{
  trades: Trade[];
  metrics: Metric[];
  tradeGrades: TradeGrade[];
  trailingWindow: number;
  onTrailingWindowChange: (next: number) => void;
  layoutMode: 'stacked' | 'grid';
  onLayoutModeChange: (next: 'stacked' | 'grid') => void;
}> = ({ trades, metrics, tradeGrades, trailingWindow, onTrailingWindowChange, layoutMode, onLayoutModeChange }) => {

  interface ChartRowBase {
    tradeIndex: number;
    ticker: string;
    date: string;
    // Dynamic metric option counts will be added as numeric properties
    [optionName: string]: string | number; // counts are numbers, metadata are strings
  }

  const chartData = useMemo(() => {
    const sortedTrades = [...trades].sort(
      (a, b) => new Date(a.Entry_Date).getTime() - new Date(b.Entry_Date).getTime()
    );

    // Determine the last trade (by Entry_Date order) that has at least one metric grade.
    // We only want to include trades up to (and including) this one in the analytics charts.
    let lastGradedIndex = -1;
    if (tradeGrades.length > 0) {
      // Create a quick lookup of graded trade IDs
      const gradedTradeIds = new Set(tradeGrades.map(g => g.tradeId));
      for (let i = 0; i < sortedTrades.length; i++) {
        if (gradedTradeIds.has(sortedTrades[i].ID)) {
          lastGradedIndex = i; // because sorted ascending, this will end at the last graded trade
        }
      }
    }

    // If no trades have been graded yet, we produce empty datasets so user doesn't see future stats.
    const effectiveTrades = lastGradedIndex >= 0 ? sortedTrades.slice(0, lastGradedIndex + 1) : [];

    const data: { [key: string]: ChartRowBase[] } = {};

    metrics.forEach(metric => {
      const metricData: ChartRowBase[] = [];

      for (let i = 0; i < effectiveTrades.length; i++) {
        const windowStart = Math.max(0, i - (trailingWindow - 1));
        const windowTrades = effectiveTrades.slice(windowStart, i + 1);

        const dataPoint: ChartRowBase = {
          tradeIndex: i + 1,
          ticker: effectiveTrades[i].Ticker,
          date: effectiveTrades[i].Entry_Date,
        };

        metric.options.forEach(option => {
          const count = windowTrades.reduce((acc, trade) => {
            const grade = tradeGrades.find(
              g => g.tradeId === trade.ID && parseInt(g.metricId) === metric.id
            );
            return acc + (grade?.selectedOptionId === option.id.toString() ? 1 : 0);
          }, 0);

          dataPoint[option.name] = count;
        });

        metricData.push(dataPoint);
      }

      data[metric.id] = metricData;
    });

    return data;
  }, [trades, metrics, tradeGrades, trailingWindow]);

  // Helper to fetch a color for a given option name with fallback to palette order.
  const getOptionColor = (name: string, index: number) => {
    // Direct lookup; customize (e.g. toLowerCase) if you prefer case-insensitive keys.
    return optionColorMap[name] || vibrantColors[index % vibrantColors.length];
  };

  if (metrics.length === 0) {
    return (
      <div className="bg-background rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4 flex items-center text-foreground">
          <BarChart3 className="mr-2" />
          Analytics
        </h2>
        <p className="text-muted-foreground">Create metrics and grade trades to see analytics.</p>
      </div>
    );
  }

  return (
    <div className="bg-background rounded-lg shadow-md p-6 mb-6">
      {/* <h2 className="text-2xl font-bold mb-6 flex items-center text-foreground">
        <BarChart3 className="mr-2" />
        Analytics Dashboard
      </h2> */}

      {/* ✅ Controls: trailing window + layout toggle */}
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center">
          <label className="text-sm font-medium text-foreground mr-2 whitespace-nowrap">
            Trailing Window (trades):
          </label>
          <input
            type="number"
            value={trailingWindow}
            onChange={e => onTrailingWindowChange(Math.max(1, Number(e.target.value)))}
            className="border border-input rounded-md px-2 py-1 w-24 text-center bg-background text-foreground"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-foreground">Layout:</span>
          <div className="inline-flex rounded-md overflow-hidden border border-border">
            <button
              type="button"
              onClick={() => onLayoutModeChange('stacked')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${layoutMode === 'stacked' ? 'bg-primary text-primary-foreground' : 'bg-card text-foreground hover:bg-muted'}`}
              aria-pressed={layoutMode === 'stacked'}
            >
              Stacked
            </button>
            <button
              type="button"
              onClick={() => onLayoutModeChange('grid')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors border-l border-border ${layoutMode === 'grid' ? 'bg-primary text-primary-foreground' : 'bg-card text-foreground hover:bg-muted'}`}
              aria-pressed={layoutMode === 'grid'}
            >
              2 x 2
            </button>
          </div>
        </div>
      </div>

      {/** Container switches between vertical stack and responsive 2-column grid */}
      <div className={layoutMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 gap-8' : 'space-y-8'}>
        {metrics.map(metric => {
          const data = chartData[metric.id];
          if (!data || data.length === 0) return null;

          return (
            <div
              key={metric.id}
              className={`border border-border rounded-lg p-4 bg-card ${layoutMode === 'grid' ? '' : ''}`}
            >
              <h3 className="text-xl font-semibold mb-4 flex items-center text-foreground">
                <TrendingUp className="mr-2 w-5 h-5" />
                {metric.name} Trends (Trailing {trailingWindow} Trades)
              </h3>

              {metric.description && (
                <p className="text-muted-foreground mb-4 text-sm">{metric.description}</p>
              )}

              <div className={layoutMode === 'grid' ? 'h-64' : 'h-80'}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
                    <XAxis
                      dataKey="tradeIndex"
                      label={{ value: 'Trade Number', position: 'insideBottom', offset: -5 }}
                      stroke="hsl(var(--foreground))"
                      tick={{ fill: 'hsl(var(--foreground))' }}
                    />
                    <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} stroke="hsl(var(--foreground))" tick={{ fill: 'hsl(var(--foreground))' }} />
                    <Tooltip
                      labelFormatter={value => `Trade #${value}`}
                      formatter={(value: number, name: string) => [`${value} trades`, name]}
                      labelStyle={{ color: 'hsl(var(--foreground))' }}
                      contentStyle={{ backgroundColor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', color: 'hsl(var(--foreground))' }}
                    />
                    <Legend wrapperStyle={{ color: 'hsl(var(--foreground))' }} />
                    {metric.options.map((option, index) => {
                      const strokeColor = getOptionColor(option.name, index);
                      return (
                        <Line
                          key={option.id}
                          type="monotone"
                          dataKey={option.name}
                          stroke={strokeColor}
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 5 }}
                          connectNulls={true}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default MetricAnalytics;
