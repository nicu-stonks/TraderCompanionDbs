import React, { useState, useEffect, useMemo } from 'react';
import { balanceAPI } from './services/balanceAPI';
import { MonthlyStatistics } from './components/MonthlyStatistics';
import { YearlyStatistics } from './components/YearlyStatistics';
import { TradeFilterer } from './components/TradeFilterer';
import { useTradeStats } from './hooks/useTradeStats';
import { RiskPoolStats } from './components/RiskPoolStats';
import { TradeDistribution } from './components/TradeDistribution';
import { EquityCurve } from './components/EquityCurve';
import { MarketTimingChart } from './components/MarketTimingChart';
import { TradeTrendCharts } from './components/TradeTrendCharts';
import { TopPercentTradesTable } from './components/TopPercentTradesTable';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { addMonths, format, parseISO } from 'date-fns';
import { PostAnalysisFilters } from './components/PostAnalysisFilters';
import { RiskOfRuinStats } from './components/RiskOfRuinStats';
import type { MetricOptionFilters, ExtendedFilters } from './types';
import { Filter } from 'lucide-react';

export const TradingStatsPage: React.FC = () => {
  const [filters, setFilters] = useState<ExtendedFilters>({});
  const [metricFilters, setMetricFilters] = useState<MetricOptionFilters>({});
  const [customColumnFilters, setCustomColumnFilters] = useState<Record<number, string>>({});
  const [startDate, setStartDate] = useState<string>(() => {
    const twelveMonthsAgo = addMonths(new Date(), -11);
    return format(twelveMonthsAgo, 'yyyy-MM');
  });
  const [endDate, setEndDate] = useState<string>(() => {
    return format(new Date(), 'yyyy-MM');
  });
  const [currentBalance, setCurrentBalance] = useState<number>(1000);
  const [targetReturnPercent, setTargetReturnPercent] = useState<number>(100);
  const [useMonthFilterForRisk, setUseMonthFilterForRisk] = useState(false);
  const [useMonthFilterForEquity, setUseMonthFilterForEquity] = useState(false);
  const [useMonthFilterForMarket, setUseMonthFilterForMarket] = useState(false);
  const [useMonthFilterForTrends, setUseMonthFilterForTrends] = useState(false);

  const [topPercentFilter, setTopPercentFilter] = useState<number>(0);
  const [topPercentMode, setTopPercentMode] = useState<'winners' | 'losers'>('winners');
  const [topPercentSortBy, setTopPercentSortBy] = useState<'amount' | 'percent'>('percent');
  const [useMonthFilterForTopPercent, setUseMonthFilterForTopPercent] = useState(false);
  const [streakLength, setStreakLength] = useState<number>(0);
  const [streakType, setStreakType] = useState<'winners' | 'losers'>('winners');
  const [pinnedTradeIds, setPinnedTradeIds] = useState<Set<number> | null>(null);

  useEffect(() => {
    const fetchBalance = async () => {
      try {
        const balance = await balanceAPI.getBalance();
        setCurrentBalance(balance);
      } catch (error) {
        console.error('Error fetching balance:', error);
      }
    };
    fetchBalance();
  }, []);

  const { monthlyStats, yearlyStats, loading, toggleMonth, setMonthsChecked, filteredTrades, preTopPercentTrades, selectedMonths, totalExitedTrades, totalExitedInSelectedMonths, customColumns, columnOrder, customColumnValues } = useTradeStats(
    filters,
    startDate,
    endDate,
    metricFilters,
    customColumnFilters,
    currentBalance,
    topPercentFilter,
    topPercentMode,
    topPercentSortBy,
    useMonthFilterForTopPercent,
    streakLength,
    streakType,
    pinnedTradeIds
  );

  const topPercentSourceCount = useMemo(() => {
    if (useMonthFilterForTopPercent) {
      return preTopPercentTrades.filter(trade => {
        const month = format(parseISO(trade.Entry_Date), 'MMM yy');
        return selectedMonths.has(month);
      }).length;
    }
    return preTopPercentTrades.length;
  }, [preTopPercentTrades, selectedMonths, useMonthFilterForTopPercent]);

  // The effective trades used by all components (top% filter is already applied in useTradeStats)
  const effectiveTrades = useMemo(
    () => pinnedTradeIds ? filteredTrades.filter(t => pinnedTradeIds.has(t.ID)) : filteredTrades,
    [filteredTrades, pinnedTradeIds]
  );

  // Active filter pipeline labels
  const activeBaseFilterLabels = useMemo(() => {
    const parts: string[] = [];
    const fmtLabel = (key: string) =>
      key.split(/(?=[A-Z])|_/).map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
    Object.entries(filters).forEach(([k, v]) => {
      if (v === undefined) return;
      const label = fmtLabel(k);
      if (v === 'true') parts.push(`${label} = ✓`);
      else if (v === 'false') parts.push(`${label} = ✗`);
      else parts.push(`${label} = ${v}`);
    });
    const metricCount = Object.values(metricFilters).filter(arr => arr.length > 0).length;
    if (metricCount) parts.push(`Post Analysis (${metricCount} filter${metricCount > 1 ? 's' : ''})`);
    const customCount = Object.keys(customColumnFilters).length;
    if (customCount) parts.push(`Custom (${customCount})`);
    return parts;
  }, [filters, metricFilters, customColumnFilters]);

  const anyBaseFilterActive = activeBaseFilterLabels.length > 0;
  const anyFilterActive = anyBaseFilterActive || topPercentFilter > 0 || streakLength > 1;

  // Trades filtered by selected months (for optional checkbox filtering)
  const monthFilteredTrades = useMemo(() => {
    return effectiveTrades.filter(trade => {
      const month = format(parseISO(trade.Entry_Date), 'MMM yy');
      return selectedMonths.has(month);
    });
  }, [effectiveTrades, selectedMonths]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="text-lg text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <RiskPoolStats summaryStats={yearlyStats} />

      <Card>
        <CardContent>
          <div className="flex items-center gap-4 mb-4">
            <div className="flex items-center gap-2 my-2">
              <label htmlFor="start-date" className="text-sm font-medium">From:</label>
              <input
                id="start-date"
                type="month"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="flex h-10 w-40 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [color-scheme:light] dark:[color-scheme:dark]"
              />
            </div>

            <div className="flex items-center gap-2">
              <label htmlFor="end-date" className="text-sm font-medium">To:</label>
              <input
                id="end-date"
                type="month"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="flex h-10 w-40 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [color-scheme:light] dark:[color-scheme:dark]"
              />
            </div>
          </div>
          <MonthlyStatistics
            monthlyStats={monthlyStats}
            onToggleMonth={toggleMonth}
            onRangeToggleMonths={setMonthsChecked}
            targetReturnPercent={targetReturnPercent}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          {/* Was "Yearly Statistics" before */}
          <CardTitle>Summary Statistics(Selected Months)</CardTitle>
        </CardHeader>
        <CardContent>
          <YearlyStatistics yearlyStats={yearlyStats} />
        </CardContent>
      </Card>

      <RiskOfRuinStats
        monthlyStats={monthlyStats}
        targetReturnPercent={targetReturnPercent}
        onTargetReturnChange={setTargetReturnPercent}
        filteredTrades={useMonthFilterForRisk ? monthFilteredTrades : effectiveTrades}
        currentBalance={currentBalance}
        useMonthFilter={useMonthFilterForRisk}
        onUseMonthFilterChange={setUseMonthFilterForRisk}
      />

      <TradeDistribution
        filteredTrades={effectiveTrades}
        selectedMonths={selectedMonths}
      />

      <EquityCurve
        filteredTrades={useMonthFilterForEquity ? monthFilteredTrades : effectiveTrades}
        initialBalance={currentBalance}
        useMonthFilter={useMonthFilterForEquity}
        onUseMonthFilterChange={setUseMonthFilterForEquity}
      />

      <MarketTimingChart
        filteredTrades={useMonthFilterForMarket ? monthFilteredTrades : effectiveTrades}
        currentBalance={currentBalance}
        useMonthFilter={useMonthFilterForMarket}
        onUseMonthFilterChange={setUseMonthFilterForMarket}
      />

      <TradeTrendCharts
        filteredTrades={useMonthFilterForTrends ? monthFilteredTrades : effectiveTrades}
        useMonthFilter={useMonthFilterForTrends}
        onUseMonthFilterChange={setUseMonthFilterForTrends}
      />

      <Card>
        <CardContent>
          <div className="space-y-6">
            {/* Top % Winners / Losers Filter */}
            <div className="pt-2">
              <div className="text-sm font-medium mb-3 flex items-center gap-2">
                <Filter className="h-4 w-4" />
                Top % Winners / Losers Filter
              </div>

              <div className="mb-4">
                <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useMonthFilterForTopPercent}
                    onChange={e => setUseMonthFilterForTopPercent(e.target.checked)}
                    className="accent-primary h-3.5 w-3.5"
                  />
                  Use selected months
                </label>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground whitespace-nowrap">Top %:</label>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={1}
                    value={topPercentFilter}
                    onChange={e => {
                      const v = Number(e.target.value);
                      setTopPercentFilter(Math.max(0, Math.min(100, v)));
                    }}
                    className="border border-input rounded-md px-2 py-1 w-20 text-center bg-background text-foreground text-sm"
                    placeholder="0 = off"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground whitespace-nowrap">Mode:</label>
                  <select
                    value={topPercentMode}
                    onChange={e => setTopPercentMode(e.target.value as 'winners' | 'losers')}
                    className="border border-input rounded-md px-2 py-1 text-sm bg-background text-foreground"
                  >
                    <option value="winners">Biggest Winners</option>
                    <option value="losers">Biggest Losers</option>
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground whitespace-nowrap">Sort by:</label>
                  <select
                    value={topPercentSortBy}
                    onChange={e => setTopPercentSortBy(e.target.value as 'amount' | 'percent')}
                    className="border border-input rounded-md px-2 py-1 text-sm bg-background text-foreground"
                  >
                    <option value="amount">Return ($)</option>
                    <option value="percent">Return (%)</option>
                  </select>
                </div>

                {topPercentFilter > 0 && (
                  <span className="text-xs text-muted-foreground">
                    Showing {effectiveTrades.length} of {topPercentSourceCount} trades ({topPercentFilter}% {topPercentMode}, by {topPercentSortBy === 'amount' ? 'Return $' : 'Return %'})
                  </span>
                )}
                {topPercentFilter > 0 && (
                  <button
                    type="button"
                    onClick={() => setTopPercentFilter(0)}
                    className="text-xs text-muted-foreground hover:text-foreground underline"
                  >
                    Clear Filter
                  </button>
                )}
              </div>

              {/* Streaks filter */}
              <div className="flex flex-wrap items-center gap-3 mt-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground whitespace-nowrap">Streak of N trades:</label>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={1}
                    value={streakLength}
                    onChange={e => {
                      const v = Number(e.target.value);
                      setStreakLength(Math.max(0, v));
                    }}
                    className="border border-input rounded-md px-2 py-1 w-20 text-center bg-background text-foreground text-sm"
                    placeholder="0 = off"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground whitespace-nowrap">Type:</label>
                  <select
                    value={streakType}
                    onChange={e => setStreakType(e.target.value as 'winners' | 'losers')}
                    className="border border-input rounded-md px-2 py-1 text-sm bg-background text-foreground"
                  >
                    <option value="winners">Winning Streaks</option>
                    <option value="losers">Losing Streaks</option>
                  </select>
                </div>

                {streakLength > 1 && (
                  <span className="text-xs text-muted-foreground">
                    Found {effectiveTrades.length} trades in {streakType} streaks of size ≥ {streakLength}
                  </span>
                )}
                {streakLength > 0 && (
                  <button
                    type="button"
                    onClick={() => setStreakLength(0)}
                    className="text-xs text-muted-foreground hover:text-foreground underline"
                  >
                    Clear
                  </button>
                )}
              </div>

              {/* Collapsible trades table showing matched trades */}
              {anyFilterActive && (
                  <div className="mt-3 text-xs text-muted-foreground">
                    <div className="text-xs font-medium text-foreground mb-1.5 flex items-center gap-1">
                      <Filter className="h-3 w-3" />
                      Filters applied in order
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="rounded bg-muted px-2 py-0.5">
                        {useMonthFilterForTopPercent
                          ? <>All exits <span className="text-muted-foreground">(selected months)</span>: {totalExitedInSelectedMonths}</>
                          : <>All exits: {totalExitedTrades}</>}
                      </span>
                      {anyBaseFilterActive && (
                        <>
                          <span className="text-muted-foreground">→</span>
                          <span className="rounded border border-blue-300 bg-blue-50 px-2 py-0.5 dark:border-blue-700 dark:bg-blue-950">
                            ① Trade History &amp; Post Analysis: {activeBaseFilterLabels.join(' · ')}
                          </span>
                          {!(topPercentFilter > 0 || streakLength > 1) && (
                            <>
                              <span className="text-muted-foreground">→</span>
                              <span className="rounded bg-muted px-2 py-0.5 font-medium text-foreground">{topPercentSourceCount} matched</span>
                            </>
                          )}
                          {(topPercentFilter > 0 || streakLength > 1) && (
                            <>
                              <span className="text-muted-foreground">→</span>
                              <span className="rounded bg-muted px-2 py-0.5">{topPercentSourceCount}</span>
                            </>
                          )}
                        </>
                      )}
                      {topPercentFilter > 0 && (
                        <>
                          <span className="text-muted-foreground">→</span>
                          <span className="rounded border border-amber-300 bg-amber-50 px-2 py-0.5 dark:border-amber-700 dark:bg-amber-950">
                            {anyBaseFilterActive ? '②' : '①'} Top {topPercentFilter}% {topPercentMode === 'winners' ? 'Biggest Winners' : 'Biggest Losers'} (by {topPercentSortBy === 'amount' ? 'Return $' : 'Return %'})
                          </span>
                        </>
                      )}
                      {streakLength > 1 && (
                        <>
                          <span className="text-muted-foreground">→</span>
                          <span className="rounded border border-amber-300 bg-amber-50 px-2 py-0.5 dark:border-amber-700 dark:bg-amber-950">
                            {(anyBaseFilterActive && topPercentFilter > 0) ? '③' : (anyBaseFilterActive || topPercentFilter > 0) ? '②' : '①'} {streakType === 'winners' ? 'Winning' : 'Losing'} Streak ≥ {streakLength}
                          </span>
                        </>
                      )}
                      {(topPercentFilter > 0 || streakLength > 1) && (
                        <>
                          <span className="text-muted-foreground">→</span>
                          <span className="rounded bg-muted px-2 py-0.5 font-medium text-foreground">{effectiveTrades.length} matched</span>
                        </>
                      )}
                    </div>
                  </div>
              )}
              <TopPercentTradesTable
                trades={effectiveTrades}
                sortOrder={streakLength > 1 ? 'chronological' : 'return'}
                sortBy={topPercentSortBy}
                mode={topPercentMode}
                customColumns={customColumns}
                columnOrder={columnOrder}
                customColumnValues={customColumnValues}
                onPin={ids => setPinnedTradeIds(ids)}
              />
              {pinnedTradeIds && (
                <div className="mt-2 flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Hand-picked: {pinnedTradeIds.size} trade{pinnedTradeIds.size !== 1 ? 's' : ''} used in statistics</span>
                  <button
                    type="button"
                    onClick={() => setPinnedTradeIds(null)}
                    className="text-muted-foreground hover:text-foreground underline"
                  >
                    Clear selection
                  </button>
                </div>
              )}
            </div>

            <PostAnalysisFilters selected={metricFilters} onChange={setMetricFilters} />
            <div>
              <div className="text-sm font-medium mb-2">Trade History Filters</div>
              <TradeFilterer
                filters={filters}
                onFilterChange={setFilters}
                customColumnFilters={customColumnFilters}
                onCustomColumnFilterChange={setCustomColumnFilters}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};