import { useState, useEffect, useMemo } from 'react';
import { MonthlyStats, YearlyStats, ExtendedFilters, MetricOptionFilters } from '../types';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import { tradeAPI } from '../services/tradeAPI';
import { addMonths, format, parseISO, isAfter, differenceInDays } from 'date-fns';
import { gradeService } from '@/PostAnalysisPage/services/postAnalysis';
import type { TradeGrade } from '@/PostAnalysisPage/types/types';
import { customTradeDataAPI } from '@/TradeHistoryPage/services/customTradeDataAPI';
import type { CustomColumn, ColumnOrder, CustomColumnValue } from '@/TradeHistoryPage/types/CustomTradeData';

// Type guard to check if a trade is exited with valid exit price and exit date
export type ExitedTrade = Trade & { Exit_Price: number; Exit_Date: string };
const isExitedTrade = (trade: Trade): trade is ExitedTrade => {
  return trade.Status === 'Exited' && trade.Exit_Price !== null && trade.Exit_Date !== null;
};

export const useTradeStats = (
  filters: ExtendedFilters,
  startDate?: string,
  endDate?: string,
  metricOptionFilters: MetricOptionFilters = {},
  customColumnFilters: Record<number, string> = {},
  currentBalance: number = 1000,
  topPercentFilter: number = 0,
  topPercentMode: 'winners' | 'losers' = 'winners',
  topPercentSortBy: 'amount' | 'percent' = 'percent',
  useMonthFilterForTopPercent: boolean = true,
  streakLength: number = 0,
  streakType: 'winners' | 'losers' = 'winners',
  pinnedTradeIds: Set<number> | null = null
) => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [grades, setGrades] = useState<TradeGrade[]>([]);
  const [customColumnValues, setCustomColumnValues] = useState<CustomColumnValue[]>([]);
  const [customColumns, setCustomColumns] = useState<CustomColumn[]>([]);
  const [columnOrder, setColumnOrder] = useState<ColumnOrder[]>([]);
  const [selectedMonths, setSelectedMonths] = useState<Set<string>>(() => {
    const initialMonths = new Set<string>();
    const start = startDate ? parseISO(`${startDate}-01`) : addMonths(new Date(), -11);
    const end = endDate ? parseISO(`${endDate}-01`) : new Date();

    let currentDate = start;
    while (currentDate <= end) {
      initialMonths.add(format(currentDate, 'MMM yy'));
      currentDate = addMonths(currentDate, 1);
    }
    return initialMonths;
  });

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const [tradesResp, allGrades, customValsResp, customColsResp, colOrderResp] = await Promise.all([
          tradeAPI.getTrades(),
          gradeService.getGrades(),
          customTradeDataAPI.getColumnValues(),
          customTradeDataAPI.getColumns(),
          customTradeDataAPI.getColumnOrder(),
        ]);
        setTrades(tradesResp.data);
        setGrades(allGrades);
        setCustomColumnValues(customValsResp.data);
        setCustomColumns(customColsResp.data);
        setColumnOrder(colOrderResp.data);
      } catch (error) {
        console.error('Error fetching trades:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchTrades();
  }, []);

  const preTopPercentTrades = useMemo(() => {
    // Build a quick lookup: for each trade, which metric option ids are selected
    const tradeToMetricOptionIds = new Map<number, Map<number, Set<number>>>();
    if (grades.length) {
      for (const g of grades) {
        const tradeMap = tradeToMetricOptionIds.get(g.tradeId) ?? new Map<number, Set<number>>();
        tradeMap.set(Number(g.metricId), new Set<number>([Number(g.selectedOptionId)]));
        tradeToMetricOptionIds.set(g.tradeId, tradeMap);
      }
    }

    return trades
      .filter(isExitedTrade)
      .filter(trade => {
        const baseFiltersPass = Object.entries(filters).every(([key, value]) => {
          if (value === undefined) return true;

          switch (key) {
            // case 'minEarningsQuality':
            //   return trade.Earnings_Quality >= value;
            // case 'minFundamentalsQuality':
            //   return Number(trade.Fundamentals_Quality) >= value;
            case 'maxPriceTightness':
              return trade.Price_Tightness_1_Week_Before <= value;
            case 'maxNrBases':
              return trade.Nr_Bases <= value;
            case 'pctOff52WHigh':
              return trade.Pct_Off_52W_High <= value;
            default:
              if (typeof trade[key as keyof Trade] === 'boolean') {
                return trade[key as keyof Trade] === (value === 'true');
              }
              return trade[key as keyof Trade] === value;
          }
        });

        if (!baseFiltersPass) return false;

        // Apply custom column filters
        const customFilterEntries = Object.entries(customColumnFilters);
        if (customFilterEntries.length > 0) {
          const tradeCustomVals = customColumnValues.filter(v => v.trade_id === trade.ID);
          const customFiltersPass = customFilterEntries.every(([colIdStr, expectedValue]) => {
            const colId = parseInt(colIdStr);
            const val = tradeCustomVals.find(v => v.column === colId);
            return val?.value === expectedValue;
          });
          if (!customFiltersPass) return false;
        }

        // Apply post-analysis metric option filters: for each metric with selections,
        // the trade must have a grade whose selected option is in the selected set.
        const metricIds = Object.keys(metricOptionFilters);
        if (!metricIds.length) return true;

        const tradeMap = tradeToMetricOptionIds.get(trade.ID);
        if (!tradeMap) return false;

        return metricIds.every(midStr => {
          const mid = Number(midStr);
          const selectedOptions = new Set<number>(metricOptionFilters[mid]);
          const selectedForTrade = tradeMap.get(mid);
          if (!selectedForTrade) return false;
          // intersect
          for (const optId of selectedForTrade) {
            if (selectedOptions.has(optId)) return true;
          }
          return false;
        });
      });
  }, [trades, filters, grades, metricOptionFilters, customColumnFilters, customColumnValues]);

  const filteredTrades = useMemo((): ExitedTrade[] => {
    const baseTrades = preTopPercentTrades;

    type ExitedTradeType = ExitedTrade;

    // Helper: apply month filter (controlled by useMonthFilterForTopPercent)
    const applyMonthFilter = (source: ExitedTrade[]): ExitedTrade[] => {
      if (!useMonthFilterForTopPercent) return source;
      return source.filter(t => {
        const month = format(parseISO(t.Entry_Date), 'MMM yy');
        return selectedMonths.has(month);
      });
    };

    // --- Helper: compute streak-qualifying trades ---
    // Streaks are detected on ALL trades chronologically (ignoring month filter)
    // so a single long streak isn't split. Month filter is applied after.
    const computeStreakTrades = (source: ExitedTrade[]): Map<number, ExitedTradeType> => {
      const chronTrades = [...source]
        .filter(t => t.Status === 'Exited' && t.Exit_Date != null && t.Entry_Price != null && t.Exit_Price != null)
        .sort((a, b) => new Date(a.Entry_Date).getTime() - new Date(b.Entry_Date).getTime());

      const isMatch = (t: ExitedTradeType) => {
        const isWin = t.Exit_Price > (t.Entry_Price ?? 0);
        const isLoss = t.Exit_Price < (t.Entry_Price ?? 0);
        return streakType === 'winners' ? isWin : isLoss;
      };

      const result = new Map<number, ExitedTradeType>();
      let currentStreak: ExitedTradeType[] = [];
      let currentStreakId = 1;

      const processStreak = (streak: ExitedTradeType[]) => {
        if (streak.length >= streakLength) {
          streak.forEach(t => result.set(t.ID, { ...t, streakId: currentStreakId } as unknown as ExitedTradeType));
          currentStreakId++;
        }
      };

      for (const trade of chronTrades as ExitedTradeType[]) {
        if (isMatch(trade)) {
          currentStreak.push(trade);
        } else {
          processStreak(currentStreak);
          currentStreak = [];
        }
      }
      processStreak(currentStreak);

      return result;
    };

    // --- Helper: compute top % trades ---
    const computeTopPercentTrades = (source: Trade[]): Set<number> => {
      const exited = source.filter(t => t.Status === 'Exited' && t.Exit_Price != null && t.Entry_Price != null && t.Entry_Price > 0);

      if (exited.length === 0) return new Set();

      const withReturn = exited.map(trade => ({
        trade,
        returnAmount: trade.Return ?? 0,
        returnPct: ((trade.Exit_Price! - trade.Entry_Price) / trade.Entry_Price) * 100,
      }));

      const valueOf = (x: { returnAmount: number; returnPct: number }) =>
        topPercentSortBy === 'amount' ? x.returnAmount : x.returnPct;

      const sorted = [...withReturn].sort((a, b) =>
        topPercentMode === 'winners'
          ? valueOf(b) - valueOf(a)
          : valueOf(a) - valueOf(b)
      );

      const count = Math.max(1, Math.ceil(sorted.length * (topPercentFilter / 100)));
      return new Set(sorted.slice(0, count).map(x => x.trade.ID));
    };

    const streakActive = streakLength > 1;
    const topPercentActive = topPercentFilter > 0;
    const monthFiltered = applyMonthFilter(baseTrades);

    // Neither filter active
    if (!streakActive && !topPercentActive) {
      return monthFiltered;
    }

    // Only streak filter active: detect on all, then apply month filter
    if (streakActive && !topPercentActive) {
      const streakMap = computeStreakTrades(baseTrades);
      if (!useMonthFilterForTopPercent) return Array.from(streakMap.values());
      return Array.from(streakMap.values()).filter(t => {
        const month = format(parseISO(t.Entry_Date), 'MMM yy');
        return selectedMonths.has(month);
      });
    }

    // Only top % filter active: filter months first, then top %
    if (!streakActive && topPercentActive) {
      const topIds = computeTopPercentTrades(monthFiltered);
      return monthFiltered.filter(t => topIds.has(t.ID));
    }

    // Both active: compute independently, return intersection
    const streakMap = computeStreakTrades(baseTrades);
    const topIds = computeTopPercentTrades(monthFiltered);
    return Array.from(streakMap.values())
      .filter(t => {
        const month = format(parseISO(t.Entry_Date), 'MMM yy');
        return selectedMonths.has(month);
      })
      .filter(t => topIds.has(t.ID));
  }, [
    preTopPercentTrades,
    topPercentFilter,
    topPercentMode,
    topPercentSortBy,
    useMonthFilterForTopPercent,
    selectedMonths,
    streakLength,
    streakType,
  ]);

  // When the user has hand-picked trades, use only those for all statistics
  const statsTrades = useMemo(
    () => pinnedTradeIds ? filteredTrades.filter(t => pinnedTradeIds.has(t.ID)) : filteredTrades,
    [filteredTrades, pinnedTradeIds]
  );

  const monthlyStats = useMemo((): MonthlyStats[] => {
    // Parse start and end dates, with defaults
    const start = startDate ? parseISO(`${startDate}-01`) : addMonths(new Date(), -11);
    const end = endDate ? parseISO(`${endDate}-01`) : new Date();

    // Generate array of months between start and end dates
    const months: string[] = [];
    let currentDate = start;

    while (currentDate <= end) {
      months.push(format(currentDate, 'MMM yy'));
      currentDate = addMonths(currentDate, 1);
    }

    const lastYear = addMonths(new Date(), -12);

    const result = months.map(month => {
      const monthTrades = statsTrades.filter(trade =>
        format(parseISO(trade.Entry_Date), 'MMM yy') === month
      );

      const gains = monthTrades.filter(t => t.Exit_Price > t.Entry_Price);
      const losses = monthTrades.filter(t => t.Exit_Price < t.Entry_Price);

      const totalGain = gains.reduce((acc, t) => acc + (t.Exit_Price - t.Entry_Price) / t.Entry_Price * 100, 0);
      const totalLoss = losses.reduce((acc, t) => acc + (t.Exit_Price - t.Entry_Price) / t.Entry_Price * 100, 0);

      const avgGain = gains.length ? totalGain / gains.length : 0;
      const avgLoss = losses.length ? totalLoss / losses.length : 0;

      const monthDate = parseISO(`01 ${month}`);
      const isInTrailingYear = isAfter(monthDate, lastYear);

      const avgDaysGains = gains.length ?
        gains.reduce((acc, t) => acc + differenceInDays(parseISO(t.Exit_Date), parseISO(t.Entry_Date)), 0) / gains.length : 0;

      const avgDaysLoss = losses.length ?
        losses.reduce((acc, t) => acc + differenceInDays(parseISO(t.Exit_Date), parseISO(t.Entry_Date)), 0) / losses.length : 0;

      // Calculate compounded net profit/loss: (1+r1) * (1+r2) * ... * (1+rN) - 1
      let netProfitLoss = 0;
      if (monthTrades.length > 0) {
        const compoundedReturn = monthTrades.reduce((acc, t) => {
          const returnPct = (t.Exit_Price - t.Entry_Price) / t.Entry_Price;
          return acc * (1 + returnPct);
        }, 1);
        netProfitLoss = (compoundedReturn - 1) * 100;
      }

      return {
        tradingMonth: month,
        averageGain: avgGain,
        averageLoss: avgLoss,
        winningPercentage: monthTrades.length ? (gains.length / monthTrades.length) * 100 : 0,
        totalTrades: monthTrades.length,
        largestGain: Math.max(...gains.map(t => (t.Exit_Price - t.Entry_Price) / t.Entry_Price * 100), 0),
        largestLoss: Math.min(...losses.map(t => (t.Exit_Price - t.Entry_Price) / t.Entry_Price * 100), 0),
        avgDaysGains,
        avgDaysLoss,
        isInTrailingYear,
        useInYearly: selectedMonths.has(month),
        netProfitLoss,
        equityChangePct: 0 // placeholder, will be computed below
      };
    });

    // Compute equityChangePct using running equity from dollar Returns
    // currentBalance from props is the INITIAL balance (from balanceAPI).
    // Compute actual current balance by adding all statsTrades returns.
    const allFilteredReturns = statsTrades.reduce((sum, t) => sum + (t.Return ?? 0), 0);
    const actualCurrentBalance = currentBalance + allFilteredReturns;

    // Get dollar returns for each displayed month
    const allMonthReturns = result.map(m => {
      const monthTrades = statsTrades.filter(trade =>
        format(parseISO(trade.Entry_Date), 'MMM yy') === m.tradingMonth
      );
      return monthTrades.reduce((sum, t) => sum + (t.Return ?? 0), 0);
    });
    const displayedReturns = allMonthReturns.reduce((sum, r) => sum + r, 0);
    let runningEquity = actualCurrentBalance - displayedReturns; // equity before first displayed month

    for (let i = 0; i < result.length; i++) {
      const monthDollarReturn = allMonthReturns[i];
      if (runningEquity > 0 && result[i].totalTrades > 0) {
        result[i].equityChangePct = (monthDollarReturn / runningEquity) * 100;
      }
      runningEquity += monthDollarReturn;
    }

    return result;
  }, [statsTrades, selectedMonths, startDate, endDate, currentBalance]);

  const yearlyStats = useMemo((): YearlyStats => {
    // ✅ FIXED: Filter by Entry_Date to match monthly stats calculation
    const selectedTrades = statsTrades.filter(trade => {
      const month = format(parseISO(trade.Entry_Date), 'MMM yy'); // Changed from Exit_Date to Entry_Date
      return selectedMonths.has(month);
    });

    const gains = selectedTrades.filter(t => t.Exit_Price > t.Entry_Price);
    const losses = selectedTrades.filter(t => t.Exit_Price < t.Entry_Price);

    const totalGain = gains.reduce((acc, t) => acc + (t.Exit_Price - t.Entry_Price) / t.Entry_Price * 100, 0);
    const totalLoss = losses.reduce((acc, t) => acc + (t.Exit_Price - t.Entry_Price) / t.Entry_Price * 100, 0);

    const avgGain = gains.length ? totalGain / gains.length : 0;
    const avgLoss = losses.length ? totalLoss / losses.length : 0;

    // For compounding returns, use the expected growth rate formula
    const winRate = selectedTrades.length ? gains.length / selectedTrades.length : 0;
    const lossRate = selectedTrades.length ? losses.length / selectedTrades.length : 0;

    // Calculate expected growth rate (log mean)
    let expectedGrowthRate = 0;

    if (selectedTrades.length) {
      const winComponent = winRate * Math.log(1 + avgGain / 100);
      const lossComponent = lossRate * Math.log(1 + avgLoss / 100);
      expectedGrowthRate = winComponent + lossComponent;
    }

    // Expected growth per trade (with compounding effect)
    const expectedGrowthPerTrade = (Math.exp(expectedGrowthRate) - 1) * 100;

    // Position sizing calculations
    const positionSize125 = 0.125; // 12.5% position sizing
    const positionSize25 = 0.25;   // 25% position sizing

    // Calculate expected returns with different position sizing
    const expectedReturnOn10Trades_125 = (Math.exp(expectedGrowthRate * positionSize125 * 10) - 1) * 100;
    const expectedReturnOn50Trades_125 = (Math.exp(expectedGrowthRate * positionSize125 * 50) - 1) * 100;
    const expectedReturnOn10Trades_25 = (Math.exp(expectedGrowthRate * positionSize25 * 10) - 1) * 100;
    const expectedReturnOn50Trades_25 = (Math.exp(expectedGrowthRate * positionSize25 * 50) - 1) * 100;

    // Calculate average of largest gains/losses from each month (like in summary table)
    // Use all selected months instead of just trailing 12 months
    const selectedMonthsArray = Array.from(selectedMonths);

    const selectedMonthsData = selectedMonthsArray
      .map(month => {
        // ✅ FIXED: Use Entry_Date here too for consistency
        const monthTrades = statsTrades.filter(trade =>
          format(parseISO(trade.Entry_Date), 'MMM yy') === month // Changed from Exit_Date to Entry_Date
        );

        const gains = monthTrades.filter(t => t.Exit_Price > t.Entry_Price);
        const losses = monthTrades.filter(t => t.Exit_Price < t.Entry_Price);

        const largestGain = gains.length > 0 ?
          Math.max(...gains.map(t => (t.Exit_Price - t.Entry_Price) / t.Entry_Price * 100)) : 0;
        const largestLoss = losses.length > 0 ?
          Math.min(...losses.map(t => (t.Exit_Price - t.Entry_Price) / t.Entry_Price * 100)) : 0;

        return { largestGain, largestLoss };
      });

    const monthsWithGains = selectedMonthsData.filter(month => month.largestGain > 0);
    const monthsWithLosses = selectedMonthsData.filter(month => month.largestLoss < 0);

    const avgLargestGain = monthsWithGains.length > 0
      ? monthsWithGains.reduce((sum, month) => sum + month.largestGain, 0) / monthsWithGains.length
      : 0;

    const avgLargestLoss = monthsWithLosses.length > 0
      ? monthsWithLosses.reduce((sum, month) => sum + month.largestLoss, 0) / monthsWithLosses.length
      : 0;

    const avgLargestGainLossRatio = avgLargestLoss !== 0 ? Math.abs(avgLargestGain / avgLargestLoss) : 0;

    // Calculate average days held for gains and losses (like in summary table)
    const totalWinningTrades = selectedTrades.filter(t => t.Exit_Price > t.Entry_Price).length;
    const totalLosingTrades = selectedTrades.filter(t => t.Exit_Price <= t.Entry_Price).length;

    const avgDaysGains = totalWinningTrades > 0
      ? selectedMonthsData.reduce((sum, _monthData, index) => {
        const month = selectedMonthsArray[index];
        // ✅ FIXED: Use Entry_Date here too for consistency
        const monthTrades = statsTrades.filter(trade =>
          format(parseISO(trade.Entry_Date), 'MMM yy') === month // Changed from Exit_Date to Entry_Date
        );
        const monthGains = monthTrades.filter(t => t.Exit_Price > t.Entry_Price);
        const monthAvgDaysGains = monthGains.length ?
          monthGains.reduce((acc, t) => acc + differenceInDays(parseISO(t.Exit_Date), parseISO(t.Entry_Date)), 0) / monthGains.length : 0;
        return sum + (monthAvgDaysGains * monthGains.length);
      }, 0) / totalWinningTrades
      : 0;

    const avgDaysLoss = totalLosingTrades > 0
      ? selectedMonthsData.reduce((sum, _monthData, index) => {
        const month = selectedMonthsArray[index];
        // ✅ FIXED: Use Entry_Date here too for consistency
        const monthTrades = statsTrades.filter(trade =>
          format(parseISO(trade.Entry_Date), 'MMM yy') === month // Changed from Exit_Date to Entry_Date
        );
        const monthLosses = monthTrades.filter(t => t.Exit_Price <= t.Entry_Price);
        const monthAvgDaysLoss = monthLosses.length ?
          monthLosses.reduce((acc, t) => acc + differenceInDays(parseISO(t.Exit_Date), parseISO(t.Entry_Date)), 0) / monthLosses.length : 0;
        return sum + (monthAvgDaysLoss * monthLosses.length);
      }, 0) / totalLosingTrades
      : 0;

    const avgDaysRatio = avgDaysLoss !== 0 ? avgDaysGains / Math.abs(avgDaysLoss) : 0;

    return {
      winningPercentage: selectedTrades.length ? winRate * 100 : 0,
      averageGain: avgGain,
      averageLoss: avgLoss,
      winLossRatio: avgLoss !== 0 ? Math.abs(avgGain / avgLoss) : 0,
      expectedValuePerTrade: expectedGrowthPerTrade,
      expectedReturnOn10Trades_125: expectedReturnOn10Trades_125,
      expectedReturnOn50Trades_125: expectedReturnOn50Trades_125,
      expectedReturnOn10Trades_25: expectedReturnOn10Trades_25,
      expectedReturnOn50Trades_25: expectedReturnOn50Trades_25,
      avgLargestGain,
      avgLargestLoss,
      avgLargestGainLossRatio,
      avgDaysGains,
      avgDaysLoss,
      avgDaysRatio
    };
  }, [statsTrades, selectedMonths]);

  const toggleMonth = (month: string) => {
    setSelectedMonths(prev => {
      const newSet = new Set(prev);
      if (newSet.has(month)) {
        newSet.delete(month);
      } else {
        newSet.add(month);
      }
      return newSet;
    });
  };

  const setMonthsChecked = (months: string[], checked: boolean) => {
    setSelectedMonths(prev => {
      const newSet = new Set(prev);
      months.forEach(m => checked ? newSet.add(m) : newSet.delete(m));
      return newSet;
    });
  };

  const totalExitedTrades = useMemo(() => trades.filter(isExitedTrade).length, [trades]);

  const totalExitedInSelectedMonths = useMemo(
    () => trades.filter(isExitedTrade).filter(t => selectedMonths.has(format(parseISO(t.Entry_Date), 'MMM yy'))).length,
    [trades, selectedMonths]
  );

  // Now also returning filteredTrades and selectedMonths for the distribution component
  return {
    monthlyStats,
    yearlyStats,
    loading,
    toggleMonth,
    setMonthsChecked,
    filteredTrades,
    preTopPercentTrades,
    selectedMonths,
    trades, // Return all trades
    totalExitedTrades,
    totalExitedInSelectedMonths,
    customColumns,
    columnOrder,
    customColumnValues,
  };
};