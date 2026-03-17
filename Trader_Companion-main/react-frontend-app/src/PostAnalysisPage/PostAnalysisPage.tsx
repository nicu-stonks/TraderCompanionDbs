import React, { useEffect, useState, useCallback } from 'react';
import { APIError } from './types/types';
import TradeGrader from './components/TradeGrader';
import MetricManager from './components/MetricManager';
import MetricAnalytics from './components/MetricAnalytics';
import MetricOptionBenchmarks from './components/MetricOptionBenchmarks';
import MetricOptionRecommendationManager from './components/MetricOptionRecommendationManager';
import { gradeService, metricCheckSettingService, metricService, recommendationService, tradeService, percentBaseSettingService } from './services/postAnalysis';
import ErrorDisplay from './components/ErrorDisplay';
import LoadingSpinner from './components/LoadingSpinner';

const useAsync = <T,>(asyncFunction: () => Promise<T>, dependencies: React.DependencyList) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<APIError | null>(null);

  const execute = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await asyncFunction();
      setData(result);
    } catch (err) {
      setError({
        message: err instanceof Error ? err.message : 'An unknown error occurred',
        details: err
      });
    } finally {
      setLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    execute();
  }, [execute]);

  return { data, loading, error, refetch: execute };
};

// // Sample trades data
// const sampleTrades: Trade[] = [
//   {
//     ID: 1,
//     Ticker: "AAPL",
//     Status: "Closed",
//     Entry_Date: "2024-01-15",
//     Exit_Date: "2024-02-15",
//     Entry_Price: 150.00,
//     Exit_Price: 165.00,
//     Return: 10.0,
//     Pattern: "Cup with Handle",
//     Price_Tightness_1_Week_Before: 2.5,
//     Exit_Reason: "Target reached",
//     Market_Condition: "Uptrend",
//     Category: "Growth",
//     Nr_Bases: 2,
//     Has_Earnings_Acceleration: true,
//     IPO_Last_10_Years: false,
//     Is_BioTech: false,
//     Under_30M_Shares: false,
//     Case: JSON.stringify({}),
//     If_You_Could_Only_Make_10_Trades: true,
//     Pct_Off_52W_High: 5.2,
//     C: true,
//     A: true,
//     N: true,
//     S: true,
//     L: true,
//     I: true,
//     M: true
//   },
//   {
//     ID: 2,
//     Ticker: "NVDA",
//     Status: "Closed",
//     Entry_Date: "2024-02-01",
//     Exit_Date: "2024-03-01",
//     Entry_Price: 220.00,
//     Exit_Price: 198.00,
//     Return: -10.0,
//     Pattern: "Flat Base",
//     Price_Tightness_1_Week_Before: 1.8,
//     Exit_Reason: "Stop loss",
//     Market_Condition: "Sideways",
//     Category: "Growth",
//     Nr_Bases: 1,
//     Has_Earnings_Acceleration: false,
//     IPO_Last_10_Years: false,
//     Is_BioTech: false,
//     Under_30M_Shares: false,
//     Case: JSON.stringify({}),
//     If_You_Could_Only_Make_10_Trades: false,
//     Pct_Off_52W_High: 12.1,
//     C: false,
//     A: true,
//     N: false,
//     S: true,
//     L: false,
//     I: true,
//     M: false
//   }
// ];

const PostAnalysisPage: React.FC = () => {
  const [trailingWindow, setTrailingWindow] = useState(50);
  const [layoutMode, setLayoutMode] = useState<'stacked' | 'grid'>('grid');

  // Data fetching
  const { 
    data: trades, 
    loading: tradesLoading, 
    error: tradesError,
    refetch: refetchTrades 
  } = useAsync(() => tradeService.getTrades(), []);

  const { 
    data: metrics, 
    loading: metricsLoading, 
    error: metricsError,
    refetch: refetchMetrics 
  } = useAsync(() => metricService.getMetrics(), []);

  const { 
    data: tradeGrades, 
    loading: gradesLoading, 
    error: gradesError,
    refetch: refetchGrades 
  } = useAsync(() => gradeService.getGrades(), []);

  const {
    data: recommendations,
    loading: recommendationsLoading,
    error: recommendationsError,
    refetch: refetchRecommendations
  } = useAsync(() => recommendationService.getRecommendations(), []);

  const {
    data: metricCheckSettings,
    loading: metricCheckSettingsLoading,
    error: metricCheckSettingsError,
    refetch: refetchMetricCheckSettings
  } = useAsync(() => metricCheckSettingService.getSettings(), []);

  const {
    data: percentBaseSettings,
    loading: percentBaseSettingsLoading,
    error: percentBaseSettingsError,
    refetch: refetchPercentBaseSettings
  } = useAsync(() => percentBaseSettingService.getSettings(), []);

  // Loading states
  const isLoading = tradesLoading || metricsLoading || gradesLoading || recommendationsLoading || metricCheckSettingsLoading || percentBaseSettingsLoading;
  const hasError = tradesError || metricsError || gradesError || recommendationsError || metricCheckSettingsError || percentBaseSettingsError;

  // Handle errors
  if (hasError) {
    return (
      <div className="min-h-screen bg-background p-2">
        <header className="mb-4">
          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-2">Trade Analysis Dashboard</h1>
        </header>
        <div className="space-y-2 md:space-y-4">
          {tradesError && (
            <ErrorDisplay error={tradesError} onRetry={refetchTrades} />
          )}
          {metricsError && (
            <ErrorDisplay error={metricsError} onRetry={refetchMetrics} />
          )}
          {gradesError && (
            <ErrorDisplay error={gradesError} onRetry={refetchGrades} />
          )}
          {recommendationsError && (
            <ErrorDisplay error={recommendationsError} onRetry={refetchRecommendations} />
          )}
          {metricCheckSettingsError && (
            <ErrorDisplay error={metricCheckSettingsError} onRetry={refetchMetricCheckSettings} />
          )}
          {percentBaseSettingsError && (
            <ErrorDisplay error={percentBaseSettingsError} onRetry={refetchPercentBaseSettings} />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-2 md:p-3">
      <header className="mb-4 md:mb-6">
        <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-2">Trade Analysis Dashboard</h1>
        <p className="text-base md:text-lg text-muted-foreground">
          Analyze your trading patterns and improve your performance
        </p>
        {isLoading && (
          <div className="mt-3">
            <LoadingSpinner message="Loading dashboard data..." />
          </div>
        )}
      </header>

      {!isLoading && (
        <>
          <MetricAnalytics
            trades={trades || []}
            metrics={metrics || []}
            tradeGrades={tradeGrades || []}
            trailingWindow={trailingWindow}
            onTrailingWindowChange={setTrailingWindow}
            layoutMode={layoutMode}
            onLayoutModeChange={setLayoutMode}
          />

          <MetricOptionBenchmarks
            trades={trades || []}
            metrics={metrics || []}
            tradeGrades={tradeGrades || []}
            trailingWindow={trailingWindow}
            recommendations={recommendations || []}
            layoutMode={layoutMode}
            percentBaseSettings={percentBaseSettings || []}
            onRefetchPercentBaseSettings={refetchPercentBaseSettings}
          />

          <TradeGrader
            trades={trades || []}
            metrics={metrics || []}
            tradeGrades={tradeGrades || []}
            checkSettings={metricCheckSettings || []}
            onRefetchCheckSettings={refetchMetricCheckSettings}
          />

          <MetricOptionRecommendationManager
            metrics={metrics || []}
            recommendations={recommendations || []}
            onRefetch={refetchRecommendations}
          />

          <MetricManager
            metrics={metrics || []}
            onRefetch={refetchMetrics}
          />
        </>
      )}
    </div>
  );
};


export default PostAnalysisPage;