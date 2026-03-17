import React, { useState, useEffect, useCallback, memo } from 'react';
import axios from 'axios';
import { AlertCircle, XCircle, Clock, RefreshCw, Zap } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';

interface TickerError {
  message: string;
  timestamp: string;
  type: string;
}

interface FetchErrorsResponse {
  errors: Record<string, TickerError>;
  count: number;
}

interface RequestStatsResponse {
  last_1s: number;
  last_5s: number;
  last_10s: number;
  per_second_avg_5s: number;
  per_second_avg_10s: number;
  tickers_last_1s: number;
  tickers_last_5s: number;
  tickers_last_10s: number;
  tickers_per_second_avg_5s: number;
  tickers_per_second_avg_10s: number;
  est_seconds_per_ticker: number | null;
  est_requests_per_ticker: number | null;
  est_loop_seconds: number | null;
  tracked_ticker_count: number;
}

interface TickersResponse {
  tickers: string[];
  total_count: number;
}

interface ProviderSettings {
  max_requests_per_10s: number;
  active_provider: string;
}

interface FetchErrorBannerProps {
  refreshInterval?: number; // in milliseconds, default 30000 (30s)
}

const FetchErrorBannerComponent: React.FC<FetchErrorBannerProps> = ({
  refreshInterval = 30000
}) => {
  const [errors, setErrors] = useState<Record<string, TickerError>>({});
  const [requestStats, setRequestStats] = useState<RequestStatsResponse | null>(null);
  const [tickerCount, setTickerCount] = useState<number | null>(null);
  const [settings, setSettings] = useState<ProviderSettings | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUpdatingSettings, setIsUpdatingSettings] = useState(false);
  const [liveMode, setLiveMode] = useState(false);

  const formatDurationSeconds = (seconds: number): string => {
    if (!Number.isFinite(seconds) || seconds < 0) return 'N/A';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs.toString().padStart(2, '0')}s`;
  };

  const formatRate = (value: number | null | undefined): string => {
    if (value == null || !Number.isFinite(value)) return 'N/A';
    return `${value.toFixed(2)}/s`;
  };

  const fetchData = useCallback(async () => {
    try {
      setIsLoading(true);
      const host = 'http://localhost:8000/ticker_data/api/ticker_data';
      const [errorsRes, statsRes, tickersRes, settingsRes] = await Promise.all([
        axios.get<FetchErrorsResponse>(`${host}/errors`),
        axios.get<RequestStatsResponse>(`${host}/request-stats`),
        axios.get<TickersResponse>(`${host}/tickers`),
        axios.get<ProviderSettings>(`${host}/settings`)
      ]);
      setErrors(errorsRes.data.errors);
      setRequestStats(statsRes.data);
      setTickerCount(tickersRes.data.total_count ?? tickersRes.data.tickers?.length ?? null);
      if (!isUpdatingSettings) {
        setSettings(settingsRes.data);
      }
    } catch {
      // Silently fail - the server might not be running
    } finally {
      setIsLoading(false);
    }
  }, [isUpdatingSettings]);

  const handleSettingsUpdate = async (newLimit: number) => {
    if (!settings) return;
    try {
      setIsUpdatingSettings(true);
      const host = 'http://localhost:8000/ticker_data/api/ticker_data';
      const res = await axios.put<ProviderSettings>(`${host}/settings`, {
        ...settings,
        max_requests_per_10s: newLimit
      });
      setSettings(res.data);
    } catch (e) {
      console.error("Failed to update settings", e);
    } finally {
      setIsUpdatingSettings(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Use 500ms interval in live mode, otherwise use the normal refresh interval
    const intervalMs = liveMode ? 500 : refreshInterval;
    const interval = setInterval(fetchData, intervalMs);
    return () => clearInterval(interval);
  }, [refreshInterval, liveMode, fetchData]);

  // Check if error is recent (within last 10 minutes)
  const isRecentError = (timestamp: string): boolean => {
    const errorTime = new Date(timestamp).getTime();
    const tenMinutesAgo = Date.now() - 10 * 60 * 1000;
    return errorTime > tenMinutesAgo;
  };

  // Format relative time
  const formatTimeAgo = (timestamp: string): string => {
    const now = Date.now();
    const errorTime = new Date(timestamp).getTime();
    const diffMs = now - errorTime;
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'just now';
    if (diffMins === 1) return '1 minute ago';
    if (diffMins < 60) return `${diffMins} minutes ago`;

    const diffHours = Math.floor(diffMins / 60);
    if (diffHours === 1) return '1 hour ago';
    return `${diffHours} hours ago`;
  };

  const errorEntries = Object.entries(errors);
  const hasRecentErrors = errorEntries.some(([, error]) => isRecentError(error.timestamp));

  const estimatedLoopSeconds = requestStats?.est_loop_seconds != null
    ? formatDurationSeconds(requestStats.est_loop_seconds)
    : null;

  const estimatedReqPerTicker = requestStats?.est_requests_per_ticker != null
    ? requestStats.est_requests_per_ticker.toFixed(2)
    : null;

  return (
    <div className="space-y-3" style={{ contain: 'layout paint' }}>
      {/* Request Stats - Always show */}
      {requestStats && (
        <div className="p-3 rounded-lg bg-muted/50 border text-sm">
          {/* Header: label + controls */}
          <div className="flex items-center justify-between mb-2.5">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Fetch Activity</span>
              {settings && (
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-bold font-mono bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400">
                  {settings.active_provider === 'webull' ? 'WB' : 'YF'}
                  {estimatedLoopSeconds && <span>{estimatedLoopSeconds}/loop</span>}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {settings && (
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-muted-foreground">Limit/10s</span>
                  <input
                    type="number"
                    value={settings.max_requests_per_10s || ''}
                    className="w-12 h-6 px-1 border rounded text-xs bg-background text-foreground font-mono text-center"
                    onBlur={(e) => handleSettingsUpdate(Number(e.target.value))}
                    onChange={(e) => setSettings({ ...settings, max_requests_per_10s: Number(e.target.value) })}
                    disabled={isUpdatingSettings}
                  />
                </div>
              )}
              <Button
                variant={liveMode ? "default" : "outline"}
                size="sm"
                onClick={() => setLiveMode(!liveMode)}
                className={`h-6 px-2 text-xs ${liveMode ? 'bg-green-600 hover:bg-green-700' : ''}`}
              >
                <Zap className={`h-3 w-3 mr-1 ${liveMode ? 'animate-pulse' : ''}`} />
                Live
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={fetchData}
                disabled={isLoading}
                className="h-6 w-6 p-0"
              >
                <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>

          {/* Two-column stat cards */}
          <div className="grid grid-cols-2 gap-2">
            {/* Requests */}
            <div className="rounded-md bg-background border p-2 space-y-1.5">
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-blue-500 shrink-0" />
                <span className="text-xs font-semibold">API Requests</span>
                <span className="ml-auto text-xs text-muted-foreground font-mono">{formatRate(requestStats.per_second_avg_10s)}/s</span>
              </div>
              <div className="grid grid-cols-3 gap-1">
                {([
                  { label: '1s', value: requestStats.last_1s, highlight: requestStats.last_1s > 1 },
                  { label: '5s', value: requestStats.last_5s, highlight: false },
                  { label: '10s', value: requestStats.last_10s, highlight: false },
                ] as { label: string; value: number; highlight: boolean }[]).map(({ label, value, highlight }) => (
                  <div key={label} className="bg-muted/60 rounded p-1 text-center">
                    <div className={`font-mono font-bold text-base leading-none ${highlight ? 'text-amber-500' : ''}`}>{value}</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">{label}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Tickers */}
            <div className="rounded-md bg-background border p-2 space-y-1.5">
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-emerald-500 shrink-0" />
                <span className="text-xs font-semibold">Tickers</span>
                <span className="ml-auto text-xs text-muted-foreground font-mono">{formatRate(requestStats.tickers_per_second_avg_10s)}/s</span>
              </div>
              <div className="grid grid-cols-3 gap-1">
                {([
                  { label: '1s', value: requestStats.tickers_last_1s },
                  { label: '5s', value: requestStats.tickers_last_5s },
                  { label: '10s', value: requestStats.tickers_last_10s },
                ] as { label: string; value: number }[]).map(({ label, value }) => (
                  <div key={label} className="bg-muted/60 rounded p-1 text-center">
                    <div className="font-mono font-bold text-base leading-none">{value}</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">{label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Footer summary row */}
          {(estimatedReqPerTicker || tickerCount != null) && (
            <div className="flex items-center gap-4 mt-2 pt-2 border-t text-xs text-muted-foreground">
              {estimatedReqPerTicker && (
                <span>Req/ticker: <span className="font-mono font-semibold text-foreground">{estimatedReqPerTicker}</span></span>
              )}
              {tickerCount != null && (
                <span className="ml-auto">Tracked: <span className="font-mono font-semibold text-foreground">{tickerCount}</span></span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Errors - Only show if there are errors */}
      {errorEntries.length > 0 && (
        <Alert
          variant={hasRecentErrors ? "destructive" : "default"}
          className={`${hasRecentErrors ? 'border-red-500 bg-red-50 dark:bg-red-950/30' : ''}`}
        >
          <AlertCircle className="h-4 w-4" />
          <AlertTitle className="flex items-center justify-between">
            <span>Data Fetch Errors ({errorEntries.length})</span>
          </AlertTitle>
          <AlertDescription>
            <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
              {errorEntries.map(([ticker, error]) => {
                const recent = isRecentError(error.timestamp);
                return (
                  <div
                    key={ticker}
                    className={`flex items-center gap-2 text-sm p-1 rounded ${recent
                      ? 'bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-200 font-medium'
                      : 'text-muted-foreground'
                      }`}
                  >
                    <XCircle className={`h-3 w-3 flex-shrink-0 ${recent ? 'text-red-600' : ''}`} />
                    <span className="font-mono font-semibold">{ticker}</span>
                    <span className="flex-1 truncate">{error.message}</span>
                    <span className="flex items-center gap-1 text-xs whitespace-nowrap">
                      <Clock className="h-3 w-3" />
                      {formatTimeAgo(error.timestamp)}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="mt-3 pt-2 border-t border-red-200 dark:border-red-800 text-xs text-muted-foreground">
              💡 <strong>Tip:</strong> Outdated yfinance api might be the cause of the errors: <code className="bg-muted px-1 py-0.5 rounded">pip install --upgrade yfinance</code>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

// Memoize to prevent re-renders when parent state changes
export const FetchErrorBanner = memo(FetchErrorBannerComponent);
