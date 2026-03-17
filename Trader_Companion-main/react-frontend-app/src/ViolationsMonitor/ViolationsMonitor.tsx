import { useState, useEffect, useCallback, useRef, useLayoutEffect } from 'react';
import axios from 'axios';
import { Plus, Settings, BookOpen, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react';
import {
  fetchTrades,
  createTrade,
  deleteTrade,
  computeAllViolations,
  computeViolations,
  forceComputeAll
} from './api';
import { API_CONFIG } from '../config';
import type { MonAlertTrade, TradeViolationsResult } from './types';

interface ProviderInfo {
  active_provider: string;
  est_loop_seconds: number | null;
}
import { AddTradeModal } from './components/AddTradeModal';
import { SettingsModal } from './components/SettingsModal';
import { GlossaryModal } from './components/GlossaryModal';
import { TradeCard } from './components/TradeCard';
import { TradeDailyChartModal } from './components/TradeDailyChartModal';

const POLL_INTERVAL = 500; // 500ms — matches backend compute interval

export function ViolationsMonAlert() {
  const [trades, setTrades] = useState<MonAlertTrade[]>([]);
  const [results, setResults] = useState<Record<number, TradeViolationsResult>>({});
  const [loading, setLoading] = useState(false);
  const [loadingTrades, setLoadingTrades] = useState<Set<number>>(new Set());
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem('violations_monalert_collapsed') === 'true';
    } catch {
      return false;
    }
  });
  const [showAddModal, setShowAddModal] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showGlossary, setShowGlossary] = useState(false);
  const [showAllGroupedDates, setShowAllGroupedDates] = useState(false);
  const [providerInfo, setProviderInfo] = useState<ProviderInfo | null>(null);
  const [switchingProvider, setSwitchingProvider] = useState(false);
  const [quickChartTicker, setQuickChartTicker] = useState('');
  const [quickChartBusy, setQuickChartBusy] = useState(false);
  const [quickChartError, setQuickChartError] = useState<string | null>(null);
  const [quickChartTrade, setQuickChartTrade] = useState<MonAlertTrade | null>(null);
  const [quickChartInitialResult, setQuickChartInitialResult] = useState<TradeViolationsResult | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval>>();
  const computeAllInFlightRef = useRef(false);
  const tickerColRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const quickChartTempTradeIdsRef = useRef<Set<number>>(new Set());
  const pendingAddTraceRef = useRef<{
    tradeId: number;
    ticker: string;
    startMark: number;
    stateReadyAt?: number;
    paintedLogged: boolean;
  } | null>(null);

  // Load trades
  const loadTrades = useCallback(async () => {
    try {
      const data = await fetchTrades();
      const activeTrades = data.filter((t: MonAlertTrade) => t.is_active);
      setTrades(activeTrades);
    } catch (e) {
      console.error('Failed to load MonAlert trades:', e);
    }
  }, []);

  // Compute violations for all trades (silent = background poll, no loading state)
  const computeAll = useCallback(async (silent = false) => {
    if (trades.length === 0) return;
    if (computeAllInFlightRef.current) return;
    computeAllInFlightRef.current = true;
    if (!silent) setLoading(true);
    try {
      const allResults = await computeAllViolations();

      const map: Record<number, TradeViolationsResult> = {};
      allResults.forEach((r: TradeViolationsResult) => {
        map[r.trade_id] = r;
      });
      // Spread prev first so results added by handleAddTrade (or any
      // other path) are never dropped by a stale closure.  Fresh API
      // data overwrites on top.
      setResults((prev) => ({
        ...prev,
        ...map,
      }));
    } catch (e) {
      console.error('Failed to compute violations:', e);
    } finally {
      if (!silent) setLoading(false);
      computeAllInFlightRef.current = false;
    }
  }, [trades]);

  // Compute violations for a single trade
  const computeSingle = useCallback(async (tradeId: number) => {
    setLoadingTrades((prev) => new Set(prev).add(tradeId));
    try {
      const result = await computeViolations(tradeId);
      setResults((prev) => ({
        ...prev,
        [tradeId]: result,
      }));
    } catch (e) {
      console.error(`Failed to compute violations for trade ${tradeId}:`, e);
    } finally {
      setLoadingTrades((prev) => {
        const next = new Set(prev);
        next.delete(tradeId);
        return next;
      });
    }
  }, []);

  // Fetch provider info (settings + request stats)
  const fetchProviderInfo = useCallback(async () => {
    try {
      const base = API_CONFIG.tickerDataBaseURL;
      const [settingsRes, statsRes] = await Promise.all([
        axios.get<{ active_provider: string }>(`${base}/settings`),
        axios.get<{ est_loop_seconds: number | null }>(`${base}/request-stats`),
      ]);
      setProviderInfo({
        active_provider: settingsRes.data.active_provider,
        est_loop_seconds: statsRes.data.est_loop_seconds ?? null,
      });
    } catch {
      // non-critical, ignore
    }
  }, []);

  const switchToYf = useCallback(async () => {
    setSwitchingProvider(true);
    try {
      const base = API_CONFIG.tickerDataBaseURL;
      await axios.put(`${base}/settings`, { active_provider: 'yfinance' });
      await fetchProviderInfo();
    } catch (e) {
      console.error('Failed to switch provider to yfinance:', e);
    } finally {
      setSwitchingProvider(false);
    }
  }, [fetchProviderInfo]);

  // Initial load
  useEffect(() => {
    loadTrades();
    fetchProviderInfo();
  }, [loadTrades, fetchProviderInfo]);

  // Refresh provider info every 10s
  useEffect(() => {
    const id = setInterval(fetchProviderInfo, 10_000);
    return () => clearInterval(id);
  }, [fetchProviderInfo]);

  // Compute when trades change
  useEffect(() => {
    if (trades.length > 0) {
      computeAll();
    }
  }, [trades.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // Polling (silent — no loading spinner flicker)
  useEffect(() => {
    pollRef.current = setInterval(() => {
      if (trades.length > 0 && !collapsed) {
        computeAll(true);
      }
    }, POLL_INTERVAL);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [trades.length, collapsed, computeAll]);

  // Persist collapsed state
  useEffect(() => {
    localStorage.setItem('violations_monalert_collapsed', String(collapsed));
  }, [collapsed]);

  // Sync ticker column widths: direct DOM manipulation, no state needed
  useLayoutEffect(() => {
    const refs = tickerColRefs.current;
    if (refs.size === 0) return;
    // Pass 1: reset all to natural width
    refs.forEach((el) => { el.style.minWidth = ''; });
    // Force synchronous reflow so measurements reflect natural sizes
    void document.body.offsetHeight;
    // Pass 2: find widest
    let maxW = 0;
    refs.forEach((el) => {
      const w = el.scrollWidth;
      if (w > maxW) maxW = w;
    });
    // Pass 3: apply to all
    if (maxW > 0) {
      refs.forEach((el) => { el.style.minWidth = `${maxW}px`; });
    }
  });

  useEffect(() => {
    const pending = pendingAddTraceRef.current;
    if (!pending || pending.paintedLogged) return;

    const hasTrade = trades.some((t) => t.id === pending.tradeId);
    const hasResult = Boolean(results[pending.tradeId]);
    if (!hasTrade || !hasResult) return;

    pending.stateReadyAt = performance.now();
    if (collapsed) {
      console.log(
        `[UI] Data ready for ${pending.ticker} at ${(pending.stateReadyAt - pending.startMark).toFixed(2)}ms but MonAlert is collapsed.`
      );
      pending.paintedLogged = true;
      return;
    }

    pending.paintedLogged = true;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const paintedAt = performance.now();
        const card = document.querySelector(`[data-trade-id=\"${pending.tradeId}\"]`);
        console.log(
          `[UI] Actually shown in UI for ${pending.ticker}: state_ready=${((pending.stateReadyAt ?? paintedAt) - pending.startMark).toFixed(2)}ms ` +
          `painted=${(paintedAt - pending.startMark).toFixed(2)}ms card_present=${Boolean(card)}`
        );
      });
    });
  }, [results, trades, collapsed]);

  const handleAddTrade = async (
    ticker: string,
    startDate: string,
    endDate: string | null,
    useLatest: boolean
  ) => {
    const startMark = performance.now();
    // Close modal instantly for a snappy UI experience 
    setShowAddModal(false);

    try {
      const newTrade = await createTrade({
        ticker,
        start_date: startDate,
        end_date: endDate,
        use_latest_end_date: useLatest,
      });

      // Compute violations BEFORE adding the trade to state so the card
      // never renders without its data (no blank-card flash).
      let tradeResult = newTrade.initial_result ?? null;
      if (!tradeResult) {
        tradeResult = await computeViolations(newTrade.id);
      }

      // Both updates in the same synchronous block → React 18 batches
      // them into a single render.  Set results first so the data is
      // already present when TradeCard mounts.
      setResults((prev) => ({
        ...prev,
        [newTrade.id]: tradeResult
      }));
      setTrades((prev) => [...prev, newTrade]);

      pendingAddTraceRef.current = {
        tradeId: newTrade.id,
        ticker: ticker.toUpperCase(),
        startMark,
        stateReadyAt: performance.now(),
        paintedLogged: false,
      };
    } catch (e) {
      console.error('Failed to add trade:', e);
    }
  };

  const handleTradeUpdated = async () => {
    await loadTrades();
    // Force synchronous recompute so date changes are reflected instantly
    try {
      const allResults = await forceComputeAll();

      const map: Record<number, TradeViolationsResult> = {};
      allResults.forEach((r: TradeViolationsResult) => {
        map[r.trade_id] = r;
      });
      setResults(map);
    } catch (e) {
      console.error('Failed to force-compute violations:', e);
    }
  };

  const removeQuickChartTempTrade = useCallback(async (tradeId: number) => {
    if (!quickChartTempTradeIdsRef.current.has(tradeId)) return;
    try {
      await deleteTrade(tradeId);
    } catch (e) {
      console.warn(`Could not delete temporary quick-chart trade ${tradeId}:`, e);
    } finally {
      quickChartTempTradeIdsRef.current.delete(tradeId);
      setTrades((prev) => prev.filter((t) => t.id !== tradeId));
      setResults((prev) => {
        const next = { ...prev };
        delete next[tradeId];
        return next;
      });
    }
  }, []);

  const handleCloseQuickChartModal = useCallback(async () => {
    const trade = quickChartTrade;
    setQuickChartTrade(null);
    setQuickChartInitialResult(null);
    if (trade) {
      await removeQuickChartTempTrade(trade.id);
    }
  }, [quickChartTrade, removeQuickChartTempTrade]);

  const handleOpenQuickChart = useCallback(async () => {
    const ticker = quickChartTicker.trim().toUpperCase();
    if (!ticker) return;
    if (!/^[A-Z0-9.-]{1,10}$/.test(ticker)) {
      setQuickChartError('Ticker must be 1-10 chars (A-Z, 0-9, ., -).');
      return;
    }

    setQuickChartBusy(true);
    setQuickChartError(null);

    try {
      if (quickChartTrade) {
        await removeQuickChartTempTrade(quickChartTrade.id);
        setQuickChartTrade(null);
        setQuickChartInitialResult(null);
      }

      const anchorDate = new Date().toISOString().slice(0, 10);
      const tempTrade = await createTrade({
        ticker,
        start_date: anchorDate,
        end_date: anchorDate,
        use_latest_end_date: false,
      });
      quickChartTempTradeIdsRef.current.add(tempTrade.id);

      const tempResult = tempTrade.initial_result ?? await computeViolations(tempTrade.id);

      setTrades((prev) => [...prev, tempTrade]);
      setResults((prev) => ({ ...prev, [tempTrade.id]: tempResult }));
      setQuickChartTrade(tempTrade);
      setQuickChartInitialResult(tempResult);
      setQuickChartTicker('');
    } catch (e) {
      console.error('Failed opening quick chart:', e);
      setQuickChartError(`Could not open chart for ${ticker}.`);
    } finally {
      setQuickChartBusy(false);
    }
  }, [quickChartTicker, quickChartTrade, removeQuickChartTempTrade]);

  // Total counts across all trades
  const totalV = Object.values(results).reduce((s, r) => s + (r?.total_violations ?? 0), 0);
  const totalC = Object.values(results).reduce((s, r) => s + (r?.total_confirmations ?? 0), 0);

  return (
    <div className="w-full bg-background border-b border-border">
      <div className="w-full px-3 sm:px-4 lg:px-8">
        {/* Header bar - always visible */}
        <div className="flex items-center gap-2 py-1.5">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            {collapsed ? <ChevronDown className="h-3 w-3" /> : <ChevronUp className="h-3 w-3" />}
            <span>MonAlert</span>
          </button>

          {/* Quick summary */}
          {trades.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-muted-foreground">{trades.length} position{trades.length !== 1 ? 's' : ''}</span>
              <span className="font-bold text-green-50 bg-green-700/75 px-1.5 py-0.5 rounded text-[11px]">
                {totalC}
              </span>
              <span className="font-bold text-red-50 bg-red-800/70 px-1.5 py-0.5 rounded text-[11px]">
                {totalV}
              </span>
              {providerInfo && (() => {
                const isWb = providerInfo.active_provider === 'webull';
                const label = isWb ? 'WB' : 'YF';
                const estStr = providerInfo.est_loop_seconds != null
                  ? `${providerInfo.est_loop_seconds.toFixed(1)}s/loop`
                  : null;
                const tickerCountStr = trades.length > 0 ? ` (${trades.length} ticker${trades.length !== 1 ? 's' : ''})` : '';
                return (
                  <>
                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-bold font-mono bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400">
                      {label}{estStr && ` ${estStr}`}{tickerCountStr}
                    </span>
                    {isWb && (
                      <button
                        onClick={switchToYf}
                        disabled={switchingProvider}
                        className="px-2 py-0.5 rounded text-[11px] font-semibold bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-60"
                        title="Switch active provider to YFinance"
                      >
                        {switchingProvider ? 'Switching…' : 'Switch to YF'}
                      </button>
                    )}
                  </>
                );
              })()}
              {loading && <RefreshCw className="h-3 w-3 animate-spin text-muted-foreground" />}
            </div>
          )}

          {/* Action buttons */}
          <div className="flex items-center gap-1 ml-auto">
            <input
              type="text"
              value={quickChartTicker}
              onChange={(e) => {
                setQuickChartTicker(e.target.value.toUpperCase());
                if (quickChartError) setQuickChartError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  if (!quickChartBusy) {
                    void handleOpenQuickChart();
                  }
                }
              }}
              placeholder="Ticker"
              maxLength={10}
              className="h-7 w-24 rounded border border-border bg-background px-2 text-xs uppercase"
              title="Type ticker and press Enter to open quick chart"
            />
            <button
              onClick={() => {
                if (!quickChartBusy) {
                  void handleOpenQuickChart();
                }
              }}
              disabled={quickChartBusy || !quickChartTicker.trim()}
              className="h-7 px-2 text-xs rounded border border-border text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-50"
              title="Open quick chart"
            >
              {quickChartBusy ? 'Opening…' : 'Go'}
            </button>
            <button
              onClick={() => setShowAddModal(true)}
              className="flex items-center gap-1 px-2 py-1 text-xs bg-primary text-primary-foreground rounded hover:opacity-90"
              title="Add position"
            >
              <Plus className="h-3 w-3" />
              Add
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted rounded"
              title="Settings"
            >
              <Settings className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => setShowGlossary(true)}
              className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted rounded"
              title="Glossary"
            >
              <BookOpen className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => setShowAllGroupedDates((v) => !v)}
              className={`px-2 py-1 text-xs rounded border ${showAllGroupedDates
                ? 'bg-primary text-primary-foreground border-primary'
                : 'text-muted-foreground border-border hover:text-foreground hover:bg-muted'
                }`}
              title="Toggle grouped date details"
            >
              Dates {showAllGroupedDates ? 'On' : 'Off'}
            </button>
            {trades.length > 0 && (
              <button
                onClick={() => computeAll(false)}
                disabled={loading}
                className="p-1 text-muted-foreground hover:text-foreground hover:bg-muted rounded disabled:opacity-50"
                title="Refresh all"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
              </button>
            )}
          </div>
        </div>

        {/* Expanded content */}
        {!collapsed && (
          <div className="pb-2 space-y-1">
            {quickChartError && (
              <p className="text-xs text-red-400">{quickChartError}</p>
            )}
            {trades.length === 0 ? (
              <p className="text-xs text-muted-foreground py-1">
                No positions in MonAlert. Click "Add" to start tracking violations.
              </p>
            ) : (
              trades.map((trade) => (
                <TradeCard
                  key={trade.id}
                  trade={trade}
                  result={results[trade.id] ?? null}
                  loading={loadingTrades.has(trade.id) || (loading && !results[trade.id])}
                  onTradeUpdated={handleTradeUpdated}
                  onRefresh={() => computeSingle(trade.id)}
                  showAllGroupedDates={showAllGroupedDates}
                  tickerColRef={(el: HTMLDivElement | null) => {
                    if (el) tickerColRefs.current.set(trade.id, el);
                    else tickerColRefs.current.delete(trade.id);
                  }}
                />
              ))
            )}
          </div>
        )}
      </div>

      {/* Modals */}
      {showAddModal && (
        <AddTradeModal onAdd={handleAddTrade} onClose={() => setShowAddModal(false)} />
      )}
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
      {showGlossary && <GlossaryModal onClose={() => setShowGlossary(false)} />}
      {quickChartTrade && (
        <TradeDailyChartModal
          trade={quickChartTrade}
          initialResult={quickChartInitialResult}
          onClose={() => {
            void handleCloseQuickChartModal();
          }}
        />
      )}
    </div>
  );
}
