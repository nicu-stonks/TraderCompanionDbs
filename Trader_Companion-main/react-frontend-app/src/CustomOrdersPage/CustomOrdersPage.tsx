// COMPLETE CLEAN REWRITE (previous file was corrupted with legacy fragments)
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Plus, Eye, AlertCircle, RefreshCw, CheckCircle2, AlertTriangle } from 'lucide-react';
import { balanceAPI } from '../TradeStatisticsPage/services/balanceAPI';
import { tradeAPI } from '../TradeStatisticsPage/services/tradeAPI';
import { OrderConfig, NewTrade, NewTradeStop, ServerStatus, ErrorLog, IbConnectionStatus } from './types';
import { OrderConfigTab } from './components/OrderConfigTab';
import { TradesTab } from './components/TradesTab';
import { StatusTab } from './components/StatusTab';
import { ErrorsTab } from './components/ErrorsTab';

type PendingRiskFixTrade = {
  payload: unknown;
  ticker: string;
  requiredRisk: number;
  availableRisk: number;
};

type PendingRiskFixExecute = {
  ticker: string;
  lower_price_range: number;
  higher_price_range: number;
  requiredRisk: number;
  availableRisk: number;
};

export function CustomOrdersPage() {
  const ORDER_CONFIG_KEY = 'customOrdersPage.orderConfig.v1';
  const NEW_TRADE_KEY = 'customOrdersPage.newTrade.v2';
  const PIVOT_KEY = 'customOrdersPage.pivotPositions.v1';
  const SHOW_ADV_KEY = 'customOrdersPage.showAdvanced.v1';
  const SEEN_ERRORS_KEY = 'customOrdersPage.seenErrors.v1';

  const defaultOrderConfig: OrderConfig = {
    ticker: '', lower_price: 0, higher_price: 0, volume_requirements: [], pivot_adjustment: '0.0',
    day_high_max_percent_off: 3, time_in_pivot: 30, time_in_pivot_positions: '', data_server: 'http://localhost:8000/ticker_data/api/ticker_data',
    trade_server: 'http://localhost:5002', volume_multipliers: [1, 1, 1], max_day_low: null, min_day_low: null,
    wait_after_open_minutes: 1.01, breakout_lookback_minutes: 60, breakout_exclude_minutes: 0.5,
    start_minutes_before_close: null, stop_minutes_before_close: 0, request_lower_price: null, request_higher_price: null,
  };
  const defaultPivotPositions = { any: false, lower: false, middle: false, upper: false };
  const defaultNewTrade: NewTrade = {
    ticker: '', shares: 0, risk_amount: 0, risk_percent_of_equity: 0, lower_price_range: 0, higher_price_range: 0,
    order_type: 'MKT', adaptive_priority: 'Urgent', timeout_seconds: 5,
    sell_stops: [{ price: 0, position_pct: 1, percent_below_fill: undefined, __ui_mode: 'price' }],
    consider_zero_risk: false
  };

  const safeLoad = <T,>(k: string, fb: T): T => { if (typeof window === 'undefined') return fb; try { const raw = localStorage.getItem(k); if (!raw) return fb; const p = JSON.parse(raw); return typeof p === 'object' && p !== null ? { ...fb, ...p } : fb; } catch { return fb; } };

  const [activeTab, setActiveTab] = useState<'order' | 'trades' | 'status' | 'errors'>('status');
  const [orderConfig, setOrderConfig] = useState<OrderConfig>(() => safeLoad(ORDER_CONFIG_KEY, defaultOrderConfig));
  const [pivotPositions, setPivotPositions] = useState(() => safeLoad(PIVOT_KEY, defaultPivotPositions));
  const [newTrade, setNewTrade] = useState<NewTrade>(() => safeLoad(NEW_TRADE_KEY, defaultNewTrade));
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [errors, setErrors] = useState<ErrorLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [riskAmount, setRiskAmount] = useState(0);
  const [showVolumeWarningModal, setShowVolumeWarningModal] = useState(false);
  const [pendingRiskFixTrade, setPendingRiskFixTrade] = useState<PendingRiskFixTrade | null>(null);
  const [pendingRiskFixExecute, setPendingRiskFixExecute] = useState<PendingRiskFixExecute | null>(null);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(() => { if (typeof window === 'undefined') return false; try { return JSON.parse(localStorage.getItem(SHOW_ADV_KEY) || 'false'); } catch { return false; } });
  const [flash, setFlash] = useState({ order: false, trade: false, advanced: false, refresh: false });
  const subtleFlashClass = 'ring-2 ring-primary/50 bg-muted/60 shadow-sm';
  const triggerFlash = (k: keyof typeof flash, d = 180) => { setFlash(p => ({ ...p, [k]: true })); window.setTimeout(() => setFlash(p => ({ ...p, [k]: false })), d); };
  const [currentEquity, setCurrentEquity] = useState<number | null>(null);
  const [ibStatus, setIbStatus] = useState<IbConnectionStatus | null>(null);
  const [ibStatusLoading, setIbStatusLoading] = useState(false);
  const tickerInputRef = useRef<HTMLInputElement>(null);
  const riskFixConfirmButtonRef = useRef<HTMLButtonElement>(null);
  const riskFixExecuteConfirmButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!pendingRiskFixTrade) return;
    window.setTimeout(() => riskFixConfirmButtonRef.current?.focus(), 0);
  }, [pendingRiskFixTrade]);

  useEffect(() => {
    if (!pendingRiskFixExecute) return;
    window.setTimeout(() => riskFixExecuteConfirmButtonRef.current?.focus(), 0);
  }, [pendingRiskFixExecute]);

  // Error tracking
  const getErrorId = (error: ErrorLog): string => `${error.timestamp}_${error.error_type}_${error.error_message}`;
  const [seenErrorIds, setSeenErrorIds] = useState<Set<string>>(() => {
    if (typeof window === 'undefined') return new Set();
    try {
      const stored = localStorage.getItem(SEEN_ERRORS_KEY);
      if (stored) {
        const ids = JSON.parse(stored) as string[];
        return new Set(ids);
      }
    } catch {
      // ignore localStorage parse errors
    }
    return new Set();
  });
  const unseenErrorCount = errors.filter(e => !seenErrorIds.has(getErrorId(e))).length;

  // Equity polling
  useEffect(() => { let m = true; const load = async () => { try { const balance = await balanceAPI.getBalance(); const resp = await tradeAPI.getTrades(); const trades: Array<{ Return: number | null }> = resp.data; const ret = trades.reduce((s, t) => s + (t.Return ?? 0), 0); if (m) setCurrentEquity(Math.round((balance + ret) * 100) / 100); } catch (e) { console.error(e); } }; load(); const iv = setInterval(load, 20000); return () => { m = false; clearInterval(iv); }; }, []);

  // Status & errors polling
  const fetchStatus = async () => { try { const r = await fetch('http://localhost:5002/status'); setServerStatus(await r.json()); } catch (e) { console.error(e); } };
  const fetchErrors = async () => { try { const r = await fetch('http://localhost:5002/errors'); const j = await r.json(); if (j.success) setErrors(j.errors); } catch (e) { console.error(e); } };
  useEffect(() => { fetchStatus(); fetchErrors(); const iv = setInterval(() => { fetchStatus(); fetchErrors(); }, 5000); return () => clearInterval(iv); }, []);
  const fetchIbStatus = useCallback(async () => {
    setIbStatusLoading(true);
    try {
      const response = await fetch('http://localhost:5002/ib_status');
      const data = await response.json();
      setIbStatus({
        success: !!data.success,
        stage: data.stage,
        message: data.message || (data.success ? 'IBKR Web API responded successfully.' : 'Unable to reach IBKR Web API.'),
        sample_symbol: data.sample_symbol,
        sample_conid: data.sample_conid,
        checked_at: data.checked_at || new Date().toISOString(),
      });
    } catch {
      setIbStatus({
        success: false,
        stage: 'network',
        message: 'Unable to reach IBKR status endpoint. Ensure the Stock Buyer server is running.',
        checked_at: new Date().toISOString(),
      });
    } finally {
      setIbStatusLoading(false);
    }
  }, []);
  useEffect(() => {
    fetchIbStatus();
    const iv = setInterval(() => { fetchIbStatus(); }, 30000);
    return () => clearInterval(iv);
  }, [fetchIbStatus]);

  // Pivot position update
  const updatePivotPositions = (position: string, checked: boolean) => {
    setPivotPositions(p => ({ ...p, [position]: checked }));
    const updated = { ...pivotPositions, [position]: checked };
    const selected = Object.entries(updated).filter(([, v]) => v).map(([k]) => k).join(',');
    setOrderConfig(o => ({ ...o, time_in_pivot_positions: selected }));
  };

  // Risk update
  const updateRisk = async () => { setLoading(true); try { const r = await fetch('http://localhost:5002/update_risk', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ amount: riskAmount }) }); const j = await r.json(); if (!j.success) alert(`Error: ${j.error}`); if (j.success) fetchStatus(); } catch (e) { alert(`Network error: ${e}`); } finally { setLoading(false); } };

  const updateRiskToAmount = async (amount: number) => {
    const normalized = Math.round((Number(amount) || 0) * 100) / 100;
    const r = await fetch('http://localhost:5002/update_risk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount: normalized })
    });
    const j = await r.json();
    if (!j?.success) {
      throw new Error(j?.error || 'Unable to update risk');
    }
    setRiskAmount(normalized);
    fetchStatus();
  };

  const parseRiskExceededError = (raw: unknown): { requiredRisk: number; availableRisk: number } | null => {
    const msg = typeof raw === 'string' ? raw : '';
    if (!msg) return null;
    if (!/required\s+risk\s*\$/i.test(msg) || !/available\s+risk\s*\$/i.test(msg)) return null;
    if (!/exceeds/i.test(msg)) return null;

    const requiredMatch = msg.match(/required\s+risk\s*\$\s*([0-9]+(?:\.[0-9]+)?)/i);
    const availableMatch = msg.match(/available\s+risk\s*\$\s*([0-9]+(?:\.[0-9]+)?)/i);
    const requiredRisk = requiredMatch ? Number(requiredMatch[1]) : NaN;
    const availableRisk = availableMatch ? Number(availableMatch[1]) : NaN;
    if (!isFinite(requiredRisk) || !isFinite(availableRisk)) return null;
    return { requiredRisk, availableRisk };
  };

  // Parser for execute trade insufficient risk error: "Insufficient available risk: have $X.XX, need $Y.YY for trade TICKER"
  const parseInsufficientRiskError = (raw: unknown): { requiredRisk: number; availableRisk: number } | null => {
    const msg = typeof raw === 'string' ? raw : '';
    if (!msg) return null;
    if (!/insufficient\s+available\s+risk/i.test(msg)) return null;

    const haveMatch = msg.match(/have\s+\$\s*([0-9]+(?:\.[0-9]+)?)/i);
    const needMatch = msg.match(/need\s+\$\s*([0-9]+(?:\.[0-9]+)?)/i);
    const availableRisk = haveMatch ? Number(haveMatch[1]) : NaN;
    const requiredRisk = needMatch ? Number(needMatch[1]) : NaN;
    if (!isFinite(requiredRisk) || !isFinite(availableRisk)) return null;
    return { requiredRisk, availableRisk };
  };

  // Trade actions
  const deleteTrade = async (tradeId: string) => { if (!confirm('Delete trade?')) return; setLoading(true); try { const r = await fetch('http://localhost:5002/remove_trade', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ trade_id: tradeId }) }); const j = await r.json(); if (!j.success) alert(`Error: ${j.error}`); if (j.success) fetchStatus(); } catch (e) { alert(`Network error: ${e}`); } finally { setLoading(false); } };
  const deleteAllTrades = async () => {
    if (!serverStatus?.trades || serverStatus.trades.length === 0) return;
    if (!confirm(`Delete all ${serverStatus.trades.length} trade(s)?`)) return;
    setLoading(true);
    try {
      for (const trade of serverStatus.trades) {
        const r = await fetch('http://localhost:5002/remove_trade', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ trade_id: trade.trade_id })
        });
        const j = await r.json();
        if (!j.success) {
          alert(`Error deleting trade ${trade.ticker}: ${j.error}`);
          break;
        }
      }
      fetchStatus();
    } catch (e) {
      alert(`Network error: ${e}`);
    } finally {
      setLoading(false);
    }
  };
  const executeTradeNow = async ({ ticker, lower_price_range, higher_price_range }: { ticker: string; lower_price_range: number; higher_price_range: number; }) => {
    if (!confirm(`Execute now?\n${ticker} $${lower_price_range.toFixed(2)}-$${higher_price_range.toFixed(2)}`)) return;
    setLoading(true);
    try {
      const r = await fetch('http://localhost:5002/execute_trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, lower_price: lower_price_range, higher_price: higher_price_range })
      });
      const j = await r.json();
      if (j.success) {
        alert('Trade executed successfully.');
        fetchStatus();
      } else {
        const parsed = parseInsufficientRiskError(j.error);
        if (parsed) {
          setPendingRiskFixExecute({
            ticker,
            lower_price_range,
            higher_price_range,
            requiredRisk: parsed.requiredRisk,
            availableRisk: parsed.availableRisk
          });
          return;
        }
        alert(`Execution error: ${j.error || 'Unknown'}`);
      }
    } catch (e) {
      alert(`Network error: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  // Update trade risk: delete old trade and create new one with recalculated shares
  const updateTradeRisk = async (
    tradeId: string,
    newRiskAmount: number,
    settings?: { order_type: 'MKT' | 'IBALGO'; adaptive_priority?: 'Patient' | 'Normal' | 'Urgent'; timeout_seconds?: number }
  ) => {
    // Find the trade in serverStatus
    const trade = serverStatus?.trades?.find(t => t.trade_id === tradeId);
    if (!trade) {
      alert('Trade not found');
      return;
    }

    // Calculate new shares using the same formula
    const entry = (trade.lower_price_range + trade.higher_price_range) / 2;
    let weightedDrop = 0;
    for (const stop of trade.sell_stops) {
      const pct = trade.shares > 0 ? stop.shares / trade.shares : 0;
      if (pct <= 0) continue;
      let stopPrice: number | null = null;
      if (stop.percent_below_fill !== undefined && stop.percent_below_fill !== null) {
        stopPrice = entry * (1 - stop.percent_below_fill / 100);
      } else {
        stopPrice = stop.price ?? null;
      }
      if (!stopPrice || !isFinite(stopPrice)) continue;
      const drop = entry - stopPrice;
      if (drop <= 0) continue;
      weightedDrop += pct * drop;
    }

    if (weightedDrop <= 0) {
      alert('Unable to calculate shares');
      return;
    }

    const targetOrderType = settings?.order_type ?? ((trade.order_type ?? 'MKT') as 'MKT' | 'IBALGO');
    const isIbalgo = targetOrderType === 'IBALGO';
    const targetAdaptivePriority = settings?.adaptive_priority ?? ((trade.adaptive_priority ?? 'Urgent') as 'Patient' | 'Normal' | 'Urgent');
    const targetTimeout = settings?.timeout_seconds ?? (trade.timeout_seconds ?? (isIbalgo ? 30 : 5));
    const newSharesRaw = newRiskAmount / weightedDrop;
    const newShares = isIbalgo
      ? Math.max(1, Math.round(newSharesRaw))
      : Math.round(newSharesRaw * 100) / 100;

    // Rebuild sell stops with new share allocations (maintaining same proportions)
    const proportionalStops = trade.sell_stops.map(stop => {
      const pct = trade.shares > 0 ? stop.shares / trade.shares : 0;
      const newStopShares = isIbalgo ? (newShares * pct) : Math.round(newShares * pct * 1000) / 1000;
      if (stop.percent_below_fill !== undefined && stop.percent_below_fill !== null) {
        return { shares: newStopShares, percent_below_fill: stop.percent_below_fill };
      } else {
        return { shares: newStopShares, price: stop.price ?? entry };
      }
    });

    const newSellStops = isIbalgo
      ? proportionalStops.map((stop) => {
          const roundedShares = Math.floor(stop.shares);
          return { ...stop, shares: roundedShares };
        })
      : proportionalStops;

    if (isIbalgo) {
      let assigned = newSellStops.reduce((sum, stop) => sum + stop.shares, 0);
      let remainder = newShares - assigned;
      let cursor = 0;
      while (remainder > 0 && newSellStops.length > 0) {
        newSellStops[cursor % newSellStops.length].shares += 1;
        remainder -= 1;
        cursor += 1;
      }
    }

    const newTradePayload = {
      ticker: trade.ticker,
      shares: newShares,
      risk_amount: newRiskAmount,
      lower_price_range: trade.lower_price_range,
      higher_price_range: trade.higher_price_range,
      order_type: targetOrderType,
      adaptive_priority: isIbalgo ? targetAdaptivePriority : undefined,
      timeout_seconds: Math.max(1, Math.round(targetTimeout)),
      sell_stops: newSellStops,
    };

    setLoading(true);
    try {
      // Delete the old trade
      const deleteResp = await fetch('http://localhost:5002/remove_trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trade_id: tradeId }),
      });
      const deleteResult = await deleteResp.json();
      if (!deleteResult.success) {
        alert(`Error deleting old trade: ${deleteResult.error}`);
        return;
      }

      // Create the new trade
      const addResp = await fetch('http://localhost:5002/add_trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTradePayload),
      });
      const addResult = await addResp.json();
      if (addResult.success) {
        fetchStatus();
      } else {
        alert(`Error creating new trade: ${addResult.error}`);
      }
    } catch (e) {
      alert(`Network error: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  // Share / stop math
  const mid = (l: number, u: number) => (!isFinite(l) || !isFinite(u) || l <= 0 || u <= 0 || l === u) ? null : (l + u) / 2;
  const autoShares = (t: NewTrade) => { const entry = mid(t.lower_price_range, t.higher_price_range); if (entry == null || !isFinite(t.risk_amount) || t.risk_amount <= 0 || t.sell_stops.length === 0) return null; let wd = 0; for (const s of t.sell_stops) { const pct = Number(s.position_pct) || 0; if (pct <= 0) continue; let stopPrice: number | null = null; if ((s.__ui_mode ?? 'price') === 'percent') { stopPrice = entry * (1 - (Number(s.percent_below_fill) || 0) / 100); } else { stopPrice = Number(s.price); } if (!stopPrice || !isFinite(stopPrice)) continue; const drop = entry - stopPrice; if (drop <= 0) continue; wd += pct * drop; } if (wd <= 0) return null; const sh = t.risk_amount / wd; if (!isFinite(sh) || sh <= 0) return null; if ((t.order_type ?? 'MKT') === 'IBALGO') return Math.max(1, Math.round(sh)); return Math.round(sh * 100) / 100; };
  const allocate = (total: number, stops: NewTradeStop[], integerOnly = false) => { const unitScale = integerOnly ? 1 : 100; const units = Math.max(0, Math.round((total || 0) * unitScale)); if (!stops.length) return [] as number[]; const p = stops.map(s => Math.max(0, Number(s.position_pct) || 0)); const sum = p.reduce((a, b) => a + b, 0); if (sum <= 0) { const r = new Array(stops.length).fill(0); r[0] = units / unitScale; return r; } const raw = p.map(x => units * (x / sum)); const base = raw.map(x => Math.floor(x)); let rem = units - base.reduce((a, b) => a + b, 0); const order = raw.map((v, i) => ({ i, f: v - Math.floor(v) })).sort((a, b) => b.f - a.f).map(o => o.i); let c = 0; while (rem > 0) { base[order[c % order.length]] += 1; rem--; c++; } return base.map(v => v / unitScale); };
  const buildPayload = (t: NewTrade, auto: boolean) => { const orderType = t.order_type ?? 'MKT'; const isIbalgo = orderType === 'IBALGO'; const totalRaw = auto ? ((autoShares(t) ?? t.shares) || 0) : (t.shares || 0); const total = isIbalgo ? Math.max(0, Math.round(totalRaw)) : Math.max(0, Math.round(totalRaw * 100) / 100); const alloc = allocate(total, t.sell_stops, isIbalgo); const entry = mid(t.lower_price_range, t.higher_price_range) ?? 0; return { ticker: t.ticker, shares: total, risk_amount: t.risk_amount, lower_price_range: t.lower_price_range, higher_price_range: t.higher_price_range, order_type: orderType, adaptive_priority: isIbalgo ? (t.adaptive_priority ?? 'Urgent') : undefined, timeout_seconds: Math.max(1, Math.round((t.timeout_seconds ?? (isIbalgo ? 30 : 5)))), sell_stops: t.sell_stops.map((s, i) => (s.__ui_mode ?? 'price') === 'percent' ? { shares: alloc[i] ?? 0, percent_below_fill: s.percent_below_fill ?? 0 } : { shares: alloc[i] ?? 0, price: s.price ?? entry }) }; };
  const [autoCalcEnabled, setAutoCalcEnabled] = useState(true);
  const [autoCalcReady, setAutoCalcReady] = useState(false);
  useEffect(() => {
    if (!autoCalcEnabled) { setAutoCalcReady(autoShares(newTrade) != null); return; } const s = autoShares(newTrade); const ready = s != null; setAutoCalcReady(!!ready); if (ready && Number(newTrade.shares) !== s) setNewTrade(t => ({ ...t, shares: s! })); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoCalcEnabled, newTrade.lower_price_range, newTrade.higher_price_range, newTrade.risk_amount, newTrade.order_type, JSON.stringify(newTrade.sell_stops)]);
  const addSellStop = () => setNewTrade(t => ({ ...t, sell_stops: [...t.sell_stops, { price: 0, position_pct: 0, percent_below_fill: undefined, __ui_mode: 'price' }] }));
  const removeSellStop = (i: number) => setNewTrade(t => ({ ...t, sell_stops: t.sell_stops.filter((_, idx) => idx !== i) }));
  const updateSellStop = (i: number, field: 'price' | 'position_pct' | 'percent_below_fill' | '__ui_mode', value: number | string) => setNewTrade(t => ({ ...t, sell_stops: t.sell_stops.map((s, idx) => { if (idx !== i) return s; if (field === '__ui_mode') { const mode = value as 'price' | 'percent'; return mode === 'price' ? { ...s, __ui_mode: 'price', percent_below_fill: undefined, price: s.price ?? 0 } : { ...s, __ui_mode: 'percent', price: undefined, percent_below_fill: s.percent_below_fill ?? 1 }; } const num = typeof value === 'string' ? (parseFloat(value) || 0) : value; return { ...s, [field]: num } as NewTradeStop; }) }));
  const addTrade = async () => {
    if (newTrade.lower_price_range >= newTrade.higher_price_range) { alert('Lower pivot price must be lower than higher pivot price!'); return; }
    const totalPct = newTrade.sell_stops.reduce((s, st) => s + (Number(st.position_pct) || 0), 0);
    if (Math.abs(totalPct - 1) > 0.001) { alert(`Sell stop percentages must sum to 1.0. Current: ${totalPct.toFixed(4)}`); return; }

    setLoading(true);
    try {
      const payload = buildPayload(newTrade, autoCalcEnabled);
      const r = await fetch('http://localhost:5002/add_trade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const j = await r.json();
      if (j.success) {
        try {
          await fetch('http://localhost:8000/ticker_data/api/ticker_data/tickers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: newTrade.ticker.toUpperCase() })
          });
        } catch (e) {
          console.error('ticker add', e);
        }
        setNewTrade(defaultNewTrade);
        fetchStatus();
        setTimeout(() => tickerInputRef.current?.focus(), 0);
      } else {
        const parsed = parseRiskExceededError(j.error);
        if (parsed) {
          setPendingRiskFixTrade({
            payload,
            ticker: (newTrade.ticker || '').toUpperCase(),
            requiredRisk: parsed.requiredRisk,
            availableRisk: parsed.availableRisk
          });
          return;
        }
        alert(`Error: ${j.error}`);
      }
    } catch (e) {
      alert(`Network error: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  // Persistence
  useEffect(() => { try { localStorage.setItem(ORDER_CONFIG_KEY, JSON.stringify(orderConfig)); } catch { /* ignore persistence error */ } }, [orderConfig]);
  useEffect(() => { try { localStorage.setItem(NEW_TRADE_KEY, JSON.stringify(newTrade)); } catch { /* ignore persistence error */ } }, [newTrade]);
  useEffect(() => { try { localStorage.setItem(PIVOT_KEY, JSON.stringify(pivotPositions)); } catch { /* ignore persistence error */ } }, [pivotPositions]);
  useEffect(() => { try { localStorage.setItem(SHOW_ADV_KEY, JSON.stringify(showAdvanced)); } catch { /* ignore persistence error */ } }, [showAdvanced]);
  useEffect(() => { try { localStorage.setItem(SEEN_ERRORS_KEY, JSON.stringify(Array.from(seenErrorIds))); } catch { /* ignore persistence error */ } }, [seenErrorIds]);

  // Mark errors as seen when user visits errors tab
  useEffect(() => {
    if (activeTab === 'errors' && errors.length > 0) {
      const newSeenIds = new Set(seenErrorIds);
      errors.forEach(error => {
        newSeenIds.add(getErrorId(error));
      });
      setSeenErrorIds(newSeenIds);
    }
  }, [activeTab, errors]); // eslint-disable-line react-hooks/exhaustive-deps
  const clearSavedOrderConfig = () => { setOrderConfig(defaultOrderConfig); try { localStorage.removeItem(ORDER_CONFIG_KEY); } catch { /* ignore */ } triggerFlash('order'); };
  const clearSavedTrade = () => { setNewTrade(defaultNewTrade); try { localStorage.removeItem(NEW_TRADE_KEY); } catch { /* ignore */ } triggerFlash('trade'); };

  // Start order (volume warning)
  const performStartOrder = async () => { setLoading(true); try { const resp = await fetch('http://localhost:5003/start_bot', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(orderConfig) }); const j = await resp.json(); if (!j.success) alert(`Error: ${j.error}`); } catch (e) { alert(`Network error: ${e}`); } finally { setLoading(false); } };
  const startOrderHandler = () => { if (orderConfig.volume_requirements.length === 0) { setShowVolumeWarningModal(true); return; } performStartOrder(); };

  const TabButton = ({ tab, label, icon: Icon, badgeCount }: { tab: string; label: string; icon: React.ComponentType<{ className?: string }>; badgeCount?: number }) => (
    <button onClick={() => setActiveTab(tab as typeof activeTab)} className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors relative ${activeTab === tab ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground'}`}>
      <Icon className="w-4 h-4" />{label}
      {badgeCount !== undefined && badgeCount > 0 && (
        <span className="absolute -top-1 -right-1 bg-red-600 text-white text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center">
          {badgeCount > 99 ? '99+' : badgeCount}
        </span>
      )}
    </button>
  );
  const isUnavailable = ibStatus != null && !ibStatus.success;
  const ibStatusVariantClass = ibStatus == null
    ? 'bg-muted/40 border-border'
    : ibStatus.success
      ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-900'
      : 'bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-900 border-2';

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="bg-card text-card-foreground rounded-lg shadow-sm border p-6">
        <div className={`mb-4 border rounded flex items-center justify-between gap-3 ${ibStatusVariantClass} ${isUnavailable ? 'px-4 py-3 shadow-md' : 'px-3 py-2'}`}>
          <div className="flex items-center gap-2">
            {ibStatus?.success ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
            ) : (
              <AlertTriangle className={`text-rose-600 dark:text-rose-400 flex-shrink-0 ${isUnavailable ? 'w-5 h-5' : 'w-4 h-4'}`} />
            )}
            <span className={isUnavailable ? 'text-base font-semibold' : 'text-sm font-medium'}>
              {ibStatus == null ? 'Checking IBKR...' : ibStatus.success ? 'IBKR API Ready' : 'IBKR API Unavailable'}
            </span>
            {ibStatus?.success && (
              <span className="text-xs text-emerald-700 dark:text-emerald-300">• Application can submit orders</span>
            )}
            {!ibStatus?.success && (
              <span className={isUnavailable ? 'text-sm text-rose-700 dark:text-rose-300 font-medium' : 'text-xs text-muted-foreground'}>
                {isUnavailable ? "Can't place orders • Start Docker Desktop and ibeam container" : '• Start Docker ibeam container'}
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => fetchIbStatus()}
            disabled={ibStatusLoading}
            className={`inline-flex items-center gap-1 rounded border border-input hover:bg-muted/60 disabled:opacity-60 ${isUnavailable ? 'px-3 py-1.5 text-sm' : 'px-2 py-1 text-xs'}`}
          >
            <RefreshCw className={`${isUnavailable ? 'w-4 h-4' : 'w-3 h-3'} ${ibStatusLoading ? 'animate-spin' : ''}`} />
            {ibStatusLoading ? 'Checking' : 'Refresh'}
          </button>
        </div>
        <div className="flex gap-2 mb-6">
          <TabButton tab="status" label="Trades" icon={Eye} />
          <TabButton tab="trades" label="New Trade" icon={Plus} />
          <TabButton tab="order" label="Start Order" icon={Play} />
          <TabButton tab="errors" label="Errors" icon={AlertCircle} badgeCount={unseenErrorCount} />
        </div>
        {activeTab === 'order' && (
          <OrderConfigTab
            orderConfig={orderConfig}
            setOrderConfig={setOrderConfig}
            pivotPositions={pivotPositions}
            updatePivotPositions={updatePivotPositions}
            clearSavedOrderConfig={clearSavedOrderConfig}
            showAdvanced={showAdvanced}
            setShowAdvanced={setShowAdvanced}
            startOrder={startOrderHandler}
            loading={loading}
            flashAdvanced={flash.advanced}
            flashOrder={flash.order}
            triggerFlash={(k: string) => triggerFlash(k as keyof typeof flash)}
            subtleFlashClass={subtleFlashClass}
          />
        )}
        {activeTab === 'trades' && (
          <TradesTab
            newTrade={newTrade}
            setNewTrade={setNewTrade}
            clearSavedTrade={clearSavedTrade}
            addTrade={addTrade}
            loading={loading}
            flashTrade={flash.trade}
            triggerFlash={(k: string) => triggerFlash(k as keyof typeof flash)}
            subtleFlashClass={subtleFlashClass}
            computeMidPrice={mid}
            addSellStop={addSellStop}
            removeSellStop={removeSellStop}
            updateSellStop={updateSellStop}
            autoCalcEnabled={autoCalcEnabled}
            setAutoCalcEnabled={setAutoCalcEnabled}
            autoCalcReady={autoCalcReady}
            currentEquity={currentEquity}
            tickerInputRef={tickerInputRef}
          />
        )}
        {activeTab === 'status' && (
          <StatusTab
            serverStatus={serverStatus}
            executeTradeNow={executeTradeNow}
            deleteTrade={deleteTrade}
            deleteAllTrades={deleteAllTrades}
            loading={loading}
            riskAmount={riskAmount}
            setRiskAmount={setRiskAmount}
            updateRisk={updateRisk}
            updateRiskToAmount={updateRiskToAmount}
            updateTradeRisk={updateTradeRisk}
          />
        )}
        {activeTab === 'errors' && (
          <ErrorsTab
            errors={errors}
            fetchErrors={fetchErrors}
            triggerFlash={(k: string) => triggerFlash(k as keyof typeof flash)}
            flashRefresh={flash.refresh}
            subtleFlashClass={subtleFlashClass}
          />
        )}
      </div>
      {showVolumeWarningModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border rounded-lg shadow-lg max-w-md w-full mx-4 p-6">
            <div className="flex items-center gap-3 mb-4">
              <AlertCircle className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              <h3 className="text-lg font-semibold text-foreground">No Volume Requirements Set</h3>
            </div>
            <p className="text-muted-foreground mb-6">You haven't set any volume requirements. Continue anyway?</p>
            <div className="flex gap-3 justify-end">
              <button onClick={() => setShowVolumeWarningModal(false)} className="px-4 py-2 text-muted-foreground border border-input rounded-lg hover:bg-muted/50">Cancel</button>
              <button onClick={performStartOrder} className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 dark:bg-red-700 dark:hover:bg-red-800">Start Anyway</button>
            </div>
          </div>
        </div>
      )}

      {pendingRiskFixTrade && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border rounded-lg shadow-lg max-w-md w-full mx-4 p-6">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              <h3 className="text-lg font-semibold text-foreground">Increase Current Risk?</h3>
            </div>
            <p className="text-muted-foreground mb-4">
              This trade requires <span className="font-medium">${pendingRiskFixTrade.requiredRisk.toFixed(2)}</span> risk,
              but available risk is <span className="font-medium">${pendingRiskFixTrade.availableRisk.toFixed(2)}</span>.
            </p>
            <p className="text-muted-foreground mb-6">
              Set current risk to <span className="font-medium">${pendingRiskFixTrade.requiredRisk.toFixed(2)}</span>
              {pendingRiskFixTrade.ticker ? <> and add <span className="font-medium">{pendingRiskFixTrade.ticker}</span>?</> : ' and add this trade?'}
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setPendingRiskFixTrade(null)}
                disabled={loading}
                className="px-4 py-2 text-muted-foreground border border-input rounded-lg hover:bg-muted/50 disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  const pending = pendingRiskFixTrade;
                  if (!pending) return;

                  setLoading(true);
                  try {
                    await updateRiskToAmount(pending.requiredRisk);
                    const addResp = await fetch('http://localhost:5002/add_trade', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(pending.payload)
                    });
                    const addResult = await addResp.json();
                    if (addResult.success) {
                      try {
                        await fetch('http://localhost:8000/ticker_data/api/ticker_data/tickers', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ symbol: pending.ticker })
                        });
                      } catch (e) {
                        console.error('ticker add', e);
                      }
                      setPendingRiskFixTrade(null);
                      setNewTrade(defaultNewTrade);
                      fetchStatus();
                      setTimeout(() => tickerInputRef.current?.focus(), 0);
                    } else {
                      alert(`Error: ${addResult.error}`);
                    }
                  } catch (e) {
                    alert(`Error: ${e}`);
                  } finally {
                    setLoading(false);
                  }
                }}
                disabled={loading}
                ref={riskFixConfirmButtonRef}
                autoFocus
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-60"
              >
                Set Risk & Add Trade
              </button>
            </div>
          </div>
        </div>
      )}

      {pendingRiskFixExecute && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border rounded-lg shadow-lg max-w-md w-full mx-4 p-6">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              <h3 className="text-lg font-semibold text-foreground">Increase Current Risk?</h3>
            </div>
            <p className="text-muted-foreground mb-4">
              This action needs <span className="font-medium">${pendingRiskFixExecute.requiredRisk.toFixed(2)}</span> risk,
              but available risk is <span className="font-medium">${pendingRiskFixExecute.availableRisk.toFixed(2)}</span>.
            </p>
            <p className="text-muted-foreground mb-6">
              Set current risk to <span className="font-medium">${pendingRiskFixExecute.requiredRisk.toFixed(2)}</span>
              {' '}and execute <span className="font-medium">{pendingRiskFixExecute.ticker}</span>?
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setPendingRiskFixExecute(null)}
                disabled={loading}
                className="px-4 py-2 text-muted-foreground border border-input rounded-lg hover:bg-muted/50 disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  const pending = pendingRiskFixExecute;
                  if (!pending) return;

                  setLoading(true);
                  try {
                    await updateRiskToAmount(pending.requiredRisk);
                    const execResp = await fetch('http://localhost:5002/execute_trade', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        ticker: pending.ticker,
                        lower_price: pending.lower_price_range,
                        higher_price: pending.higher_price_range
                      })
                    });
                    const execResult = await execResp.json();
                    setPendingRiskFixExecute(null);
                    if (execResult.success) {
                      alert('Trade executed successfully.');
                      fetchStatus();
                    } else {
                      alert(`Execution error: ${execResult.error}`);
                    }
                  } catch (e) {
                    setPendingRiskFixExecute(null);
                    alert(`Error: ${e}`);
                  } finally {
                    setLoading(false);
                  }
                }}
                disabled={loading}
                ref={riskFixExecuteConfirmButtonRef}
                autoFocus
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-60"
              >
                Set Risk & Execute
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CustomOrdersPage;