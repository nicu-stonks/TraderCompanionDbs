import React, { useEffect, useRef, useState } from 'react';
import { Eye, DollarSign, RefreshCw, Play, Trash2, Check } from 'lucide-react';
import { ServerStatus } from '../types';

interface Props {
  serverStatus: ServerStatus | null;
  executeTradeNow: (p: { ticker: string; lower_price_range: number; higher_price_range: number }) => void;
  deleteTrade: (id: string) => void;
  deleteAllTrades: () => void;
  loading: boolean;
  riskAmount: number;
  setRiskAmount: React.Dispatch<React.SetStateAction<number>>;
  updateRisk: () => Promise<void> | void;
  updateRiskToAmount?: (amount: number) => Promise<void>;
  updateTradeRisk?: (tradeId: string, newRiskAmount: number, settings?: { order_type: 'MKT' | 'IBALGO'; adaptive_priority?: 'Patient' | 'Normal' | 'Urgent'; timeout_seconds?: number }) => Promise<void>;
}

type TradeSettings = { order_type: 'MKT' | 'IBALGO'; adaptive_priority?: 'Patient' | 'Normal' | 'Urgent'; timeout_seconds?: number };
type TradeDraft = {
  risk_amount: number;
  shares: number;
  order_type: 'MKT' | 'IBALGO';
  adaptive_priority: 'Patient' | 'Normal' | 'Urgent';
  timeout_seconds: number;
};

type PendingRiskModal =
  | { kind: 'trade-risk'; requiredRisk: number; availableRisk: number; tradeId: string; settings: TradeSettings }
  | { kind: 'pool-risk'; requestedRisk: number; availableRisk: number };

export const StatusTab: React.FC<Props> = ({ serverStatus, executeTradeNow, deleteTrade, deleteAllTrades, loading, riskAmount, setRiskAmount, updateRisk, updateRiskToAmount, updateTradeRisk }) => {
  const [tradeDrafts, setTradeDrafts] = useState<Record<string, TradeDraft>>({});
  const [pendingRiskModal, setPendingRiskModal] = useState<PendingRiskModal | null>(null);
  const confirmButtonRef = useRef<HTMLButtonElement>(null);
  const [flashUpdateRisk, setFlashUpdateRisk] = useState(false);
  const flashTimerRef = useRef<number | null>(null);

  useEffect(() => {
    const trades = serverStatus?.trades ?? [];
    setTradeDrafts(prev => {
      const next: Record<string, TradeDraft> = {};
      for (const trade of trades) {
        const existing = prev[trade.trade_id];
        const orderType = (trade.order_type ?? 'MKT') as 'MKT' | 'IBALGO';
        next[trade.trade_id] = existing ?? {
          risk_amount: trade.risk_amount,
          shares: trade.shares,
          order_type: orderType,
          adaptive_priority: (trade.adaptive_priority ?? 'Urgent') as 'Patient' | 'Normal' | 'Urgent',
          timeout_seconds: trade.timeout_seconds ?? (orderType === 'IBALGO' ? 30 : 5)
        };
      }
      return next;
    });
  }, [serverStatus?.trades]);

  useEffect(() => {
    if (!pendingRiskModal) return;
    window.setTimeout(() => confirmButtonRef.current?.focus(), 0);
  }, [pendingRiskModal]);

  useEffect(() => {
    return () => {
      if (flashTimerRef.current != null) {
        window.clearTimeout(flashTimerRef.current);
      }
    };
  }, []);

  const triggerUpdateRiskFlash = () => {
    setFlashUpdateRisk(true);
    if (flashTimerRef.current != null) {
      window.clearTimeout(flashTimerRef.current);
    }
    flashTimerRef.current = window.setTimeout(() => setFlashUpdateRisk(false), 180);
  };

  const parseRiskExceededMessage = (msg: unknown): { requestedRisk: number; availableRisk: number } | null => {
    const raw = typeof msg === 'string' ? msg : '';
    if (!raw) return null;

    const requestedMatch = raw.match(/Requested\s+risk\s*\(\$\s*([0-9]+(?:\.[0-9]+)?)\)/i);
    const availableMatch = raw.match(/available\s+risk\s*\(\$\s*([0-9]+(?:\.[0-9]+)?)\)/i);
    const requestedRisk = requestedMatch ? Number(requestedMatch[1]) : NaN;
    const availableRisk = availableMatch ? Number(availableMatch[1]) : NaN;
    if (!isFinite(requestedRisk) || !isFinite(availableRisk)) return null;
    if (!/exceeds/i.test(raw)) return null;
    return { requestedRisk, availableRisk };
  };

  const updateDraft = (tradeId: string, updater: (current: TradeDraft) => TradeDraft) => {
    setTradeDrafts(prev => {
      const current = prev[tradeId];
      if (!current) return prev;
      return { ...prev, [tradeId]: updater(current) };
    });
  };

  const getDraft = (trade: NonNullable<ServerStatus['trades']>[0]): TradeDraft => {
    const existing = tradeDrafts[trade.trade_id];
    if (existing) return existing;
    const orderType = (trade.order_type ?? 'MKT') as 'MKT' | 'IBALGO';
    return {
      risk_amount: trade.risk_amount,
      shares: trade.shares,
      order_type: orderType,
      adaptive_priority: (trade.adaptive_priority ?? 'Urgent') as 'Patient' | 'Normal' | 'Urgent',
      timeout_seconds: trade.timeout_seconds ?? (orderType === 'IBALGO' ? 30 : 5)
    };
  };

  const calculateWeightedDrop = (trade: NonNullable<ServerStatus['trades']>[0]) => {
    const entry = (trade.lower_price_range + trade.higher_price_range) / 2;
    if (!isFinite(entry) || entry <= 0 || trade.sell_stops.length === 0) return null;

    let weightedDrop = 0;
    for (const stop of trade.sell_stops) {
      // Calculate position_pct from shares ratio
      const totalShares = trade.shares;
      const pct = totalShares > 0 ? stop.shares / totalShares : 0;
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

    return weightedDrop > 0 ? weightedDrop : null;
  };

  const calculateSharesFromRisk = (trade: NonNullable<ServerStatus['trades']>[0], orderType: 'MKT' | 'IBALGO', newRiskAmount: number): number | null => {
    const weightedDrop = calculateWeightedDrop(trade);
    if (!weightedDrop) return null;

    const shares = newRiskAmount / weightedDrop;
    if (!isFinite(shares) || shares <= 0) return null;
    if (orderType === 'IBALGO') {
      return Math.max(1, Math.round(shares));
    }
    return Math.round(shares * 100) / 100;
  };

  const calculateRiskFromShares = (trade: NonNullable<ServerStatus['trades']>[0], shares: number): number | null => {
    const weightedDrop = calculateWeightedDrop(trade);
    if (!weightedDrop || !isFinite(shares) || shares <= 0) return null;
    const risk = shares * weightedDrop;
    return isFinite(risk) && risk > 0 ? Math.round(risk * 100) / 100 : null;
  };

  const formatShares = (orderType: 'MKT' | 'IBALGO', shares: number) => {
    if (orderType === 'IBALGO') {
      return Math.round(shares).toString();
    }
    return shares.toFixed(3);
  };

  const handleApplyTradeUpdate = async (trade: NonNullable<ServerStatus['trades']>[0]) => {
    const draft = getDraft(trade);
    const newShares = calculateSharesFromRisk(trade, draft.order_type, draft.risk_amount);
    if (newShares === null) {
      alert('Unable to calculate new shares. Check stop prices.');
      return;
    }

    // Check if new risk exceeds available risk
    if (serverStatus && draft.risk_amount > serverStatus.available_risk) {
      setPendingRiskModal({
        kind: 'trade-risk',
        requiredRisk: draft.risk_amount,
        availableRisk: serverStatus.available_risk,
        tradeId: trade.trade_id,
        settings: {
          order_type: draft.order_type,
          adaptive_priority: draft.order_type === 'IBALGO' ? draft.adaptive_priority : undefined,
          timeout_seconds: Math.max(1, Math.round(draft.timeout_seconds || 0))
        }
      });
      return;
    }

    if (updateTradeRisk) {
      await updateTradeRisk(trade.trade_id, draft.risk_amount, {
        order_type: draft.order_type,
        adaptive_priority: draft.order_type === 'IBALGO' ? draft.adaptive_priority : undefined,
        timeout_seconds: Math.max(1, Math.round(draft.timeout_seconds || 0))
      });
    }
  };

  // Removed confirmUpdate and cancelUpdate functions

  return (
    <div className="space-y-4">
      {serverStatus && serverStatus.trades && serverStatus.trades.length > 0 && (
        <div className="bg-card border rounded-lg overflow-hidden">
          <div className="bg-muted/50 px-4 py-2 border-b flex items-center justify-between">
            <h3 className="text-base font-semibold text-foreground">Active Trades Details</h3>
            <button
              onClick={deleteAllTrades}
              disabled={loading}
              className="inline-flex items-center gap-1 px-2 py-1 bg-destructive text-destructive-foreground rounded hover:bg-destructive/90 disabled:opacity-50 text-xs"
              title="Delete all trades"
            >
              <Trash2 className="w-3 h-3" />
              Delete All
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-muted/50">
                <tr>
                  <th className="px-3 py-1.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Ticker</th>
                  <th className="px-3 py-1.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Price Range</th>
                  <th className="px-3 py-1.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Risk</th>
                  <th className="px-3 py-1.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Shares</th>
                  <th className="px-3 py-1.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Type</th>
                  <th className="px-3 py-1.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Timeout</th>
                  <th className="px-3 py-1.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Actions</th>
                  <th className="px-3 py-1.5 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Sell Stops</th>
                </tr>
              </thead>
              <tbody className="bg-card divide-y divide-border">
                {serverStatus.trades.map(trade => {
                  const draft = getDraft(trade);
                  const savedOrderType = (trade.order_type ?? 'MKT') as 'MKT' | 'IBALGO';
                  const savedAdaptivePriority = (trade.adaptive_priority ?? 'Urgent') as 'Patient' | 'Normal' | 'Urgent';
                  const savedTimeout = trade.timeout_seconds ?? (savedOrderType === 'IBALGO' ? 30 : 5);
                  const riskChanged = Math.abs(draft.risk_amount - trade.risk_amount) > 0.0001;
                  const sharesChanged = Math.abs(draft.shares - trade.shares) > 0.0001;
                  const typeChanged = draft.order_type !== savedOrderType;
                  const adaptiveChanged = draft.order_type === 'IBALGO' && (typeChanged || draft.adaptive_priority !== savedAdaptivePriority);
                  const timeoutChanged = Math.round(draft.timeout_seconds) !== Math.round(savedTimeout);
                  return (
                    <tr key={trade.trade_id} className="hover:bg-muted/50">
                      <td className="px-3 py-1.5 whitespace-nowrap text-sm font-medium text-foreground">{trade.ticker}</td>
                      <td className="px-3 py-1.5 whitespace-nowrap text-sm text-foreground">{trade.lower_price_range.toFixed(2)} - {trade.higher_price_range.toFixed(2)}</td>
                      <td className="px-3 py-1.5 whitespace-nowrap text-sm text-foreground">
                        <input
                          type="number"
                          value={draft.risk_amount}
                          onChange={(e) => updateDraft(trade.trade_id, cur => {
                            const nextRisk = Math.max(0, parseFloat(e.target.value) || 0);
                            const nextShares = calculateSharesFromRisk(trade, cur.order_type, nextRisk);
                            return {
                              ...cur,
                              risk_amount: nextRisk,
                              shares: nextShares ?? cur.shares,
                            };
                          })}
                          className={`w-20 px-1 py-0.5 border border-input bg-background rounded focus:ring-2 focus:ring-ring focus:border-ring text-xs ${riskChanged ? 'text-amber-600 dark:text-amber-400' : 'text-foreground'}`}
                          step="0.01"
                        />
                      </td>
                      <td className="px-3 py-1.5 whitespace-nowrap text-sm text-foreground">
                        <input
                          type="number"
                          value={draft.shares}
                          onChange={(e) => updateDraft(trade.trade_id, cur => {
                            const rawShares = Math.max(0, parseFloat(e.target.value) || 0);
                            const normalizedShares = cur.order_type === 'IBALGO' ? Math.round(rawShares) : Math.round(rawShares * 100) / 100;
                            const nextRisk = calculateRiskFromShares(trade, normalizedShares);
                            return {
                              ...cur,
                              shares: normalizedShares,
                              risk_amount: nextRisk ?? cur.risk_amount,
                            };
                          })}
                          className={`w-20 px-1 py-0.5 border border-input bg-background rounded focus:ring-2 focus:ring-ring focus:border-ring text-xs ${sharesChanged ? 'text-amber-600 dark:text-amber-400' : 'text-foreground'}`}
                          step={draft.order_type === 'IBALGO' ? '1' : '0.01'}
                          min={0}
                        />
                      </td>
                      <td className="px-3 py-1.5 whitespace-nowrap text-xs text-foreground">
                        <div className="flex items-center gap-1">
                          <select
                            value={draft.order_type}
                            onChange={(e) => {
                              const nextType = e.target.value as 'MKT' | 'IBALGO';
                              updateDraft(trade.trade_id, cur => ({
                                ...cur,
                                order_type: nextType,
                                adaptive_priority: nextType === 'IBALGO' ? (cur.adaptive_priority ?? 'Urgent') : 'Urgent',
                                timeout_seconds: cur.timeout_seconds > 0 ? cur.timeout_seconds : (nextType === 'IBALGO' ? 30 : 5),
                                shares: nextType === 'IBALGO' ? Math.max(0, Math.round(cur.shares)) : cur.shares
                              }));
                            }}
                            className={`px-1 py-0.5 border border-input bg-background rounded text-xs ${typeChanged ? 'text-amber-600 dark:text-amber-400' : 'text-foreground'}`}
                          >
                            <option value="MKT">MKT</option>
                            <option value="IBALGO">IBALGO MKT</option>
                          </select>
                          {draft.order_type === 'IBALGO' && (
                            <select
                              value={draft.adaptive_priority}
                              onChange={(e) => updateDraft(trade.trade_id, cur => ({ ...cur, adaptive_priority: e.target.value as 'Patient' | 'Normal' | 'Urgent' }))}
                              className={`px-1 py-0.5 border border-input bg-background rounded text-xs ${adaptiveChanged ? 'text-amber-600 dark:text-amber-400' : 'text-foreground'}`}
                            >
                              <option value="Patient">Patient</option>
                              <option value="Normal">Normal</option>
                              <option value="Urgent">Urgent</option>
                            </select>
                          )}
                        </div>
                      </td>
                      <td className="px-3 py-1.5 whitespace-nowrap text-sm text-foreground">
                        <input
                          type="number"
                          min={1}
                          step={1}
                          value={draft.timeout_seconds}
                          onChange={(e) => updateDraft(trade.trade_id, cur => ({ ...cur, timeout_seconds: Math.max(1, Math.round(parseFloat(e.target.value) || 0)) }))}
                          className={`w-16 px-1 py-0.5 border border-input bg-background rounded text-xs ${timeoutChanged ? 'text-amber-600 dark:text-amber-400' : 'text-foreground'}`}
                        />
                      </td>
                      <td className="px-3 py-1.5 whitespace-nowrap text-xs text-foreground">
                        <div className="flex items-center gap-1">
                          <button onClick={() => handleApplyTradeUpdate(trade)} disabled={loading || draft.risk_amount <= 0} className="inline-flex items-center gap-0.5 px-2 py-0.5 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 text-xs dark:bg-green-700 dark:hover:bg-green-800" title="Apply current row values">
                            <Check className="w-3 h-3" />
                            Update
                          </button>
                          <button onClick={() => executeTradeNow({ ticker: trade.ticker, lower_price_range: trade.lower_price_range, higher_price_range: trade.higher_price_range })} disabled={loading} className="inline-flex items-center gap-0.5 px-2 py-0.5 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-xs dark:bg-blue-700 dark:hover:bg-blue-800" title="Execute this trade immediately">
                            <Play className="w-3 h-3" />
                            Execute Now
                          </button>
                          <button onClick={() => deleteTrade(trade.trade_id)} disabled={loading} className="inline-flex items-center gap-0.5 px-2 py-0.5 bg-destructive text-destructive-foreground rounded hover:bg-destructive/90 disabled:opacity-50 text-xs">
                            <Trash2 className="w-3 h-3" />
                            Delete
                          </button>
                        </div>
                      </td>
                      <td className="px-3 py-1.5 text-xs text-foreground">
                        <div className="flex flex-wrap gap-1">
                          {trade.sell_stops.map((stop, i) => {
                            const isPercent = stop.percent_below_fill !== undefined && stop.percent_below_fill !== null;
                            return (
                              <span key={i} className="text-xs bg-muted px-1.5 py-0.5 rounded whitespace-nowrap">
                                {isPercent ? (
                                  <>{stop.percent_below_fill}% ({formatShares(draft.order_type, stop.shares)} sh)</>
                                ) : (
                                  <>{(stop.price ?? 0).toFixed(2)} ({formatShares(draft.order_type, stop.shares)} sh)</>
                                )}
                              </span>
                            );
                          })}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        {serverStatus && (
          <>
            <div className="flex-1 min-w-0 bg-blue-50 dark:bg-blue-950/50 px-3 py-2 rounded-lg border border-blue-200 dark:border-blue-800 flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Eye className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium text-blue-800 dark:text-blue-200">Active Trades</span>
              </div>
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400">{serverStatus.active_trades}</span>
            </div>
            <div className="flex-1 min-w-0 bg-green-50 dark:bg-green-950/50 px-3 py-2 rounded-lg border border-green-200 dark:border-green-800 flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <DollarSign className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-green-800 dark:text-green-200">Available Risk</span>
              </div>
              <span className="text-lg font-bold text-green-600 dark:text-green-400">${serverStatus.available_risk}</span>
            </div>
            <div className="flex-1 min-w-0 bg-purple-50 dark:bg-purple-950/50 px-3 py-2 rounded-lg border border-purple-200 dark:border-purple-800 flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <RefreshCw className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <span className="text-sm font-medium text-purple-800 dark:text-purple-200">Uptime</span>
              </div>
              <span className="text-sm font-bold text-purple-600 dark:text-purple-400">{serverStatus.server_uptime}</span>
            </div>
          </>
        )}
      </div>

      <div className="bg-muted/50 px-3 py-2 rounded-lg">
        <div className="flex gap-3 items-center">
          <span className="text-sm font-medium text-foreground whitespace-nowrap">Update Risk</span>
          <input type="number" value={riskAmount} onChange={(e) => setRiskAmount(parseFloat(e.target.value))} className="flex-1 max-w-[120px] p-1.5 border border-input bg-background text-foreground rounded focus:ring-2 focus:ring-ring focus:border-ring text-sm" step="0.01" />
          <button
            onClick={async () => {
              triggerUpdateRiskFlash();
              if (!updateRiskToAmount) {
                await updateRisk();
                return;
              }
              try {
                await updateRiskToAmount(riskAmount);
              } catch (e) {
                const parsed = parseRiskExceededMessage(String(e));
                if (parsed) {
                  setPendingRiskModal({ kind: 'pool-risk', requestedRisk: parsed.requestedRisk, availableRisk: parsed.availableRisk });
                  return;
                }
                alert(`Error: ${e}`);
              }
            }}
            disabled={loading}
            className={`bg-primary text-primary-foreground py-1.5 px-3 rounded hover:bg-primary/90 disabled:opacity-50 flex items-center gap-1.5 text-sm transition-shadow ${flashUpdateRisk ? 'ring-2 ring-ring animate-pulse' : ''}`}
          >
            {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <DollarSign className="w-3.5 h-3.5" />}
            Update
          </button>
        </div>
      </div>

      {pendingRiskModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border rounded-lg shadow-lg max-w-md w-full mx-4 p-6">
            <div className="flex items-center gap-3 mb-4">
              <DollarSign className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              <h3 className="text-lg font-semibold text-foreground">Update Risk?</h3>
            </div>

            {pendingRiskModal.kind === 'trade-risk' ? (
              <>
                <p className="text-muted-foreground mb-4">
                  This action needs <span className="font-medium">${pendingRiskModal.requiredRisk.toFixed(2)}</span> risk,
                  but available risk is <span className="font-medium">${pendingRiskModal.availableRisk.toFixed(2)}</span>.
                </p>
                <p className="text-muted-foreground mb-6">
                  Set current risk to <span className="font-medium">${pendingRiskModal.requiredRisk.toFixed(2)}</span> and update the trade?
                </p>
              </>
            ) : (
              <>
                <p className="text-muted-foreground mb-4">
                  Requested risk is <span className="font-medium">${pendingRiskModal.requestedRisk.toFixed(2)}</span>,
                  but available risk is <span className="font-medium">${pendingRiskModal.availableRisk.toFixed(2)}</span>.
                </p>
                <p className="text-muted-foreground mb-6">
                  Set risk to <span className="font-medium">${pendingRiskModal.availableRisk.toFixed(2)}</span> instead?
                </p>
              </>
            )}

            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setPendingRiskModal(null)}
                disabled={loading}
                className="px-4 py-2 text-muted-foreground border border-input rounded-lg hover:bg-muted/50 disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                ref={confirmButtonRef}
                autoFocus
                onClick={async () => {
                  if (!updateRiskToAmount) {
                    alert('Risk update function unavailable');
                    return;
                  }

                  const modal = pendingRiskModal;
                  if (!modal) return;

                  try {
                    if (modal.kind === 'trade-risk') {
                      await updateRiskToAmount(modal.requiredRisk);
                      if (updateTradeRisk) {
                        await updateTradeRisk(modal.tradeId, modal.requiredRisk, {
                          order_type: modal.settings.order_type,
                          adaptive_priority: modal.settings.order_type === 'IBALGO' ? modal.settings.adaptive_priority : undefined,
                          timeout_seconds: Math.max(1, Math.round(modal.settings.timeout_seconds ?? 0))
                        });
                      }
                    } else {
                      setRiskAmount(modal.availableRisk);
                      await updateRiskToAmount(modal.availableRisk);
                    }
                    setPendingRiskModal(null);
                  } catch (e) {
                    alert(`Error: ${e}`);
                  }
                }}
                disabled={loading}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-60"
              >
                Set Risk & Continue
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
