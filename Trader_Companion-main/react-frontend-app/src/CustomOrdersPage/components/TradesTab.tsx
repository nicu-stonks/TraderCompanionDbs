import React, { useState, useEffect, useMemo } from 'react';
import { Plus, RefreshCw, Trash2, AlertTriangle } from 'lucide-react';
import { NewTrade, NewTradeStop } from '../types';

interface Props {
  newTrade: NewTrade;
  setNewTrade: React.Dispatch<React.SetStateAction<NewTrade>>;
  clearSavedTrade: () => void;
  addTrade: () => Promise<void> | void;
  loading: boolean;
  flashTrade: boolean;
  triggerFlash: (key: string) => void;
  subtleFlashClass: string;
  computeMidPrice: (l: number, u: number) => number | null;
  addSellStop: () => void;
  removeSellStop: (i: number) => void;
  updateSellStop: (index: number, field: 'price' | 'position_pct' | 'percent_below_fill' | '__ui_mode', value: number | string) => void;
  autoCalcEnabled: boolean;
  setAutoCalcEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  autoCalcReady: boolean;
  currentEquity: number | null;
  tickerInputRef?: React.RefObject<HTMLInputElement>;
}

export const TradesTab: React.FC<Props> = ({
  newTrade,
  setNewTrade,
  clearSavedTrade,
  addTrade,
  loading,
  flashTrade,
  triggerFlash,
  subtleFlashClass,
  computeMidPrice,
  addSellStop,
  removeSellStop,
  updateSellStop,
  autoCalcEnabled,
  setAutoCalcEnabled,
  autoCalcReady,
  currentEquity,
  tickerInputRef
}) => {
  const [localShares, setLocalShares] = useState<string>('');
  const [localDollars, setLocalDollars] = useState<string>('');
  const [activeInput, setActiveInput] = useState<'shares' | 'dollars' | null>(null);
  const isIbalgo = (newTrade.order_type ?? 'MKT') === 'IBALGO';

  const entry = computeMidPrice(newTrade.lower_price_range, newTrade.higher_price_range);
  const dollarAmount = entry ? Math.round((newTrade.shares * entry) * 10000) / 10000 : 0;

  // Sync local states when global state changes from other sources (e.g. autoCalc or initial load)
  useEffect(() => {
    if (activeInput !== 'shares') {
      const sharesValue = isIbalgo ? Math.trunc(newTrade.shares || 0) : newTrade.shares;
      setLocalShares(sharesValue === 0 ? '' : sharesValue.toString());
    }
  }, [newTrade.shares, activeInput, isIbalgo]);

  useEffect(() => {
    if (activeInput !== 'dollars') {
      setLocalDollars(dollarAmount === 0 ? '' : dollarAmount.toString());
    }
  }, [dollarAmount, activeInput]);

  // Calculate risk from shares when in manual mode
  const calculatedRisk = useMemo(() => {
    if (autoCalcEnabled) return null;

    const entry = computeMidPrice(newTrade.lower_price_range, newTrade.higher_price_range);
    if (entry == null || !isFinite(newTrade.shares) || newTrade.shares <= 0 || newTrade.sell_stops.length === 0) {
      return null;
    }

    let weightedDrop = 0;
    for (const stop of newTrade.sell_stops) {
      const pct = Number(stop.position_pct) || 0;
      if (pct <= 0) continue;

      let stopPrice: number | null = null;
      const mode = (stop.__ui_mode ?? (stop.percent_below_fill != null ? 'percent' : 'price')) as 'price' | 'percent';
      if (mode === 'percent') {
        stopPrice = entry * (1 - (Number(stop.percent_below_fill) || 0) / 100);
      } else {
        stopPrice = Number(stop.price);
      }

      if (!stopPrice || !isFinite(stopPrice)) continue;
      const drop = entry - stopPrice;
      if (drop <= 0) continue;
      weightedDrop += pct * drop;
    }

    if (weightedDrop <= 0) {
      return null;
    }

    const riskAmount = newTrade.shares * weightedDrop;
    if (!isFinite(riskAmount) || riskAmount <= 0) {
      return null;
    }

    const roundedRiskAmount = Math.round(riskAmount * 100) / 100;
    let riskPercent: number | null = null;
    if (currentEquity != null && currentEquity > 0) {
      riskPercent = Math.round(((roundedRiskAmount / currentEquity) * 100) * 100) / 100;
    }

    return { riskAmount: roundedRiskAmount, riskPercent };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoCalcEnabled, newTrade.shares, newTrade.lower_price_range, newTrade.higher_price_range, JSON.stringify(newTrade.sell_stops), currentEquity, computeMidPrice]);

  // Auto-update risk when in manual mode and calculation is ready
  useEffect(() => {
    if (!autoCalcEnabled && calculatedRisk?.riskAmount != null) {
      setNewTrade(prev => ({
        ...prev,
        risk_amount: calculatedRisk.riskAmount!,
        risk_percent_of_equity: calculatedRisk.riskPercent ?? prev.risk_percent_of_equity
      }));
    }
  }, [autoCalcEnabled, calculatedRisk?.riskAmount, calculatedRisk?.riskPercent]); // eslint-disable-line react-hooks/exhaustive-deps

  // Check if position size exceeds 25% of equity (high risk warning)
  const isHighRisk = useMemo(() => {
    if (currentEquity == null || currentEquity <= 0) return false;
    const midPrice = computeMidPrice(newTrade.lower_price_range, newTrade.higher_price_range);
    if (midPrice == null || !isFinite(newTrade.shares) || newTrade.shares <= 0) return false;
    const positionSize = midPrice * newTrade.shares;
    const threshold = currentEquity * 0.25;
    return positionSize > threshold;
  }, [currentEquity, newTrade.lower_price_range, newTrade.higher_price_range, newTrade.shares, computeMidPrice]);

  return (
    <div className="space-y-4">
      <div className="bg-muted/50 p-3 rounded-lg">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h3 className="text-base font-semibold">Add New Trade</h3>
          <button
            type="button"
            onClick={() => { clearSavedTrade(); triggerFlash('trade'); }}
            className={`text-xs px-2 py-1 rounded-md border border-input hover:bg-muted/60 transition-shadow relative duration-200 ${flashTrade ? subtleFlashClass : ''}`}
          >
            Reset
          </button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-7 gap-2">
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">Ticker</label>
            <input
              ref={tickerInputRef}
              type="text"
              value={newTrade.ticker}
              onChange={(e) => setNewTrade(prev => ({ ...prev, ticker: e.target.value }))}
              className="w-full p-2 border border-input bg-background text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">Type</label>
            <select
              value={newTrade.order_type ?? 'MKT'}
              onChange={(e) => {
                const nextType = e.target.value as 'MKT' | 'IBALGO';
                setNewTrade(prev => {
                  const nextShares = nextType === 'IBALGO' ? Math.max(0, Math.round(prev.shares || 0)) : prev.shares;
                  const nextTimeout = prev.timeout_seconds && prev.timeout_seconds > 0
                    ? prev.timeout_seconds
                    : (nextType === 'IBALGO' ? 30 : 5);
                  return {
                    ...prev,
                    order_type: nextType,
                    adaptive_priority: nextType === 'IBALGO' ? (prev.adaptive_priority ?? 'Urgent') : undefined,
                    timeout_seconds: nextTimeout,
                    shares: nextShares,
                  };
                });
              }}
              className="w-full p-2 border border-input bg-background text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm"
            >
              <option value="MKT">MKT</option>
              <option value="IBALGO">IBALGO MKT</option>
            </select>
          </div>
          {isIbalgo && (
            <div>
              <label className="block text-xs font-medium text-foreground mb-1">Adaptive</label>
              <select
                value={newTrade.adaptive_priority ?? 'Urgent'}
                onChange={(e) => setNewTrade(prev => ({ ...prev, adaptive_priority: e.target.value as 'Patient' | 'Normal' | 'Urgent' }))}
                className="w-full p-2 border border-input bg-background text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm"
              >
                <option value="Patient">Patient</option>
                <option value="Normal">Normal</option>
                <option value="Urgent">Urgent</option>
              </select>
            </div>
          )}
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">Timeout (s)</label>
            <input
              type="number"
              min={1}
              step={1}
              value={newTrade.timeout_seconds ?? ((newTrade.order_type ?? 'MKT') === 'IBALGO' ? 30 : 5)}
              onChange={(e) => setNewTrade(prev => ({ ...prev, timeout_seconds: Math.max(1, Math.round(parseFloat(e.target.value) || 0)) }))}
              className="w-full p-2 border border-input bg-background text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">Lower Price</label>
            <input type="number" value={newTrade.lower_price_range || ''} onChange={(e) => setNewTrade(prev => ({ ...prev, lower_price_range: e.target.value === '' ? 0 : parseFloat(e.target.value) }))} className="w-full p-2 border border-input bg-background text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm" step="0.01" />
          </div>
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">Higher Price</label>
            <input type="number" value={newTrade.higher_price_range || ''} onChange={(e) => setNewTrade(prev => ({ ...prev, higher_price_range: e.target.value === '' ? 0 : parseFloat(e.target.value) }))} className="w-full p-2 border border-input bg-background text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm" step="0.01" />
          </div>
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">
              Risk ($)
              {!autoCalcEnabled && calculatedRisk?.riskAmount != null && (
                <span className="ml-1 text-xs text-muted-foreground font-normal">(calc)</span>
              )}
            </label>
            <input
              type="number"
              value={newTrade.risk_amount || ''}
              onChange={(e) => {
                if (autoCalcEnabled) {
                  const dollars = e.target.value === '' ? 0 : parseFloat(e.target.value);
                  let pct = newTrade.risk_percent_of_equity ?? 0;
                  if (currentEquity != null && currentEquity > 0) {
                    pct = Math.round(((dollars / currentEquity) * 100) * 100) / 100;
                  }
                  setNewTrade(prev => ({ ...prev, risk_amount: dollars, risk_percent_of_equity: pct }));
                }
              }}
              readOnly={!autoCalcEnabled}
              className={`w-full p-2 border border-input ${!autoCalcEnabled ? 'bg-muted' : 'bg-background'} text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm`}
              step="0.01"
              min={0}
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-foreground mb-1">
              Equity Risk % (P/L added)
              {!autoCalcEnabled && calculatedRisk?.riskPercent != null && (
                <span className="ml-1 text-xs text-muted-foreground font-normal">(calc)</span>
              )}
            </label>
            <input
              type="number"
              value={newTrade.risk_percent_of_equity || ''}
              onChange={(e) => {
                if (autoCalcEnabled) {
                  const pct = e.target.value === '' ? 0 : parseFloat(e.target.value);
                  let dollars = newTrade.risk_amount;
                  if (currentEquity != null && currentEquity > 0 && pct > 0) {
                    dollars = Math.round((currentEquity * pct / 100) * 100) / 100;
                  }
                  setNewTrade(prev => ({ ...prev, risk_percent_of_equity: pct, risk_amount: dollars }));
                }
              }}
              readOnly={!autoCalcEnabled}
              className={`w-full p-2 border border-input ${!autoCalcEnabled ? 'bg-muted' : 'bg-background'} text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm`}
              step="0.01"
              min={0}
              max={100}
            />
          </div>
        </div>

        {/* High Risk Warning */}
        {isHighRisk && (
          <div className="mt-2 flex items-start gap-2 p-2 bg-amber-500/10 border border-amber-500/30 rounded-lg">
            <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-amber-500">
              <span className="font-semibold">High Risk Warning:</span> You are risking more than 25% of your account on a single position.
              This significantly increases your risk of ruin. Are you sure this is what you want?
            </div>
          </div>
        )}



        <div className="mt-3">
          <label className="block text-xs font-medium text-foreground mb-1">Sell Stops</label>
          <div className="space-y-1">
            <button onClick={addSellStop} className="w-full p-1.5 border-2 border-dashed border-border rounded-lg hover:border-border/80 text-muted-foreground hover:text-foreground text-sm">
              + Add Sell Stop
            </button>
            {newTrade.sell_stops.map((stop: NewTradeStop, index: number) => {
              const mode = (stop.__ui_mode || (stop.percent_below_fill != null ? 'percent' : 'price')) as 'price' | 'percent';
              const entry = computeMidPrice(newTrade.lower_price_range, newTrade.higher_price_range);
              let effectiveStop: number | null = null;
              if (entry != null) {
                effectiveStop = mode === 'percent'
                  ? entry * (1 - (Number(stop.percent_below_fill) || 0) / 100)
                  : Number(stop.price) || null;
              }
              return (
                <div key={index} className="flex items-center gap-2 flex-wrap md:flex-nowrap">
                  <div className="w-24">
                    <select value={mode} onChange={(e) => updateSellStop(index, '__ui_mode', e.target.value)} className="w-full p-1.5 border border-input bg-background text-foreground rounded text-xs">
                      <option value="price">Price</option>
                      <option value="percent">% Below</option>
                    </select>
                  </div>
                  {mode === 'price' ? (
                    <div className="flex-1 min-w-[100px]">
                      <input type="number" value={stop.price ?? 0} onChange={(e) => updateSellStop(index, 'price', parseFloat(e.target.value))} className="w-full p-1.5 border border-input bg-background text-foreground rounded focus:ring-2 focus:ring-ring focus:border-ring text-sm" placeholder="Price" step="0.01" />
                    </div>
                  ) : (
                    <div className="flex-1 min-w-[100px]">
                      <input type="number" value={stop.percent_below_fill ?? 1} onChange={(e) => updateSellStop(index, 'percent_below_fill', parseFloat(e.target.value))} className="w-full p-1.5 border border-input bg-background text-foreground rounded focus:ring-2 focus:ring-ring focus:border-ring text-sm" placeholder="%" step="0.1" min="0.1" />
                    </div>
                  )}
                  <div className="flex-1 min-w-[100px]">
                    <input type="number" value={stop.position_pct ?? 0} onChange={(e) => updateSellStop(index, 'position_pct', parseFloat(e.target.value) || 0)} className="w-full p-1.5 border border-input bg-background text-foreground rounded focus:ring-2 focus:ring-ring focus:border-ring text-sm" placeholder="Sell %" step="0.01" min="0" max="1" />
                  </div>
                  {entry != null && effectiveStop != null && (
                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                      -${(entry - effectiveStop).toFixed(2)}
                    </span>
                  )}
                  <button onClick={() => removeSellStop(index)} className="p-1 text-destructive hover:bg-destructive/10 rounded" disabled={newTrade.sell_stops.length === 1}>
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              );
            })}
            <div className="flex flex-col gap-1 mt-2">
              <label className="flex items-center gap-2 text-xs select-none">
                <input type="checkbox" className="h-3.5 w-3.5" checked={autoCalcEnabled} onChange={(e) => setAutoCalcEnabled(e.target.checked)} />
                <span>Auto-calculate shares {autoCalcReady ? '' : '(incomplete)'}</span>
              </label>
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div>
                  <label className="block text-xs font-medium text-foreground mb-1">Shares {autoCalcEnabled ? '(auto)' : ''}</label>
                  <input
                    type="number"
                    value={activeInput === 'shares' ? localShares : (newTrade.shares || '')}
                    onFocus={() => setActiveInput('shares')}
                    onBlur={() => setActiveInput(null)}
                    onChange={(e) => {
                      const val = e.target.value;
                      setLocalShares(val);
                      if (!autoCalcEnabled) {
                        const parsed = parseFloat(val);
                        const sharesValue = val === '' ? 0 : (isIbalgo ? Math.max(0, Math.round(parsed || 0)) : Math.round((parsed || 0) * 10000) / 10000);
                        setNewTrade(prev => ({
                          ...prev,
                          shares: sharesValue
                        }));
                      }
                    }}
                    readOnly={autoCalcEnabled}
                    className={`w-full p-2 border border-input ${autoCalcEnabled ? 'bg-muted' : 'bg-background'} text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm`}
                    step={isIbalgo ? '1' : '0.01'}
                    min="0"
                    placeholder={autoCalcEnabled ? 'Auto' : 'Manual'}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-foreground mb-1">Dollar Amount</label>
                  <input
                    type="number"
                    value={activeInput === 'dollars' ? localDollars : (dollarAmount || '')}
                    onFocus={() => setActiveInput('dollars')}
                    onBlur={() => setActiveInput(null)}
                    onChange={(e) => {
                      const val = e.target.value;
                      setLocalDollars(val);
                      if (!autoCalcEnabled && entry) {
                        const dollars = val === '' ? 0 : parseFloat(val);
                        const shRaw = dollars / entry;
                        const sh = isIbalgo ? Math.max(0, Math.round(shRaw)) : Math.round(shRaw * 10000) / 10000;
                        setNewTrade(prev => ({ ...prev, shares: sh }));
                      }
                    }}
                    readOnly={autoCalcEnabled}
                    className={`w-full p-2 border border-input ${autoCalcEnabled ? 'bg-muted' : 'bg-background'} text-foreground rounded-lg focus:ring-2 focus:ring-ring focus:border-ring text-sm`}
                    step="0.01"
                    min="0"
                    placeholder={autoCalcEnabled ? 'Auto' : 'Manual'}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* <div className="mt-3 flex items-center gap-4 flex-wrap">
          <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={newTrade.consider_zero_risk ?? false}
              onChange={(e) => {
                const checked = e.target.checked;
                setNewTrade(prev => ({
                  ...prev,
                  consider_zero_risk: checked,
                  risk_amount: checked ? 0 : prev.risk_amount,
                  risk_percent_of_equity: checked ? 0 : prev.risk_percent_of_equity
                }));
              }}
            />
            <span>Consider 0 Risk For This Trade</span>
          </label>
        </div> */}

        <button onClick={addTrade} disabled={loading} className="mt-3 bg-green-600 text-white py-1.5 px-3 rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center gap-1.5 text-sm dark:bg-green-700 dark:hover:bg-green-800">
          {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
          {loading ? 'Adding...' : 'Add Trade'}
        </button>
      </div>
    </div>
  );
};
