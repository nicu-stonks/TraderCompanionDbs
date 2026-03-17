import React, { useState } from 'react';
import { HelpCircle, Info, Play, RefreshCw, Trash2 } from 'lucide-react';
import { OrderConfig } from '../types';
import { FetchErrorBanner } from '@/components/FetchErrorBanner';
import { TickerManager } from '@/components/TickerManager';
import { DataSourceSelector } from '@/components/DataSourceSelector';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

// Helper component for field labels with info tooltips
const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <div className="flex items-center gap-1 mb-1">
    <label className="block text-xs font-medium">{label}</label>
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="text-muted-foreground hover:text-foreground transition-colors">
          <HelpCircle className="w-3.5 h-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs text-xs">
        <p>{tooltip}</p>
      </TooltipContent>
    </Tooltip>
  </div>
);

interface PivotPositions { [key: string]: boolean; }

interface Props {
  orderConfig: OrderConfig;
  setOrderConfig: React.Dispatch<React.SetStateAction<OrderConfig>>;
  pivotPositions: PivotPositions;
  updatePivotPositions: (position: string, checked: boolean) => void;
  clearSavedOrderConfig: () => void;
  showAdvanced: boolean;
  setShowAdvanced: React.Dispatch<React.SetStateAction<boolean>>;
  startOrder: () => Promise<void> | void;
  loading: boolean;
  flashAdvanced: boolean;
  flashOrder: boolean;
  triggerFlash: (key: string) => void;
  subtleFlashClass: string;
}

/**
 * @deprecated This component is deprecated and should no longer be used.
 * It is kept for reference only and will be removed in a future version.
 */
export const OrderConfigTab: React.FC<Props> = ({
  orderConfig,
  setOrderConfig,
  pivotPositions,
  updatePivotPositions,
  clearSavedOrderConfig,
  showAdvanced,
  setShowAdvanced,
  startOrder,
  loading,
  flashAdvanced,
  flashOrder,
  triggerFlash,
  subtleFlashClass
}) => {
  const [newVolumeReq, setNewVolumeReq] = useState('');
  const [addFractionalVolumes, setAddFractionalVolumes] = useState(true);
  const [convertDayToHourly, setConvertDayToHourly] = useState(true);

  const addVolumeRequirement = () => {
    if (newVolumeReq.trim()) {
      let baseReq = newVolumeReq.trim();
      const dayMatch = baseReq.match(/^day\s*=\s*([\d,.]+)$/i);
      if (convertDayToHourly && dayMatch) {
        const dayVolume = parseFloat(dayMatch[1].replace(/,/g, ''));
        if (!isNaN(dayVolume) && dayVolume > 0) {
          const hourlyVolume = Math.round(dayVolume / 6.5);
          baseReq = `60=${hourlyVolume}`;
        }
      }

      setOrderConfig(prev => {
        const existing = new Set(prev.volume_requirements.map(v => v.trim()));
        const additions: string[] = [];
        if (!existing.has(baseReq)) { additions.push(baseReq); existing.add(baseReq); }
        const match = baseReq.match(/^(\d+)\s*=\s*(\d+)$/);
        if (addFractionalVolumes && match) {
          const minutes = parseInt(match[1], 10);
          const volume = parseInt(match[2], 10);
          if (minutes > 1) {
            const halfMinutes = Math.round(minutes / 2);
            const quarterMinutes = Math.round(minutes / 4);
            const halfVolume = Math.round(volume / 2);
            const quarterVolume = Math.round(volume / 4);
            const halfReq = `${halfMinutes}=${halfVolume}`;
            const quarterReq = `${quarterMinutes}=${quarterVolume}`;
            if (!existing.has(halfReq)) { additions.push(halfReq); existing.add(halfReq); }
            if (!existing.has(quarterReq)) { additions.push(quarterReq); existing.add(quarterReq); }
          }
        }
        return { ...prev, volume_requirements: [...prev.volume_requirements, ...additions] };
      });
      setNewVolumeReq('');
    }
  };

  const removeVolumeRequirement = (index: number) => {
    setOrderConfig(prev => ({
      ...prev,
      volume_requirements: prev.volume_requirements.filter((_, i) => i !== index)
    }));
  };

  return (
    <div className="relative space-y-5">
      {/* Deprecation overlay – blocks all interaction */}
      <div
        className="absolute inset-0 z-50 flex flex-col items-center justify-start pt-44 gap-3 rounded-lg"
        style={{ background: 'rgba(0,0,0,0.4)' }}
        aria-hidden="false"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="w-10 h-10 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
        </svg>
        <div className="text-center px-6">
          <p className="text-white font-semibold text-base">This feature is deprecated</p>
          <p className="text-white/70 text-sm mt-1">This tab is no longer available because no one uses this shit anymore.</p>
        </div>
      </div>
      {/* Fetch Error Banner */}
      <FetchErrorBanner />

      {/* Data Source Selector */}
      <DataSourceSelector />

      {/* Ticker Manager */}
      <div className="p-3 rounded-lg border bg-muted/30">
        <TickerManager compact title="Monitored Tickers" />
      </div>

      <div className="bg-blue-50 border border-blue-200 text-sm text-blue-900 dark:bg-blue-950 dark:border-blue-800 dark:text-blue-100 rounded-md p-3 flex gap-3 items-start">
        <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
        <div>
          <p>
            To capture full trading-day data for a ticker, add the trade in the <strong>New Trades</strong> tab before the market opens (or add it in Monitored Tickers above).
            You can also add a ticker just before placing a trade&mdash;it will use whatever data is available. Preferably add it at least <strong>1 hour before</strong>, since the default lookback is 1 hour of price action.
          </p>
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 flex-wrap">
        <h2 className="text-lg font-semibold">Basic Breakout Order Settings</h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => { setShowAdvanced(s => !s); triggerFlash('advanced'); }}
            className={`text-xs px-3 py-1.5 rounded-md border border-input hover:bg-muted/60 flex items-center gap-1 relative transition-shadow duration-200 ${flashAdvanced ? subtleFlashClass : ''}`}
          >
            {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
          </button>
          <button
            type="button"
            onClick={() => { clearSavedOrderConfig(); triggerFlash('order'); }}
            className={`text-xs px-3 py-1.5 rounded-md border border-input hover:bg-muted/60 transition-shadow relative duration-200 ${flashOrder ? subtleFlashClass : ''}`}
            title="Reset & remove saved order config"
          >
            Reset Order
          </button>
        </div>
      </div>

      <p className="text-sm text-muted-foreground">
        Configure an automated order that triggers when price breaks out inside the pivot range with sufficient volume.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3">
        <div>
          <label className="block text-xs font-medium text-foreground mb-1">Ticker</label>
          <input
            type="text"
            value={orderConfig.ticker}
            onChange={(e) => setOrderConfig(prev => ({ ...prev, ticker: e.target.value }))}
            className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring"
            placeholder="AAPL"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-foreground mb-1">Lower Pivot Price</label>
          <input
            type="number"
            value={orderConfig.lower_price}
            onChange={(e) => setOrderConfig(prev => ({ ...prev, lower_price: parseFloat(e.target.value) }))}
            className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring"
            step="0.01"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-foreground mb-1">Higher Pivot Price</label>
          <input
            type="number"
            value={orderConfig.higher_price}
            onChange={(e) => setOrderConfig(prev => ({ ...prev, higher_price: parseFloat(e.target.value) }))}
            className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring"
            step="0.01"
          />
        </div>
      </div>

      <div className="space-y-2">
        <label className="block text-xs font-medium text-foreground">Volume Requirements (passed if any passes)</label>
        <div className="flex flex-col md:flex-row gap-2 items-start md:items-end">
          <input
            type="text"
            value={newVolumeReq}
            onChange={(e) => setNewVolumeReq(e.target.value)}
            className="flex-1 w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring text-sm"
            placeholder="minutes=volume or day=volume (e.g. 60=100000)"
          />
          <button
            onClick={addVolumeRequirement}
            className="px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 text-sm"
            disabled={!newVolumeReq.trim()}
          >Add</button>
        </div>
        <label className="flex items-center gap-2 text-xs cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-4 w-4"
            checked={addFractionalVolumes}
            onChange={(e) => setAddFractionalVolumes(e.target.checked)}
          />
          Auto add 1/2 & 1/4 (e.g. 60=100000 ➜ 30=50000 & 15=25000)
        </label>
        <label className="flex items-center gap-2 text-xs cursor-pointer select-none">
          <input
            type="checkbox"
            className="h-4 w-4"
            checked={convertDayToHourly}
            onChange={(e) => setConvertDayToHourly(e.target.checked)}
          />
          Auto-convert daily volume to hourly (day=volume ➜ 60=volume/6.5)
        </label>
        {orderConfig.volume_requirements.length === 0 && (
          <p className="text-xs text-muted-foreground">No volume requirements added yet.</p>
        )}
        {orderConfig.volume_requirements.length > 0 && (
          <ul className="space-y-1 max-h-40 overflow-auto pr-1">
            {orderConfig.volume_requirements.map((req, index) => (
              <li key={index} className="flex items-center gap-2 bg-muted/40 px-2 py-1 rounded text-xs">
                <span className="flex-1 font-mono">{req}</span>
                <button
                  onClick={() => removeVolumeRequirement(index)}
                  className="p-1 text-destructive hover:bg-destructive/10 rounded"
                  title="Remove"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {showAdvanced && (
        <TooltipProvider delayDuration={200}>
          <div className="space-y-4 border border-dashed border-border rounded-md p-4 bg-muted/30">
            <h3 className="text-sm font-semibold tracking-wide uppercase text-muted-foreground">Advanced Settings</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3">
              <div>
                <FieldLabel
                  label="Request Lower Trade Price (override)"
                  tooltip="Manually override the lower price boundary for the trade. When set, the bot will use this price when it sends the command for a trade execution(normally trades are sent by the order using lower pivot price and higher pivot price outside of the advanced settings). But it will use the prices Lower Pivot Price and Higher Pivot Price to monitor for breakouts."
                />
                <input
                  type="number"
                  step="0.01"
                  value={orderConfig.request_lower_price ?? ''}
                  onChange={(e) => setOrderConfig(prev => ({ ...prev, request_lower_price: e.target.value === '' ? null : parseFloat(e.target.value) }))}
                  className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring text-sm"
                  placeholder="Optional"
                />
              </div>
              <div>
                <FieldLabel
                  label="Request Higher Trade Price (override)"
                  tooltip="Manually override the higher price boundary for the trade. When set, the bot will use this price when it sends the command for a trade execution(normally trades are sent by the order using lower pivot price and higher pivot price outside of the advanced settings). But it will use the prices Lower Pivot Price and Higher Pivot Price to monitor for breakouts."
                />
                <input
                  type="number"
                  step="0.01"
                  value={orderConfig.request_higher_price ?? ''}
                  onChange={(e) => setOrderConfig(prev => ({ ...prev, request_higher_price: e.target.value === '' ? null : parseFloat(e.target.value) }))}
                  className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring text-sm"
                  placeholder="Optional"
                />
              </div>
              <div>
                <FieldLabel
                  label="Start Minutes Before Close"
                  tooltip="Begin monitoring for trade opportunities this many minutes before market close. For example, 60 means start watching 1 hour before close."
                />
                <input type="number" step="1" value={orderConfig.start_minutes_before_close ?? ''} onChange={(e) => setOrderConfig(prev => ({ ...prev, start_minutes_before_close: e.target.value === '' ? null : parseFloat(e.target.value) }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring text-sm" min={1} placeholder="e.g. 60" />
              </div>
              <div>
                <FieldLabel
                  label="Stop Minutes Before Close"
                  tooltip="Stop monitoring for new trades this many minutes before market close. Set to 0 to monitor until close. This prevents entering trades too close to market close."
                />
                <input type="number" step="1" value={orderConfig.stop_minutes_before_close ?? 0} onChange={(e) => setOrderConfig(prev => ({ ...prev, stop_minutes_before_close: parseFloat(e.target.value) || 0 }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring text-sm" min={0} placeholder="0" />
              </div>
              <div>
                <FieldLabel
                  label="Max Day Low"
                  tooltip="Maximum acceptable day low price. Orders will only trigger if the stock's daily low is at or below this value. Leave empty to ignore this filter."
                />
                <input type="number" value={orderConfig.max_day_low || ''} onChange={(e) => setOrderConfig(prev => ({ ...prev, max_day_low: e.target.value ? parseFloat(e.target.value) : null }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus-border-ring text-sm" step="0.01" placeholder="Optional" />
              </div>
              <div>
                <FieldLabel
                  label="Min Day Low"
                  tooltip="Minimum acceptable day low price. Orders will only trigger if the stock's daily low is at or above this value. Use this to avoid stocks that have dropped too much."
                />
                <input type="number" value={orderConfig.min_day_low || ''} onChange={(e) => setOrderConfig(prev => ({ ...prev, min_day_low: e.target.value ? parseFloat(e.target.value) : null }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus-border-ring text-sm" step="0.01" placeholder="Optional" />
              </div>
              <div>
                <FieldLabel
                  label="Day High Max % Off"
                  tooltip="Maximum allowed percentage the current price can be below the day's high. For example, 3 means the price must be within 3% of the day high to trigger a trade."
                />
                <input type="number" value={orderConfig.day_high_max_percent_off} onChange={(e) => setOrderConfig(prev => ({ ...prev, day_high_max_percent_off: parseFloat(e.target.value) }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus-border-ring text-sm" step="0.01" />
              </div>
              <div>
                <FieldLabel
                  label="Wait After Open (minutes)"
                  tooltip="Number of minutes to wait after market open before starting to monitor. This helps avoid the volatility of the opening minutes. Decimals allowed (e.g., 1.5 = 1 min 30 sec)."
                />
                <input type="number" step="0.1" value={orderConfig.wait_after_open_minutes ?? 0} onChange={(e) => setOrderConfig(prev => ({ ...prev, wait_after_open_minutes: parseFloat(e.target.value) || 0 }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus-border-ring text-sm" min={0} placeholder="0" />
              </div>
              <div className="col-span-2 md:col-span-3 xl:col-span-4 space-y-2">
                <FieldLabel
                  label="Volume Multipliers"
                  tooltip="Multipliers applied to volume requirements based on pivot position (Lower/Middle/Upper). A value of 1 means no change; 2 would double the volume requirement for that position. E.g. [1, 1, 0.5] means full volume required in Lower and Middle, but order can trigger with half volume in Upper position. Useful if you want to be more lenient when price is near the top of the pivot range."
                />
                <div className="grid grid-cols-3 gap-2">
                  {['Lower', 'Middle', 'Upper'].map((label, idx) => (
                    <input key={idx} type="number" value={orderConfig.volume_multipliers[idx]} onChange={(e) => { const newMultipliers = [...orderConfig.volume_multipliers]; newMultipliers[idx] = parseFloat(e.target.value); setOrderConfig(prev => ({ ...prev, volume_multipliers: newMultipliers })); }} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring text-sm" step="0.01" placeholder={label} />
                  ))}
                </div>
              </div>
              <div className="col-span-2 md:col-span-3 xl:col-span-4">
                <FieldLabel
                  label="Time in Pivot Positions"
                  tooltip="Select which pivot positions the Time in Pivot applies to. 'Any' allows any position, while 'Lower', 'Middle', and 'Upper' restrict to specific zones within the pivot range."
                />
                <div className="flex flex-wrap gap-3">
                  {(['any', 'lower', 'middle', 'upper'] as const).map(position => (
                    <label key={position} className="flex items-center gap-1 text-xs bg-muted/40 px-2 py-1 rounded">
                      <input type="checkbox" checked={pivotPositions[position]} onChange={(e) => updatePivotPositions(position, e.target.checked)} className="rounded border-input text-primary focus:ring-ring h-3 w-3" />
                      <span className="capitalize">{position}</span>
                    </label>
                  ))}
                </div>
              </div>
              {/* <div>
              <FieldLabel 
                label="Pivot Adjustment" 
                tooltip="Percentage adjustment applied to pivot levels. Use this to shift the pivot boundaries slightly higher or lower. 0% means no adjustment."
              />
              <select value={orderConfig.pivot_adjustment} onChange={(e) => setOrderConfig(prev => ({ ...prev, pivot_adjustment: e.target.value }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus:border-ring text-sm">
                <option value="0.0">0.0%</option>
                <option value="0.5">0.5%</option>
                <option value="1.0">1.0%</option>
              </select>
            </div> */}
              <div>
                <FieldLabel
                  label="Time in Pivot (s)"
                  tooltip="Minimum time in seconds the price must stay within the pivot zone(s) chosen in the 'Time in Pivot Positions' setting before triggering a trade. Longer times help confirm the price level is holding."
                />
                <input type="number" value={orderConfig.time_in_pivot} onChange={(e) => setOrderConfig(prev => ({ ...prev, time_in_pivot: parseInt(e.target.value) }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus-border-ring text-sm" placeholder="seconds" />
              </div>
              <div>
                <FieldLabel
                  label="Breakout Lookback (m)"
                  tooltip="Number of minutes to look back when checking for breakout conditions. If you set it to 60, the stock must be higher than the highest price in the last 60 minutes to qualify as a breakout."
                />
                <input type="number" value={orderConfig.breakout_lookback_minutes} onChange={(e) => setOrderConfig(prev => ({ ...prev, breakout_lookback_minutes: parseInt(e.target.value) || 0 }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus-border-ring text-sm" min={1} />
              </div>
              <div>
                <FieldLabel
                  label="Lookback Exclude Recent (m)"
                  tooltip="Exclude the most recent X minutes from breakout analysis, but the latest price is still used—only the rest of the data in that window is ignored. For example, 0.5 excludes the last 30 seconds except for the most recent tick. This helps when a spike occurs at breakout (e.g., to $11) and immediately drops back (e.g., to $10.90), which happens often."
                />
                <input type="number" step="0.1" value={orderConfig.breakout_exclude_minutes} onChange={(e) => setOrderConfig(prev => ({ ...prev, breakout_exclude_minutes: parseFloat(e.target.value) || 0 }))} className="w-full p-2 border border-input bg-background text-foreground rounded-md focus:ring-2 focus:ring-ring focus-border-ring text-sm" min={0} />
              </div>
            </div>
          </div>
        </TooltipProvider>
      )}

      <button onClick={startOrder} disabled={loading} className="w-full bg-primary text-primary-foreground py-3 px-4 rounded-lg hover:bg-primary/90 disabled:opacity-50 flex items-center justify-center gap-2">
        {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
        {loading ? 'Starting Order...' : 'Start Trading Order'}
      </button>
    </div>
  );
};
