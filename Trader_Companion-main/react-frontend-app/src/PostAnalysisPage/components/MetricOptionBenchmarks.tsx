import React, { useMemo } from 'react';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import { Metric, MetricOptionRecommendation, TradeGrade, MetricPercentBaseSetting } from '../types/types';
import { BarChart3 } from 'lucide-react';
import { percentBaseSettingService } from '../services/postAnalysis';

interface MetricOptionBenchmarksProps {
  trades: Trade[];
  metrics: Metric[];
  tradeGrades: TradeGrade[];
  trailingWindow: number;
  recommendations: MetricOptionRecommendation[];
  layoutMode: 'stacked' | 'grid';
  percentBaseSettings: MetricPercentBaseSetting[];
  onRefetchPercentBaseSettings: () => void;
}

const MetricOptionBenchmarks: React.FC<MetricOptionBenchmarksProps> = ({
  trades,
  metrics,
  tradeGrades,
  trailingWindow,
  recommendations,
  layoutMode,
  percentBaseSettings,
  onRefetchPercentBaseSettings
}) => {
  const { windowTrades, countByOption, totalCountByMetric } = useMemo(() => {
    const sortedTrades = [...trades].sort(
      (a, b) => new Date(a.Entry_Date).getTime() - new Date(b.Entry_Date).getTime()
    );

    let lastGradedIndex = -1;
    if (tradeGrades.length > 0) {
      const gradedTradeIds = new Set(tradeGrades.map(g => g.tradeId));
      for (let i = 0; i < sortedTrades.length; i++) {
        if (gradedTradeIds.has(sortedTrades[i].ID)) {
          lastGradedIndex = i;
        }
      }
    }

    const effectiveTrades = lastGradedIndex >= 0 ? sortedTrades.slice(0, lastGradedIndex + 1) : [];
    const windowStart = Math.max(0, effectiveTrades.length - trailingWindow);
    const windowTrades = effectiveTrades.slice(windowStart);

    const tradeGradeMap = new Map<number, Map<number, number>>();
    for (const grade of tradeGrades) {
      const metricMap = tradeGradeMap.get(grade.tradeId) ?? new Map<number, number>();
      metricMap.set(Number(grade.metricId), Number(grade.selectedOptionId));
      tradeGradeMap.set(grade.tradeId, metricMap);
    }

    const percentByOption = new Map<string, number>();
    const countByOption = new Map<string, number>();
    const totalCountByMetric = new Map<number, number>();
    for (const metric of metrics) {
      totalCountByMetric.set(metric.id, 0);
      for (const option of metric.options) {
        let count = 0;
        for (const trade of windowTrades) {
          const metricMap = tradeGradeMap.get(trade.ID);
          if (metricMap && metricMap.get(metric.id) === option.id) {
            count += 1;
          }
        }
        totalCountByMetric.set(metric.id, (totalCountByMetric.get(metric.id) ?? 0) + count);
        countByOption.set(`${metric.id}-${option.id}`, count);
      }
      const metricTotal = totalCountByMetric.get(metric.id) ?? 0;
      for (const option of metric.options) {
        const optionCount = countByOption.get(`${metric.id}-${option.id}`) ?? 0;
        const pct = metricTotal > 0 ? (optionCount / metricTotal) * 100 : 0;
        percentByOption.set(`${metric.id}-${option.id}`, pct);
      }
    }

    return { windowTrades, countByOption, totalCountByMetric };
  }, [trades, tradeGrades, metrics, trailingWindow]);

  const recommendationMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const rec of recommendations) {
      const metricId = Number(rec.metric);
      const optionId = Number(rec.option);
      map.set(`${metricId}-${optionId}`, Number(rec.recommended_pct));
    }
    return map;
  }, [recommendations]);

  const percentBaseMap = useMemo(() => {
    const map = new Map<number, boolean>();
    for (const setting of percentBaseSettings) {
      map.set(Number(setting.metric_id), Boolean(setting.use_total_trades));
    }
    return map;
  }, [percentBaseSettings]);

  const handleTogglePercentBase = async (metricId: number, nextValue: boolean) => {
    try {
      await percentBaseSettingService.upsertSetting(metricId, nextValue);
      onRefetchPercentBaseSettings();
    } catch (error) {
      console.error('Failed to save percent base setting:', error);
    }
  };

  if (metrics.length === 0) {
    return null;
  }

  return (
    <div className="bg-background rounded-lg shadow-md p-4 mb-4">
      <h2 className="text-2xl font-bold mb-1 flex items-center text-foreground">
        <BarChart3 className="mr-2" />
        Metric Option Benchmarks
      </h2>

      {windowTrades.length === 0 ? (
        <p className="text-muted-foreground">Grade trades to see benchmark bars for the trailing window.</p>
      ) : (
        <div className={layoutMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 gap-2' : 'space-y-2'}>
          {metrics.map(metric => (
            <div key={metric.id} className="border border-border rounded-lg p-2 bg-card">
              <div className="flex items-center gap-2 mb-1">
                <div className="relative group">
                  <input
                    type="checkbox"
                    checked={percentBaseMap.get(metric.id) ?? metric.name === 'Missed Opportunities'}
                    onChange={(e) => handleTogglePercentBase(metric.id, e.target.checked)}
                  />
                  <div className="pointer-events-none absolute -top-8 left-0 hidden group-hover:flex z-10">
                    <div className="px-2 py-1 rounded-md bg-foreground text-background text-xs shadow whitespace-nowrap text-center">
                      Use total trades as % base
                    </div>
                  </div>
                </div>
                <h3 className="text-lg font-semibold text-foreground">
                  {metric.name} <span className="text-muted-foreground font-normal">{totalCountByMetric.get(metric.id) ?? 0}</span>
                </h3>
              </div>
              <div className="space-y-1">
                {metric.options.map(option => {
                  const key = `${metric.id}-${option.id}`;
                  const useTotalTrades = percentBaseMap.get(metric.id) ?? metric.name === 'Missed Opportunities';
                  const metricTotal = totalCountByMetric.get(metric.id) ?? 0;
                  const optionCount = countByOption.get(key) ?? 0;
                  const pct = useTotalTrades
                    ? (windowTrades.length > 0 ? (optionCount / windowTrades.length) * 100 : 0)
                    : (metricTotal > 0 ? (optionCount / metricTotal) * 100 : 0);
                  const count = countByOption.get(key) ?? 0;
                  const recommended = recommendationMap.get(key);
                  const rec = recommendations.find(r => Number(r.metric) === metric.id && Number(r.option) === option.id);
                  const isMinimum = rec?.is_minimum ?? true;
                  const meetsRecommended = recommended !== undefined
                    ? (isMinimum ? pct >= recommended : pct <= recommended)
                    : false;
                  const fillColor = recommended === undefined
                    ? 'bg-primary'
                    : (meetsRecommended ? 'bg-emerald-500' : 'bg-yellow-400');
                  const pctColor = recommended === undefined
                    ? 'text-muted-foreground'
                    : (meetsRecommended ? 'text-emerald-500' : 'text-yellow-500');
                  const tooltip = recommended !== undefined
                    ? `Recommended ${isMinimum ? '>=' : '<='}${recommended}%`
                    : 'No recommendation set';

                  return (
                    <div key={option.id} className="flex items-center gap-1 text-sm">
                      <span className="text-foreground min-w-[110px]">{option.name}</span>
                      <span className="text-muted-foreground w-6 text-right">{count}</span>
                      <div className="relative flex-1 group">
                        <div className="w-full h-2 rounded-full bg-muted/60 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${fillColor}`}
                            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                          />
                        </div>
                        <div className="pointer-events-none absolute -top-8 left-0 hidden group-hover:flex">
                          <div className="px-2 py-1 rounded-md bg-foreground text-background text-xs shadow">
                            {tooltip}
                          </div>
                        </div>
                      </div>
                      <span className={`${pctColor} w-12 text-right`}>{pct.toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MetricOptionBenchmarks;
