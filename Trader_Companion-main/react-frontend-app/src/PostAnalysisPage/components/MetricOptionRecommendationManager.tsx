import React, { useEffect, useMemo, useState } from 'react';
import { Edit2, Loader, Save } from 'lucide-react';
import { Metric, MetricOptionRecommendation } from '../types/types';
import { recommendationService } from '../services/postAnalysis';

interface MetricOptionRecommendationManagerProps {
  metrics: Metric[];
  recommendations: MetricOptionRecommendation[];
  onRefetch: () => void;
}

const MetricOptionRecommendationManager: React.FC<MetricOptionRecommendationManagerProps> = ({
  metrics,
  recommendations,
  onRefetch
}) => {
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [directions, setDirections] = useState<Record<string, boolean>>({});
  const [savingKey, setSavingKey] = useState<string | null>(null);

  const recommendationMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const rec of recommendations) {
      map.set(`${rec.metric}-${rec.option}`, rec.recommended_pct);
    }
    return map;
  }, [recommendations]);

  useEffect(() => {
    const nextDrafts: Record<string, string> = {};
    const nextDirections: Record<string, boolean> = {};
    for (const metric of metrics) {
      for (const option of metric.options) {
        const key = `${metric.id}-${option.id}`;
        const existing = recommendationMap.get(key);
        if (existing !== undefined) {
          nextDrafts[key] = String(existing);
        }
        const existingRec = recommendations.find(r => r.metric === metric.id && r.option === option.id);
        if (existingRec) {
          nextDirections[key] = existingRec.is_minimum;
        } else {
          nextDirections[key] = true;
        }
      }
    }
    setDrafts(nextDrafts);
    setDirections(nextDirections);
  }, [metrics, recommendationMap, recommendations]);

  const handleChange = (key: string, value: string) => {
    setDrafts(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = async (metricId: number, optionId: number, key: string) => {
    const raw = drafts[key];
    const parsed = Number(raw);
    if (Number.isNaN(parsed)) return;
    const isMinimum = directions[key] ?? true;

    setSavingKey(key);
    try {
      await recommendationService.upsertRecommendation(metricId, optionId, parsed, isMinimum);
      onRefetch();
    } catch (error) {
      console.error('Failed to save recommendation:', error);
    } finally {
      setSavingKey(null);
    }
  };

  if (metrics.length === 0) {
    return null;
  }

  return (
    <div className="rounded-lg shadow-md p-3 mb-3 bg-background text-foreground">
      <h2 className="text-2xl font-bold mb-2 flex items-center">
        <Edit2 className="mr-2" />
        Metric Option Recommendations
      </h2>
      <p className="text-muted-foreground mb-1 text-xs">Set the acceptable % of trades in the trailing window for each option.</p>

      <div className="space-y-1">
        {metrics.map(metric => (
          <div key={metric.id} className="border rounded-lg p-1.5 border-border bg-card text-card-foreground">
            <h3 className="text-lg font-semibold mb-1">{metric.name}</h3>
            <div className="space-y-1">
              {metric.options.map(option => {
                const key = `${metric.id}-${option.id}`;
                return (
                  <div key={option.id} className="flex items-center justify-between gap-1">
                    <div className="text-sm font-medium">{option.name}</div>
                    <div className="flex items-center gap-1">
                      <select
                        value={(directions[key] ?? true) ? 'above' : 'below'}
                        onChange={(e) => setDirections(prev => ({ ...prev, [key]: e.target.value === 'above' }))}
                        className="px-1 py-0.5 border rounded-md bg-background text-foreground border-input text-[11px]"
                      >
                        <option value="above">≥</option>
                        <option value="below">≤</option>
                      </select>
                      <input
                        type="number"
                        min={0}
                        max={100}
                        step={1}
                        value={drafts[key] ?? ''}
                        onChange={e => handleChange(key, e.target.value)}
                        className="w-16 px-2 py-0.5 border rounded-md bg-background text-foreground border-input text-sm"
                        placeholder="%"
                      />
                      <button
                        type="button"
                        onClick={() => handleSave(metric.id, option.id, key)}
                        disabled={savingKey === key || drafts[key] === undefined || drafts[key] === ''}
                        className="px-1.5 py-0.5 bg-primary text-primary-foreground text-[11px] rounded-md hover:bg-primary/90 disabled:opacity-50 flex items-center"
                      >
                        {savingKey === key ? <Loader className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                        <span className="ml-1">Save</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MetricOptionRecommendationManager;
