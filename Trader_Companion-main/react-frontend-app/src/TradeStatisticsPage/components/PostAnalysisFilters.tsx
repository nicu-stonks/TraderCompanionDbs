import React, { useEffect, useMemo, useState } from 'react';
import { metricService } from '@/PostAnalysisPage/services/postAnalysis';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import type { Metric } from '@/PostAnalysisPage/types/types';
import type { MetricOptionFilters } from '../types';

type Props = {
  selected: MetricOptionFilters;
  onChange: (next: MetricOptionFilters) => void;
};

export const PostAnalysisFilters: React.FC<Props> = ({ selected, onChange }) => {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const all = await metricService.getMetrics();
        if (mounted) setMetrics(all);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  const handleToggle = (metricId: number, optionId: number, checked: boolean | string) => {
    const isChecked = checked === true || checked === 'true';
    const current = selected[metricId] ?? [];
    const nextForMetric = isChecked
      ? Array.from(new Set<number>([...current, optionId]))
      : current.filter((id: number) => id !== optionId);

    const next: MetricOptionFilters = { ...selected };
    if (nextForMetric.length) next[metricId] = nextForMetric;
    else delete next[metricId];
    onChange(next);
  };

  const hasAnySelection = useMemo(() => Object.keys(selected).length > 0, [selected]);

  if (loading) {
    return <div className="text-sm text-muted-foreground">Loading post-analysis filters…</div>;
  }

  if (!metrics.length) {
    return <div className="text-sm text-muted-foreground">No post-analysis metrics found.</div>;
  }

  return (
    <Card className="border-0 shadow-none">
      <CardContent className="p-0">
      <div className="mt-3 mb-2 text-sm font-medium">Post Analysis Filters {hasAnySelection ? '' : '(none selected)'}</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
          {metrics.map(metric => (
            <div key={metric.id} className="rounded-md border p-3">
              <div className="mb-2 font-medium text-sm">{metric.name}</div>
              <div className="space-y-2 max-h-52 overflow-auto pr-1">
                {metric.options.map(opt => {
                  const checked = (selected[metric.id] ?? []).includes(opt.id);
                  return (
                    <label key={opt.id} className="flex items-center gap-2 text-sm">
                      <Checkbox
                        checked={checked}
                        onCheckedChange={(val) => handleToggle(metric.id, opt.id, val)}
                        id={`metric-${metric.id}-opt-${opt.id}`}
                      />
                      <Label htmlFor={`metric-${metric.id}-opt-${opt.id}`} className="cursor-pointer text-sm font-normal">
                        {opt.name}
                      </Label>
                    </label>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        {hasAnySelection && (
          <div className="mt-3">
            <button
              type="button"
              className="text-xs text-muted-foreground hover:text-foreground underline"
              onClick={() => onChange({})}
            >
              Clear post-analysis filters
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default PostAnalysisFilters;
