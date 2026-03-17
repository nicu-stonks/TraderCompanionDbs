import { useMemo, useRef, useState } from 'react';
import { X, AlertTriangle } from 'lucide-react';
import type { RiskAnalysis, RiskDistribution } from '../types';

interface Props {
  ticker: string;
  risk: RiskAnalysis;
  onClose: () => void;
}

interface DistributionCardProps {
  title: string;
  subtitle: string;
  distribution: RiskDistribution;
}

function DistributionCard({ title, subtitle, distribution }: DistributionCardProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{ x: number; density_pct: number; tail_pct: number; cdf_pct?: number; exceed_pct?: number } | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null);
  const bins = distribution.bins || [];

  const width = 640;
  const height = 210;
  const leftPad = 42;
  const rightPad = 18;
  const bottomPad = 74;

  const xMin = bins[0].x;
  const xMax = bins[bins.length - 1].x;
  const yMax = Math.max(...bins.map(b => b.density_pct || 0)) * 1.05;

  const tickIndices = useMemo(() => {
    if (bins.length === 0) return [] as number[];
    const plotWidthTarget = width - leftPad - rightPad;
    const minLabelSpacingPx = 44; // dense but readable
    const maxTickCount = Math.max(2, Math.floor(plotWidthTarget / minLabelSpacingPx));
    const step = Math.max(1, Math.ceil(bins.length / maxTickCount));

    const indices: number[] = [];
    for (let i = 0; i < bins.length; i += step) {
      indices.push(i);
    }
    if (indices[indices.length - 1] !== bins.length - 1) {
      indices.push(bins.length - 1);
    }

    // Ensure the forced right-most label does not overlap the previous one.
    while (indices.length >= 2) {
      const last = indices[indices.length - 1];
      const prev = indices[indices.length - 2];
      const pixelGap = ((last - prev) / Math.max(1, bins.length - 1)) * plotWidthTarget;
      if (pixelGap >= minLabelSpacingPx) break;
      indices.splice(indices.length - 2, 1);
    }

    return indices;
  }, [bins.length, width, leftPad, rightPad]);

  if (bins.length === 0) {
    return (
      <div className="rounded-md border border-border p-3 bg-muted/20">
        <h4 className="text-sm font-semibold">{title}</h4>
        <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
        <p className="text-xs text-muted-foreground mt-3">Not enough historical data yet.</p>
      </div>
    );
  }

  const xScale = (x: number) => {
    const plotWidthTarget = width - leftPad - rightPad;
    if (xMax === xMin) return leftPad;
    return leftPad + ((x - xMin) / (xMax - xMin)) * plotWidthTarget;
  };

  const medianVal = distribution.median ?? distribution.mean;
  const isSpread = title.toLowerCase().includes('spread');

  // Reference lines: Q1/Q3 for spread, ±1σ/±2σ for gap & daily change
  const line1Val = isSpread ? (distribution.q1 ?? medianVal) : (distribution.mean - distribution.std);
  const line2Val = isSpread ? (distribution.q3 ?? medianVal) : (distribution.mean + distribution.std);
  const line1X = xScale(line1Val);
  const line2X = xScale(line2Val);
  const line1Label = isSpread ? `${line1Val.toFixed(2)}%(Q1)` : `${line1Val.toFixed(2)}%(-1σ)`;
  const line2Label = isSpread ? `${line2Val.toFixed(2)}%(Q3)` : `${line2Val.toFixed(2)}%(+1σ)`;

  // Outer lines: P10/P90 for spread, ±2σ for gap & daily change
  const line3Val = isSpread ? (distribution.p10 ?? line1Val) : (distribution.mean - 2 * distribution.std);
  const line4Val = isSpread ? (distribution.p90 ?? line2Val) : (distribution.mean + 2 * distribution.std);
  const line3X = xScale(line3Val);
  const line4X = xScale(line4Val);
  const line3Label = isSpread ? `${line3Val.toFixed(2)}%(P10)` : `${line3Val.toFixed(2)}%(-2σ)`;
  const line4Label = isSpread ? `${line4Val.toFixed(2)}%(P90)` : `${line4Val.toFixed(2)}%(+2σ)`;

  // Pre-calculate label collisions to determine required top padding
  const refLines = [
    { x: line1X, label: line1Label, cls: 'text-muted-foreground', fillCls: 'fill-muted-foreground', dash: '3 3' },
    { x: line2X, label: line2Label, cls: 'text-muted-foreground', fillCls: 'fill-muted-foreground', dash: '3 3' },
    { x: line3X, label: line3Label, cls: 'text-muted-foreground/75', fillCls: 'fill-muted-foreground/75', dash: '6 3' },
    { x: line4X, label: line4Label, cls: 'text-muted-foreground/75', fillCls: 'fill-muted-foreground/75', dash: '6 3' },
  ];
  const sortedRefLines = [...refLines].sort((a, b) => a.x - b.x);
  const minSpacing = 46;
  const labelHeight = 10;
  const yOffsets: number[] = [];
  let maxStack = 0;

  for (let i = 0; i < sortedRefLines.length; i++) {
    let offset = 0;
    for (let j = i - 1; j >= 0; j--) {
      // Check for overlap, allowing spacing
      if (Math.abs(sortedRefLines[i].x - sortedRefLines[j].x) < minSpacing && offset >= yOffsets[j] - labelHeight) {
        offset = yOffsets[j] - labelHeight;
      }
    }
    yOffsets.push(offset);
    maxStack = Math.max(maxStack, Math.abs(offset) / labelHeight);
  }

  // Dynamic top pad based on how many labels stack
  const topPad = 16 + (maxStack * labelHeight);
  const plotWidth = width - leftPad - rightPad;
  const plotHeight = height - topPad - bottomPad;

  const yScale = (y: number) => {
    return topPad + (1 - y / yMax) * plotHeight;
  };

  const path = bins
    .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${xScale(point.x)} ${yScale(point.density_pct)}`)
    .join(' ');
  const getInterpolatedPoint = (clientX: number, svgRect: DOMRect) => {
    const relativeXPx = clientX - svgRect.left;
    const scaleX = svgRect.width > 0 ? (width / svgRect.width) : 1;
    const relativeX = relativeXPx * scaleX;
    const clampedX = Math.max(leftPad, Math.min(width - rightPad, relativeX));
    const pct = (clampedX - leftPad) / plotWidth;
    const idxFloat = pct * (bins.length - 1);
    const lowIdx = Math.max(0, Math.min(bins.length - 1, Math.floor(idxFloat)));
    const highIdx = Math.max(0, Math.min(bins.length - 1, Math.ceil(idxFloat)));

    const low = bins[lowIdx];
    const high = bins[highIdx];
    if (!low || !high || lowIdx === highIdx) {
      return low ?? { x: 0.0, density_pct: 0.0, tail_pct: 0.0, cdf_pct: 0.0, exceed_pct: 100.0 };
    }

    const alpha = idxFloat - lowIdx;
    const lowCdf = low.cdf_pct ?? low.tail_pct;
    const highCdf = high.cdf_pct ?? high.tail_pct;
    const lowExceed = low.exceed_pct ?? (100 - (low.cdf_pct ?? low.tail_pct));
    const highExceed = high.exceed_pct ?? (100 - (high.cdf_pct ?? high.tail_pct));
    return {
      x: low.x + (high.x - low.x) * alpha,
      density_pct: low.density_pct + (high.density_pct - low.density_pct) * alpha,
      tail_pct: low.tail_pct + (high.tail_pct - low.tail_pct) * alpha,
      cdf_pct: lowCdf + (highCdf - lowCdf) * alpha,
      exceed_pct: lowExceed + (highExceed - lowExceed) * alpha,
    };
  };

  const hoveredBin = hoveredPoint;

  const metricLabel = title.toLowerCase().includes('gap')
    ? 'gap'
    : title.toLowerCase().includes('spread')
      ? 'spread'
      : 'daily move';
  const thresholdAxisLabel = `X axis: ${metricLabel} threshold (%)`;
  const probabilityAxisLabel = metricLabel === 'spread'
    ? 'Lower: P(spread ≥ x)'
    : `Lower: left P(${metricLabel} ≤ x), right P(${metricLabel} > x)`;
  const splitPoint = distribution.split_point ?? distribution.mean;
  const usesLeftTail = hoveredBin ? hoveredBin.x <= splitPoint : false;

  return (
    <div ref={containerRef} className="rounded-md border border-border p-3 bg-background relative">
      <div className="mb-2">
        <h4 className="text-sm font-semibold">{title}</h4>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const containerRect = containerRef.current?.getBoundingClientRect();
          const point = getInterpolatedPoint(e.clientX, rect);
          setHoveredPoint(point);
          setHoverPos({
            x: containerRect ? (e.clientX - containerRect.left) : (e.clientX - rect.left),
            y: containerRect ? (e.clientY - containerRect.top) : (e.clientY - rect.top),
          });
        }}
        onMouseLeave={() => {
          setHoveredPoint(null);
          setHoverPos(null);
        }}
      >
        <line x1={leftPad} y1={yScale(0)} x2={width - rightPad} y2={yScale(0)} stroke="currentColor" className="text-border" strokeWidth="1" />
        <line x1={leftPad} y1={topPad} x2={leftPad} y2={yScale(0)} stroke="currentColor" className="text-border" strokeWidth="1" />

        {/* Reference lines with collision-avoiding labels */}
        {(() => {
          const refLines = [
            { x: line1X, label: line1Label, cls: 'text-muted-foreground', fillCls: 'fill-muted-foreground', dash: '3 3' },
            { x: line2X, label: line2Label, cls: 'text-muted-foreground', fillCls: 'fill-muted-foreground', dash: '3 3' },
            { x: line3X, label: line3Label, cls: 'text-muted-foreground/75', fillCls: 'fill-muted-foreground/75', dash: '6 3' },
            { x: line4X, label: line4Label, cls: 'text-muted-foreground/75', fillCls: 'fill-muted-foreground/75', dash: '6 3' },
          ];
          // Sort by x so we can detect adjacency
          const sorted = [...refLines].sort((a, b) => a.x - b.x);
          const minSpacing = 46; // min px between label centers before bumping
          const labelHeight = 10;
          const baseY = topPad - 3;
          // Assign y offsets: if too close to previous, bump up
          const yPositions: number[] = [];
          for (let i = 0; i < sorted.length; i++) {
            let y = baseY;
            // Check against all already-placed labels for overlap
            for (let j = i - 1; j >= 0; j--) {
              if (Math.abs(sorted[i].x - sorted[j].x) < minSpacing && y >= yPositions[j] - labelHeight) {
                y = yPositions[j] - labelHeight;
              }
            }
            yPositions.push(y);
          }
          return sorted.map((line, i) => (
            <g key={`ref-${i}`}>
              <line x1={line.x} y1={topPad} x2={line.x} y2={yScale(0)} stroke="currentColor" className={line.cls} strokeDasharray={line.dash} strokeWidth="1" />
              <text x={line.x} y={yPositions[i]} textAnchor="middle" className={line.fillCls} style={{ fontSize: '9px' }}>
                {line.label}
              </text>
            </g>
          ));
        })()}

        <path d={path} fill="none" stroke="currentColor" className="text-blue-500" strokeWidth="2" />

        {hoveredBin && (
          <>
            <line
              x1={xScale(hoveredBin.x)}
              y1={topPad}
              x2={xScale(hoveredBin.x)}
              y2={yScale(0)}
              stroke="currentColor"
              className="text-blue-400"
              strokeDasharray="2 2"
              strokeWidth="1"
            />
            <circle cx={xScale(hoveredBin.x)} cy={yScale(hoveredBin.density_pct)} r="3" fill="currentColor" className="text-blue-500" />
          </>
        )}

        {tickIndices.map((idx) => {
          const point = bins[idx];
          const x = xScale(point.x);
          return (
            <g key={`tick-${idx}`}>
              <line x1={x} y1={yScale(0)} x2={x} y2={yScale(0) + 4} stroke="currentColor" className="text-border" strokeWidth="1" />
              <text x={x} y={yScale(0) + 16} textAnchor="middle" className="fill-muted-foreground" style={{ fontSize: '10px' }}>
                {point.x.toFixed(1)}%
              </text>
              <text x={x} y={yScale(0) + 31} textAnchor="middle" className="fill-muted-foreground" style={{ fontSize: '10px' }}>
                {point.tail_pct.toFixed(1)}%
              </text>
            </g>
          );
        })}

        <text x={leftPad} y={height - 6} textAnchor="start" className="fill-muted-foreground" style={{ fontSize: '10px' }}>
          {thresholdAxisLabel}
        </text>
        <text x={width - rightPad} y={height - 6} textAnchor="end" className="fill-muted-foreground" style={{ fontSize: '10px' }}>
          {probabilityAxisLabel}
        </text>
      </svg>

      {hoveredBin && hoverPos && (
        (() => {
          const tooltipWidth = 320;
          const tooltipHeight = 82;
          const containerWidth = containerRef.current?.clientWidth ?? width;
          const left = Math.min(Math.max(hoverPos.x + 10, 8), Math.max(8, containerWidth - tooltipWidth - 8));
          const top = Math.max(40, hoverPos.y - tooltipHeight);

          return (
            <div
              className="absolute z-10 pointer-events-none rounded border border-border bg-background/95 px-2 py-1.5 shadow text-[11px]"
              style={{ left, top, maxWidth: `${tooltipWidth}px` }}
            >
              <div><span className="font-semibold">{metricLabel === 'spread' ? 'Spread:' : 'Move:'}</span> {hoveredBin.x.toFixed(2)}%</div>
              <div>
                {(() => {
                  const xVal = hoveredBin.x.toFixed(2);
                  const leftProb = (hoveredBin.cdf_pct ?? hoveredBin.tail_pct).toFixed(2);
                  const rightProb = (
                    hoveredBin.exceed_pct
                    ?? (100 - (hoveredBin.cdf_pct ?? hoveredBin.tail_pct))
                  ).toFixed(2);

                  if (metricLabel === 'gap') {
                    return usesLeftTail
                      ? `Probability of a gap down ≤ ${xVal}%: ${leftProb}%`
                      : `Probability of a gap up > ${xVal}%: ${rightProb}%`;
                  }

                  if (metricLabel === 'daily move') {
                    return usesLeftTail
                      ? `Probability of a daily move ≤ ${xVal}%: ${leftProb}%`
                      : `Probability of a daily move > ${xVal}%: ${rightProb}%`;
                  }

                  return `Probability of a spread ≥ ${xVal}%: ${rightProb}%`;
                })()}
              </div>
            </div>
          );
        })()
      )}

      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-2 text-[11px]">
        <div className="rounded bg-muted px-2 py-1">N: <span className="font-semibold">{distribution.sample_size}</span></div>
        <div className="rounded bg-muted px-2 py-1">Median: <span className="font-semibold">{medianVal.toFixed(2)}%</span></div>
        <div className="rounded bg-muted px-2 py-1">Mean: <span className="font-semibold">{distribution.mean.toFixed(2)}%</span></div>
        {metricLabel === 'spread' && distribution.q1 != null && distribution.q3 != null
          ? <div className="rounded bg-muted px-2 py-1">IQR: <span className="font-semibold">[{distribution.q1.toFixed(2)}%, {distribution.q3.toFixed(2)}%]</span></div>
          : <div className="rounded bg-muted px-2 py-1">Std: <span className="font-semibold">{distribution.std.toFixed(2)}%</span></div>
        }
        <div className="rounded bg-muted px-2 py-1">Range: <span className="font-semibold">[{distribution.min.toFixed(1)}%, {distribution.max.toFixed(1)}%]</span></div>
      </div>
    </div>
  );
}

export function RiskAnalysisModal({ ticker, risk, onClose }: Props) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-background border border-border rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div>
            <h3 className="text-lg font-semibold">Risk Analysis — {ticker}</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Using up to last {risk.lookback_days} trading days (~1 year). As of {risk.as_of_date ?? 'N/A'}. Warning: these are statistical estimates based on historical data and may not reflect future outcomes.
            </p>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-muted rounded">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="overflow-y-auto p-4 space-y-4">
          <DistributionCard
            title="Gaps Distribution"
            subtitle="Daily gap % = Open vs Previous Close"
            distribution={risk.gap_risk}
          />
          <DistributionCard
            title="Daily % Change Distribution"
            subtitle="Daily move % = Close vs Previous Close"
            distribution={risk.daily_change}
          />
          <DistributionCard
            title="Daily Spread Distribution"
            subtitle="Spread % = (High - Low) / Low"
            distribution={risk.daily_spread}
          />

          <div className="rounded-md border border-blue-500/30 bg-blue-500/5 p-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-blue-500">
              <AlertTriangle className="h-4 w-4" />
              Risk Glossary
            </div>
            <ul className="text-xs text-muted-foreground mt-2 space-y-1.5">
              <li>• Window: calculations use up to the most recent 252 trading days (~1 year).</li>
              <li>• Curve: KDE-smoothed density from historical data with peak normalized to 100 for visual scale; 100 is the highest-density region, not a 100% chance outcome.</li>
              <li>• Tail axis: left of the mean shows P(move ≤ x); right of the mean shows P(move ≥ x).</li>
              <li>• Gap Risk: open-to-previous-close % jump/drop distribution.</li>
              <li>• Daily % Change: close-to-previous-close % move distribution.</li>
              <li>• Daily Spread: intraday spread magnitude (High-Low)/Low, capped at 0 minimum (no negative values).</li>
              <li>• ±1σ lines (gap/daily change): ~68% of trading days fall within this range around the mean.</li>
              <li>• ±2σ lines (gap/daily change): ~95% of trading days fall within this range around the mean. Moves beyond ±2σ are statistically unusual.</li>
              <li>• Q1/Q3 lines (spread): the interquartile range — 50% of all trading days have a spread between Q1 and Q3.</li>
              <li>• P10/P90 lines (spread): 80% of all trading days have a spread between P10 and P90. Spreads beyond P90 are unusually wide days. Spreads below P10 are unusually tight days.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
