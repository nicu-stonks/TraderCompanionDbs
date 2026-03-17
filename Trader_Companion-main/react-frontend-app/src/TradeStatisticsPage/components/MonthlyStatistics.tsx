import React, { useMemo, useRef } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from '@/components/ui/checkbox';
import { MonthlyStats } from '../types';

interface MonthlyStatisticsProps {
  monthlyStats: MonthlyStats[];
  onToggleMonth: (month: string) => void;
  onRangeToggleMonths: (months: string[], checked: boolean) => void;
  targetReturnPercent: number;
}

const computeProjection = (monthlyRatePct: number, targetReturnPercent: number): { text: string; color: string } => {
  const r = monthlyRatePct / 100;
  if (Math.abs(r) < 0.0001) return { text: 'N/A', color: '' };
  if (r > 0) {
    const months = Math.log(1 + targetReturnPercent / 100) / Math.log(1 + r);
    const years = months / 12;
    return { text: `${years.toFixed(1)}y → +${targetReturnPercent}%`, color: '#34d399' };
  } else {
    const months = Math.log(0.5) / Math.log(1 + r);
    const years = months / 12;
    return { text: `${years.toFixed(1)}y → −50%`, color: '#f87171' };
  }
};

export const MonthlyStatistics: React.FC<MonthlyStatisticsProps> = ({
  monthlyStats,
  onToggleMonth,
  onRangeToggleMonths,
  targetReturnPercent
}) => {
  const lastClickedIdx = useRef<number>(-1);

  const handleMonthClick = (e: React.MouseEvent, month: MonthlyStats, idx: number) => {
    if (e.shiftKey && lastClickedIdx.current >= 0) {
      const from = Math.min(lastClickedIdx.current, idx);
      const to = Math.max(lastClickedIdx.current, idx);
      const range = monthlyStats.slice(from, to + 1).map(m => m.tradingMonth);
      // target state = what clicking this month would do
      onRangeToggleMonths(range, !month.useInYearly);
    } else {
      onToggleMonth(month.tradingMonth);
    }
    lastClickedIdx.current = idx;
  };
  const summaryStats = useMemo(() => {
    const filteredStats = monthlyStats.filter(month => month.useInYearly);
    const totalTrades = filteredStats.reduce((sum, month) => sum + month.totalTrades, 0);

    const totalWinningTrades = filteredStats.reduce((sum, month) =>
      sum + Math.round((month.winningPercentage / 100) * month.totalTrades), 0);

    const totalLosingTrades = totalTrades - totalWinningTrades;

    const averageGain = totalWinningTrades
      ? filteredStats.reduce((sum, month) => sum + (month.averageGain * Math.round((month.winningPercentage / 100) * month.totalTrades)), 0) / totalWinningTrades
      : 0;

    const averageLoss = totalLosingTrades
      ? filteredStats.reduce((sum, month) => sum + (month.averageLoss * (month.totalTrades - Math.round((month.winningPercentage / 100) * month.totalTrades))), 0) / totalLosingTrades
      : 0;


    const winningPercentage = filteredStats.reduce((sum, month) =>
      sum + (month.winningPercentage * month.totalTrades), 0) / totalTrades;

    const avgDaysGains = totalWinningTrades > 0
      ? filteredStats.reduce((sum, month) => {
        const winningTrades = Math.round((month.winningPercentage / 100) * month.totalTrades);
        return sum + (month.avgDaysGains * winningTrades);
      }, 0) / totalWinningTrades
      : 0;


    const avgDaysLoss = totalLosingTrades > 0
      ? filteredStats.reduce((sum, month) => {
        const losingTrades = month.totalTrades - Math.round((month.winningPercentage / 100) * month.totalTrades);
        return sum + (month.avgDaysLoss * losingTrades);
      }, 0) / totalLosingTrades
      : 0;



    // Calculate average largest gain and loss - only include months with valid values
    const monthsWithGains = filteredStats.filter(month =>
      month.largestGain !== null && month.largestGain !== undefined && month.largestGain !== 0
    );
    const monthsWithLosses = filteredStats.filter(month =>
      month.largestLoss !== null && month.largestLoss !== undefined && month.largestLoss !== 0
    );

    const avgOfLargestGains = monthsWithGains.length > 0
      ? monthsWithGains.reduce((sum, month) => sum + month.largestGain, 0) / monthsWithGains.length
      : 0;

    const avgOfLargestLosses = monthsWithLosses.length > 0
      ? monthsWithLosses.reduce((sum, month) => sum + month.largestLoss, 0) / monthsWithLosses.length
      : 0;

    // Calculate compounded net profit/loss across all selected months
    const netProfitLoss = filteredStats.length > 0
      ? (filteredStats.reduce((acc, month) => acc * (1 + month.netProfitLoss / 100), 1) - 1) * 100
      : 0;

    // Sum of equity change % across all selected months
    const totalEquityChange = filteredStats.reduce((sum, month) => sum + month.equityChangePct, 0);

    return {
      averageGain,
      averageLoss,
      winningPercentage,
      totalTrades,
      avgOfLargestGains,
      avgOfLargestLosses,
      avgDaysGains,
      avgDaysLoss,
      netProfitLoss,
      totalEquityChange,
      monthCount: filteredStats.length
    };
  }, [monthlyStats]);

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/50">
            <TableHead className="w-12 text-center" title="Shift+click to select a range">Use In Statistics <span className="text-xs font-normal text-muted-foreground">(shift+click for range)</span></TableHead>
            <TableHead className="text-center">Trading Month</TableHead>
            <TableHead className="text-center">Average GAIN</TableHead>
            <TableHead className="text-center">Average LOSS</TableHead>
            <TableHead className="text-center">WINNING %</TableHead>
            <TableHead className="text-center">TOTAL TRADES</TableHead>
            <TableHead className="text-center">LG GAIN</TableHead>
            <TableHead className="text-center">LG LOSS</TableHead>
            <TableHead className="text-center">Avg Days Gains Held</TableHead>
            <TableHead className="text-center">Avg Days Loss Held</TableHead>
            <TableHead className="text-center">Adj. Risk:Reward Ratio</TableHead>
            <TableHead className="text-center">Equity Risked/Gained</TableHead>
            <TableHead className="text-center">Projected Ruin/Return (by Equity Chg)</TableHead>
            <TableHead className="text-center">Expected Growth/Trade</TableHead>
            <TableHead className="text-center">Compounded Total % Return</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {monthlyStats.map((month, idx) => (
            <TableRow
              key={month.tradingMonth}
              className={month.isInTrailingYear ? 'bg-muted/50' : ''}
            >
              <TableCell className="py-0">
                <div onClick={(e) => handleMonthClick(e, month, idx)} className="cursor-pointer">
                  <Checkbox
                    checked={month.useInYearly}
                    className="mt-1 pointer-events-none"
                  />
                </div>
              </TableCell>
              <TableCell className="py-1">{month.tradingMonth}</TableCell>
              <TableCell className="text-center py-1">{month.averageGain.toFixed(2)}%</TableCell>
              <TableCell className="text-center py-1">{month.averageLoss.toFixed(2)}%</TableCell>
              <TableCell className="text-center py-1">{month.winningPercentage.toFixed(2)}%</TableCell>
              <TableCell className="text-center py-1">{month.totalTrades}</TableCell>
              <TableCell className="text-center py-1">{month.largestGain.toFixed(2)}%</TableCell>
              <TableCell className="text-center py-1">{month.largestLoss.toFixed(2)}%</TableCell>
              <TableCell className="text-center py-1">{month.avgDaysGains.toFixed(1)}</TableCell>
              <TableCell className="text-center py-1">{month.avgDaysLoss.toFixed(1)}</TableCell>
              <TableCell className="text-center py-1">
                {(() => {
                  const winRate = month.winningPercentage / 100;
                  const lossRate = 1 - winRate;

                  if (winRate === 0 && month.averageLoss !== 0) {
                    return "0:1";
                  }

                  if (lossRate === 0 && month.averageGain !== 0) {
                    return "∞:1";
                  }

                  if (month.totalTrades === 0) {
                    return "N/A";
                  }

                  // Risk-reward ratio adjusted by win rate
                  // Formula: (Win Rate * Avg Gain) / (Loss Rate * |Avg Loss|)
                  // The Holy Grail
                  const adjustedRatio = (winRate * month.averageGain) / (lossRate * Math.abs(month.averageLoss));

                  return `${adjustedRatio.toFixed(2)}:1`;
                })()}
              </TableCell>
              <TableCell className="text-center py-1" style={{ color: month.equityChangePct === 0 ? undefined : (month.equityChangePct > 0 ? '#34d399' : '#f87171') }}>
                {month.equityChangePct !== 0 ? `${month.equityChangePct.toFixed(2)}%` : ""}
              </TableCell>
              <TableCell className="text-center py-1 whitespace-nowrap" style={{ color: month.totalTrades > 0 && month.equityChangePct !== 0 ? computeProjection(month.equityChangePct, targetReturnPercent).color : undefined }}>
                {month.totalTrades > 0 && month.equityChangePct !== 0 ? computeProjection(month.equityChangePct, targetReturnPercent).text : ""}
              </TableCell>
              <TableCell className="text-center py-1" style={{
                color: (() => {
                  const winRate = month.winningPercentage / 100;
                  const lossRate = 1 - winRate;
                  if (month.totalTrades === 0) return undefined;
                  const logGrowthRate = winRate * Math.log(1 + month.averageGain / 100) +
                    lossRate * Math.log(1 + month.averageLoss / 100);
                  const geometricExpectancy = (Math.exp(logGrowthRate) - 1) * 100;
                  return geometricExpectancy === 0 ? undefined : (geometricExpectancy > 0 ? '#34d399' : '#f87171');
                })()
              }}>
                {(() => {
                  const winRate = month.winningPercentage / 100;
                  const lossRate = 1 - winRate;

                  if (month.totalTrades === 0) {
                    return "N/A";
                  }

                  const logGrowthRate = winRate * Math.log(1 + month.averageGain / 100) +
                    lossRate * Math.log(1 + month.averageLoss / 100);
                  const geometricExpectancy = (Math.exp(logGrowthRate) - 1) * 100;

                  if (geometricExpectancy === 0) return "";
                  return `${geometricExpectancy.toFixed(2)}%`;
                })()}
              </TableCell>
              <TableCell className="text-center py-1" style={{ color: month.netProfitLoss === 0 ? undefined : (month.netProfitLoss > 0 ? '#34d399' : '#f87171') }}>
                {month.netProfitLoss !== 0 ? `${month.netProfitLoss.toFixed(2)}%` : ""}
              </TableCell>
            </TableRow>
          ))}

          {/* Summary Row */}
          <TableRow className="bg-muted">
            <TableCell className="py-1"></TableCell>
            <TableCell className="py-1">Summary</TableCell>
            <TableCell className="text-center py-1">{summaryStats.averageGain.toFixed(2)}%</TableCell>
            <TableCell className="text-center py-1">{summaryStats.averageLoss.toFixed(2)}%</TableCell>
            <TableCell className="text-center py-1">{summaryStats.winningPercentage.toFixed(2)}%</TableCell>
            <TableCell className="text-center py-1">{summaryStats.totalTrades}</TableCell>
            <TableCell className="text-center py-1">{summaryStats.avgOfLargestGains.toFixed(2)}%</TableCell>
            <TableCell className="text-center py-1">{summaryStats.avgOfLargestLosses.toFixed(2)}%</TableCell>
            <TableCell className="text-center py-1">{summaryStats.avgDaysGains.toFixed(1)}</TableCell>
            <TableCell className="text-center py-1">{summaryStats.avgDaysLoss.toFixed(1)}</TableCell>
            <TableCell className="text-center py-1">
              {(() => {
                const winRate = summaryStats.winningPercentage / 100;
                const lossRate = 1 - winRate;

                if (winRate === 0 && summaryStats.averageLoss !== 0) {
                  return "0:1";
                }

                if (lossRate === 0 && summaryStats.averageGain !== 0) {
                  return "∞:1";
                }

                if (summaryStats.totalTrades === 0) {
                  return "N/A";
                }

                const adjustedRatio = (winRate * summaryStats.averageGain) / (lossRate * Math.abs(summaryStats.averageLoss));

                return `${adjustedRatio.toFixed(2)}:1`;
              })()}
            </TableCell>
            <TableCell className="text-center py-1" style={{ color: summaryStats.totalEquityChange === 0 ? undefined : (summaryStats.totalEquityChange > 0 ? '#34d399' : '#f87171') }}>
              {summaryStats.totalEquityChange !== 0 ? `${summaryStats.totalEquityChange.toFixed(2)}%` : ""}
            </TableCell>
            <TableCell className="text-center py-1 whitespace-nowrap" style={{
              color: (() => {
                if (summaryStats.monthCount === 0 || summaryStats.totalEquityChange === 0) return undefined;
                // Geometric mean: (1 + total)^(1/n) - 1
                const avgMonthlyRate = (Math.pow(1 + summaryStats.totalEquityChange / 100, 1 / summaryStats.monthCount) - 1) * 100;
                return computeProjection(avgMonthlyRate, targetReturnPercent).color;
              })()
            }}>
              {(() => {
                if (summaryStats.monthCount === 0 || summaryStats.totalEquityChange === 0) return "";
                const avgMonthlyRate = (Math.pow(1 + summaryStats.totalEquityChange / 100, 1 / summaryStats.monthCount) - 1) * 100;
                return computeProjection(avgMonthlyRate, targetReturnPercent).text;
              })()}
            </TableCell>
            <TableCell className="text-center py-1" style={{
              color: (() => {
                const winRate = summaryStats.winningPercentage / 100;
                const lossRate = 1 - winRate;
                if (summaryStats.totalTrades === 0) return undefined;
                const logGrowthRate = winRate * Math.log(1 + summaryStats.averageGain / 100) +
                  lossRate * Math.log(1 + summaryStats.averageLoss / 100);
                const geometricExpectancy = (Math.exp(logGrowthRate) - 1) * 100;
                return geometricExpectancy === 0 ? undefined : (geometricExpectancy > 0 ? '#34d399' : '#f87171');
              })()
            }}>
              {(() => {
                const winRate = summaryStats.winningPercentage / 100;
                const lossRate = 1 - winRate;

                if (summaryStats.totalTrades === 0) {
                  return "N/A";
                }

                const logGrowthRate = winRate * Math.log(1 + summaryStats.averageGain / 100) +
                  lossRate * Math.log(1 + summaryStats.averageLoss / 100);
                const geometricExpectancy = (Math.exp(logGrowthRate) - 1) * 100;

                if (geometricExpectancy === 0) return "";
                return `${geometricExpectancy.toFixed(2)}%`;
              })()}
            </TableCell>
            <TableCell className="text-center py-1" style={{ color: summaryStats.netProfitLoss === 0 ? undefined : (summaryStats.netProfitLoss > 0 ? '#34d399' : '#f87171') }}>
              {summaryStats.netProfitLoss !== 0 ? `${summaryStats.netProfitLoss.toFixed(2)}%` : ""}
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </div>
  );
};