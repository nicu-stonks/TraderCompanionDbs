import React, { useState, useMemo } from 'react';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import { Slider } from '@/components/ui/slider';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { format, parseISO } from 'date-fns';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceArea } from 'recharts';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface TradeDistributionProps {
  filteredTrades: Trade[];
  selectedMonths: Set<string>;
}

export const TradeDistribution: React.FC<TradeDistributionProps> = ({
  filteredTrades,
  selectedMonths
}) => {
  // Range sliders for distribution limits (in percentage)
  const [minRange, setMinRange] = useState(-30);
  const [maxRange, setMaxRange] = useState(90);
  // End interval slider for table (0-300%, default 30%)
  const [endIntervalMax, setEndIntervalMax] = useState(30);

  // Constants for distribution configuration
  const INTERVAL_SIZE = 2; // 2% intervals
  const MAX_ALLOWED_RANGE = 300;
  const MIN_ALLOWED_RANGE = -100;
  const END_INTERVAL_MAX_ALLOWED = 300;

  interface Interval {
    range: string;
    label: string;
    count: number;
    percentage: number;
    start: number;
    end: number;
    gainsCount: number;
    lossesCount: number;
    gainsPercent: number;
    lossesPercent: number;
    netProfit: number;
    returns: number[]; // Store individual returns for compounding
  }

  const distributionData = useMemo(() => {
    // Filter trades based on selected months
    const tradesInSelectedMonths = filteredTrades.filter(trade => {
      if (!trade.Entry_Date || trade.Status !== 'Exited' || trade.Exit_Price === null) return false;
      const month = format(parseISO(trade.Entry_Date), 'MMM yy');
      return selectedMonths.has(month);
    });

    // Calculate returns for each trade
    const returns = tradesInSelectedMonths.map(trade => {
      if (!trade.Exit_Price || !trade.Entry_Price) return 0;
      return ((trade.Exit_Price - trade.Entry_Price) / trade.Entry_Price) * 100;
    });

    // Create intervals from min to max range
    const intervals: Interval[] = [];

    // Calculate number of intervals
    const numIntervals = Math.ceil((maxRange - minRange) / INTERVAL_SIZE);

    // Initialize intervals
    for (let i = 0; i < numIntervals; i++) {
      const start = minRange + (i * INTERVAL_SIZE);
      const end = start + INTERVAL_SIZE;
      intervals.push({
        range: `${start}-${end}`,
        label: `${start}-${end}`,
        count: 0,
        percentage: 0,
        start,
        end,
        gainsCount: 0,
        lossesCount: 0,
        gainsPercent: 0,
        lossesPercent: 0,
        netProfit: 0,
        returns: []
      });
    }

    // Count trades in each interval and track gains/losses
    returns.forEach(returnValue => {
      if (returnValue < minRange || returnValue >= maxRange) return;

      // Find the correct interval
      const intervalIndex = Math.floor((returnValue - minRange) / INTERVAL_SIZE);
      if (intervalIndex >= 0 && intervalIndex < intervals.length) {
        intervals[intervalIndex].count++;
        intervals[intervalIndex].returns.push(returnValue);
        if (returnValue >= 0) {
          intervals[intervalIndex].gainsCount++;
        } else {
          intervals[intervalIndex].lossesCount++;
        }
      }
    });

    // Calculate percentages and compounded net profit
    const totalTrades = returns.length;
    if (totalTrades > 0) {
      intervals.forEach(interval => {
        interval.percentage = (interval.count / totalTrades) * 100;
        interval.gainsPercent = totalTrades > 0 ? (interval.gainsCount / totalTrades) * 100 : 0;
        interval.lossesPercent = totalTrades > 0 ? (interval.lossesCount / totalTrades) * 100 : 0;

        // Calculate compounded net profit: (1+r1) * (1+r2) * ... * (1+rN) - 1
        if (interval.returns.length > 0) {
          const compoundedReturn = interval.returns.reduce((acc, r) => acc * (1 + r / 100), 1);
          interval.netProfit = (compoundedReturn - 1) * 100;
        }
      });
    }

    return {
      intervals,
      totalTrades,
      maxCount: Math.max(...intervals.map(interval => interval.count), 1)
    };
  }, [filteredTrades, selectedMonths, minRange, maxRange]);

  // Separate data for the gains/losses table - uses absolute return values
  const tableData = useMemo(() => {
    // Filter trades based on selected months
    const tradesInSelectedMonths = filteredTrades.filter(trade => {
      if (!trade.Entry_Date || trade.Status !== 'Exited' || trade.Exit_Price === null) return false;
      const month = format(parseISO(trade.Entry_Date), 'MMM yy');
      return selectedMonths.has(month);
    });

    // Calculate returns for each trade
    const returns = tradesInSelectedMonths.map(trade => {
      if (!trade.Exit_Price || !trade.Entry_Price) return 0;
      return ((trade.Exit_Price - trade.Entry_Price) / trade.Entry_Price) * 100;
    });

    // Create intervals from 0 to endIntervalMax using absolute returns
    interface TableInterval {
      range: string;
      start: number;
      end: number;
      gainsCount: number;
      lossesCount: number;
      gainsPercent: number;
      lossesPercent: number;
      netProfit: number;
      allReturns: number[];
    }

    const numIntervals = Math.ceil(endIntervalMax / INTERVAL_SIZE);
    const intervals: TableInterval[] = [];

    for (let i = 0; i < numIntervals; i++) {
      const start = i * INTERVAL_SIZE;
      const end = start + INTERVAL_SIZE;
      intervals.push({
        range: `${start} - ${end}%`,
        start,
        end,
        gainsCount: 0,
        lossesCount: 0,
        gainsPercent: 0,
        lossesPercent: 0,
        netProfit: 0,
        allReturns: []
      });
    }

    const totalTrades = returns.length;

    // Place each trade in the correct interval based on ABSOLUTE return value
    returns.forEach(returnValue => {
      const absReturn = Math.abs(returnValue);
      if (absReturn >= endIntervalMax) return; // Outside range

      const intervalIndex = Math.floor(absReturn / INTERVAL_SIZE);
      if (intervalIndex >= 0 && intervalIndex < intervals.length) {
        intervals[intervalIndex].allReturns.push(returnValue);
        if (returnValue >= 0) {
          intervals[intervalIndex].gainsCount++;
        } else {
          intervals[intervalIndex].lossesCount++;
        }
      }
    });

    // Calculate percentages and compounded net profit
    if (totalTrades > 0) {
      intervals.forEach(interval => {
        interval.gainsPercent = (interval.gainsCount / totalTrades) * 100;
        interval.lossesPercent = (interval.lossesCount / totalTrades) * 100;

        // Calculate compounded net profit: (1+r1) * (1+r2) * ... * (1+rN) - 1
        if (interval.allReturns.length > 0) {
          const compoundedReturn = interval.allReturns.reduce((acc, r) => acc * (1 + r / 100), 1);
          interval.netProfit = (compoundedReturn - 1) * 100;
        }
      });
    }

    return intervals;
  }, [filteredTrades, selectedMonths, endIntervalMax, INTERVAL_SIZE]);

  // Custom tooltip for the bar chart
  interface TooltipProps {
    active?: boolean;
    payload?: Array<{
      payload: {
        range: string;
        count: number;
        percentage: number;
      };
    }>;
    label?: string;
  }

  const CustomTooltip: React.FC<TooltipProps> = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border rounded p-2 shadow-md">
          <p className="font-medium">{`${data.range}%`}</p>
          <p className="text-sm">{`${data.count} trades`}</p>
          <p className="text-sm">{`${data.percentage.toFixed(1)}% of total`}</p>
        </div>
      );
    }
    return null;
  };

  // Find the indices for the -2-0 and 0-2 ranges
  const zeroAreaIndices = useMemo(() => {
    const startIndex = distributionData.intervals.findIndex(
      interval => interval.start === -2 || (interval.start < -2 && interval.end > -2)
    );
    const endIndex = distributionData.intervals.findIndex(
      interval => interval.end > 2 && interval.start <= 2
    );
    return { startIndex, endIndex };
  }, [distributionData.intervals]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Trade Return Distribution(Selected Months)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Range controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm">Minimum Range: {minRange}%</span>
                <span className="text-sm text-muted-foreground">Min: {MIN_ALLOWED_RANGE}%</span>
              </div>
              <Slider
                value={[minRange]}
                min={MIN_ALLOWED_RANGE}
                max={maxRange - INTERVAL_SIZE} // Don't allow min to overlap max
                step={INTERVAL_SIZE}
                onValueChange={(values) => setMinRange(values[0])}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm">Maximum Range: {maxRange}%</span>
                <span className="text-sm text-muted-foreground">Max: {MAX_ALLOWED_RANGE}%</span>
              </div>
              <Slider
                value={[maxRange]}
                min={minRange + INTERVAL_SIZE} // Don't allow max to overlap min
                max={MAX_ALLOWED_RANGE}
                step={INTERVAL_SIZE}
                onValueChange={(values) => setMaxRange(values[0])}
              />
            </div>
          </div>

          {/* Distribution visualization using Recharts */}
          <div className="border rounded-md p-4">
            <div className="flex justify-between items-end mb-2">
              <span className="text-sm">Total Trades in Range: {distributionData.totalTrades}</span>
              <span className="text-sm text-muted-foreground">Max Count: {distributionData.maxCount}</span>
            </div>

            {/* Recharts Bar Chart */}
            <div className="h-64 w-full">
              {distributionData.totalTrades > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={distributionData.intervals}
                    margin={{ top: 5, right: 5, left: 5, bottom: 20 }}
                  >
                    <XAxis
                      dataKey="label"
                      angle={-45}
                      textAnchor="end"
                      height={60}
                      tick={{ fontSize: 10, fill: "#60656e" }}  // Added fill color for light gray
                      interval={0}
                    // Remove the stroke="currentColor" or change it if needed
                    />
                    <YAxis stroke="currentColor"
                      tick={{ fill: "#60656e" }} // Added fill color for light gray
                      interval={0} tickMargin={5} // Add some margin between the ticks and the axis line
                      width={40} // Adjust the width of the Y-axis to make space for the labels
                      tickLine={{ stroke: "#60656e" }} // Added stroke color for light gray
                      axisLine={{ stroke: "#60656e" }} // Added stroke color for light gray
                    />
                    <Tooltip content={<CustomTooltip />} />

                    {/* Reference area for -2 to 0 and 0 to 2 range */}
                    {zeroAreaIndices.startIndex !== -1 && zeroAreaIndices.endIndex !== -1 && (
                      <ReferenceArea
                        x1={distributionData.intervals[zeroAreaIndices.startIndex].label}
                        x2={distributionData.intervals[zeroAreaIndices.endIndex - 1].label}
                        fill="hsl(var(--primary))"
                        fillOpacity={0.2}
                      />
                    )}

                    <Bar
                      dataKey="count"
                      fill="hsl(var(--primary))"
                      minPointSize={2}
                      isAnimationActive={false}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                  No trades in the selected range
                </div>
              )}
            </div>
          </div>
          {/* say a title here saying it uses selcted months  */}
          <h3 className="text-lg font-semibold mb-4">Trades Compounded by percentage (Selected Months)</h3>

          {/* End Interval Slider */}
          <div className="space-y-2 mt-6">
            <div className="flex justify-between">
              <span className="text-sm">Table End Interval: {endIntervalMax}%</span>
              <span className="text-sm text-muted-foreground">Range: 0% - {END_INTERVAL_MAX_ALLOWED}%</span>
            </div>
            <Slider
              value={[endIntervalMax]}
              min={0}
              max={END_INTERVAL_MAX_ALLOWED}
              step={INTERVAL_SIZE}
              onValueChange={(values) => setEndIntervalMax(values[0])}
            />
          </div>

          {/* Distribution Table */}
          <div className="border rounded-md overflow-x-auto mt-4">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/50">
                  <TableHead className="text-center">Range</TableHead>
                  <TableHead className="text-center"># Gains</TableHead>
                  <TableHead className="text-center"># Losses</TableHead>
                  <TableHead className="text-center"><span style={{ color: '#34d399' }}>% ↑</span> %</TableHead>
                  <TableHead className="text-center"><span style={{ color: '#f87171' }}>% ↓</span> %</TableHead>
                  <TableHead className="text-center">Net</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {tableData.map((interval) => (
                  <TableRow key={interval.range}>
                    <TableCell className="text-center py-1">{interval.start} - {interval.end}%</TableCell>
                    <TableCell className="text-center py-1">{interval.gainsCount}</TableCell>
                    <TableCell className="text-center py-1">{interval.lossesCount}</TableCell>
                    <TableCell className="text-center py-1">
                      {interval.gainsCount > 0 ? `${interval.gainsPercent.toFixed(0)}%` : ""}
                    </TableCell>
                    <TableCell className="text-center py-1">
                      {interval.lossesCount > 0 ? `${interval.lossesPercent.toFixed(0)}%` : ""}
                    </TableCell>
                    <TableCell className="text-center py-1" style={{ color: interval.netProfit === 0 ? undefined : (interval.netProfit > 0 ? '#34d399' : '#f87171') }}>
                      {interval.netProfit !== 0 ? `${interval.netProfit.toFixed(2)}%` : ""}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};