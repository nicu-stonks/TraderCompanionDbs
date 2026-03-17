import React, { useState, useMemo, useEffect } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { format, parseISO, addMonths } from 'date-fns';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Checkbox } from '@/components/ui/checkbox';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface EquityCurveProps {
  filteredTrades: Trade[];
  initialBalance: number;
  useMonthFilter: boolean;
  onUseMonthFilterChange: (checked: boolean) => void;
}

type ChartType = 'balance' | 'pnl' | 'return';

export const EquityCurve: React.FC<EquityCurveProps> = ({ filteredTrades = [], initialBalance, useMonthFilter, onUseMonthFilterChange }) => {
  const [chartType, setChartType] = useState<ChartType>('return');
  const [internalStartDate, setInternalStartDate] = useState<string>('');
  const [internalEndDate, setInternalEndDate] = useState<string>('');

  // Default to showing all history when the component mounts
  useEffect(() => {
    if (filteredTrades.length > 0 && !internalStartDate && !internalEndDate) {
      const dates = filteredTrades
        .filter(t => t.Exit_Date)
        .map(t => new Date(t.Exit_Date!).getTime());

      if (dates.length > 0) {
        const minDate = new Date(Math.min(...dates));
        const maxDate = new Date(Math.max(...dates));

        // "Date before first trade" (approx 1 month before for month picker)
        const start = addMonths(minDate, -1);
        // "Date after last trade" (approx 1 month after)
        const end = addMonths(maxDate, 1);

        setInternalStartDate(format(start, 'yyyy-MM'));
        setInternalEndDate(format(end, 'yyyy-MM'));
      }
    }
  }, [filteredTrades, internalStartDate, internalEndDate]);

  const chartData = useMemo(() => {
    // Use filteredTrades which now contains All Global Dates 
    const sourceTrades = filteredTrades.length > 0 ? filteredTrades : [];
    if (sourceTrades.length === 0) return [];

    // 1. Sort trades chronologically
    const sortedTrades = [...sourceTrades]
      .filter(t => t.Status === 'Exited' && t.Exit_Date && t.Return !== null)
      .sort((a, b) => new Date(a.Exit_Date!).getTime() - new Date(b.Exit_Date!).getTime());

    if (sortedTrades.length === 0) return [];

    // 2. Determine "Start of Time" Balance for this specific set of trades
    // Based on user feedback and past task context, 'initialBalance' prop IS the Starting Capital.
    // So we do NOT back-calculate. We start FROM initialBalance.
    const startOfSetBalance = initialBalance;

    // 3. Build Full History Curve
    let currentBalance = startOfSetBalance;
    let currentPnL = 0;

    // Add an initial "Start" point at date before first trade? 
    // This helps visualize the start balance.
    // The first trade's exit date might be "2024-01-15".
    // We can insert a point "2024-01-14" with balance = Initial, PnL = 0.

    // For simplicity with the mapping, let's just map the trades. 
    // If we want a localized "Day before first trade" point, we can prepend it.

    const tradePoints = sortedTrades.map(trade => {
      const tradeReturn = trade.Return || 0;

      currentBalance += tradeReturn;
      currentPnL += tradeReturn;

      // Return % = (Current Balance - Initial) / Initial * 100
      const returnPct = startOfSetBalance !== 0
        ? ((currentBalance - startOfSetBalance) / startOfSetBalance) * 100
        : 0;

      return {
        date: trade.Exit_Date!.split('T')[0],
        fullDate: format(parseISO(trade.Exit_Date!), 'MMM d, yyyy'),
        balance: currentBalance,
        pnl: currentPnL,
        return: returnPct
      };
    });

    // Prepend start point if we have trades
    let fullHistory = tradePoints;
    if (sortedTrades.length > 0) {
      const firstTradeDate = parseISO(sortedTrades[0].Exit_Date!);
      // Use 1 day before first trade exit
      const starDateObj = new Date(firstTradeDate);
      starDateObj.setDate(starDateObj.getDate() - 1);

      const startPoint = {
        date: format(starDateObj, 'yyyy-MM-dd'),
        fullDate: format(starDateObj, 'MMM d, yyyy'),
        balance: startOfSetBalance,
        pnl: 0,
        return: 0
      };

      fullHistory = [startPoint, ...tradePoints];
    }

    // 4. Apply Local Date Filter (Month-based)
    // internalStartDate/EndDate are in 'yyyy-MM' format if set

    let filterStart = ''; // yyyy-MM-dd
    let filterEnd = '';   // yyyy-MM-dd (Exclusive upper bound)

    if (internalStartDate) {
      filterStart = `${internalStartDate}-01`;
    }

    if (internalEndDate) {
      // Logic: If user selects "2026-01", valid range includes all of Jan.
      // So valid < 2026-02-01.
      const endMonthStart = parseISO(`${internalEndDate}-01`);
      // User requested "To January 2026". 
      // This usually means "Through the end of January 2026".
      // addMonths(..., 1) gives Feb 1st. < Feb 1st includes Jan 31st. Correct.
      const nextMonthStart = addMonths(endMonthStart, 1);
      filterEnd = format(nextMonthStart, 'yyyy-MM-dd');
    }

    const filteredHistory = fullHistory.filter(pt => {
      if (filterStart && pt.date < filterStart) return false;
      if (filterEnd && pt.date >= filterEnd) return false;
      return true;
    });

    if (filteredHistory.length === 0) return [];

    // 5. Re-normalize for Relative Performance
    // Find the balance *just before* the filtered set starts to use as the baseline.
    let baselineBalance = startOfSetBalance;

    // Find the index of the first point in filteredHistory within the fullHistory
    // Since fullHistory is sorted and filteredHistory is a subset, we can find the first match.
    const firstVisibleIndex = fullHistory.findIndex(p => p.date === filteredHistory[0].date);

    if (firstVisibleIndex > 0) {
      // The baseline is the balance of the trade immediately preceding the window
      baselineBalance = fullHistory[firstVisibleIndex - 1].balance;
    }
    // If index is 0, baseline remains startOfSetBalance (Initial Balance)

    return filteredHistory.map(pt => ({
      ...pt,
      // Recalculate PnL and Return relative to the window start
      pnl: pt.balance - baselineBalance,
      return: baselineBalance !== 0
        ? ((pt.balance - baselineBalance) / baselineBalance) * 100
        : 0
    }));

  }, [filteredTrades, initialBalance, internalStartDate, internalEndDate]);

  // Gradient offsets
  const gradientOffset = () => {
    if (chartData.length === 0) return 0;

    // Balance chart doesn't use split gradient
    if (chartType === 'balance') return 0;

    const key = chartType;
    const dataMax = Math.max(...chartData.map((i) => i[key]));
    const dataMin = Math.min(...chartData.map((i) => i[key]));

    if (dataMax <= 0) return 0;
    if (dataMin >= 0) return 1;

    return dataMax / (dataMax - dataMin);
  };

  const off = gradientOffset();
  const greenColor = "#34d399";
  const redColor = "#f87171";

  const formatYAxis = (value: number) => {
    if (chartType === 'return') return `${value.toFixed(1)}%`;
    return `$${value.toLocaleString()}`;
  };

  return (
    <Card className="col-span-1 md:col-span-2 lg:col-span-3">
      <CardHeader>
        <div className="flex flex-col space-y-4 sm:space-y-0 sm:flex-row sm:items-center sm:justify-between">
          <div className="space-y-1.5">
            <CardTitle>Equity Curve</CardTitle>
            <p className="text-sm text-muted-foreground">
              Performance over time based on closed trades
            </p>
          </div>

          <div className="flex flex-col md:flex-row items-end md:items-center gap-2 sm:gap-4">
            <div className="flex items-center gap-2">
              <Checkbox
                id="use-month-filter-equity"
                checked={useMonthFilter}
                onCheckedChange={(checked) => onUseMonthFilterChange(checked === true)}
              />
              <label htmlFor="use-month-filter-equity" className="text-sm text-muted-foreground cursor-pointer whitespace-nowrap">
                Use selected months
              </label>
            </div>
            {/* Date Range Inputs (Month Picker) */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium whitespace-nowrap">Range:</span>
              <input
                type="month"
                className="flex h-10 w-40 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [color-scheme:dark]"
                value={internalStartDate}
                onChange={(e) => setInternalStartDate(e.target.value)}
                placeholder="Start Month"
                aria-label="Start Month"
              />
              <span className="text-muted-foreground">-</span>
              <input
                type="month"
                className="flex h-10 w-40 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [color-scheme:dark]"
                value={internalEndDate}
                onChange={(e) => setInternalEndDate(e.target.value)}
                placeholder="End Month"
                aria-label="End Month"
              />
            </div>

            <Select
              value={chartType}
              onValueChange={(value) => setChartType(value as ChartType)}
            >
              <SelectTrigger className="w-[180px] h-10 text-sm">
                <SelectValue placeholder="Select chart type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="return">Total Return (%)</SelectItem>
                <SelectItem value="pnl">Cumulative PnL ($)</SelectItem>
                <SelectItem value="balance">Account Balance ($)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[400px] w-full">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="splitColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={off} stopColor={greenColor} stopOpacity={1} />
                    <stop offset={off} stopColor={redColor} stopOpacity={1} />
                  </linearGradient>
                  <linearGradient id="splitColorFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={off} stopColor={greenColor} stopOpacity={0.3} />
                    <stop offset={off} stopColor={redColor} stopOpacity={0.3} />
                  </linearGradient>
                  <linearGradient id="balanceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8884d8" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--muted-foreground))" strokeOpacity={0.1} />
                <XAxis
                  dataKey="date"
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                  tickFormatter={(val) => {
                    if (!val) return '';
                    const date = parseISO(val);
                    return format(date, 'MMM yy');
                  }}
                  minTickGap={30}
                />
                <YAxis
                  domain={['auto', 'auto']}
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                  tickFormatter={formatYAxis}
                  width={60}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    borderColor: 'hsl(var(--border))',
                    borderRadius: 'var(--radius)',
                    color: 'hsl(var(--card-foreground))'
                  }}
                  itemStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    color: 'hsl(var(--card-foreground))'
                  }}
                  formatter={(value: number) => {
                    const val = Number(value);
                    const label = chartType === 'return' ? 'Total Return' :
                      chartType === 'pnl' ? 'Cumulative PnL' : 'Account Balance';
                    const formatted = chartType === 'return' ? `${val.toFixed(2)}%` : `$${val.toLocaleString()}`;
                    return [formatted, label];
                  }}
                  labelFormatter={(label) => {
                    if (!label) return '';
                    return format(parseISO(label), 'MMM d, yyyy');
                  }}
                />
                <Area
                  type="monotone"
                  dataKey={chartType}
                  stroke={chartType === 'balance' ? "#8884d8" : "url(#splitColor)"}
                  strokeWidth={2}
                  fill={chartType === 'balance' ? "url(#balanceGradient)" : "url(#splitColorFill)"}
                  activeDot={{ r: 4, fill: 'hsl(var(--primary))' }}
                />
                {/* Reference line at 0 for PnL/Return charts */}
                {chartType !== 'balance' && (
                  <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeOpacity={0.5} />
                )}
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center justify-center text-muted-foreground">
              No trade data available for the selected range.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
