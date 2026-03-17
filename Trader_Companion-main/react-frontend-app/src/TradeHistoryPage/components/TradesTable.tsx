import React, { useState, useEffect, useMemo } from 'react';
import { addMonths, format, parseISO } from 'date-fns';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent } from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Trade } from '../types/Trade';
import type { UseCustomTradeDataReturn, ColumnDef } from '../hooks/useCustomTradeData';

// Fields to hide from the UI (kept in backend for compatibility)
const HIDDEN_FIELDS = new Set([
  'ID',  // Auto-generated, not shown in table
  'Price_Tightness_1_Week_Before',
  'Nr_Bases',
  'Has_Earnings_Acceleration',
  'IPO_Last_10_Years',
  'Is_BioTech',
  'Under_30M_Shares',
  'If_You_Could_Only_Make_10_Trades',
  'Pct_Off_52W_High',
]);

interface TradesTableProps {
  trades: Trade[];
  onUpdate: (updatedTrade: Trade) => void;
  onDelete: (id: number) => void;
  columnWidths?: { [key in keyof Trade]?: string };
  customTradeData: UseCustomTradeDataReturn;
}

export const TradesTable: React.FC<TradesTableProps> = ({
  trades,
  onUpdate,
  onDelete,
  columnWidths = {},
  customTradeData,
}) => {
  const { orderedColumns, getCustomValue, updateCustomValue } = customTradeData;
  const [editedTrades, setEditedTrades] = useState<{ [key: number]: Trade }>({});
  const [editedCustomValues, setEditedCustomValues] = useState<Record<string, string>>({});
  const [displayCount, setDisplayCount] = useState<number>(20);
  const [tradeToDelete, setTradeToDelete] = useState<Trade | null>(null);
  const [flashingTrades, setFlashingTrades] = useState<Set<number>>(new Set());
  const [expandedCell, setExpandedCell] = useState<string | null>(null); // "tradeId-field"

  const defaultColumnWidths: { [key: string]: string } = {
    Ticker: 'w-14',
    Status: 'w-24',
    Entry_Date: 'w-32',
    Exit_Date: 'w-32',
    Entry_Price: 'w-20',
    Exit_Price: 'w-20',
    Return: 'w-16',
    Pct_Of_Equity: 'w-20',
    Pattern: 'w-28',
    // Days_In_Pattern_Before_Entry: 'w-14',
    Price_Tightness_1_Week_Before: 'w-20',
    Exit_Reason: 'w-40',
    Market_Condition: 'w-20',
    Case: 'w-40',
    Category: 'w-20',
    // Earnings_Quality: 'w-16',
    // Fundamentals_Quality: 'w-20',
    Nr_Bases: 'w-10',
  };

  useEffect(() => {
    setEditedTrades(prev => {
      const newEditedTrades = { ...prev };

      trades.forEach(trade => {
        // Only add new trades that don't exist in editedTrades yet
        if (!prev[trade.ID]) {
          newEditedTrades[trade.ID] = { ...trade };
        }
      });

      return newEditedTrades;
    });
  }, [trades]);

  const handleInputChange = (tradeId: number, field: keyof Trade, value: Trade[keyof Trade]) => {
    setEditedTrades(prev => ({
      ...prev,
      [tradeId]: {
        ...prev[tradeId],
        [field]: value
      }
    }));
  };

  // Custom column value handlers
  const getEditedCustomValue = (tradeId: number, customColumnId: number): string => {
    const key = `${tradeId}_${customColumnId}`;
    if (key in editedCustomValues) return editedCustomValues[key];
    return getCustomValue(tradeId, customColumnId);
  };

  const handleCustomValueChange = (tradeId: number, customColumnId: number, value: string) => {
    const key = `${tradeId}_${customColumnId}`;
    setEditedCustomValues(prev => ({ ...prev, [key]: value }));
  };

  const handleUpdate = (trade: Trade) => {
    const updatedTrade = editedTrades[trade.ID];
    onUpdate(updatedTrade);

    // Also persist any edited custom column values for this trade
    const customEntries = Object.entries(editedCustomValues)
      .filter(([key]) => key.startsWith(`${trade.ID}_`));
    for (const [key, value] of customEntries) {
      const colId = parseInt(key.split('_')[1]);
      updateCustomValue(trade.ID, colId, value);
    }

    // Add to flashing set for visual feedback
    setFlashingTrades(prev => new Set(prev).add(trade.ID));
    setTimeout(() => {
      setFlashingTrades(prev => {
        const next = new Set(prev);
        next.delete(trade.ID);
        return next;
      });
    }, 1000);
  };

  const handleConfirmDelete = () => {
    if (tradeToDelete) {
      onDelete(tradeToDelete.ID);
      setTradeToDelete(null);
    }
  };

  const handleDisplayCountChange = (value: number) => {
    const newCount = Math.min(Math.max(1, value), trades.length);
    setDisplayCount(newCount);
  };

  const getColumnWidth = (field: keyof Trade) => {
    return columnWidths[field] || defaultColumnWidths[field] || 'w-40';
  };

  const getTradeRowClass = (trade: Trade) => {
    const baseClass = 'transition-colors duration-500 ';

    if (flashingTrades.has(trade.ID)) {
      return baseClass + 'bg-blue-500/20 dark:bg-blue-400/20';
    }

    if (!trade.Exit_Price) return baseClass + 'hover:bg-muted/50';
    const profitPercent = ((trade.Exit_Price - trade.Entry_Price) / trade.Entry_Price) * 100;

    if (profitPercent > 0) {
      return baseClass + 'bg-emerald-500/20 hover:bg-emerald-500/30 dark:bg-emerald-600/20 dark:hover:bg-emerald-600/35';
    }

    return baseClass + 'hover:bg-muted/50';
  };

  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [useDateFilter, setUseDateFilter] = useState<boolean>(false);

  // Initialize default dates based on "Latest 20 trades" rule
  useEffect(() => {
    if (trades.length > 0 && !startDate && !endDate) {
      // Sort to find the relevant range (descending)
      const sorted = [...trades].sort((a, b) => {
        const dateA = a.Entry_Date ? new Date(a.Entry_Date).getTime() : 0;
        const dateB = b.Entry_Date ? new Date(b.Entry_Date).getTime() : 0;
        return dateB - dateA;
      });

      // Get latest 20 (or fewer)
      const latestTrades = sorted.slice(0, 20);

      if (latestTrades.length > 0) {
        // Range covers these 20 trades
        // Start: Date of the oldest of these 20
        const oldestEntry = latestTrades[latestTrades.length - 1];
        const newestEntry = latestTrades[0];

        if (oldestEntry?.Entry_Date && newestEntry?.Entry_Date) {
          // Buffer: 1 month before oldest, 1 month after newest
          const minDate = new Date(oldestEntry.Entry_Date);
          const maxDate = new Date(newestEntry.Entry_Date);

          const start = addMonths(minDate, -1);
          const end = addMonths(maxDate, 1);

          setStartDate(format(start, 'yyyy-MM'));
          setEndDate(format(end, 'yyyy-MM'));
        }
      } else {
        // Fallback if dates missing or something
        const now = new Date();
        setStartDate(format(addMonths(now, -1), 'yyyy-MM'));
        setEndDate(format(addMonths(now, 1), 'yyyy-MM'));
      }
    }
  }, [trades, startDate, endDate]);

  const sortedTrades = useMemo(() => {
    let filtered = [...trades];

    if (useDateFilter) {
      // Month-based filtering
      let filterStart = '';
      let filterEnd = '';

      if (startDate) {
        // Start of selected month
        filterStart = `${startDate}-01`;
      }

      if (endDate) {
        // End of selected month (Start of next month)
        const endMonthStart = parseISO(`${endDate}-01`);
        const nextMonthStart = addMonths(endMonthStart, 1);
        filterEnd = format(nextMonthStart, 'yyyy-MM-dd');
      }

      filtered = filtered.filter(t => {
        if (!t.Entry_Date) return false;
        if (filterStart && t.Entry_Date < filterStart) return false;
        if (filterEnd && t.Entry_Date >= filterEnd) return false;
        return true;
      });
    }

    // Sort Descending
    return filtered.sort((a, b) => {
      const dateA = a.Entry_Date ? new Date(a.Entry_Date).getTime() : 0;
      const dateB = b.Entry_Date ? new Date(b.Entry_Date).getTime() : 0;
      return dateB - dateA;
    });
  }, [trades, useDateFilter, startDate, endDate]);

  const renderCell = (trade: Trade, field: keyof Trade, pixelWidth?: number, isTextarea?: boolean) => {
    const editedTrade = editedTrades[trade.ID];
    if (!editedTrade) return null;

    const value = editedTrade[field];
    const width = pixelWidth ? 'w-full' : getColumnWidth(field);

    if (field === 'ID') {
      return <span className={`${width}`}>{value}</span>;
    }

    if (typeof value === 'boolean') {
      return (
        <div className="flex items-center justify-center h-8">
          <Checkbox
            checked={value}
            onCheckedChange={(checked) =>
              handleInputChange(trade.ID, field, checked)
            }
            className="h-4 w-4"
          />
        </div>
      );
    }

    if (field === 'Entry_Date' || field === 'Exit_Date') {
      return (
        <Input
          type="date"
          value={value || ''}
          onChange={(e) => handleInputChange(trade.ID, field, e.target.value)}
          className={`${width} h-6 px-0.5 text-xs`}
        />
      );
    }



    if (typeof value === 'number') {
      return (
        <Input
          type="number"
          value={value ?? ''}
          onChange={(e) => handleInputChange(trade.ID, field, Number(e.target.value))}
          className={`${width} h-8 px-1`}
          step={field.includes('Price') || field === 'Pct_Of_Equity' || field === 'Return' ? '0.01' : '1'}
        />
      );
    }

    // Textarea columns: show as Input, click to expand into textarea
    if (isTextarea) {
      const cellKey = `${trade.ID}-${field}`;
      if (expandedCell === cellKey) {
        return (
          <div className="relative">
            <textarea
              autoFocus
              value={value || ''}
              onChange={(e) => handleInputChange(trade.ID, field, e.target.value)}
              onBlur={() => setExpandedCell(null)}
              className={`absolute -top-3 left-0 w-[400px] min-w-[100px] h-[240px] min-h-[100px] px-1.5 py-1 text-xs rounded-md border-2 border-primary bg-background shadow-xl placeholder:text-muted-foreground focus-visible:outline-none resize z-50`}
              style={{ animation: 'textarea-pop 100ms ease-out' }}
            />
          </div>
        );
      }
      return (
        <Input
          type="text"
          value={value || ''}
          onChange={(e) => handleInputChange(trade.ID, field, e.target.value)}
          onFocus={() => setExpandedCell(cellKey)}
          className={`${width} h-8 px-1`}
        />
      );
    }

    return (
      <Input
        type="text"
        value={value || ''}
        onChange={(e) => handleInputChange(trade.ID, field, e.target.value)}
        className={`${width} h-8 px-1`}
      />
    );
  };

  // Format Entry_Date to "Jan 25" style
  const formatTradeMonth = (entryDate: string | undefined): string => {
    if (!entryDate) return '-';
    try {
      const date = parseISO(entryDate);
      return format(date, 'MMM yy'); // e.g., "Jan 25", "Feb 24"
    } catch {
      return '-';
    }
  };

  // Use orderedColumns instead of deriving from trade keys
  const visibleColumns: ColumnDef[] = orderedColumns.filter(col => !col.isCustom ? !HIDDEN_FIELDS.has(col.key) : true);

  if (trades.length === 0) {
    return <div className="text-center py-4">No trades available</div>;
  }

  return (
    <div className="space-y-2">
      <Card className="py-0">
        <CardContent className="flex flex-col space-y-2 p-2">
          {/* Top Row: Latest Trades Slider */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="use-date-filter"
                checked={useDateFilter}
                onCheckedChange={(checked) => setUseDateFilter(checked === true)}
              />
              <label htmlFor="use-date-filter" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Filter by Date Range
              </label>
            </div>

            {!useDateFilter && (
              <>
                <div className="w-px h-4 bg-border mx-2" />
                <span className="text-xs font-medium">
                  Show latest trades:
                </span>
                <Slider
                  min={1}
                  max={trades.length}
                  value={[displayCount]}
                  onValueChange={([value]) => handleDisplayCountChange(value)}
                  className="w-48 py-0"
                />
                <Input
                  type="number"
                  value={displayCount}
                  onChange={(e) => handleDisplayCountChange(Number(e.target.value))}
                  className="w-16 h-6 text-xs"
                  min={1}
                  max={trades.length}
                />
                <span className="text-xs text-muted-foreground">
                  of {trades.length} trades
                </span>
              </>
            )}
          </div>

          {/* Bottom Row: Date Inputs (Visible only if checked) */}
          {useDateFilter && (
            <div className="flex items-center space-x-4 animate-in fade-in slide-in-from-top-1 duration-200">
              <div className="flex items-center gap-2">
                <label htmlFor="table-start-date" className="text-sm font-medium">From:</label>
                <input
                  id="table-start-date"
                  type="month"
                  className="flex h-10 w-40 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [color-scheme:dark]"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div className="flex items-center gap-2">
                <label htmlFor="table-end-date" className="text-sm font-medium">To:</label>
                <input
                  id="table-end-date"
                  type="month"
                  className="flex h-10 w-40 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [color-scheme:dark]"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
              <span className="text-xs text-muted-foreground">
                Showing {sortedTrades.length} trades
              </span>
            </div>
          )}
        </CardContent>
      </Card>
      <Card className="mt-2">
        <CardContent className="p-4">
          <h3 className="text-sm font-semibold mb-2">Considerations</h3>
          <ul className=" text-muted-foreground pl-4 space-y-1">
            <li>Only Trades with <strong>Status</strong> column set as <strong>Exited</strong> are used in statistics.</li>
            <li>All trades will be used for recommended risk calculation, even those without <strong>Exited</strong> status, it's the <strong>Return</strong> from all trades it is calculated with.</li>
          </ul>
        </CardContent>
      </Card>
      <div className="rounded-md border">
        <div>
          <Table>
            <TableHeader className="sticky top-0 bg-muted">
              <TableRow className="border-b">
                <TableHead className="w-32 py-0.5 px-0.5 text-xs text-center border-r">Actions</TableHead>
                <TableHead className="py-0.5 px-1 text-xs text-center border-r font-semibold whitespace-nowrap">Month</TableHead>
                {visibleColumns.map(col => (
                  <React.Fragment key={col.key}>
                    <TableHead
                      className={`py-0.5 px-0.5 border-r text-xs text-center`}
                      style={col.width ? { width: `${col.width}px`, minWidth: `${col.width}px`, maxWidth: `${col.width}px` } : undefined}
                    >
                      {col.label}
                    </TableHead>
                    {col.key === 'Return' && (
                      <TableHead className="w-16 py-0.5 px-0.5 border-r text-xs text-center font-semibold">Return %</TableHead>
                    )}
                  </React.Fragment>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {(useDateFilter ? sortedTrades : sortedTrades.slice(0, displayCount)).map(trade => (
                <TableRow
                  key={trade.ID}
                  className={`border-b ${getTradeRowClass(trade)}`}
                >
                  <TableCell className="p-0 w-32 border-r">
                    <div className="flex space-x-0.5">
                      <Button
                        onClick={() => handleUpdate(trade)}
                        variant={flashingTrades.has(trade.ID) ? "secondary" : "default"}
                        size="sm"
                        className={`h-6 text-xs px-1 w-[68px] transition-all duration-300 ${flashingTrades.has(trade.ID) ? 'bg-emerald-500 hover:bg-emerald-600 text-white' : ''}`}
                      >
                        {flashingTrades.has(trade.ID) ? '✓ Updated' : 'Update'}
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            onClick={() => setTradeToDelete(trade)}
                            variant="destructive"
                            size="sm"
                            className="h-6 text-xs px-1"
                          >
                            Remove
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Confirm Deletion</AlertDialogTitle>
                            <AlertDialogDescription>
                              Are you sure you want to remove the trade for {trade.Ticker}?
                              This action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel onClick={() => setTradeToDelete(null)}>
                              Cancel
                            </AlertDialogCancel>
                            <AlertDialogAction onClick={handleConfirmDelete}>
                              Remove Trade
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </TableCell>
                  <TableCell className="px-1 py-0 border-r text-center whitespace-nowrap">
                    <span className="text-xs text-muted-foreground">{formatTradeMonth(trade.Entry_Date)}</span>
                  </TableCell>
                  {visibleColumns.map(col => (
                    <React.Fragment key={col.key}>
                      {col.isCustom ? (
                        <TableCell
                          className="p-0 border-r"
                          style={col.width ? { width: `${col.width}px`, minWidth: `${col.width}px`, maxWidth: `${col.width}px` } : { width: '128px' }}
                        >
                          {col.isTextarea ? (() => {
                            const cellKey = `${trade.ID}-${col.key}`;
                            if (expandedCell === cellKey) {
                              return (
                                <div className="relative">
                                  <textarea
                                    autoFocus
                                    value={getEditedCustomValue(trade.ID, col.customColumnId!)}
                                    onChange={(e) => handleCustomValueChange(trade.ID, col.customColumnId!, e.target.value)}
                                    onBlur={() => setExpandedCell(null)}
                                    className="absolute -top-2 left-0 w-[400px] min-w-[100px] min-h-[240px] px-1.5 py-1 text-xs rounded-md border-2 border-primary bg-background shadow-xl placeholder:text-muted-foreground focus-visible:outline-none resize z-50"
                                    style={{ animation: 'textarea-pop 100ms ease-out' }}
                                  />
                                </div>
                              );
                            }
                            return (
                              <Input
                                type="text"
                                value={getEditedCustomValue(trade.ID, col.customColumnId!)}
                                onChange={(e) => handleCustomValueChange(trade.ID, col.customColumnId!, e.target.value)}
                                onFocus={() => setExpandedCell(cellKey)}
                                className="h-8 px-1"
                                style={col.width ? { width: `${col.width}px` } : { width: '128px' }}
                              />
                            );
                          })() : (
                            <Input
                              type="text"
                              value={getEditedCustomValue(trade.ID, col.customColumnId!)}
                              onChange={(e) => handleCustomValueChange(trade.ID, col.customColumnId!, e.target.value)}
                              className="h-8 px-1"
                              style={col.width ? { width: `${col.width}px` } : { width: '128px' }}
                            />
                          )}
                        </TableCell>
                      ) : (
                        <TableCell
                          className={`p-0 border-r ${!col.width ? getColumnWidth(col.key as keyof Trade) : ''}`}
                          style={col.width ? { width: `${col.width}px`, minWidth: `${col.width}px`, maxWidth: `${col.width}px` } : undefined}
                        >
                          {renderCell(trade, col.key as keyof Trade, col.width, col.isTextarea)}
                        </TableCell>
                      )}
                      {col.key === 'Return' && (
                        <TableCell className="p-1 border-r text-center w-16">
                          {trade.Exit_Price ? (
                            <span className={`text-xs font-semibold ${((trade.Exit_Price - trade.Entry_Price) / trade.Entry_Price) * 100 >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {(((trade.Exit_Price - trade.Entry_Price) / trade.Entry_Price) * 100).toFixed(2)}%
                            </span>
                          ) : (
                            <span className="text-xs text-muted-foreground">-</span>
                          )}
                        </TableCell>
                      )}
                    </React.Fragment>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>


    </div>
  );
};