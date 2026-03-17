import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Trade } from '../types/Trade';
import { stockPicksApi } from '@/StockRankingPage/services/stockPick';
import { globalCharacteristicsApi } from '@/StockRankingPage/services/globalCharacteristics';
import type { StockPick, GlobalCharacteristic } from '@/StockRankingPage/types';
import { Upload, Loader2 } from "lucide-react";
import type { UseCustomTradeDataReturn, ColumnDef } from '../hooks/useCustomTradeData';

interface IBKRImportComponentProps {
  onAdd: (trade: Trade) => Promise<void>;
  existingTrades?: Trade[];
  customTradeData: UseCustomTradeDataReturn;
  currentBalance?: number;
}

interface ParsedTransaction {
  symbol: string;
  dateTime: Date;
  quantity: number;
  price: number;
  proceeds: number;
  commission: number;
}

interface AggregatedTrade {
  ticker: string;
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  returnPct: number;
  returnDollar: number;  // Actual dollar return from proceeds
  totalBuyValue: number;  // Total cost basis at entry (for deposit-adjusted recalculation)
  pctOfEquity: number;
  status: string;
  pattern: string;
  marketCondition: string;
  category: string;
  case: string;
  tradeNote: string;  // Maps to Exit_Reason in database
  // CANSLIM checkboxes
  C: boolean;
  A: boolean;
  N: boolean;
  S: boolean;
  L: boolean;
  I: boolean;
  M: boolean;
  // Track if case is loading
  caseLoading?: boolean;
  // Track if added
  added?: boolean;
  // Custom column values keyed by column id
  customValues?: Record<number, string>;
}

interface DepositEntry {
  id: number;
  date: string;   // YYYY-MM-DD
  amount: number;
}

export const IBKRImportComponent: React.FC<IBKRImportComponentProps> = ({ onAdd, existingTrades = [], customTradeData, currentBalance }) => {
  const [csvContent, setCsvContent] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');
  const [importedFileNames, setImportedFileNames] = useState<string[]>([]);
  const [parsedTrades, setParsedTrades] = useState<AggregatedTrade[]>([]);
  const [rawTransactions, setRawTransactions] = useState<ParsedTransaction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [addingTrades, setAddingTrades] = useState<Set<string>>(new Set());
  const [showAllTrades, setShowAllTrades] = useState(false);
  const [expandedCell, setExpandedCell] = useState<string | null>(null); // "index-field"
  const [deposits, setDeposits] = useState<DepositEntry[]>([]);
  const [depositDate, setDepositDate] = useState<string>('');
  const [depositAmount, setDepositAmount] = useState<string>('');

  // Aggregation options - 2 simple toggles
  // Price: weighted avg (default) OR first buy / last sell
  const [useWeightedAvgPrices, setUseWeightedAvgPrices] = useState(true);
  // Dates: first buy / last sell (default) OR average of dates
  const [useFirstLastDates, setUseFirstLastDates] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const appendFileInputRef = useRef<HTMLInputElement>(null);
  const globalCharsRef = useRef<GlobalCharacteristic[] | null>(null);
  const fetchedStockCacheRef = useRef<Record<string, StockPick | null>>({});

  // Recalculate pct of equity for all parsed trades accounting for deposits
  const recalculatePctOfEquity = (trades: AggregatedTrade[]): AggregatedTrade[] => {
    return trades.map(trade => {
      let pctOfEquity = 0;
      if (currentBalance !== undefined && currentBalance > 0) {
        const priorTrades = existingTrades.filter(t =>
          t.Status === 'Exited' && t.Exit_Date && t.Entry_Date < trade.entryDate
        );
        const realizedPnL = priorTrades.reduce((sum, t) => sum + (t.Return || 0), 0);
        const depositsAfterEntry = deposits
          .filter(d => d.date > trade.entryDate)
          .reduce((sum, d) => sum + d.amount, 0);
        const equityAtEntry = currentBalance + realizedPnL - depositsAfterEntry;
        if (equityAtEntry > 0) {
          pctOfEquity = trade.totalBuyValue / equityAtEntry;
          if (pctOfEquity > 1) pctOfEquity = 1;
        }
      }
      return { ...trade, pctOfEquity: Math.round(pctOfEquity * 1000) / 1000 };
    });
  };

  // Re-run pct of equity whenever deposits or currentBalance changes
  React.useEffect(() => {
    if (parsedTrades.length > 0) {
      setParsedTrades(prev => recalculatePctOfEquity(prev));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deposits, currentBalance]);

  // Suggestions for Pattern, Market_Condition, Category
  const patternSuggestions = Array.from(new Set(existingTrades.map(t => t.Pattern).filter(Boolean)));
  const marketConditionSuggestions = Array.from(new Set(existingTrades.map(t => t.Market_Condition).filter(Boolean)));
  const categorySuggestions = Array.from(new Set(existingTrades.map(t => t.Category).filter(Boolean)));

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setCsvContent(content);
      setError(null);
    };
    reader.onerror = () => {
      setError('Failed to read file');
    };
    reader.readAsText(file);
  };

  // Load global characteristics once
  const ensureGlobalCharacteristics = async () => {
    if (!globalCharsRef.current) {
      try {
        const resp = await globalCharacteristicsApi.getAllGlobalCharacteristics();
        globalCharsRef.current = resp.data;
      } catch {
        globalCharsRef.current = [];
      }
    }
    return globalCharsRef.current || [];
  };

  // Build JSON for a stock pick (same as AddTradeComponent)
  const buildCaseJson = async (stock: StockPick): Promise<string> => {
    const globalChars = await ensureGlobalCharacteristics();
    const selectedIds = new Set(stock.characteristics.map(c => c.characteristic_id));
    const characteristicsStatus: Record<string, boolean> = {};
    globalChars.forEach(gc => {
      characteristicsStatus[gc.name] = selectedIds.has(gc.id);
    });
    const data = {
      symbol: stock.symbol,
      total_score: stock.total_score,
      personal_opinion_score: stock.personal_opinion_score,
      details: stock.case_text || '',
      demand_reason: stock.demand_reason || '',
      note: stock.note || '',
      characteristics: characteristicsStatus
    };
    return JSON.stringify(data, null, 2);
  };

  // Fetch case for a ticker
  const fetchCaseForTicker = async (ticker: string): Promise<string> => {
    const upperTicker = ticker.toUpperCase();
    try {
      if (!(upperTicker in fetchedStockCacheRef.current)) {
        const resp = await stockPicksApi.getAllStockPicks();
        const all: StockPick[] = resp.data;
        const found = all.find(sp => sp.symbol.toUpperCase() === upperTicker) || null;
        fetchedStockCacheRef.current[upperTicker] = found;
      }
      const stock = fetchedStockCacheRef.current[upperTicker];
      if (!stock) return '';
      return await buildCaseJson(stock);
    } catch {
      return '';
    }
  };

  const parseCSV = (content: string): ParsedTransaction[] => {
    const lines = content.split('\n');
    const transactions: ParsedTransaction[] = [];

    for (const line of lines) {
      // Match pattern: Trades,Data,Order,Stocks,USD,...
      if (!line.startsWith('Trades,Data,Order,Stocks,USD,')) continue;

      // Parse CSV with quoted fields handling
      const parts: string[] = [];
      let current = '';
      let inQuotes = false;

      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          parts.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      parts.push(current.trim());

      // Expected format:
      // 0: Trades, 1: Data, 2: Order, 3: Stocks, 4: USD
      // 5: Symbol, 6: Date/Time, 7: Quantity, 8: T. Price, 9: C. Price
      // 10: Proceeds, 11: Comm/Fee, ...

      if (parts.length < 12) continue;

      const symbol = parts[5];
      const dateTimeStr = parts[6]; // "YYYY-MM-DD, HH:MM:SS"
      const quantity = parseFloat(parts[7]);
      const price = parseFloat(parts[8]);
      const proceeds = parseFloat(parts[10]);
      const commission = parseFloat(parts[11]);

      if (!symbol || isNaN(quantity) || isNaN(price)) continue;

      // Parse date
      const dateMatch = dateTimeStr.match(/(\d{4}-\d{2}-\d{2})/);
      if (!dateMatch) continue;

      const dateTime = new Date(dateMatch[1]);

      transactions.push({
        symbol,
        dateTime,
        quantity,
        price,
        proceeds,
        commission
      });
    }

    return transactions;
  };

  // Tolerance for considering a position as closed (accounts for floating-point rounding)
  const POSITION_CLOSE_TOLERANCE = 0.001;

  // FIFO-based trade matching - detects multiple round-trip trades per ticker
  const detectRoundTripTrades = (transactions: ParsedTransaction[]): { symbol: string, buys: ParsedTransaction[], sells: ParsedTransaction[] }[] => {
    // Group transactions by symbol
    const bySymbol = new Map<string, ParsedTransaction[]>();
    for (const tx of transactions) {
      if (!bySymbol.has(tx.symbol)) {
        bySymbol.set(tx.symbol, []);
      }
      bySymbol.get(tx.symbol)!.push(tx);
    }

    const allTrades: { symbol: string, buys: ParsedTransaction[], sells: ParsedTransaction[] }[] = [];

    for (const [symbol, txs] of bySymbol) {
      // Sort by date chronologically
      txs.sort((a, b) => a.dateTime.getTime() - b.dateTime.getTime());

      let position = 0;  // Running share count
      let currentBuys: ParsedTransaction[] = [];
      let currentSells: ParsedTransaction[] = [];

      for (const tx of txs) {
        position += tx.quantity;  // Buys are positive, sells are negative

        if (tx.quantity > 0) {
          currentBuys.push(tx);
        } else {
          currentSells.push(tx);
        }

        // Check if position is closed (within tolerance)
        // Only close if we had a position before and now we're at ~0
        if (Math.abs(position) <= POSITION_CLOSE_TOLERANCE && currentBuys.length > 0 && currentSells.length > 0) {
          // Complete round-trip trade detected
          allTrades.push({
            symbol,
            buys: [...currentBuys],
            sells: [...currentSells]
          });

          // Reset for next trade
          position = 0;  // Reset to exactly 0 to avoid error accumulation
          currentBuys = [];
          currentSells = [];
        }
      }

      // Leftover transactions where position wasn't fully closed are intentionally
      // skipped — only fully closed round-trip trades should be shown.
    }

    return allTrades;
  };

  const calculateAggregatedTrade = async (
    symbol: string,
    buys: ParsedTransaction[],
    sells: ParsedTransaction[]
  ): Promise<AggregatedTrade | null> => {
    // Need at least one buy and one sell for a completed trade
    if (buys.length === 0 || sells.length === 0) return null;

    // Sort by date
    buys.sort((a, b) => a.dateTime.getTime() - b.dateTime.getTime());
    sells.sort((a, b) => a.dateTime.getTime() - b.dateTime.getTime());

    // DATES
    let entryDate: string;
    let exitDate: string;

    if (useFirstLastDates) {
      // First buy date, Last sell date (default)
      entryDate = buys[0].dateTime.toISOString().split('T')[0];
      exitDate = sells[sells.length - 1].dateTime.toISOString().split('T')[0];
    } else {
      // Average of all dates
      const avgBuyTime = buys.reduce((sum, b) => sum + b.dateTime.getTime(), 0) / buys.length;
      const avgSellTime = sells.reduce((sum, s) => sum + s.dateTime.getTime(), 0) / sells.length;
      entryDate = new Date(avgBuyTime).toISOString().split('T')[0];
      exitDate = new Date(avgSellTime).toISOString().split('T')[0];
    }

    // PRICES
    let entryPrice: number;
    let exitPrice: number;

    if (useWeightedAvgPrices) {
      // Weighted average prices (default)
      let totalBuyValue = 0;
      let totalBuyQuantity = 0;
      for (const buy of buys) {
        totalBuyValue += buy.price * buy.quantity;
        totalBuyQuantity += buy.quantity;
      }
      entryPrice = totalBuyQuantity > 0 ? totalBuyValue / totalBuyQuantity : 0;

      let totalSellValue = 0;
      let totalSellQuantity = 0;
      for (const sell of sells) {
        const qty = Math.abs(sell.quantity);
        totalSellValue += sell.price * qty;
        totalSellQuantity += qty;
      }
      exitPrice = totalSellQuantity > 0 ? totalSellValue / totalSellQuantity : 0;
    } else {
      // First buy price, Last sell price
      entryPrice = buys[0].price;
      exitPrice = sells[sells.length - 1].price;
    }

    // Calculate return percentage
    const returnPct = entryPrice > 0 ? ((exitPrice - entryPrice) / entryPrice) * 100 : 0;

    // Calculate actual dollar return from proceeds
    let totalProceeds = 0;
    let totalCommission = 0;
    for (const buy of buys) {
      totalProceeds += buy.proceeds;  // Negative for buys (money spent)
      totalCommission += Math.abs(buy.commission);
    }
    for (const sell of sells) {
      totalProceeds += sell.proceeds;  // Positive for sells (money received)
      totalCommission += Math.abs(sell.commission);
    }
    const returnDollar = totalProceeds - totalCommission;

    // Calculate total buy value (stored for deposit-adjusted recalculation later)
    let totalBuyValue = 0;
    for (const buy of buys) {
      totalBuyValue += buy.price * buy.quantity;
    }

    // Calculate Pct_Of_Equity
    let pctOfEquity = 0;
    if (existingTrades && currentBalance !== undefined) {
      // Find realized P&L from trades that exited BEFORE this trade's entry
      const priorTrades = existingTrades.filter(t => t.Status === 'Exited' && t.Exit_Date && t.Entry_Date < entryDate);
      const realizedPnL = priorTrades.reduce((sum, t) => sum + (t.Return || 0), 0);
      // Subtract deposits made AFTER this trade's entry (they weren't part of equity at entry)
      const depositsAfterEntry = deposits
        .filter(d => d.date > entryDate)
        .reduce((sum, d) => sum + d.amount, 0);
      const equityAtEntry = currentBalance + realizedPnL - depositsAfterEntry;

      if (equityAtEntry > 0) {
        pctOfEquity = totalBuyValue / equityAtEntry;
        if (pctOfEquity > 1) pctOfEquity = 1; // Cap at 1
      }
    }

    return {
      ticker: symbol,
      entryDate,
      exitDate,
      entryPrice: Math.round(entryPrice * 100) / 100,
      exitPrice: Math.round(exitPrice * 100) / 100,
      returnPct: Math.round(returnPct * 100) / 100,
      returnDollar: Math.round(returnDollar * 100) / 100,
      totalBuyValue: Math.round(totalBuyValue * 100) / 100,
      pctOfEquity: Math.round(pctOfEquity * 1000) / 1000,
      status: 'Exited',
      pattern: '',
      marketCondition: '',
      category: '',
      case: '',
      tradeNote: '',
      C: false,
      A: false,
      N: false,
      S: false,
      L: false,
      I: false,
      M: false,
      caseLoading: true,
      customValues: {}
    };
  };

  // Create unique key for transaction deduplication
  const getTransactionKey = (tx: ParsedTransaction): string => {
    return `${tx.symbol}|${tx.dateTime.toISOString().split('T')[0]}|${tx.quantity}|${tx.price.toFixed(4)}`;
  };

  // Re-aggregate trades from current raw transactions using FIFO matching
  const reAggregateTrades = async (transactions: ParsedTransaction[]) => {
    const roundTripTrades = detectRoundTripTrades(transactions);
    const trades: AggregatedTrade[] = [];

    // Track how many trades per symbol for labeling (e.g., "AAPL #2")
    const symbolTradeCount = new Map<string, number>();

    for (const { symbol, buys, sells } of roundTripTrades) {
      const tradeIndex = symbolTradeCount.get(symbol) || 0;
      symbolTradeCount.set(symbol, tradeIndex + 1);

      const trade = await calculateAggregatedTrade(symbol, buys, sells);
      if (trade) {
        trades.push(trade);
      }
    }

    // Sort by entry date descending
    trades.sort((a, b) => new Date(b.entryDate).getTime() - new Date(a.entryDate).getTime());

    setParsedTrades(trades);

    // Fetch cases in background (use base symbol without #N suffix)
    for (let i = 0; i < trades.length; i++) {
      const baseTicker = trades[i].ticker.split(' #')[0];  // Remove "#2" suffix if present
      fetchCaseForTicker(baseTicker).then(caseJson => {
        setParsedTrades(prev => prev.map((t, idx) =>
          idx === i ? { ...t, case: caseJson || t.case, caseLoading: false } : t
        ));
      });
    }
  };

  const handleProcessCSV = async () => {
    setError(null);
    setIsProcessing(true);

    try {
      const transactions = parseCSV(csvContent);

      if (transactions.length === 0) {
        setError('No valid trade transactions found in the CSV. Make sure the format matches IBKR statement format.');
        setIsProcessing(false);
        return;
      }

      // Store raw transactions for "Show All Trades" view
      setRawTransactions(transactions);
      setImportedFileNames([fileName]);

      await reAggregateTrades(transactions);

      if (parsedTrades.length === 0 && transactions.length > 0) {
        // Check after aggregation
        const roundTrips = detectRoundTripTrades(transactions);
        if (roundTrips.length === 0) {
          setError('No completed trades found (need both buy and sell transactions for each ticker).');
        }
      }

    } catch (err) {
      setError('Error processing CSV: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAppendFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
      const content = event.target?.result as string;
      setError(null);
      setIsProcessing(true);

      try {
        const newTransactions = parseCSV(content);

        if (newTransactions.length === 0) {
          setError('No valid trade transactions found in the appended CSV.');
          setIsProcessing(false);
          return;
        }

        // Create set of existing transaction keys for deduplication
        const existingKeys = new Set(rawTransactions.map(getTransactionKey));

        // Filter out duplicates
        const uniqueNewTransactions = newTransactions.filter(tx => !existingKeys.has(getTransactionKey(tx)));

        const duplicatesFound = newTransactions.length - uniqueNewTransactions.length;

        if (uniqueNewTransactions.length === 0) {
          setError(`All ${newTransactions.length} transactions from this file were duplicates and skipped.`);
          setIsProcessing(false);
          return;
        }

        // Merge transactions
        const mergedTransactions = [...rawTransactions, ...uniqueNewTransactions];
        setRawTransactions(mergedTransactions);
        setImportedFileNames(prev => [...prev, file.name]);

        // Re-aggregate
        await reAggregateTrades(mergedTransactions);

        // Show info about what was added
        if (duplicatesFound > 0) {
          setError(`Added ${uniqueNewTransactions.length} new transactions. Skipped ${duplicatesFound} duplicates.`);
        }

      } catch (err) {
        setError('Error processing appended CSV: ' + (err instanceof Error ? err.message : 'Unknown error'));
      } finally {
        setIsProcessing(false);
        // Reset the file input so the same file can be selected again if needed
        if (appendFileInputRef.current) {
          appendFileInputRef.current.value = '';
        }
      }
    };
    reader.onerror = () => {
      setError('Failed to read appended file');
    };
    reader.readAsText(file);
  };

  const handleTradeFieldChange = (index: number, field: keyof AggregatedTrade, value: string | number | boolean) => {
    setParsedTrades(prev => prev.map((trade, i) =>
      i === index ? { ...trade, [field]: value } : trade
    ));
  };

  const handleAddTrade = async (index: number) => {
    const trade = parsedTrades[index];
    const tradeKey = `${trade.ticker}-${index}`;

    setAddingTrades(prev => new Set(prev).add(tradeKey));

    try {
      const newTrade: Trade = {
        ID: 0, // Will be auto-generated
        Ticker: trade.ticker,
        Status: trade.status,
        Entry_Date: trade.entryDate,
        Exit_Date: trade.exitDate,
        Entry_Price: trade.entryPrice,
        Exit_Price: trade.exitPrice,
        Return: trade.returnDollar,
        Pattern: trade.pattern,
        Price_Tightness_1_Week_Before: 0,
        Exit_Reason: trade.tradeNote || '',
        Market_Condition: trade.marketCondition,
        Category: trade.category,
        Nr_Bases: 0,
        Has_Earnings_Acceleration: false,
        IPO_Last_10_Years: false,
        Is_BioTech: false,
        Under_30M_Shares: false,
        Case: trade.case,
        If_You_Could_Only_Make_10_Trades: false,
        Pct_Off_52W_High: 0,
        Pct_Of_Equity: trade.pctOfEquity,
        C: trade.C,
        A: trade.A,
        N: trade.N,
        S: trade.S,
        L: trade.L,
        I: trade.I,
        M: trade.M
      };

      await onAdd(newTrade);

      // Save custom column values for the newly added trade
      if (trade.customValues && Object.keys(trade.customValues).length > 0) {
        const existingIds = existingTrades.map(t => t.ID);
        const maxExistingId = existingIds.length > 0 ? Math.max(...existingIds) : 0;
        const newTradeId = maxExistingId + 1;
        for (const [colIdStr, value] of Object.entries(trade.customValues)) {
          if (value.trim()) {
            await customTradeData.updateCustomValue(newTradeId, parseInt(colIdStr), value);
          }
        }
      }

      // Trade added successfully - button stays as "Add" so user can re-add if needed
    } catch (err) {
      setError(`Failed to add ${trade.ticker}: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setAddingTrades(prev => {
        const newSet = new Set(prev);
        newSet.delete(tradeKey);
        return newSet;
      });
    }
  };

  const getRowClass = (trade: AggregatedTrade) => {
    if (trade.added) return 'bg-emerald-500/30';
    if (trade.returnPct > 0) return 'bg-emerald-500/10';
    if (trade.returnPct < 0) return 'bg-red-500/10';
    return '';
  };

  // Fields to hide from the IBKR import table (same as TradesTable)
  const HIDDEN_FIELDS = new Set([
    'ID', 'Price_Tightness_1_Week_Before', 'Nr_Bases',
    'Has_Earnings_Acceleration', 'IPO_Last_10_Years', 'Is_BioTech',
    'Under_30M_Shares', 'If_You_Could_Only_Make_10_Trades', 'Pct_Off_52W_High',
  ]);

  // Map ColumnDef.key (Trade field names) to AggregatedTrade field names
  const FIELD_MAP: Record<string, keyof AggregatedTrade> = {
    'Ticker': 'ticker', 'Status': 'status', 'Entry_Date': 'entryDate',
    'Exit_Date': 'exitDate', 'Entry_Price': 'entryPrice', 'Exit_Price': 'exitPrice',
    'Return': 'returnDollar', 'Pct_Of_Equity': 'pctOfEquity', 'Pattern': 'pattern', 'Exit_Reason': 'tradeNote',
    'Market_Condition': 'marketCondition', 'Category': 'category', 'Case': 'case',
    'C': 'C', 'A': 'A', 'N': 'N', 'S': 'S', 'L': 'L', 'I': 'I', 'M': 'M',
  };

  // Datalist mapping for suggestion fields
  const DATALIST_MAP: Record<string, { id: string; items: string[] }> = {
    'Pattern': { id: 'pattern-suggestions-import', items: patternSuggestions },
    'Market_Condition': { id: 'market-condition-suggestions-import', items: marketConditionSuggestions },
    'Category': { id: 'category-suggestions-import', items: categorySuggestions },
  };

  const { orderedColumns } = customTradeData;
  const visibleColumns: ColumnDef[] = orderedColumns.filter(col => !col.isCustom ? !HIDDEN_FIELDS.has(col.key) : true);

  // Default column widths in pixels (matching TradesTable's Tailwind classes)
  const DEFAULT_COL_WIDTHS: Record<string, number> = {
    Ticker: 56, Status: 96, Entry_Date: 128, Exit_Date: 128,
    Entry_Price: 80, Exit_Price: 80, Return: 80,
    Pattern: 112, Exit_Reason: 160, Market_Condition: 96,
    Category: 96, Case: 160,
    C: 32, A: 32, N: 32, S: 32, L: 32, I: 32, M: 32,
  };

  // Get style for a column cell — uses Column Manager width if set, otherwise defaults
  const getColStyle = (col: ColumnDef): React.CSSProperties | undefined => {
    if (col.width) return { width: `${col.width}px`, minWidth: `${col.width}px`, maxWidth: `${col.width}px` };
    const defaultW = col.isCustom ? 112 : DEFAULT_COL_WIDTHS[col.key];
    if (defaultW) return { minWidth: `${defaultW}px` };
    return undefined;
  };

  // Render a cell for a given column in the import table
  const renderImportCell = (trade: AggregatedTrade, index: number, col: ColumnDef) => {
    // Custom columns
    if (col.isCustom) {
      const colId = col.customColumnId!;
      if (col.isTextarea) {
        const cellKey = `${index}-custom_${colId}`;
        if (expandedCell === cellKey) {
          return (
            <textarea
              autoFocus
              value={trade.customValues?.[colId] || ''}
              onChange={(e) => {
                const newCustomValues = { ...(trade.customValues || {}), [colId]: e.target.value };
                handleTradeFieldChange(index, 'customValues' as keyof AggregatedTrade, newCustomValues as never);
              }}
              onBlur={() => setExpandedCell(null)}
              className="w-[300px] h-[120px] px-1 py-0.5 text-xs rounded-md border border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 resize-y z-50 relative"
              placeholder={col.label}
            />
          );
        }
        return (
          <Input
            value={trade.customValues?.[colId] || ''}
            onChange={(e) => {
              const newCustomValues = { ...(trade.customValues || {}), [colId]: e.target.value };
              handleTradeFieldChange(index, 'customValues' as keyof AggregatedTrade, newCustomValues as never);
            }}
            onFocus={() => setExpandedCell(cellKey)}
            className="h-6 w-full px-1 text-xs"
            placeholder={col.label}
          />
        );
      }
      return (
        <Input
          value={trade.customValues?.[colId] || ''}
          onChange={(e) => {
            const newCustomValues = { ...(trade.customValues || {}), [colId]: e.target.value };
            handleTradeFieldChange(index, 'customValues' as keyof AggregatedTrade, newCustomValues as never);
          }}
          className="h-6 w-full px-1 text-xs"
          placeholder={col.label}
        />
      );
    }

    // Default columns
    const field = FIELD_MAP[col.key];
    if (!field) return null;

    // Boolean (CANSLIM) checkboxes
    if (typeof trade[field] === 'boolean') {
      return (
        <Checkbox
          checked={trade[field] as boolean}
          onCheckedChange={(checked) => handleTradeFieldChange(index, field, !!checked)}
          className="h-4 w-4"
        />
      );
    }

    // Date fields
    if (col.key === 'Entry_Date' || col.key === 'Exit_Date') {
      return (
        <Input
          type="date"
          value={trade[field] as string}
          onChange={(e) => handleTradeFieldChange(index, field, e.target.value)}
          className="h-6 w-full px-1 text-xs"
        />
      );
    }

    // Number fields
    if (col.key === 'Entry_Price' || col.key === 'Exit_Price' || col.key === 'Return') {
      return (
        <Input
          type="number"
          value={trade[field] as number}
          onChange={(e) => handleTradeFieldChange(index, field, parseFloat(e.target.value) || 0)}
          className="h-6 w-full px-1 text-xs"
          step="0.01"
        />
      );
    }

    // Textarea fields (Case, Trade Note)
    if (col.isTextarea || col.key === 'Case' || col.key === 'Exit_Reason') {
      const cellKey = `${index}-${col.key}`;
      // Case has loading state
      if (col.key === 'Case' && trade.caseLoading) {
        return (
          <div className="flex items-center text-xs text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin mr-1" />
            Loading...
          </div>
        );
      }
      if (expandedCell === cellKey) {
        return (
          <textarea
            autoFocus
            value={trade[field] as string}
            onChange={(e) => handleTradeFieldChange(index, field, e.target.value)}
            onBlur={() => setExpandedCell(null)}
            className="w-[300px] h-[120px] px-1 py-0.5 text-xs rounded-md border border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 resize-y z-50 relative"
            placeholder={col.label}
          />
        );
      }
      return (
        <Input
          value={trade[field] as string}
          onChange={(e) => handleTradeFieldChange(index, field, e.target.value)}
          onFocus={() => setExpandedCell(cellKey)}
          className="h-6 w-full px-1 text-xs"
          placeholder={col.label}
        />
      );
    }

    // Text fields (Ticker, Status, Pattern, Market, Category)
    const datalist = DATALIST_MAP[col.key];
    return (
      <Input
        value={trade[field] as string}
        onChange={(e) => handleTradeFieldChange(index, field, e.target.value)}
        className="h-6 w-full px-1 text-xs"
        placeholder={col.label}
        list={datalist?.id}
      />
    );
  };

  return (
    <Card className="w-full">
      <CardHeader className="py-3">
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Import IBKR Statement
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3 space-y-4">
        {/* Aggregation Options - 2 simple toggles */}
        <div className="border rounded-md p-3 space-y-3">
          <div className="text-sm font-medium">Aggregation Options</div>
          <div className="flex flex-wrap gap-6">
            <div className="flex items-center gap-2">
              <Checkbox
                id="weightedAvgPrices"
                checked={useWeightedAvgPrices}
                onCheckedChange={(c) => setUseWeightedAvgPrices(!!c)}
              />
              <Label htmlFor="weightedAvgPrices" className="text-xs">
                Use weighted avg prices (unchecked = first buy / last sell price)
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="firstLastDates"
                checked={useFirstLastDates}
                onCheckedChange={(c) => setUseFirstLastDates(!!c)}
              />
              <Label htmlFor="firstLastDates" className="text-xs">
                Use first/last dates (unchecked = average of dates)
              </Label>
            </div>
          </div>
        </div>
        {/* Description of calculation methodology */}
        <div className="text-[14px] text-muted-foreground bg-muted/30 p-2 rounded-md border border-border/50">
          {/* <div className="font-semibold mb-1">Trade Calculation Methodology:</div> */}
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            <div><strong>Entry Date:</strong> {useFirstLastDates ? 'First buy date' : 'Average of all buy dates'}</div>
            <div><strong>Entry Price:</strong> {useWeightedAvgPrices ? 'Weighted average of all buys (by quantity)' : 'First buy price'}</div>
            <div><strong>Exit Date:</strong> {useFirstLastDates ? 'Last sell date' : 'Average of all sell dates'}</div>
            <div><strong>Exit Price:</strong> {useWeightedAvgPrices ? 'Weighted average of all sells (by quantity)' : 'Last sell price'}</div>
          </div>
        </div>

        {/* Deposit Adjustments */}
        <div className="border rounded-md p-3 space-y-3">
          <div className="text-sm font-medium">Deposit Adjustments <span className="text-xs font-normal text-muted-foreground">(corrects Pct of Equity for capital added after trades)</span></div>
          <div className="text-xs text-muted-foreground">
            Trades entered <strong>before</strong> a deposit date will have that deposit subtracted from their equity base, giving the true position size % at the time of entry.
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1">
              <Label htmlFor="depositDate" className="text-xs">Deposit Date</Label>
              <Input
                id="depositDate"
                type="date"
                value={depositDate}
                onChange={(e) => setDepositDate(e.target.value)}
                className="h-7 text-xs w-36"
              />
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="depositAmount" className="text-xs">Amount ($)</Label>
              <Input
                id="depositAmount"
                type="number"
                value={depositAmount}
                onChange={(e) => setDepositAmount(e.target.value)}
                className="h-7 text-xs w-32"
                placeholder="0.00"
                step="0.01"
                min="0"
              />
            </div>
            <Button
              type="button"
              size="sm"
              className="h-7"
              onClick={() => {
                const amt = parseFloat(depositAmount);
                if (!depositDate || isNaN(amt) || amt <= 0) return;
                const newDeposit: DepositEntry = { id: Date.now(), date: depositDate, amount: amt };
                setDeposits(prev => [...prev, newDeposit].sort((a, b) => b.date.localeCompare(a.date)));
                setDepositDate('');
                setDepositAmount('');
              }}
              disabled={!depositDate || !depositAmount || parseFloat(depositAmount || '0') <= 0}
            >
              Add Deposit
            </Button>
          </div>
          {deposits.length > 0 && (
            <div className="space-y-1 pt-1 border-t border-border/40">
              <div className="text-xs font-medium">Added deposits (newest first):</div>
              {deposits.map(d => (
                <div key={d.id} className="flex items-center gap-3 text-xs">
                  <span className="text-muted-foreground w-24">{d.date}</span>
                  <span className="text-emerald-400 font-medium">+${d.amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-5 px-1 text-xs text-red-400 hover:text-red-300"
                    onClick={() => setDeposits(prev => prev.filter(dep => dep.id !== d.id))}
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 py-2 border-y border-border/30">
          <Label className="text-sm font-medium whitespace-nowrap">Select IBKR CSV <strong>Activity Statement</strong>:</Label>

          <div className="flex items-center gap-2">
            <input
              type="file"
              ref={fileInputRef}
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 h-8"
            >
              <Upload className="h-4 w-4" />
              Choose File
            </Button>
            {fileName && (
              <span className="text-xs text-muted-foreground italic max-w-[200px] truncate" title={fileName}>
                {fileName}
              </span>
            )}
          </div>

          <Button
            onClick={handleProcessCSV}
            disabled={isProcessing || !csvContent.trim()}
            size="sm"
            className="h-8"
          >
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              'Parse CSV & Show Trades'
            )}
          </Button>

          {/* Append Statement - only show after initial parse */}
          {rawTransactions.length > 0 && (
            <>
              <input
                type="file"
                ref={appendFileInputRef}
                accept=".csv"
                onChange={handleAppendFileChange}
                className="hidden"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => appendFileInputRef.current?.click()}
                disabled={isProcessing}
                className="flex items-center gap-2 h-8"
              >
                <Upload className="h-4 w-4" />
                Import Another Statement(duplicates will be ignored)
              </Button>
            </>
          )}
        </div>

        {/* Show imported file names */}
        {importedFileNames.length > 0 && (
          <div className="text-xs text-muted-foreground">
            <span className="font-medium">Imported statements:</span>{' '}
            {importedFileNames.join(', ')}
            <span className="ml-2">({rawTransactions.length} total transactions)</span>
          </div>
        )}

        {parsedTrades.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium">
              Found {parsedTrades.length} completed trade(s) - Review and add to history:
            </div>
            <div className="rounded-md border overflow-x-auto">
              <Table>
                <TableHeader className="sticky top-0 bg-muted">
                  <TableRow className="border-b">
                    <TableHead className="w-20 py-1 px-1 text-xs text-center">Action</TableHead>
                    {visibleColumns.map(col => (
                      <React.Fragment key={col.key}>
                        <TableHead
                          className={`py-1 px-1 text-xs ${['C', 'A', 'N', 'S', 'L', 'I', 'M'].includes(col.key) ? 'w-8 text-center' : ''}`}
                          style={getColStyle(col)}
                        >
                          {col.label}
                        </TableHead>
                        {col.key === 'Return' && (
                          <TableHead className="w-16 py-1 px-1 text-xs text-center">Return %</TableHead>
                        )}
                      </React.Fragment>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {parsedTrades.map((trade, index) => {
                    const tradeKey = `${trade.ticker}-${index}`;
                    const isAdding = addingTrades.has(tradeKey);

                    return (
                      <TableRow key={tradeKey} className={`border-b ${getRowClass(trade)}`}>
                        <TableCell className="p-1">
                          <Button
                            onClick={() => handleAddTrade(index)}
                            variant="default"
                            size="sm"
                            className="h-6 text-xs px-2"
                            disabled={isAdding}
                          >
                            {isAdding ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              'Add'
                            )}
                          </Button>
                        </TableCell>
                        {visibleColumns.map(col => (
                          <React.Fragment key={col.key}>
                            <TableCell
                              className={`p-1 ${['C', 'A', 'N', 'S', 'L', 'I', 'M'].includes(col.key) ? 'text-center' : ''}`}
                              style={getColStyle(col)}
                            >
                              {renderImportCell(trade, index, col)}
                            </TableCell>
                            {col.key === 'Return' && (
                              <TableCell className="p-1 text-xs font-semibold text-center">
                                <span className={trade.returnPct >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                  {trade.returnPct.toFixed(2)}%
                                </span>
                              </TableCell>
                            )}
                          </React.Fragment>
                        ))}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>

            <datalist id="pattern-suggestions-import">
              {patternSuggestions.map(p => <option key={p} value={p} />)}
            </datalist>
            <datalist id="market-condition-suggestions-import">
              {marketConditionSuggestions.map(mc => <option key={mc} value={mc} />)}
            </datalist>
            <datalist id="category-suggestions-import">
              {categorySuggestions.map(c => <option key={c} value={c} />)}
            </datalist>
          </div>
        )}

        {/* Show All Trades - Raw Transactions */}
        {rawTransactions.length > 0 && (
          <div className="space-y-2">
            <div className="flex justify-start">
              <Button
                variant="default"
                size="sm"
                onClick={() => setShowAllTrades(!showAllTrades)}
              >
                {showAllTrades ? 'Hide' : 'Show'} All Raw Trades ({rawTransactions.length} transactions)
              </Button>
            </div>

            {showAllTrades && (
              <div className="rounded-md border overflow-x-auto">
                <Table>
                  <TableHeader className="sticky top-0 bg-muted">
                    <TableRow className="border-b">
                      <TableHead className="py-1 px-2 text-xs">Type</TableHead>
                      <TableHead className="py-1 px-2 text-xs">Symbol</TableHead>
                      <TableHead className="py-1 px-2 text-xs">Date</TableHead>
                      <TableHead className="py-1 px-2 text-xs">Qty</TableHead>
                      <TableHead className="py-1 px-2 text-xs">T. Price</TableHead>
                      <TableHead className="py-1 px-2 text-xs">Proceeds</TableHead>
                      <TableHead className="py-1 px-2 text-xs">Commission</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(() => {
                      // Group by symbol and calculate totals
                      const sorted = [...rawTransactions].sort((a, b) =>
                        a.symbol.localeCompare(b.symbol) || a.dateTime.getTime() - b.dateTime.getTime()
                      );
                      const rows: React.ReactNode[] = [];
                      let currentSymbol = '';
                      let symbolProceeds = 0;
                      let symbolCommission = 0;

                      // Grand totals
                      let grandTotalProceeds = 0;
                      let grandTotalCommission = 0;

                      sorted.forEach((tx, idx) => {
                        // Add subtotal row when symbol changes
                        if (currentSymbol && tx.symbol !== currentSymbol) {
                          const realizedPL = symbolProceeds - symbolCommission;
                          rows.push(
                            <TableRow key={`total-${currentSymbol}`} className="bg-muted/50 font-semibold">
                              <TableCell colSpan={5} className="py-1 px-2 text-xs text-right">
                                {currentSymbol} Total P/L:
                              </TableCell>
                              <TableCell className={`py-1 px-2 text-xs font-semibold ${realizedPL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                ${realizedPL.toFixed(2)}
                              </TableCell>
                              <TableCell className="py-1 px-2 text-xs text-red-400">
                                -${Math.abs(symbolCommission).toFixed(2)}
                              </TableCell>
                            </TableRow>
                          );
                          symbolProceeds = 0;
                          symbolCommission = 0;
                        }
                        currentSymbol = tx.symbol;
                        symbolProceeds += tx.proceeds;
                        symbolCommission += Math.abs(tx.commission);
                        grandTotalProceeds += tx.proceeds;
                        grandTotalCommission += Math.abs(tx.commission);

                        rows.push(
                          <TableRow key={idx} className={tx.quantity > 0 ? 'bg-green-500/10' : 'bg-red-500/10'}>
                            <TableCell className="py-1 px-2 text-xs font-medium">
                              {tx.quantity > 0 ? 'BUY' : 'SELL'}
                            </TableCell>
                            <TableCell className="py-1 px-2 text-xs font-medium">{tx.symbol}</TableCell>
                            <TableCell className="py-1 px-2 text-xs">
                              {tx.dateTime.toISOString().split('T')[0]}
                            </TableCell>
                            <TableCell className="py-1 px-2 text-xs">{Math.abs(tx.quantity).toFixed(2)}</TableCell>
                            <TableCell className="py-1 px-2 text-xs">${tx.price.toFixed(2)}</TableCell>
                            <TableCell className="py-1 px-2 text-xs">${tx.proceeds.toFixed(2)}</TableCell>
                            <TableCell className="py-1 px-2 text-xs">${tx.commission.toFixed(2)}</TableCell>
                          </TableRow>
                        );

                        // Add final subtotal for last symbol
                        if (idx === sorted.length - 1) {
                          const realizedPL = symbolProceeds - symbolCommission;
                          rows.push(
                            <TableRow key={`total-${currentSymbol}-final`} className="bg-muted/50 font-semibold">
                              <TableCell colSpan={5} className="py-1 px-2 text-xs text-right">
                                {currentSymbol} Total P/L:
                              </TableCell>
                              <TableCell className={`py-1 px-2 text-xs font-semibold ${realizedPL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                ${realizedPL.toFixed(2)}
                              </TableCell>
                              <TableCell className="py-1 px-2 text-xs text-red-400">
                                -${Math.abs(symbolCommission).toFixed(2)}
                              </TableCell>
                            </TableRow>
                          );

                          // Add grand total row
                          const grandRealizedPL = grandTotalProceeds - grandTotalCommission;
                          rows.push(
                            <TableRow key="grand-total" className="bg-muted/50 font-semibold">
                              <TableCell colSpan={5} className="py-1 px-2 text-xs text-right">
                                GRAND TOTAL:
                              </TableCell>
                              <TableCell className={`py-1 px-2 text-xs font-semibold ${grandRealizedPL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                ${grandRealizedPL.toFixed(2)}
                              </TableCell>
                              <TableCell className="py-1 px-2 text-xs text-red-400">
                                -${grandTotalCommission.toFixed(2)}
                              </TableCell>
                            </TableRow>
                          );
                        }
                      });
                      return rows;
                    })()}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
