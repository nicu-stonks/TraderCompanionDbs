import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Trade } from '../types/Trade';
import { stockPicksApi } from '@/StockRankingPage/services/stockPick';
import { globalCharacteristicsApi } from '@/StockRankingPage/services/globalCharacteristics';
import type { StockPick, GlobalCharacteristic } from '@/StockRankingPage/types';
import { AxiosError } from 'axios';
import type { UseCustomTradeDataReturn } from '../hooks/useCustomTradeData';

// Fields to hide from the UI (kept in backend for compatibility)
const HIDDEN_FIELDS = new Set([
  'ID',  // Auto-generated
  'Price_Tightness_1_Week_Before',
  'Nr_Bases',
  'Has_Earnings_Acceleration',
  'IPO_Last_10_Years',
  'Is_BioTech',
  'Under_30M_Shares',
  'If_You_Could_Only_Make_10_Trades',
  'Pct_Off_52W_High',
]);

// Custom display labels for fields (backend name -> UI label)
const FIELD_DISPLAY_LABELS: Record<string, string> = {
  'Exit_Reason': 'Trade Note',
};

interface AddTradeComponentProps {
  onAdd: (trade: Trade) => Promise<void>;
  existingTrades?: Trade[];
  customTradeData: UseCustomTradeDataReturn;
}

export const AddTradeComponent: React.FC<AddTradeComponentProps> = ({ onAdd, existingTrades = [], customTradeData }) => {
  const initialTrade: Trade = {
    ID: 0,
    Ticker: '',
    Status: '',
    Entry_Date: new Date().toISOString().split('T')[0],
    Exit_Date: null,
    Entry_Price: 0,
    Return: 0,
    Exit_Price: 0,
    Pattern: '',
    // Days_In_Pattern_Before_Entry: 0,
    Price_Tightness_1_Week_Before: 0,
    Exit_Reason: '',
    Market_Condition: '',
    Category: '',
    // Earnings_Quality: 0,
    Nr_Bases: 0,
    Case: '',
    // Fundamentals_Quality: 0,
    Has_Earnings_Acceleration: false,
    // Has_Catalyst: false,
    // Earnings_Last_Q_20_Pct: false,
    IPO_Last_10_Years: false,
    // Volume_Confirmation: false,
    Is_BioTech: false,
    // Earnings_Surprises: false,
    // Expanding_Margins: false,
    // EPS_breakout: false,
    // Strong_annual_EPS: false,
    // Signs_Acceleration_Will_Continue: false,
    // Sudden_Growth_Change: false,
    // Strong_Quarterly_Sales: false,
    // Strong_Yearly_Sales: false,
    // Positive_Analysts_EPS_Revisions: false,
    // Positive_Analysts_Price_Revisions: false,
    // Ownership_Pct_Change_Past_Earnings: false,
    // Quarters_With_75pct_Surprise: false,
    // Over_10_pct_Avg_Surprise: false,
    Under_30M_Shares: false,
    // Spikes_On_Volume: false,
    // Started_Off_Correction: false,
    // All_Trendlines_Up: false,
    If_You_Could_Only_Make_10_Trades: false,
    Pct_Off_52W_High: 0,
    C: false,
    A: false,
    N: false,
    S: false,
    L: false,
    I: false,
    M: false,
    Pct_Of_Equity: 0,
  };

  const [newTrade, setNewTrade] = useState<Trade>(initialTrade);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPrefilling, setIsPrefilling] = useState(false);
  const globalCharsRef = useRef<GlobalCharacteristic[] | null>(null);
  const userEditedCaseRef = useRef(false);
  const fetchedStockCacheRef = useRef<Record<string, StockPick | null>>({});
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Custom column values for the new trade
  const [customFieldValues, setCustomFieldValues] = useState<Record<number, string>>({});

  // Suggestions for Pattern, Market_Condition, Category
  const patternSuggestions = Array.from(new Set(existingTrades.map(t => t.Pattern).filter(Boolean)));
  const marketConditionSuggestions = Array.from(new Set(existingTrades.map(t => t.Market_Condition).filter(Boolean)));
  const categorySuggestions = Array.from(new Set(existingTrades.map(t => t.Category).filter(Boolean)));

  // Load global characteristics once (lazy on first ticker attempt)
  const ensureGlobalCharacteristics = async () => {
    if (!globalCharsRef.current) {
      try {
        const resp = await globalCharacteristicsApi.getAllGlobalCharacteristics();
        globalCharsRef.current = resp.data;
      } catch {
        // Silent fail; prefill just won't happen
        globalCharsRef.current = [];
      }
    }
    return globalCharsRef.current || [];
  };

  // Build JSON (same structure as RankingItem download) for a stock pick
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

  const attemptPrefillCase = async (tickerRaw: string) => {
    const ticker = tickerRaw.trim().toUpperCase();
    if (!ticker) return;
    // Don't override if user manually edited case after prefill
    if (userEditedCaseRef.current && newTrade.Case) return;
    setIsPrefilling(true);
    try {
      // Cached lookup
      if (!(ticker in fetchedStockCacheRef.current)) {
        // Fetch all stock picks once (could optimize later with backend filter)
        const resp = await stockPicksApi.getAllStockPicks();
        const all: StockPick[] = resp.data;
        const found = all.find(sp => sp.symbol.toUpperCase() === ticker) || null;
        fetchedStockCacheRef.current[ticker] = found;
      }
      const stock = fetchedStockCacheRef.current[ticker];
      if (!stock) return; // nothing to prefill
      const caseJson = await buildCaseJson(stock);
      setNewTrade(prev => ({ ...prev, Case: caseJson, Ticker: ticker }));
    } catch {
      // Ignore errors silently
    } finally {
      setIsPrefilling(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = e.target;
    const checked = e.target.checked;

    setNewTrade(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked :
        type === 'number' ? (value === '' || value === '-' || value === '-.' ? value : Number(value)) :
          value
    }));

    if (name === 'Ticker') {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        attemptPrefillCase(value);
      }, 600);
    }

    if (name === 'Case') {
      userEditedCaseRef.current = true; // prevent overriding after user edits
    }

    // Clear error when user starts typing
    if (error) setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      await onAdd(newTrade);

      // Save custom column values for the newly added trade
      // We need to get the ID from the response - onAdd returns void but the trade
      // gets added via tradeAPI which returns the data. We'll save custom values
      // after the trade is added using the latest trades list.
      if (Object.keys(customFieldValues).length > 0) {
        // Use a small delay to let the trade be created, then save custom values
        // The trade ID will be determined by the parent polling
        const existingIds = existingTrades.map(t => t.ID);
        const maxExistingId = existingIds.length > 0 ? Math.max(...existingIds) : 0;
        const newTradeId = maxExistingId + 1;
        for (const [colIdStr, value] of Object.entries(customFieldValues)) {
          if (value.trim()) {
            await customTradeData.updateCustomValue(newTradeId, parseInt(colIdStr), value);
          }
        }
      }

      setNewTrade(initialTrade); // Only reset form on successful submission
      setCustomFieldValues({}); // Reset custom field values
    } catch (err) {
      // Handle different types of errors
      if (err instanceof AxiosError) {
        const data = err.response?.data;
        let errorMessage = 'Failed to add trade. Please try again.';

        if (data) {
          if (typeof data === 'string') {
            errorMessage = data;
          } else if (data.detail) {
            errorMessage = data.detail;
          } else if (data.message) {
            errorMessage = data.message;
          } else {
            // DRF returns errors as {field: ['error1', 'error2']}
            const fieldErrors = Object.entries(data)
              .map(([field, errors]) => `${field}: ${Array.isArray(errors) ? errors.join(', ') : errors}`)
              .join('; ');
            if (fieldErrors) errorMessage = fieldErrors;
          }
        }
        console.error('Trade add error:', data);
        setError(errorMessage);
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  type InputValue = string | number | boolean | null;

  const renderFormField = (key: keyof Trade, value: InputValue) => {
    if (typeof value === 'boolean' || typeof initialTrade[key] === 'boolean') {
      return (
        <div key={key} className="flex items-center space-x-2">
          <Checkbox
            id={key}
            name={key}
            checked={Boolean(value)}
            onCheckedChange={(checked: boolean) =>
              setNewTrade(prev => ({ ...prev, [key]: checked }))
            }
            disabled={isSubmitting}
          />
          <Label htmlFor={key} className="text-sm">
            {FIELD_DISPLAY_LABELS[key] || key.replace(/_/g, ' ')}
          </Label>
        </div>
      );
    }

    // Determine input type based on initial value type, not current value
    const isNumeric = typeof initialTrade[key] === 'number';
    const inputType = key.includes('Date') ? 'date' : isNumeric ? 'number' : 'text';

    return (
      <div key={key} className="space-y-1">
        <Label htmlFor={key} className="text-sm">
          {FIELD_DISPLAY_LABELS[key] || key.replace(/_/g, ' ')}
        </Label>
        <Input
          id={key}
          type={inputType}
          name={key}
          value={value ?? ''}
          onChange={handleInputChange}
          step={key.includes('Price') ? '0.01' : '0.01'}
          className="h-8"
          disabled={isSubmitting}
          list={key === 'Pattern' ? 'pattern-suggestions' :
            key === 'Market_Condition' ? 'market-condition-suggestions' :
              key === 'Category' ? 'category-suggestions' : undefined}
        />
      </div>
    );
  };

  return (
    <Card className="w-full h-full flex flex-col">
      <CardHeader className="py-3 flex-shrink-0">
        <CardTitle className="text-lg font-semibold">Add New Trade</CardTitle>
      </CardHeader>
      <CardContent className="p-3 flex-grow overflow-hidden">
        <form onSubmit={handleSubmit} className="h-full flex flex-col">
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <ScrollArea className="flex-grow pr-4 -mr-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4 gap-4 mb-4">
              {Object.entries(newTrade)
                .filter(([key]) => !HIDDEN_FIELDS.has(key))
                .map(([key, value]) =>
                  renderFormField(key as keyof Trade, value as InputValue)
                )}
              {isPrefilling && (
                <div className="text-xs text-muted-foreground mt-2">Prefilling case from ranking data...</div>
              )}

              {/* Custom column fields */}
              {customTradeData.customColumns.map(col => (
                <div key={`custom_${col.id}`} className="space-y-1">
                  <Label htmlFor={`custom_${col.id}`} className="text-sm">
                    {col.name}
                  </Label>
                  <Input
                    id={`custom_${col.id}`}
                    type="text"
                    value={customFieldValues[col.id] || ''}
                    onChange={(e) => setCustomFieldValues(prev => ({ ...prev, [col.id]: e.target.value }))}
                    className="h-8"
                    disabled={isSubmitting}
                  />
                </div>
              ))}
            </div>

            <datalist id="pattern-suggestions">
              {patternSuggestions.map(p => <option key={p} value={p} />)}
            </datalist>
            <datalist id="market-condition-suggestions">
              {marketConditionSuggestions.map(mc => <option key={mc} value={mc} />)}
            </datalist>
            <datalist id="category-suggestions">
              {categorySuggestions.map(c => <option key={c} value={c} />)}
            </datalist>
          </ScrollArea>

          <Button
            type="submit"
            className="w-full mt-4"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Adding Trade...' : 'Add Trade'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};