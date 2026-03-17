// hooks/useCustomTradeData.ts
import { useState, useEffect, useCallback, useRef } from 'react';
import { CustomColumn, ColumnOrder, CustomColumnValue } from '../types/CustomTradeData';
import { customTradeDataAPI } from '../services/customTradeDataAPI';

// Default column keys in original order (these match Trade fields shown in TradesTable)
const DEFAULT_COLUMNS: string[] = [
  'Ticker', 'Status', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price',
  'Return', 'Pct_Of_Equity', 'Pattern', 'Exit_Reason', 'Market_Condition', 'Category', 'Case',
  'C', 'A', 'N', 'S', 'L', 'I', 'M',
];

export interface ColumnDef {
  key: string;         // For default cols: field name. For custom: `custom_${id}`
  label: string;       // Display label
  isCustom: boolean;
  customColumnId?: number; // Only for custom columns
  position: number;
  width: number;       // pixel width, 0 means use default
  isTextarea: boolean; // render as expandable textarea
}

export function useCustomTradeData() {
  const [customColumns, setCustomColumns] = useState<CustomColumn[]>([]);
  const [columnOrder, setColumnOrder] = useState<ColumnOrder[]>([]);
  const [columnValues, setColumnValues] = useState<CustomColumnValue[]>([]);
  const [loading, setLoading] = useState(true);
  const initialLoadDone = useRef(false);

  // Load all custom trade data
  const loadAll = useCallback(async () => {
    try {
      const [colsResp, orderResp, valsResp] = await Promise.all([
        customTradeDataAPI.getColumns(),
        customTradeDataAPI.getColumnOrder(),
        customTradeDataAPI.getColumnValues(),
      ]);
      setCustomColumns(colsResp.data);
      setColumnOrder(orderResp.data);
      setColumnValues(valsResp.data);
    } catch (err) {
      console.error('Error loading custom trade data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!initialLoadDone.current) {
      initialLoadDone.current = true;
      loadAll();
    }
  }, [loadAll]);

  // Build the ordered list of all columns
  const orderedColumns: ColumnDef[] = (() => {
    // Start with default columns + custom columns
    const allDefs: ColumnDef[] = [];

    // Default textarea columns
    const DEFAULT_TEXTAREA_KEYS = new Set(['Exit_Reason', 'Case']);

    // Build lookup from saved order
    const orderMap = new Map<string, ColumnOrder>();
    columnOrder.forEach(o => orderMap.set(o.column_key, o));

    // Default columns
    DEFAULT_COLUMNS.forEach((key, idx) => {
      const FIELD_DISPLAY_LABELS: Record<string, string> = {
        'Exit_Reason': 'Trade Note',
      };
      const saved = orderMap.get(key);
      allDefs.push({
        key,
        label: FIELD_DISPLAY_LABELS[key] || key.replace(/_/g, ' '),
        isCustom: false,
        position: saved?.position ?? idx,
        width: saved?.width ?? 0,
        isTextarea: saved?.is_textarea ?? DEFAULT_TEXTAREA_KEYS.has(key),
      });
    });

    // Custom columns
    customColumns.forEach((col, idx) => {
      const colKey = `custom_${col.id}`;
      const saved = orderMap.get(colKey);
      allDefs.push({
        key: colKey,
        label: col.name,
        isCustom: true,
        customColumnId: col.id,
        position: saved?.position ?? (DEFAULT_COLUMNS.length + idx),
        width: saved?.width ?? 0,
        isTextarea: saved?.is_textarea ?? false,
      });
    });

    // Sort by position
    allDefs.sort((a, b) => a.position - b.position);

    return allDefs;
  })();

  // Get custom column value for a specific trade and column
  const getCustomValue = useCallback((tradeId: number, customColumnId: number): string => {
    const val = columnValues.find(v => v.trade_id === tradeId && v.column === customColumnId);
    return val?.value || '';
  }, [columnValues]);

  // Get all custom values for a trade as a map { custom_${id}: value }
  const getCustomValuesForTrade = useCallback((tradeId: number): Record<string, string> => {
    const result: Record<string, string> = {};
    columnValues
      .filter(v => v.trade_id === tradeId)
      .forEach(v => {
        result[`custom_${v.column}`] = v.value;
      });
    return result;
  }, [columnValues]);

  // Update a single custom column value (optimistic + persist)
  const updateCustomValue = useCallback(async (tradeId: number, customColumnId: number, value: string) => {
    // Optimistic update
    setColumnValues(prev => {
      const existing = prev.find(v => v.trade_id === tradeId && v.column === customColumnId);
      if (existing) {
        return prev.map(v =>
          v.trade_id === tradeId && v.column === customColumnId
            ? { ...v, value }
            : v
        );
      }
      return [...prev, { trade_id: tradeId, column: customColumnId, value }];
    });

    // Persist
    try {
      await customTradeDataAPI.bulkUpsertColumnValues([
        { trade_id: tradeId, column: customColumnId, value }
      ]);
    } catch (err) {
      console.error('Error saving custom column value:', err);
    }
  }, []);

  // CRUD for custom columns
  const addCustomColumn = useCallback(async (name: string) => {
    try {
      const resp = await customTradeDataAPI.createColumn(name);
      setCustomColumns(prev => [...prev, resp.data]);
      return resp.data;
    } catch (err) {
      console.error('Error creating custom column:', err);
      throw err;
    }
  }, []);

  const renameCustomColumn = useCallback(async (id: number, name: string) => {
    try {
      await customTradeDataAPI.updateColumn(id, name);
      setCustomColumns(prev => prev.map(c => c.id === id ? { ...c, name } : c));
    } catch (err) {
      console.error('Error renaming custom column:', err);
      throw err;
    }
  }, []);

  const deleteCustomColumn = useCallback(async (id: number) => {
    try {
      await customTradeDataAPI.deleteValuesByColumn(id);
      await customTradeDataAPI.deleteColumn(id);
      setCustomColumns(prev => prev.filter(c => c.id !== id));
      setColumnValues(prev => prev.filter(v => v.column !== id));
      // Also remove from column order
      setColumnOrder(prev => prev.filter(o => o.column_key !== `custom_${id}`));
    } catch (err) {
      console.error('Error deleting custom column:', err);
      throw err;
    }
  }, []);

  // Save column order (preserves widths)
  const saveColumnOrder = useCallback(async (newOrder: ColumnDef[]) => {
    const orders: ColumnOrder[] = newOrder.map((col, idx) => ({
      column_key: col.key,
      position: idx,
      is_custom: col.isCustom,
      width: col.width,
      is_textarea: col.isTextarea,
    }));

    setColumnOrder(orders);

    try {
      await customTradeDataAPI.bulkUpdateColumnOrder(orders);
    } catch (err) {
      console.error('Error saving column order:', err);
    }
  }, []);

  // Convenience: get all unique values of a custom column across all trades (for filters)
  const getUniqueCustomColumnValues = useCallback((customColumnId: number): string[] => {
    const values = columnValues
      .filter(v => v.column === customColumnId && v.value.trim() !== '')
      .map(v => v.value);
    return Array.from(new Set(values));
  }, [columnValues]);

  return {
    customColumns,
    columnOrder,
    columnValues,
    orderedColumns,
    loading,
    getCustomValue,
    getCustomValuesForTrade,
    updateCustomValue,
    addCustomColumn,
    renameCustomColumn,
    deleteCustomColumn,
    saveColumnOrder,
    getUniqueCustomColumnValues,
    reload: loadAll,
    DEFAULT_COLUMNS,
  };
}

export type UseCustomTradeDataReturn = ReturnType<typeof useCustomTradeData>;
