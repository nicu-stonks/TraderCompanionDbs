// services/customTradeDataAPI.ts
import axios from 'axios';
import { CustomColumn, ColumnOrder, CustomColumnValue } from '../types/CustomTradeData';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/custom_trade_data'
});

export const customTradeDataAPI = {
  // ---- Custom Columns CRUD ----
  getColumns: () =>
    api.get<CustomColumn[]>('/columns/'),

  createColumn: (name: string) =>
    api.post<CustomColumn>('/columns/', { name }),

  updateColumn: (id: number, name: string) =>
    api.put<CustomColumn>(`/columns/${id}/`, { name }),

  deleteColumn: (id: number) =>
    api.delete(`/columns/${id}/`),

  // ---- Column Order ----
  getColumnOrder: () =>
    api.get<ColumnOrder[]>('/column_order/'),

  bulkUpdateColumnOrder: (orders: ColumnOrder[]) =>
    api.post<ColumnOrder[]>('/column_order/bulk/', orders),

  // ---- Custom Column Values ----
  getColumnValues: (tradeIds?: number[]) => {
    const params = tradeIds ? { trade_id: tradeIds.join(',') } : {};
    return api.get<CustomColumnValue[]>('/values/', { params });
  },

  bulkUpsertColumnValues: (values: CustomColumnValue[]) =>
    api.post<CustomColumnValue[]>('/values/bulk/', values),

  deleteValuesByColumn: (columnId: number) =>
    api.delete(`/values/by_column/${columnId}/`),
};
