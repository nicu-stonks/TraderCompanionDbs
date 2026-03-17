// services/tradeAPI.ts
import axios from 'axios';
import { Trade } from '../types/Trade';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/trades_app'
});

export const tradeAPI = {
  getTrades: (limit?: number) =>
    api.get<Trade[]>('/trades/', {
      params: { limit }
    }),

  addTrade: async (trade: Trade) => {
    // Auto-generate next ID by finding max existing ID + 1
    const existingTrades = await api.get<Trade[]>('/trades/');
    const maxId = existingTrades.data.reduce((max, t) => Math.max(max, t.ID), 0);
    const nextId = maxId + 1;

    return api.post<Trade>('/trades/', {
      ...trade,
      ID: nextId,
      Entry_Date: trade.Entry_Date.toString(),
      Exit_Date: trade.Exit_Date?.toString() || null,
      // Required fields need defaults since backend requires them (no blank=True in Django model)
      Pattern: trade.Pattern || 'N/A',
      Market_Condition: trade.Market_Condition || 'N/A',
      Category: trade.Category || 'N/A',
      Exit_Reason: trade.Exit_Reason || 'N/A',
      Price_Tightness_1_Week_Before: trade.Price_Tightness_1_Week_Before || 0,
      Pct_Off_52W_High: trade.Pct_Off_52W_High || 0,
      Nr_Bases: trade.Nr_Bases || 0,
    });
  },

  updateTrade: (trade: Trade) =>
    api.put<Trade>(`/trades/${trade.ID}/`, trade),

  deleteTrade: (id: number) =>
    api.delete(`/trades/${id}/`)
};