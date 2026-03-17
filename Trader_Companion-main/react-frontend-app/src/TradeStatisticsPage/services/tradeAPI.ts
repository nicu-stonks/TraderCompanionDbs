// services/tradeAPI.ts
import axios from 'axios';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/trades_app'
});

export const tradeAPI = {
  getTrades: (limit?: number) => 
    api.get<Trade[]>('/trades/', {
      params: { limit }
    }),
  
  addTrade: (trade: Trade) => 
    api.post<Trade>('/trades/', {
      ...trade,
      Entry_Date: trade.Entry_Date.toString(),
      Exit_Date: trade.Exit_Date?.toString() || null,
    }),
  
  updateTrade: (trade: Trade) => 
    api.put<Trade>(`/trades/${trade.ID}/`, trade),
  
  deleteTrade: (id: number) => 
    api.delete(`/trades/${id}/`)
};