// services/stockCharacteristics.ts
import axios from 'axios';
import { StockCharacteristic } from '../types';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/personal_ranking'
});

// Stock Characteristics API
export const stockCharacteristicsApi = {
  getAllCharacteristics: () =>
    api.get('/stock-characteristics/'),
  
  getCharacteristicsByStockPick: (stockPickId: number) =>
    api.get(`/stock-characteristics/?stock_pick=${stockPickId}`),
  
  getCharacteristic: (id: number) =>
    api.get(`/stock-characteristics/${id}/`),
  
  createCharacteristic: (data: {
    stock_pick: number;
    name: string;
    score: number;
    global_characteristic_id?: number;
  }) =>
    api.post('/stock-characteristics/', data),
  
  updateCharacteristic: (id: number, data: Partial<StockCharacteristic>) =>
    api.put(`/stock-characteristics/${id}/`, data),
  
  deleteCharacteristic: (id: number) =>
    api.delete(`/stock-characteristics/${id}/`)
};