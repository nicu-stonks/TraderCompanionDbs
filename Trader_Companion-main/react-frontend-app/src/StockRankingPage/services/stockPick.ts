import axios from 'axios';
import { StockPick } from '../types';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/personal_ranking'
});

// Stock Picks API
export const stockPicksApi = {
  getAllStockPicks: () =>
    api.get('/stock-picks/'),
  
  getStockPicksByBox: (boxId: number) =>
    api.get(`/stock-picks/?ranking_box=${boxId}`),
  
  getStockPick: (id: number) =>
    api.get(`/stock-picks/${id}/`),
  
  createStockPick: (data: { ranking_box: number; symbol: string; total_score: number; case_text?: string }) =>
    api.post('/stock-picks/', data),
  
  updateStockPick: (id: number, data: Partial<StockPick>) =>
    api.put(`/stock-picks/${id}/`, data),
  
  deleteStockPick: (id: number) => 
    api.delete(`/stock-picks/${id}/`),

  deleteAllStockPicks: () =>
    api.delete('/stock-picks/delete_all/'),
    
  // Set all characteristics at once
  setCharacteristics: (stockPickId: number, characteristics: Array<{characteristic_id: number, score: number}>) =>
    api.post(`/stock-picks/${stockPickId}/set_characteristics/`, { characteristics }),
    
  // Add or update a single characteristic
  addCharacteristic: (stockPickId: number, data: {characteristic_id: number, score?: number}) =>
    api.post(`/stock-picks/${stockPickId}/add_characteristic/`, data),
    
  // Remove a single characteristic
  removeCharacteristic: (stockPickId: number, data: {characteristic_id: number}) =>
    api.post(`/stock-picks/${stockPickId}/remove_characteristic/`, data),
    
  // Update personal opinion score
  updatePersonalScore: (stockPickId: number, score: number) =>
    api.post(`/stock-picks/${stockPickId}/update_personal_score/`, { personal_opinion_score: score })
};