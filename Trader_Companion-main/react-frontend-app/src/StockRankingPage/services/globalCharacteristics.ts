// services/globalCharacteristics.ts
import axios from 'axios';
import { GlobalCharacteristic } from '../types';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/personal_ranking'
});

// Global Characteristics API
export const globalCharacteristicsApi = {
  getAllGlobalCharacteristics: () =>
    api.get('/global-characteristics/'),
  
  getGlobalCharacteristic: (id: number) =>
    api.get(`/global-characteristics/${id}/`),
  
  createGlobalCharacteristic: (data: {
    name: string;
    default_score: number;
  }) =>
    api.post('/global-characteristics/', data),
  
  updateGlobalCharacteristic: (id: number, data: Partial<GlobalCharacteristic>) =>
    api.put(`/global-characteristics/${id}/`, data),
  
  deleteGlobalCharacteristic: (id: number) =>
    api.delete(`/global-characteristics/${id}/`)
};