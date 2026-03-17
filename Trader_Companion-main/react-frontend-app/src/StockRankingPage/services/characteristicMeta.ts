import axios from 'axios';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/personal_ranking'
});

export interface OrderedCharacteristicMeta {
  id: number;
  characteristic_id: number;
  name: string;
  position: number;
}

export interface PriorityCharacteristicMeta {
  id: number;
  characteristic_id: number;
  name: string;
  created_at?: string;
}

export interface ColorCodedCharacteristicMeta {
  id: number;
  characteristic_id: number;
  name: string;
  created_at?: string;
}

export const characteristicMetaApi = {
  getOrdered: () => api.get<OrderedCharacteristicMeta[]>('/ordered-characteristics/'),
  createOrdered: (data: { characteristic_id: number; position?: number }) => api.post('/ordered-characteristics/', data),
  updateOrdered: (id: number, data: Partial<{ position: number }>) => api.put(`/ordered-characteristics/${id}/`, data),
  deleteOrdered: (id: number) => api.delete(`/ordered-characteristics/${id}/`),
  reorderOrdered: (items: Array<{ id: number; position: number }>) => api.post('/ordered-characteristics/reorder/', { items }),

  getPriority: () => api.get<PriorityCharacteristicMeta[]>('/priority-characteristics/'),
  createPriority: (data: { characteristic_id: number }) => api.post('/priority-characteristics/', data),
  deletePriority: (id: number) => api.delete(`/priority-characteristics/${id}/`),

  getColorCoded: () => api.get<ColorCodedCharacteristicMeta[]>('/color-coded-characteristics/'),
  createColorCoded: (data: { characteristic_id: number }) => api.post('/color-coded-characteristics/', data),
  deleteColorCoded: (id: number) => api.delete(`/color-coded-characteristics/${id}/`),
};
