import axios from 'axios';
import { Alert, AlarmSettings, CreateAlertData } from '../types';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/price_alerts'
});

export const priceAlertsAPI = {
  getAlerts: () =>
    api.get<Alert[]>('/alerts/'),

  createAlert: (data: CreateAlertData) =>
    api.post<Alert>('/alerts/', data),

  updateAlert: (id: number, data: Partial<Alert>) =>
    api.patch<Alert>(`/alerts/${id}/`, data),

  deleteAlert: (id: number) =>
    api.delete(`/alerts/${id}/`),

  deleteAllAlerts: () =>
    api.delete<{ message: string; deleted_count: number }>('/alerts/delete_all/'),

  getAlarmSettings: () =>
    api.get<AlarmSettings>('/alarm-settings/'),

  updateAlarmSettings: (data: Partial<AlarmSettings>) =>
    api.put<AlarmSettings>('/alarm-settings/', data),

  uploadAlarmSound: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post<{ message: string; filename: string; path: string }>(
      '/upload-alarm-sound/',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
  },

  listAlarmSounds: () =>
    api.get<{ sounds: string[] }>('/list-alarm-sounds/'),
};

