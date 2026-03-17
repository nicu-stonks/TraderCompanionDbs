// services/balanceAPI.ts
import axios from 'axios';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/trades_app'
});

export const balanceAPI = {
  getBalance: () => 
    api.get('/balance/').then(response => response.data.balance),
  
  updateBalance: (balance: number) => 
    api.put('/balance/', { balance })
};