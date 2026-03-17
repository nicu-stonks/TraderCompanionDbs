// services/personalRanking.ts
import axios from 'axios';
import { UserPageState } from '../types';
import { API_CONFIG } from '@/config';

const api = axios.create({
  baseURL: API_CONFIG.baseURL + '/personal_ranking'
});

export const userPageStateApi = {  
  getUserPageState: () => 
    api.get<UserPageState>('/user-page-state/'),
  
  updateUserPageState: (data: Partial<UserPageState>) => 
    api.post<UserPageState>('/user-page-state/', data)
};