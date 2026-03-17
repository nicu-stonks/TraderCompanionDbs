import axios from 'axios';
import { ScreeningOptions } from '../types/screenerCommander';
import { API_CONFIG } from '../../config';

export const sendScreenerCommand = async (options: ScreeningOptions) => {
  try {
    const response = await axios.post(
      `${API_CONFIG.baseURL}/stock_filtering_app/run_screening`,
      options,  // axios automatically handles JSON stringification
      {
        headers: {
          'Content-Type': 'application/json',
        }
      }
    );
    
    // axios throws on 4xx/5xx status codes, but we want to allow 409
    return response.data.message;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 409) {
      return error.response.data.message;
    }
    console.error('Error in stock screener:', error);
    throw error;
  }
};