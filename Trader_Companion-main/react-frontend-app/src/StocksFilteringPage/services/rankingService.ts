import axios from 'axios';
import { RankingListSuccessResponse } from '../types/rankingList';
import { API_CONFIG } from '../../config';

export const rankingService = {
  async fetchRankingList(fileName: string): Promise<RankingListSuccessResponse> {
    try {
      const response = await axios.get(
        `${API_CONFIG.baseURL}/stock_filtering_app/rankings/${fileName}`
      );
      
      // If response.data is already an object, use it directly
      if (typeof response.data === 'object' && response.data !== null) {
        return {
          status: 'success',
          message: response.data.message,
          stock_data_created_at: response.data.stock_data_created_at,
          rankings_created_at: response.data.rankings_created_at,
          total_stocks: response.data.total_stocks || 0, // Default to 0 if not present
          filtered_stocks: response.data.filtered_stocks || 0 // Default to 0 if not present
        };
      }

      // If it's a string, we need to handle potential Infinity values before parsing
      if (typeof response.data === 'string') {
        // Replace Infinity with a numeric value
        const sanitizedData = response.data
          .replace(/:\s*Infinity/g, ': null')
          .replace(/:\s*-Infinity/g, ': null');
        
        const parsed = JSON.parse(sanitizedData);
        return {
          status: 'success',
          message: parsed.message,
          stock_data_created_at: parsed.stock_data_created_at,
          rankings_created_at: parsed.rankings_created_at,
          total_stocks: parsed.total_stocks || 0, // Default to 0 if not present
          filtered_stocks: parsed.filtered_stocks || 0 // Default to 0 if not present
        };
      }

      throw new Error('Invalid response format');
    } catch (error) {
      console.error('Error details:', error);
      // Normalize backend / filesystem "file not found" style errors into a friendly UX hint
      const extractAxiosMessage = () => {
        if (!axios.isAxiosError(error)) return null;
        const data = error.response?.data;
        if (!data) return error.message;
        if (typeof data === 'string') return data;
        // Narrow potential object shape
        if (typeof (data as { message?: unknown }).message === 'string') {
          return (data as { message?: string }).message as string;
        }
        return error.message;
      };

      let rawMessage = extractAxiosMessage() || (error instanceof Error ? error.message : 'Failed to fetch ranking list');

      const missingFilePatterns = [
        'WinError 3', // Windows specific missing path
        'No such file or directory',
        'cannot find the path specified',
        'FileNotFoundError'
      ];

      const isMissingFile = missingFilePatterns.some(p => rawMessage?.toLowerCase().includes(p.toLowerCase()));

      if (isMissingFile) {
        rawMessage = 'No screening results yet. Run a screening (Start Screening) and wait for it to complete to generate ranking files.';
      }

      throw new Error(rawMessage || 'Failed to fetch ranking list');
    }
  }
};