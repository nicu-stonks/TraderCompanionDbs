import { useState } from 'react';
import { API_CONFIG } from '@/config';

interface BanResponse {
  status: string;
  error?: string;
}

export const useBanStock = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const banStocks = async (stocks: { ticker: string; duration: number }[]) => {
    setIsLoading(true);
    setError(null);

    try {
      const baseUrl = API_CONFIG.baseURL;
      const response = await fetch(`${baseUrl}/stock_filtering_app/ban`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ stocks }),
      });

      const data: BanResponse = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to ban stocks');
      }

      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to ban stocks';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return { banStocks, isLoading, error };
};