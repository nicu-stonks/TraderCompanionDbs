import { useState, useCallback } from 'react';
import { ScreeningOptions } from '../types/screenerCommander';
import { sendScreenerCommand } from '../services/stockScreenerService';

export const useStockScreener = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<string | null>(null);

  const sendCommand = useCallback(async (options: ScreeningOptions) => {
    try {
      setLoading(true);
      setError(null);
      const data = await sendScreenerCommand(options);
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch rankings');
    } finally {
      setLoading(false);
    }
  }, []);

  return { response, loading, error, sendCommand };
};