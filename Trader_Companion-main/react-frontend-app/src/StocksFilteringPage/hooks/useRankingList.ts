// hooks/useRankingList.ts
import { useState, useEffect, useCallback } from 'react';
import { RankingListSuccessResponse } from '../types/rankingList';
import { rankingService } from '../services/rankingService';

export const useRankingList = (fileName: string) => {
  const [rankings, setRankings] = useState<RankingListSuccessResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRankings = useCallback(async () => {
    try {
      setLoading(true);
      const data = await rankingService.fetchRankingList(fileName);
      setRankings(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [fileName]); // Memoize fetchRankings based on fileName now

  useEffect(() => {
    fetchRankings();
  }, [fileName, fetchRankings]); // Dependencies updated

  return { rankings, loading, error, refetch: fetchRankings };
};