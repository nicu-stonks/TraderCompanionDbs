// hooks/useRankingBoxes.ts
import { useState, useEffect, useCallback } from 'react';
import { RankingBox, UserPageState } from '../types';
import { rankingBoxesApi } from '../services/rankingBoxes';
import { userPageStateApi } from '../services/userPageState';

export const useRankingBoxes = () => {
  const [rankingBoxes, setRankingBoxes] = useState<RankingBox[]>([]);
  const [pageState, setPageState] = useState<UserPageState>({
    column_count: 3,
    ranking_boxes_order: [],
    updated_at: ''
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const sortBoxes = (boxes: RankingBox[]) => {
    return [...boxes].sort((a, b) => b.stock_picks.length - a.stock_picks.length);
  };

  const fetchData = useCallback(async () => {
    try {
      setIsLoading(true);
      const [boxesResponse, pageStateResponse] = await Promise.all([
        rankingBoxesApi.getRankingBoxes(),
        userPageStateApi.getUserPageState()
      ]);

      const sortedBoxes = sortBoxes(boxesResponse.data);

      setRankingBoxes(sortedBoxes);
      setPageState(pageStateResponse.data);
    } catch (err) {
      setError('Failed to fetch data');
      console.error('Error fetching data:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleColumnCountChange = async (count: number) => {
    try {
      await userPageStateApi.updateUserPageState({
        ...pageState,
        column_count: count
      });
      setPageState(prev => ({ ...prev, column_count: count }));
    } catch (err) {
      setError('Failed to update column count');
      console.error('Error updating column count:', err);
    }
  };

  const handleRemoveBox = async (id: number) => {
    try {
      await rankingBoxesApi.deleteRankingBox(id);
      setRankingBoxes(prev => sortBoxes(prev.filter(box => box.id !== id)));
    } catch (err) {
      setError('Failed to delete box');
      console.error('Error deleting box:', err);
    }
  };

  const handleUpdateStock = async (boxId: number, updatedBox: RankingBox) => {
    setRankingBoxes(prev => 
      sortBoxes(prev.map(box => box.id === boxId ? updatedBox : box))
    );
  };

  const handleDeleteAllBoxes = async () => {
    try {
      await rankingBoxesApi.deleteAllRankingBoxes();
      setRankingBoxes([]);
      setPageState(prev => ({ ...prev, ranking_boxes_order: [] }));
    } catch (err) {
      setError('Failed to delete all boxes');
      console.error('Error deleting all boxes:', err);
    }
  };

  const refreshBoxes = () => {
    fetchData();
  };

  return {
    rankingBoxes,
    pageState,
    isLoading,
    error,
    handleColumnCountChange,
    handleRemoveBox,
    handleUpdateStock,
    handleDeleteAllBoxes,
    refreshBoxes
  };
};