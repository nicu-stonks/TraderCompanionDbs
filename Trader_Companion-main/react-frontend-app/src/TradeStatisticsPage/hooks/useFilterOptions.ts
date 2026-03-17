// hooks/useFilterOptions.ts
import { useState, useEffect } from 'react';
import { Trade } from '@/TradeHistoryPage/types/Trade';
import { tradeAPI } from '../services/tradeAPI';

// Helper type to store the unique values
type FilterOptionsValue = Set<string | number | boolean>;
type FilterOptions = {
  [K in keyof Trade]?: FilterOptionsValue;
};

export const useFilterOptions = () => {
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const response = await tradeAPI.getTrades();
        const trades = response.data;

        // Initialize options object
        const options: FilterOptions = {};

        // Get unique values for each field
        trades.forEach((trade) => {
          Object.entries(trade).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
              if (!options[key as keyof Trade]) {
                options[key as keyof Trade] = new Set<string | number | boolean>();
              }
              // For boolean values, store them as-is
              // For other types, convert to string for consistent handling
              if (typeof value === 'boolean') {
                options[key as keyof Trade]?.add(value);
              } else {
                options[key as keyof Trade]?.add(value.toString());
              }
            }
          });
        });

        setFilterOptions(options);
      } catch (error) {
        console.error('Error fetching trades for filter options:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();
  }, []);

  return { filterOptions, loading };
};