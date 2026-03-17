// TradeHistoryPage.tsx
import React, { useEffect, useState } from 'react';
import { Trade } from './types/Trade';
import { tradeAPI } from './services/tradeAPI';
import { TradesTable } from './components/TradesTable';
import { AddTradeComponent } from './components/AddTradeComponent';
import { IBKRImportComponent } from './components/IBKRImportComponent';
import { ColumnManagerComponent } from './components/ColumnManagerComponent';
import { useCustomTradeData } from './hooks/useCustomTradeData';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2 } from "lucide-react";
import { balanceAPI } from '../TradeStatisticsPage/services/balanceAPI';

export const TradeHistoryPage: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [currentBalance, setCurrentBalance] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const customTradeData = useCustomTradeData();

  useEffect(() => {
    loadTrades();

    // Poll for trades every 500ms to keep table in sync
    const interval = setInterval(() => {
      tradeAPI.getTrades().then(response => {
        setTrades(response.data);
      }).catch(err => {
        console.error('Error polling trades:', err);
      });
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const loadTrades = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const [tradesResp, balance] = await Promise.all([
        tradeAPI.getTrades(),
        balanceAPI.getBalance()
      ]);
      setTrades(tradesResp.data);
      setCurrentBalance(balance);
    } catch (err) {
      setError('Failed to load data. Please refresh the page.');
      console.error('Error loading data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddTrade = async (newTrade: Trade): Promise<void> => {
    const response = await tradeAPI.addTrade(newTrade);
    // Update local state optimistically
    setTrades(prevTrades => [...prevTrades, response.data]);
  };

  const handleUpdateTrade = async (updatedTrade: Trade) => {
    try {
      const response = await tradeAPI.updateTrade(updatedTrade);
      // Update local state optimistically
      setTrades(prevTrades =>
        prevTrades.map(trade =>
          trade.ID === updatedTrade.ID ? response.data : trade
        )
      );
    } catch (err) {
      console.error('Error updating trade:', err);
      // Reload trades to ensure consistency
      loadTrades();
      throw err;
    }
  };

  const handleDeleteTrade = async (id: number) => {
    try {
      await tradeAPI.deleteTrade(id);
      // Update local state optimistically
      setTrades(prevTrades => prevTrades.filter(trade => trade.ID !== id));
    } catch (err) {
      console.error('Error deleting trade:', err);
      // Reload trades to ensure consistency
      loadTrades();
      throw err;
    }
  };

  if (isLoading) {
    return (
      <div className="w-full h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="w-full px-4 py-6 space-y-6">
      <div className="max-w-[95vw] mx-auto">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="space-y-6">
          <div className="rounded-lg shadow bg-background">
            <TradesTable
              trades={trades}
              onUpdate={handleUpdateTrade}
              onDelete={handleDeleteTrade}
              customTradeData={customTradeData}
            />
          </div>

          <div className="rounded-lg shadow bg-background">
            <IBKRImportComponent
              onAdd={handleAddTrade}
              existingTrades={trades}
              customTradeData={customTradeData}
              currentBalance={currentBalance}
            />
          </div>

          <div className="rounded-lg shadow bg-background">
            <ColumnManagerComponent customTradeData={customTradeData} />
          </div>

          <div className="bg-white rounded-lg shadow">
            <AddTradeComponent onAdd={handleAddTrade} existingTrades={trades} customTradeData={customTradeData} />
          </div>
        </div>
      </div>
    </div>
  );
};