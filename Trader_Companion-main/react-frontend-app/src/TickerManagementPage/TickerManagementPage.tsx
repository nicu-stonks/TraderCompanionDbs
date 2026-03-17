import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { Trash2, Loader2 } from 'lucide-react';
import { FetchErrorBanner } from '@/components/FetchErrorBanner';
import { DataSourceSelector } from '@/components/DataSourceSelector';

interface Ticker {
  symbol: string;
}

export const TickerManagementPage: React.FC = () => {
  const [tickers, setTickers] = useState<Ticker[]>([]);
  const [newTicker, setNewTicker] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showDeleteAllConfirm, setShowDeleteAllConfirm] = useState(false);
  const [isRemovingAll, setIsRemovingAll] = useState(false);
  const [showPurgeConfirm, setShowPurgeConfirm] = useState(false);
  const [isPurging, setIsPurging] = useState(false);
  const [purgeResult, setPurgeResult] = useState<string | null>(null);

  // Fetch tickers on component mount
  useEffect(() => {
    fetchTickers();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchTickers(true);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const fetchTickers = async (isPolling = false) => {
    if (!isPolling) setIsLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/ticker_data/api/ticker_data/tickers');
      setTickers(response.data.tickers.map((symbol: string) => ({ symbol })));
      setError('');
    } catch (err) {
      setError('Failed to fetch tickers. Please try again.');
      console.error('Error fetching tickers:', err);
    } finally {
      if (!isPolling) setIsLoading(false);
    }
  };

  const handleAddTicker = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTicker.trim()) {
      setError('Please enter a valid ticker symbol');
      return;
    }

    setIsLoading(true);
    try {
      await axios.post('http://localhost:8000/ticker_data/api/ticker_data/tickers', { symbol: newTicker.toUpperCase() });
      await fetchTickers();
      setNewTicker('');
      setError('');
    } catch (err) {
      setError('Failed to add ticker. It may already exist or be invalid.');
      console.error('Error fetching tickers:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRemoveTicker = async (symbol: string) => {
    setIsLoading(true);
    try {
      await axios.delete(`http://localhost:8000/ticker_data/api/ticker_data/tickers/${symbol}`);
      await fetchTickers();
      setError('');
    } catch (err) {
      setError(`Failed to remove ticker ${symbol}.`);
      console.error(`Error removing ticker ${symbol}:`, err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRemoveAllTickers = async () => {
    setIsRemovingAll(true);
    try {
      await axios.delete('http://localhost:8000/ticker_data/api/ticker_data/tickers');
      setTickers([]);
      setError('');
      setShowDeleteAllConfirm(false);
    } catch (err) {
      setError('Failed to remove all tickers.');
      console.error('Error removing all tickers:', err);
    } finally {
      setIsRemovingAll(false);
    }
  };

  const handlePurgeAllPriceData = async () => {
    setIsPurging(true);
    setPurgeResult(null);
    try {
      const response = await axios.delete('http://localhost:8000/ticker_data/api/ticker_data/purge-all-price-data');
      const { deleted_daily, deleted_5m, deleted_weekly } = response.data;
      setPurgeResult(`Purged ${deleted_daily.toLocaleString()} daily, ${deleted_weekly.toLocaleString()} weekly, and ${deleted_5m.toLocaleString()} 5m bars. Data will be re-fetched automatically.`);
      setShowPurgeConfirm(false);
    } catch (err) {
      setError('Failed to purge price data.');
      console.error('Error purging price data:', err);
    } finally {
      setIsPurging(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 p-4">
      <h1 className="text-2xl font-bold text-foreground">Stock Tickers Monitor</h1>

      {/* Fetch Error Banner */}
      <FetchErrorBanner />

      {/* Data Source Selector - switch between yfinance and Webull */}
      <DataSourceSelector />

      {/* Purge All Price Data */}
      <Card className="bg-background border-muted">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            Data Management
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Clear all cached historical price data (daily, weekly + 5-minute bars) for every ticker. Tracked tickers are kept and data will be re-fetched automatically.
          </p>
          {purgeResult && (
            <Alert className="bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800">
              <AlertDescription className="text-green-800 dark:text-green-200">{purgeResult}</AlertDescription>
            </Alert>
          )}
          <AlertDialog open={showPurgeConfirm} onOpenChange={setShowPurgeConfirm}>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" size="sm">
                <Trash2 className="h-4 w-4 mr-2" />
                Purge All Price Data
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Purge All Price Data?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will permanently delete <strong>all daily, weekly and 5-minute price bars</strong> for every ticker in the database.
                  <br /><br />
                  Tracked tickers will be preserved and data will start re-fetching automatically.
                  <br /><br />
                  <span className="text-amber-600 dark:text-amber-400 font-medium">
                    ⚠️ This action cannot be undone. A full re-fetch may take several minutes depending on the number of tickers.
                  </span>
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel disabled={isPurging}>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handlePurgeAllPriceData}
                  disabled={isPurging}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  {isPurging ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                  Purge All Data
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </CardContent>
      </Card>

      {/* Legacy Manual Controls (Optional) */}
      <Card className="bg-background border-muted">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            Manual Controls
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <form onSubmit={handleAddTicker} className="space-y-4">
            <div className="flex gap-4">
              <Input
                type="text"
                value={newTicker}
                onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
                placeholder="Enter ticker symbol (e.g., AAPL)"
                className="bg-background text-foreground border-input focus:ring-ring"
              />
              <Button
                type="submit"
                disabled={isLoading}
                className="bg-primary text-primary-foreground hover:bg-primary/90 disabled:bg-muted"
              >
                {isLoading ? 'Adding...' : 'Add Ticker'}
              </Button>
            </div>
            {error && <p className="text-destructive text-sm">{error}</p>}
          </form>
        </CardContent>
      </Card>

      {/* Ticker List */}
      <Card className="bg-background border-muted">
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle className="text-foreground">Currently Monitored Tickers</CardTitle>
            <p className="text-sm text-muted-foreground">
              Tickers are automatically managed based on your trading activity
            </p>
          </div>
          {tickers.length > 0 && (
            <AlertDialog open={showDeleteAllConfirm} onOpenChange={setShowDeleteAllConfirm}>
              <AlertDialogTrigger asChild>
                <Button
                  variant="destructive"
                  size="sm"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete All
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Remove All Tickers?</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to remove <strong>all {tickers.length} ticker(s)</strong> from monitoring?
                    <br /><br />
                    <span className="text-amber-600 dark:text-amber-400 font-medium">
                      ⚠️ This will permanently delete all historical price and volume data. This action cannot be undone.
                    </span>
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel disabled={isRemovingAll}>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={handleRemoveAllTickers}
                    disabled={isRemovingAll}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    {isRemovingAll ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                    Remove All
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </CardHeader>
        <CardContent>
          {isLoading && <p className="text-muted-foreground">Loading tickers...</p>}
          {!isLoading && tickers.length === 0 && (
            <p className="text-muted-foreground">
              No tickers are currently being monitored. They will be added automatically when you place trades.
            </p>
          )}
          {!isLoading && tickers.length > 0 && (
            <ul className="space-y-2">
              {tickers.map((ticker) => (
                <li
                  key={ticker.symbol}
                  className="flex justify-between items-center p-2 hover:bg-muted/50 rounded"
                >
                  <div className="flex flex-col">
                    <span className="font-medium text-foreground">{ticker.symbol}</span>
                  </div>
                  {ticker.symbol !== 'SPY' && (
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button
                          variant="destructive"
                          size="sm"
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                          Remove Now
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Remove Ticker?</AlertDialogTitle>
                          <AlertDialogDescription>
                            Are you sure you want to remove <strong>{ticker.symbol}</strong> from monitoring?
                            This action cannot be undone. The ticker will need to be added again manually or
                            will be automatically added when you create a new trade.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => handleRemoveTicker(ticker.symbol)}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                          >
                            Remove
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  )}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
};