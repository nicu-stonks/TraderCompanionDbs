import React, { useState, useEffect, memo } from 'react';
import axios from 'axios';
import { Plus, X, Loader2, AlertTriangle, Trash2 } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

interface TickerManagerProps {
  compact?: boolean;  // If true, shows a more compact layout
  title?: string;
  description?: string;
  refreshTrigger?: number;  // Increment this to force a refresh
}

interface Ticker {
  symbol: string;
}

const TickerManagerComponent: React.FC<TickerManagerProps> = ({
  compact = false,
  title = "Monitored Tickers",
  description = "Add or remove tickers from the monitoring list",
  refreshTrigger = 0
}) => {
  const [tickers, setTickers] = useState<Ticker[]>([]);
  const [newTicker, setNewTicker] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [error, setError] = useState('');
  const [tickerToRemove, setTickerToRemove] = useState<string | null>(null);
  const [isRemoving, setIsRemoving] = useState(false);
  const [showDeleteAllConfirm, setShowDeleteAllConfirm] = useState(false);
  const [isRemovingAll, setIsRemovingAll] = useState(false);
  const hasLoadedOnceRef = React.useRef(false);

  const fetchTickers = async () => {
    const shouldShowLoading = !hasLoadedOnceRef.current;
    if (shouldShowLoading) {
      setIsLoading(true);
    }
    try {
      const response = await axios.get('http://localhost:8000/ticker_data/api/ticker_data/tickers');
      setTickers(response.data.tickers.map((symbol: string) => ({ symbol })));
      setError('');
    } catch (err) {
      setError('Failed to fetch tickers');
      console.error('Error fetching tickers:', err);
    } finally {
      if (shouldShowLoading) {
        setIsLoading(false);
      }
      if (!hasLoadedOnceRef.current) {
        hasLoadedOnceRef.current = true
      }
    }
  };

  useEffect(() => {
    fetchTickers();
  }, [refreshTrigger]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchTickers();
    }, 1000);
    return () => clearInterval(interval);
  }, [refreshTrigger]);

  const handleAddTicker = async () => {
    if (!newTicker.trim()) return;

    setIsAdding(true);
    try {
      await axios.post('http://localhost:8000/ticker_data/api/ticker_data/tickers', {
        symbol: newTicker.toUpperCase().trim()
      });
      setNewTicker('');
      await fetchTickers();
      setError('');
    } catch (err) {
      setError('Failed to add ticker');
      console.error('Error adding ticker:', err);
    } finally {
      setIsAdding(false);
    }
  };

  const confirmRemoveTicker = async () => {
    if (!tickerToRemove) return;
    setIsRemoving(true);
    try {
      await axios.delete(`http://localhost:8000/ticker_data/api/ticker_data/tickers/${tickerToRemove}`);
      await fetchTickers();
      setError('');
    } catch (err) {
      setError('Failed to remove ticker');
      console.error('Error removing ticker:', err);
    } finally {
      setIsRemoving(false);
      setTickerToRemove(null);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAddTicker();
    }
  };

  const confirmRemoveAllTickers = async () => {
    setIsRemovingAll(true);
    try {
      await axios.delete('http://localhost:8000/ticker_data/api/ticker_data/tickers');
      setTickers([]);
      setError('');
    } catch (err) {
      setError('Failed to remove all tickers');
      console.error('Error removing all tickers:', err);
    } finally {
      setIsRemovingAll(false);
      setShowDeleteAllConfirm(false);
    }
  };

  if (compact) {
    return (
      <>
        <Dialog open={tickerToRemove !== null} onOpenChange={(open) => !open && setTickerToRemove(null)}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-amber-500" />
                Remove Ticker
              </DialogTitle>
              <DialogDescription>
                Are you sure you want to remove <strong className="text-foreground">{tickerToRemove}</strong> from monitoring?
                <br />
                <span className="mt-2 block text-amber-600 dark:text-amber-400 font-medium">
                  This will stop fetching price data and permanently delete all historical price and volume data for this ticker.
                </span>
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="gap-2 sm:gap-0">
              <Button variant="outline" onClick={() => setTickerToRemove(null)} disabled={isRemoving}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={confirmRemoveTicker} disabled={isRemoving}>
                {isRemoving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Remove
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
        <Dialog open={showDeleteAllConfirm} onOpenChange={(open) => !open && setShowDeleteAllConfirm(false)}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-red-600 dark:text-red-400">
                <Trash2 className="h-5 w-5" />
                Remove All Tickers
              </DialogTitle>
              <DialogDescription>
                Are you sure you want to remove <strong className="text-foreground">all {tickers.length} ticker(s)</strong> from monitoring?
                <br />
                <span className="mt-2 block text-amber-600 dark:text-amber-400 font-medium">
                  ⚠️ This will permanently delete all historical price and volume data for all tickers. This action cannot be undone.
                </span>
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="gap-2 sm:gap-0">
              <Button variant="outline" onClick={() => setShowDeleteAllConfirm(false)} disabled={isRemovingAll}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={confirmRemoveAllTickers} disabled={isRemovingAll}>
                {isRemovingAll ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Remove All
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">{title}</span>
              <span className="text-xs text-muted-foreground">({tickers.length} active)</span>
            </div>
            {tickers.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDeleteAllConfirm(true)}
                className="h-7 px-2 text-xs text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/30"
                title="Remove all tickers"
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear All
              </Button>
            )}
          </div>

          <div className="flex gap-2">
            <Input
              placeholder="Add ticker..."
              value={newTicker}
              onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
              onKeyPress={handleKeyPress}
              className="h-8 text-sm"
              disabled={isAdding}
            />
            <Button
              size="sm"
              onClick={handleAddTicker}
              disabled={!newTicker.trim() || isAdding}
              className="h-8 px-3"
            >
              {isAdding ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
            </Button>
          </div>

          {error && <p className="text-xs text-red-500">{error}</p>}

          <div className="flex flex-wrap gap-1.5">
            {isLoading ? (
              <span className="text-xs text-muted-foreground">Loading...</span>
            ) : tickers.length === 0 ? (
              <span className="text-xs text-muted-foreground">No tickers added</span>
            ) : (
              tickers.map((ticker) => (
                <Badge
                  key={ticker.symbol}
                  variant="secondary"
                  className={`pl-2 ${ticker.symbol === 'SPY' ? 'pr-2' : 'pr-1'} py-0.5 flex items-center gap-1 text-xs`}
                >
                  {ticker.symbol}
                  {ticker.symbol !== 'SPY' && (
                    <button
                      onClick={() => setTickerToRemove(ticker.symbol)}
                      className="ml-1 hover:bg-destructive/20 rounded p-0.5"
                      title="Remove ticker"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  )}
                </Badge>
              ))
            )}
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Dialog open={tickerToRemove !== null} onOpenChange={(open) => !open && setTickerToRemove(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              Remove Ticker
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to remove <strong className="text-foreground">{tickerToRemove}</strong> from monitoring?
              <br />
              <span className="mt-2 block text-amber-600 dark:text-amber-400 font-medium">
                This will stop fetching price data and permanently delete all historical price and volume data for this ticker.
              </span>
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={() => setTickerToRemove(null)} disabled={isRemoving}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmRemoveTicker} disabled={isRemoving}>
              {isRemoving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={showDeleteAllConfirm} onOpenChange={(open) => !open && setShowDeleteAllConfirm(false)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-red-600 dark:text-red-400">
              <Trash2 className="h-5 w-5" />
              Remove All Tickers
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to remove <strong className="text-foreground">all {tickers.length} ticker(s)</strong> from monitoring?
              <br />
              <span className="mt-2 block text-amber-600 dark:text-amber-400 font-medium">
                ⚠️ This will permanently delete all historical price and volume data for all tickers. This action cannot be undone.
              </span>
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={() => setShowDeleteAllConfirm(false)} disabled={isRemovingAll}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmRemoveAllTickers} disabled={isRemovingAll}>
              {isRemovingAll ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Remove All
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
          <div>
            <CardTitle className="text-base">{title}</CardTitle>
            <CardDescription className="text-sm">{description}</CardDescription>
          </div>
          {tickers.length > 0 && (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setShowDeleteAllConfirm(true)}
              title="Remove all tickers"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete All
            </Button>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter ticker symbol (e.g., AAPL)"
              value={newTicker}
              onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
              onKeyPress={handleKeyPress}
              disabled={isAdding}
            />
            <Button onClick={handleAddTicker} disabled={!newTicker.trim() || isAdding}>
              {isAdding ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Plus className="h-4 w-4 mr-2" />}
              Add
            </Button>
          </div>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex flex-wrap gap-2">
            {isLoading ? (
              <span className="text-sm text-muted-foreground">Loading tickers...</span>
            ) : tickers.length === 0 ? (
              <span className="text-sm text-muted-foreground">No tickers being monitored</span>
            ) : (
              tickers.map((ticker) => (
                <Badge
                  key={ticker.symbol}
                  variant="secondary"
                  className={`pl-3 ${ticker.symbol === 'SPY' ? 'pr-3' : 'pr-1.5'} py-1 flex items-center gap-1.5`}
                >
                  <span className="font-mono">{ticker.symbol}</span>
                  {ticker.symbol !== 'SPY' && (
                    <button
                      onClick={() => setTickerToRemove(ticker.symbol)}
                      className="hover:bg-destructive/20 rounded p-0.5 transition-colors"
                      title="Remove ticker"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                </Badge>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </>
  );
};

// Memoize to prevent re-renders when parent state changes
export const TickerManager = memo(TickerManagerComponent);
