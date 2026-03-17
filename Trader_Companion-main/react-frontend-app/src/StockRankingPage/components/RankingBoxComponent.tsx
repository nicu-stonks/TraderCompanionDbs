import React, { useState, useRef } from 'react';
import type { RankingBox } from '../types';
import { useStockOperations } from '../hooks/useStockPickOperations';
import { stockPicksApi } from '../services/stockPick';
import { rankingBoxesApi } from '../services/rankingBoxes';
import { Alert } from '@/components/ui/alert';
import { RankingItem } from './RankingItem';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { X, Plus, Check, AlertTriangle, Trash2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
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
} from "@/components/ui/alert-dialog";
import { tradeAPI } from '@/TradeStatisticsPage/services/tradeAPI';
import type { Trade } from '@/TradeHistoryPage/types/Trade';

interface Props {
  box: RankingBox;
  onRemoveBox: (id: number) => void;
  onUpdateBox: (boxId: number, updatedBox: RankingBox) => void;
  onDeleteAllStocks?: () => void;
}

export const RankingBoxComponent: React.FC<Props> = ({
  box,
  onRemoveBox,
  onUpdateBox,
  onDeleteAllStocks
}) => {
  const [showAddForm, setShowAddForm] = useState(false);
  const [newStock, setNewStock] = useState({ symbol: '' });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState(box.title);
  const [titleError, setTitleError] = useState<string | null>(null);
  const [isCheckingTrades, setIsCheckingTrades] = useState(false);
  const [missingTradeSymbols, setMissingTradeSymbols] = useState<string[]>([]);
  const checkedOnceRef = useRef(false);

  const { error, handleStockUpdate, handleRemoveStock, sortStocksByScore } = useStockOperations({
    onUpdateBox
  });

  const handleAddStock = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newStock.symbol.trim()) return;

    try {
      setIsSubmitting(true);
      const response = await stockPicksApi.createStockPick({
        symbol: newStock.symbol.trim().toUpperCase(),
        ranking_box: box.id,
        total_score: 0
      });
      onUpdateBox(box.id, { ...box, stock_picks: [...box.stock_picks, response.data] });
      setNewStock({ symbol: '' });
      setShowAddForm(false);
    } catch (err) {
      console.error('Error adding stock:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleTitleEdit = async () => {
    const trimmedTitle = editedTitle.trim();
    if (!trimmedTitle) {
      setTitleError('Title cannot be empty');
      return;
    }
    if (trimmedTitle === box.title) {
      setIsEditingTitle(false);
      return;
    }

    try {
      setIsSubmitting(true);
      setTitleError(null);
      const response = await rankingBoxesApi.updateRankingBox(box.id, trimmedTitle);
      onUpdateBox(box.id, { ...box, title: response.data.title });
      setIsEditingTitle(false);
    } catch (err) {
      console.error('Error updating title:', err);
      setTitleError('Failed to update title');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleTitleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleTitleEdit();
    } else if (e.key === 'Escape') {
      setEditedTitle(box.title);
      setIsEditingTitle(false);
      setTitleError(null);
    }
  };

  return (
    <Card className="bg-card border-2 border-primary ring-1 ring-primary/50 shadow-md hover:ring-primary transition-colors rounded-sm">
      <CardHeader className="p-2 flex-row justify-between items-center space-y-0">
        <div className="flex items-center gap-2 w-full">
          {isEditingTitle ? (
            <div className="flex items-center gap-1">
              <Input
                type="text"
                value={editedTitle}
                onChange={(e) => setEditedTitle(e.target.value)}
                onKeyDown={handleTitleKeyDown}
                onBlur={handleTitleEdit}
                className="h-8 text-sm font-semibold"
                autoFocus
                disabled={isSubmitting}
              />
              <Button
                size="sm"
                className="h-8 w-8 p-0"
                onClick={handleTitleEdit}
                disabled={isSubmitting}
              >
                <Check className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <Badge
              variant="default"
              className="text-base font-semibold cursor-pointer hover:bg-primary/90"
              onClick={() => setIsEditingTitle(true)}
            >
              {box.title}
            </Badge>
          )}

          {onDeleteAllStocks && box.stock_picks.length > 0 && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-muted-foreground hover:text-destructive"
                  title="Delete all stocks in this box"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete All Stocks in "{box.title}"</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete all {box.stock_picks.length} stocks in this box? This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    className="bg-destructive hover:bg-destructive/90"
                    onClick={onDeleteAllStocks}
                  >
                    Delete All
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
          
          {!showAddForm ? (
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs rounded-sm ml-auto"
              onClick={() => setShowAddForm(true)}
            >
              <Plus className="h-3 w-3 mr-1" /> Add Stock
            </Button>
          ) : (
            <form onSubmit={handleAddStock} className="flex gap-1 ml-auto">
              <Input
                type="text"
                value={newStock.symbol}
                onChange={(e) => setNewStock({ symbol: e.target.value })}
                placeholder="Stock symbol"
                className="h-8 text-xs rounded-sm"
                disabled={isSubmitting}
              />
              <Button
                type="submit"
                size="sm"
                disabled={isSubmitting || !newStock.symbol.trim()}
                className="h-8 text-xs"
              >
                {isSubmitting ? '...' : 'Add'}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => {
                  setShowAddForm(false);
                  setNewStock({ symbol: '' });
                }}
                className="h-8 text-xs"
              >
                Cancel
              </Button>
            </form>
          )}
          
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8 text-muted-foreground hover:text-destructive"
                onClick={async () => {
                  // Lazy check only once per open lifecycle unless stocks change
                  if (!checkedOnceRef.current) {
                    try {
                      setIsCheckingTrades(true);
                      const resp = await tradeAPI.getTrades();
                      const trades: Trade[] = resp.data;
                      const tradeSet = new Set(trades.map(t => t.Ticker.toUpperCase()));
                      const missing = box.stock_picks
                        .map(s => s.symbol.toUpperCase())
                        .filter(sym => !tradeSet.has(sym));
                      setMissingTradeSymbols(missing);
                      checkedOnceRef.current = true;
                    } catch {
                      setMissingTradeSymbols([]);
                    } finally {
                      setIsCheckingTrades(false);
                    }
                  }
                }}
              >
                <X className="h-4 w-4" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete Ranking Box</AlertDialogTitle>
                <AlertDialogDescription>
                  Are you sure you want to delete "{box.title}"? This action cannot be undone.
                  {isCheckingTrades && (
                    <div className="mt-2 text-xs text-muted-foreground">Checking trade history...</div>
                  )}
                  {!isCheckingTrades && missingTradeSymbols.length > 0 && (
                    <div className="mt-3 text-amber-600 dark:text-amber-400 text-sm flex gap-2">
                      <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                      <span>
                        The following tickers have <strong>no trade</strong> in history yet: {missingTradeSymbols.slice(0,8).join(', ')}{missingTradeSymbols.length>8?'…':''}. If you add trades first, each trade's Case field will auto-store the ranking JSON so you can review it after deletion.
                      </span>
                    </div>
                  )}
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  className="bg-destructive hover:bg-destructive/90"
                  onClick={() => onRemoveBox(box.id)}
                >
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </CardHeader>

      <CardContent className="p-2 space-y-2">
        {(error || titleError) && (
          <Alert variant="destructive" className="py-1 text-xs">
            {error || titleError}
          </Alert>
        )}

        <div className="space-y-1">
          {sortStocksByScore(box.stock_picks).map((stock) => (
            <RankingItem
              key={stock.id}
              stock={stock}
              onUpdate={(updatedStock) => handleStockUpdate(box.id, updatedStock, box)}
              onRemove={() => handleRemoveStock(box.id, stock.id, box)}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
};