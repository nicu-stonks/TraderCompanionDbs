import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { RankingItem } from './RankingItem';
import { Download, Trash2 } from 'lucide-react';
import type { StockPick } from '../types';
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

interface Props {
  allStocks: StockPick[];
  onStockUpdate?: (boxId: number, updatedStock: StockPick) => void;
  onRemoveStock?: (boxId: number, stockId: number) => void;
  onDeleteAll?: () => void;
}

export const MainRankingList: React.FC<Props> = ({
  allStocks,
  onStockUpdate,
  onRemoveStock,
  onDeleteAll
}) => {
  const [sortedStocks, setSortedStocks] = useState<StockPick[]>([]);

  useEffect(() => {
    setSortedStocks([...allStocks].sort((a, b) => b.total_score - a.total_score));
  }, [allStocks]);

  const handleStockUpdate = (updatedStock: StockPick) => {
    setSortedStocks(prev => {
      const newStocks = prev.map(stock =>
        stock.id === updatedStock.id ? updatedStock : stock
      );
      return [...newStocks].sort((a, b) => b.total_score - a.total_score);
    });
    
    if (onStockUpdate) {
      onStockUpdate(updatedStock.ranking_box, updatedStock);
    }
  };

  const handleRemoveStock = (stock: StockPick) => {
    setSortedStocks(prev => prev.filter(s => s.id !== stock.id));
    
    if (onRemoveStock) {
      onRemoveStock(stock.ranking_box, stock.id);
    }
  };

  const handleDownload = () => {
    // Generate the stock list in the specified format
    let content = 'CSVEXPORT\nCOLUMN,0\n';
    
    sortedStocks.forEach(stock => {
      content += `SYM,${stock.symbol},SMART/AMEX,\n`;
    });
    
    // Create a blob and download it
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'stock_list.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Card className="w-full rounded-sm bg-background border-border h-full">
      <CardHeader className="px-3 py-2 border-b border-border relative">
        <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-sm font-medium text-muted-foreground">
          Total stocks: {sortedStocks.length}
        </span>
        <CardTitle className="text-lg font-medium text-foreground text-center">Overall Ranking</CardTitle>
        <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
          <button 
            onClick={handleDownload} 
            className="text-muted-foreground hover:text-foreground transition-colors"
            title="Download stock list"
          >
            <Download size={16} />
          </button>

          {onDeleteAll && sortedStocks.length > 0 && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <button 
                  className="text-muted-foreground hover:text-destructive transition-colors ml-1"
                  title="Delete all rankings"
                >
                  <Trash2 size={16} />
                </button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete All Overall Rankings</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete all {sortedStocks.length} stock rankings? This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    className="bg-destructive hover:bg-destructive/90"
                    onClick={onDeleteAll}
                  >
                    Delete All
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y divide-border">
          {sortedStocks.map((stock) => (
            <div key={stock.id} className="px-2 py-1 hover:bg-muted/50 transition-colors">
              <RankingItem
                stock={stock}
                onUpdate={handleStockUpdate}
                onRemove={() => handleRemoveStock(stock)}
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};


export default MainRankingList;
