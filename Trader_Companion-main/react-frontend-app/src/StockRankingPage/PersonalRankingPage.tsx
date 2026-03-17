// PersonalRankingPage.tsx
import React from 'react';
import { ConfigurationBox } from './components/ConfigurationBox';
import { MainRankingList } from './components/MainRankingBox';
import { RankingBoxComponent } from './components/RankingBoxComponent';
import { useRankingBoxes } from './hooks/useRankingBoxes';
import { useStockOperations } from './hooks/useStockPickOperations';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2 } from 'lucide-react';
import GlobalCharacteristicsManager from './components/GlobalCharacteristicsManager';
import CharacteristicMetaManager from './components/CharacteristicMetaManager';

export const PersonalRankingPage: React.FC = () => {
  const { 
    rankingBoxes, 
    pageState, 
    isLoading, 
    error: rankingBoxError, 
    handleColumnCountChange, 
    handleRemoveBox, 
    handleUpdateStock, 
    handleDeleteAllBoxes,
    refreshBoxes 
  } = useRankingBoxes();

  const { 
    error: stockError, 
    handleStockUpdate, 
    handleRemoveStock,
    handleDeleteAllStocksInBox,
    handleDeleteAllOverallRankings
  } = useStockOperations({ onUpdateBox: handleUpdateStock });

  const allStocks = rankingBoxes.flatMap(box => 
    box.stock_picks.map(stock => ({ 
      ...stock, 
      ranking_box: box.id 
    }))
  );

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  if (rankingBoxError || stockError) {
    return (
      <Alert variant="destructive" className="m-1">
        <AlertDescription>{rankingBoxError || stockError}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="min-h-screen bg-background p-1">
      <div className="space-y-1">
        <MainRankingList 
          allStocks={allStocks}
          onStockUpdate={(boxId, updatedStock) => {
            const box = rankingBoxes.find(b => b.id === boxId);
            if (box) handleStockUpdate(boxId, updatedStock, box);
          }}
          onRemoveStock={(boxId, stockId) => {
            const box = rankingBoxes.find(b => b.id === boxId);
            if (box) handleRemoveStock(boxId, stockId, box);
          }}
          onDeleteAll={() => handleDeleteAllOverallRankings(rankingBoxes)}
        />

        <ConfigurationBox
          columnCount={pageState.column_count}
          onColumnCountChange={handleColumnCountChange}
          onRankingBoxCreated={refreshBoxes}
          onDeleteAllBoxes={handleDeleteAllBoxes}
        />

        <div 
          className="columns-1 sm:columns-2 md:columns-3 lg:columns-4 gap-1"
          style={{
            columnCount: pageState.column_count
          }}
        >
          {rankingBoxes.map(box => (
            <div key={box.id} className="break-inside-avoid mb-1">
              <RankingBoxComponent
                box={box}
                onRemoveBox={handleRemoveBox}
                onUpdateBox={handleUpdateStock}
                onDeleteAllStocks={() => handleDeleteAllStocksInBox(box.id, box)}
              />
            </div>
          ))}
        </div>
      </div>
      <GlobalCharacteristicsManager />
      <CharacteristicMetaManager />
    </div>
  );
};