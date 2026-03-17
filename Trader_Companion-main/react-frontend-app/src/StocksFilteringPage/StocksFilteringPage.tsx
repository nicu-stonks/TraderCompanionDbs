import React from 'react';
import { StocksScreenerCommander } from './components/StocksScreenerCommander';
import { RankingList } from '@/StocksFilteringPage/components/RankingList';
import { PipelineStatus } from '@/StocksFilteringPage/components/PipelineStatus';

export const StocksRankingPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="space-y-8">
        <StocksScreenerCommander />
      </div>
      <PipelineStatus pollingInterval={1500} />
      
      <div className="space-y-8 mt-8">
        
        <RankingList 
          filename="minervini_1mo/stocks_ranking_by_price.csv" 
          title="Minervini Trend Between 1 Month and 4 Months (Good for Power Plays)" 
        />

        <RankingList 
          filename="ipos/stocks_ranking_by_price.csv" 
          title="IPO's" 
        />

        <RankingList 
          filename="minervini_4mo/stocks_ranking_by_price.csv" 
          title="Minervini Trend At least 4 Months (Good for Breakouts)" 
        />
{/* 
        <RankingList 
          filename="minervini_1mo_unbanned/stocks_ranking_by_price.csv" 
          title="Minervini Trend At least 1 Month" 
        /> */}
      </div>
    </div>
  );
};

export default StocksRankingPage;