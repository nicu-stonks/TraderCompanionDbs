import React, { useState, useMemo } from 'react';
import { useRankingList } from '../hooks/useRankingList';
import { useBanStock } from '../hooks/useBanStock';

interface RankingListProps {
  filename: string;
  title?: string;
}

export const RankingList: React.FC<RankingListProps> = ({ filename, title }) => {
  const { rankings, loading, error } = useRankingList(filename);
  const { banStocks, error: banError } = useBanStock();
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [bannedStocks, setBannedStocks] = useState<Record<string, boolean>>({});
  const [pendingBans, setPendingBans] = useState<Record<string, boolean>>({});
  const [lastClickedRow, setLastClickedRow] = useState<string | null>(null);
  const [hoveredColumn, setHoveredColumn] = useState<string | null>(null);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const handleBanStock = async (ticker: string, durationWeeks: number) => {
    // Mark the stock as pending ban immediately for UI feedback
    setPendingBans(prev => ({ ...prev, [ticker]: true }));
    
    try {
      // Fire off the ban request without awaiting its completion
      banStocks([{ ticker, duration: durationWeeks }])
        .then(() => {
          // When the request completes successfully, update banned status
          setBannedStocks(prev => ({ ...prev, [ticker]: true }));
        })
        .catch((err) => {
          console.error('Failed to ban stock:', err);
        })
        .finally(() => {
          // Always remove from pending state when request finishes
          setPendingBans(prev => {
            const updated = { ...prev };
            delete updated[ticker];
            return updated;
          });
        });
    } catch (err) {
      console.error('Failed to initiate ban stock request:', err);
      // Remove from pending if we couldn't even start the request
      setPendingBans(prev => {
        const updated = { ...prev };
        delete updated[ticker];
        return updated;
      });
    }
  };

  const handleRowClick = (symbol: string) => {
    setLastClickedRow(symbol);
  };

  const sortedRankings = useMemo(() => {
    if (!rankings?.message || !sortColumn) return rankings?.message || [];

    return [...rankings.message].sort((a, b) => {
      const valueA = a[sortColumn];
      const valueB = b[sortColumn];

      // Push missing values to the bottom
      const isValueAMissing = valueA === null || valueA === undefined || valueA === '';
      const isValueBMissing = valueB === null || valueB === undefined || valueB === '';

      if (isValueAMissing && isValueBMissing) return 0;
      if (isValueAMissing) return 1; // Always move missing values down
      if (isValueBMissing) return -1;

      if (typeof valueA === 'number' && typeof valueB === 'number') {
        return sortDirection === 'asc' ? valueA - valueB : valueB - valueA;
      }

      const strA = String(valueA).toLowerCase();
      const strB = String(valueB).toLowerCase();

      return sortDirection === 'asc' ? strA.localeCompare(strB) : strB.localeCompare(strA);
    });
  }, [rankings, sortColumn, sortDirection]);

  if (loading) {
    return <div className="bg-background rounded-lg shadow-sm p-4">Loading rankings...</div>;
  }

  if (error) {
    const isNoResultsYet = /No screening results yet/i.test(error);
    if (isNoResultsYet) {
      return (
        <div className="bg-background rounded-lg shadow-sm p-4 text-sm space-y-2">
          <div className="font-semibold">No Screening Results Yet</div>
          <p className="text-muted-foreground">You haven't generated any rankings. Use the <strong>Stock Screener Commander</strong> panel and click <em>Start Screening</em> while TWS with the API enabled on port 7497 is running. When the pipeline finishes, the ranking list will appear here automatically.</p>
        </div>
      );
    }
    return <div className="bg-background rounded-lg shadow-sm p-4 text-destructive">Error: {error}</div>;
  }

  if (!sortedRankings.length) {
    return <div className="bg-background rounded-lg shadow-sm p-4">No data available</div>;
  }

  // Get filtered and total counts from API response
  const filteredStocks = rankings?.filtered_stocks ?? sortedRankings.length;
  const totalStocks = rankings?.total_stocks ?? 0;

  // All available columns
  const allAvailableColumns = Object.keys(sortedRankings[0]);
  
  // Define the exact order we want for specific columns
  const priorityColumns = ['Symbol', 'Price_Increase_Percentage', 'Screeners'];
  
  // Add EPS_Quarters and Revenue_Quarters after Screeners if they exist
  if (allAvailableColumns.includes('EPS_Quarters')) {
    priorityColumns.push('EPS_Quarters');
  }
  
  if (allAvailableColumns.includes('Revenue_Quarters')) {
    priorityColumns.push('Revenue_Quarters');
  }
  
  // Add remaining columns that weren't explicitly ordered
  const remainingColumns = allAvailableColumns.filter(
    (key) => !priorityColumns.includes(key)
  );
  
  // Final column order
  const orderedColumns = [...priorityColumns, ...remainingColumns];

  return (
    <div className="bg-background rounded-lg shadow-sm">
      <div className="p-4 border-b border-border flex justify-between items-start">
        <div className="text-lg font-semibold flex items-center gap-2">
          <div className="flex gap-2 items-center">
            <span className="text-base px-2 py-0.5 rounded-md bg-primary/20">
              {filteredStocks} Filtered Stocks
            </span>
            {totalStocks > 0 && (
              <span className="text-base px-2 py-0.5 rounded-md bg-primary/20">
                {totalStocks} Total Stocks
              </span>
            )}
          </div>
          <button 
            className="text-xs px-2 py-0.5 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            onClick={() => {
              // Format data with the correct format
              const tickers = sortedRankings.map(item => item.Symbol);
              
              let formattedContent = "CSVEXPORT\nCOLUMN,0\n";
              formattedContent += tickers.map(ticker => 
                `SYM,${ticker},SMART/AMEX,`
              ).join('\n');
              
              // Create blob and download link
              const blob = new Blob([formattedContent], { type: 'text/plain' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = 'export.csv';
              document.body.appendChild(a);
              a.click();
              
              // Clean up
              setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }, 100);
            }}
          >
            Download List
          </button>
          {title || 'Stock Rankings'}
        </div>
        <div className="text-sm text-muted-foreground text-right">
          {rankings?.rankings_created_at && (
            <p>Current Ranking List Last Update: {new Date(rankings.rankings_created_at).toLocaleString()}</p>
          )}
          {rankings?.stock_data_created_at && (
            <p>Stock Data Last Update: {new Date(rankings.stock_data_created_at).toLocaleString()}</p>
          )}
        </div>
      </div>
        
      {banError && (
        <div className="p-2 bg-destructive/10 text-destructive text-sm">
          Error when banning stocks: {banError}
        </div>
      )}
  
      <div className="overflow-auto">
        <table className="w-full text-sm border border-border">
          <thead>
            <tr className="border-b bg-muted text-muted-foreground">
              {orderedColumns.map((column) => (
                <th
                  key={column}
                  className={`px-2 py-1 cursor-pointer text-left border-r border-border transition-colors duration-150 
                    ${hoveredColumn === column ? 'bg-secondary/30' : ''}`}
                  onClick={() => handleSort(column)}
                  onMouseEnter={() => setHoveredColumn(column)}
                  onMouseLeave={() => setHoveredColumn(null)}
                >
                  {column.replace(/_/g, ' ')}{' '}
                  {sortColumn === column && (
                    <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                  )}
                </th>
              ))}
              <th 
                className={`px-2 py-1 text-left border-r border-border transition-colors duration-150
                  ${hoveredColumn === 'actions' ? 'bg-secondary/30' : ''}`}
                onMouseEnter={() => setHoveredColumn('actions')}
                onMouseLeave={() => setHoveredColumn(null)}
              >
                Ban Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedRankings.map((item, rowIndex) => (
              <tr
                key={item.Symbol}
                className={`border-b 
                  ${rowIndex % 2 === 0 ? 'bg-muted/20' : 'bg-background'} 
                  ${bannedStocks[item.Symbol] ? 'opacity-50' : ''} 
                  ${lastClickedRow === item.Symbol ? 'bg-primary/20' : ''} 
                  hover:bg-primary/20 hover:shadow-sm transition-all duration-150`}
                onClick={() => handleRowClick(item.Symbol)}
              >
                {orderedColumns.map((column) => (
                  <td 
                    key={column} 
                    className={`px-2 py-0.5 border-r border-border transition-colors duration-150
                      ${hoveredColumn === column ? 'bg-secondary/20' : ''}`}
                  >
                    {typeof item[column] === 'number'
                      ? Math.round(item[column]) // Convert numbers to integers
                      : item[column] ?? '-'}
                  </td>
                ))}
                <td 
                  className={`px-2 py-0.5 border-r border-border transition-colors duration-150
                    ${hoveredColumn === 'actions' ? 'bg-secondary/20' : ''}`}
                >
                  <div className="flex gap-2">
                    <button
                      className={`text-white w-full text-xs font-medium py-0.5 px-2 rounded
                        ${pendingBans[item.Symbol] 
                          ? 'bg-blue-500/40' 
                          : 'bg-red-500/40 hover:bg-red-600/80'}`}
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent row click when clicking the button
                        handleBanStock(item.Symbol, 4);
                      }}
                      disabled={bannedStocks[item.Symbol]}
                    >
                      {pendingBans[item.Symbol] ? '...' : '1Mo'}
                    </button>
                    <button
                      className={`text-white w-full text-xs font-medium py-0.5 px-2 rounded
                        ${pendingBans[item.Symbol] 
                          ? 'bg-blue-500/40' 
                          : 'bg-red-500/40 hover:bg-red-600/80'}`}
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent row click when clicking the button
                        handleBanStock(item.Symbol, 12);
                      }}
                      disabled={bannedStocks[item.Symbol]}
                    >
                      {pendingBans[item.Symbol] ? '...' : '3Mo'}
                    </button>
                    <button
                      className={`text-white w-full text-xs font-medium py-0.5 px-2 rounded
                        ${pendingBans[item.Symbol] 
                          ? 'bg-blue-500/40' 
                          : 'bg-red-500/40 hover:bg-red-600/80'}`}
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent row click when clicking the button
                        handleBanStock(item.Symbol, 48);
                      }}
                      disabled={bannedStocks[item.Symbol]}
                    >
                      {pendingBans[item.Symbol] ? '...' : '1Y'}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};