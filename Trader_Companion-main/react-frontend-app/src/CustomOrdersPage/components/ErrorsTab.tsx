import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { ErrorLog } from '../types';

interface Props {
  errors: ErrorLog[];
  fetchErrors: () => void;
  triggerFlash: (key: string) => void;
  flashRefresh: boolean;
  subtleFlashClass: string;
}

export const ErrorsTab: React.FC<Props> = ({ errors, fetchErrors, triggerFlash, flashRefresh, subtleFlashClass }) => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Error Log</h3>
        <button onClick={() => { fetchErrors(); triggerFlash('refresh'); }} className={`flex items-center gap-2 px-3 py-1 bg-muted rounded-lg hover:bg-muted/80 transition-shadow duration-200 ${flashRefresh ? subtleFlashClass : ''}`}>
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>
      <div className="space-y-3">
        {errors.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <AlertCircle className="w-12 h-12 mx-auto mb-4 text-muted-foreground/50" />
            <p>No errors found</p>
          </div>
        ) : (
          errors.map((error, index) => (
            <div key={index} className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/50 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5" />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-red-700 dark:text-red-300">{error.error_type}</span>
                    <span className="text-sm text-red-600 dark:text-red-400">{error.timestamp}</span>
                  </div>
                  <p className="text-red-800 dark:text-red-200 mb-2">{error.error_message}</p>
                  {error.ticker && (
                    <p className="text-sm text-red-700 dark:text-red-300 bg-red-100 dark:bg-red-900/40 p-2 rounded">Ticker: {error.ticker}</p>
                  )}
                  {error.trade_data && (
                    <p className="text-sm text-red-700 dark:text-red-300 bg-red-100 dark:bg-red-900/40 p-2 rounded mt-2">Trade Data: {JSON.stringify(error.trade_data)}</p>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
