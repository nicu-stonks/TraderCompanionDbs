import React from 'react';
import { useGlobalAlerts } from './GlobalAlertContext';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Bell, Power, Trash2, X } from 'lucide-react';

export const GlobalAlertModal: React.FC = () => {
  const {
    triggeredAlerts,
    dismissAllAlerts,
    stopAlert,
    stopAllAlerts,
    deleteAlert,
    deleteAllAlerts,
    isLoading,
  } = useGlobalAlerts();

  if (triggeredAlerts.length === 0) {
    return null;
  }

  const handleStopSingle = async (id: number) => {
    try {
      await stopAlert(id);
    } catch {
      // Error already logged in context
    }
  };

  const handleDeleteSingle = async (id: number) => {
    try {
      await deleteAlert(id);
    } catch {
      // Error already logged in context
    }
  };

  const handleStopAll = async () => {
    try {
      await stopAllAlerts();
    } catch {
      // Error already logged in context
    }
  };

  const handleDeleteAll = async () => {
    try {
      await deleteAllAlerts();
    } catch {
      // Error already logged in context
    }
  };

  const formatPrice = (price: number | null) => {
    if (price === null) return 'N/A';
    return `$${price.toFixed(2)}`;
  };

  // Check if any alerts can still be stopped
  const hasActiveAlerts = triggeredAlerts.some((a) => a.is_active || a.triggered);

  return (
    <Dialog open={true} onOpenChange={() => { }}>
      <DialogContent
        className="sm:max-w-2xl max-h-[85vh] overflow-hidden bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95 border-red-500/50 backdrop-blur-xl shadow-2xl shadow-red-500/20 [&>button]:hidden"
      >
        <DialogHeader className="space-y-3">
          <DialogTitle className="flex items-center gap-3 text-red-400">
            <div className="relative">
              <Bell className="h-6 w-6 animate-pulse" />
              <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-500 rounded-full animate-ping" />
              <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-500 rounded-full" />
            </div>
            <span className="text-xl font-bold">🚨 Alerts Triggered!</span>
            <Badge variant="secondary" className="ml-auto bg-red-500/20 text-red-300 border-red-500/30">
              {triggeredAlerts.length} alert{triggeredAlerts.length !== 1 ? 's' : ''}
            </Badge>
          </DialogTitle>
          <DialogDescription className="sr-only">
            Manage your triggered price alerts
          </DialogDescription>
        </DialogHeader>

        {/* Scrollable alert list */}
        <div className="overflow-y-auto max-h-[45vh] space-y-3 py-2 pr-1">
          {triggeredAlerts.map((alert) => {
            // Alert can be stopped if it's active or triggered
            const canStop = alert.is_active || alert.triggered;

            return (
              <div
                key={alert.id}
                className={`flex items-center gap-4 p-4 rounded-xl transition-colors ${canStop
                  ? 'bg-black/30 border border-red-500/30 hover:border-red-500/50'
                  : 'bg-black/20 border border-gray-600/30 opacity-60'
                  }`}
              >
                {/* Ticker */}
                <div className="min-w-[80px]">
                  <div className="text-2xl font-black text-white tracking-wider">
                    {alert.ticker}
                  </div>
                  {!canStop && (
                    <Badge variant="secondary" className="text-xs mt-1">
                      Stopped
                    </Badge>
                  )}
                </div>

                {/* Prices */}
                <div className="flex-1 grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Alert</div>
                    <div className="text-lg font-bold text-amber-400">
                      {formatPrice(alert.alert_price)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Current</div>
                    <div className="text-lg font-bold text-green-400">
                      {formatPrice(alert.current_price)}
                    </div>
                    {alert.percent_change !== null && (
                      <div
                        className={`text-xs font-semibold ${alert.percent_change >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}
                      >
                        {alert.percent_change >= 0 ? '+' : ''}
                        {alert.percent_change.toFixed(2)}%
                      </div>
                    )}
                  </div>
                </div>

                {/* Individual actions - Stop and Delete */}
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleStopSingle(alert.id)}
                    disabled={isLoading || !canStop}
                    className={
                      canStop
                        ? 'text-amber-400 hover:text-amber-300 hover:bg-amber-900/30'
                        : 'text-gray-500 opacity-30 cursor-not-allowed'
                    }
                    title={canStop ? 'Stop alert' : 'Already stopped'}
                  >
                    <Power className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteSingle(alert.id)}
                    disabled={isLoading}
                    className="text-red-400 hover:text-red-300 hover:bg-red-900/30"
                    title="Delete alert"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            );
          })}
        </div>

        <DialogFooter className="flex flex-col sm:flex-row gap-2 pt-4 border-t border-gray-700/50">
          <Button
            variant="outline"
            onClick={dismissAllAlerts}
            className="flex-1 bg-gray-800/50 border-gray-600 hover:bg-gray-700 text-gray-200"
          >
            <X className="h-4 w-4 mr-2" />
            Close Popup
          </Button>
          <Button
            variant="outline"
            onClick={handleStopAll}
            disabled={isLoading || !hasActiveAlerts}
            className={
              hasActiveAlerts
                ? 'flex-1 bg-amber-900/30 border-amber-600/50 hover:bg-amber-800/50 text-amber-300'
                : 'flex-1 bg-gray-800/30 border-gray-600/50 text-gray-500 opacity-50'
            }
          >
            <Power className="h-4 w-4 mr-2" />
            {isLoading ? 'Stopping...' : 'Stop All Triggered'}
          </Button>
          <Button
            variant="destructive"
            onClick={handleDeleteAll}
            disabled={isLoading}
            className="flex-1 bg-red-900/50 hover:bg-red-800/60 border-red-600/50"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            {isLoading ? 'Deleting...' : 'Delete All Triggered'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
