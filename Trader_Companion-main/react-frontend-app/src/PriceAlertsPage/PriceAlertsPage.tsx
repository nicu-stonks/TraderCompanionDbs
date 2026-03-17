import React, { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Alert, AlarmSettings, CreateAlertData } from './types';
import { priceAlertsAPI } from './services/priceAlertsAPI';
import { AlertForm } from './components/AlertForm';
import { AlertsTable } from './components/AlertsTable';
import { AlarmSettingsForm } from './components/AlarmSettingsForm';
import { TelegramSetupPage } from './components/TelegramSetupPage';
import { Alert as AlertComponent, AlertDescription } from '@/components/ui/alert';
import { Loader2, AlertTriangle, Trash2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { FetchErrorBanner } from '@/components/FetchErrorBanner';
import { TickerManager } from '@/components/TickerManager';
import { DataSourceSelector } from '@/components/DataSourceSelector';

export const PriceAlertsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'alerts' | 'telegram'>('alerts');
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alarmSettings, setAlarmSettings] = useState<AlarmSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [priceDirections, setPriceDirections] = useState<Map<number, 'up' | 'down'>>(new Map());
  const lastPricesRef = useRef<Map<number, number | null>>(new Map());

  // State for under-price alert confirmation modal
  const [underPriceAlert, setUnderPriceAlert] = useState<Alert | null>(null);

  // State for delete confirmation modal
  const [deleteConfirmAlert, setDeleteConfirmAlert] = useState<Alert | null>(null);
  const [alsoRemoveTicker, setAlsoRemoveTicker] = useState(false);
  const [alsoDeleteAllTickerAlerts, setAlsoDeleteAllTickerAlerts] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // State for delete all confirmation modal
  const [showDeleteAllConfirm, setShowDeleteAllConfirm] = useState(false);
  const [isDeletingAll, setIsDeletingAll] = useState(false);

  // Trigger for TickerManager refresh (increment to force refresh)
  const [tickerRefreshTrigger, setTickerRefreshTrigger] = useState(0);

  // State for monitored tickers (to compare with alerts)
  const [monitoredTickers, setMonitoredTickers] = useState<string[]>([]);

  const loadAlerts = useCallback(async () => {
    try {
      const response = await priceAlertsAPI.getAlerts();
      const fetchedAlerts = response.data;

      setPriceDirections(prev => {
        const newDirections = new Map(prev);
        const activeIds = new Set<number>();

        fetchedAlerts.forEach(alert => {
          activeIds.add(alert.id);
          const prevPrice = lastPricesRef.current.get(alert.id);
          const currentPrice = alert.current_price;

          if (prevPrice !== undefined && prevPrice !== null && currentPrice !== null) {
            if (currentPrice > prevPrice) {
              newDirections.set(alert.id, 'up');
            } else if (currentPrice < prevPrice) {
              newDirections.set(alert.id, 'down');
            }
          }
        });

        Array.from(newDirections.keys()).forEach(id => {
          if (!activeIds.has(id)) {
            newDirections.delete(id);
          }
        });

        return newDirections;
      });

      lastPricesRef.current = new Map(fetchedAlerts.map(alert => [alert.id, alert.current_price]));
      setAlerts(fetchedAlerts);
    } catch (err) {
      console.error('Error loading alerts:', err);
    }
  }, []);

  const loadAlarmSettings = useCallback(async () => {
    try {
      const response = await priceAlertsAPI.getAlarmSettings();
      setAlarmSettings(response.data);
    } catch (err) {
      console.error('Error loading alarm settings:', err);
    }
  }, []);

  const loadMonitoredTickers = useCallback(async () => {
    try {
      const response = await axios.get('http://localhost:8000/ticker_data/api/ticker_data/tickers');
      setMonitoredTickers(response.data.tickers || []);
    } catch (err) {
      console.error('Error loading monitored tickers:', err);
    }
  }, []);

  const loadData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      await Promise.all([loadAlerts(), loadAlarmSettings(), loadMonitoredTickers()]);
    } catch (err) {
      setError('Failed to load data. Please refresh the page.');
      console.error('Error loading data:', err);
    } finally {
      setIsLoading(false);
    }
  }, [loadAlerts, loadAlarmSettings, loadMonitoredTickers]);

  useEffect(() => {
    loadData();

    // Set up auto-refresh every 0.5 seconds to show updated prices from backend
    const interval = setInterval(() => {
      loadAlerts();
    }, 500);

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [loadData, loadAlerts]);

  // Refresh monitored tickers when tickerRefreshTrigger changes
  useEffect(() => {
    if (tickerRefreshTrigger > 0) {
      loadMonitoredTickers();
    }
  }, [tickerRefreshTrigger, loadMonitoredTickers]);

  const handleCreateAlert = async (data: CreateAlertData) => {
    setIsCreating(true);
    try {
      const response = await priceAlertsAPI.createAlert(data);
      const createdAlert = response.data;

      setAlerts(prev => {
        const updated = [createdAlert, ...prev];
        return updated;
      });

      // Check if the alert price is below the current price
      if (createdAlert.current_price !== null && createdAlert.alert_price < createdAlert.current_price) {
        setUnderPriceAlert(createdAlert);
      }
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { error?: string } } };
      const errorMessage = axiosError.response?.data?.error || 'Failed to create alert';
      alert(errorMessage);
      throw err;
    } finally {
      setIsCreating(false);
      // Refresh TickerManager to show newly added ticker
      setTickerRefreshTrigger(prev => prev + 1);
    }
  };

  const handleConfirmUnderPriceAlert = () => {
    // User wants to keep the alert - just close the modal
    setUnderPriceAlert(null);
  };

  const handleDeleteUnderPriceAlert = async () => {
    if (!underPriceAlert) return;

    try {
      await priceAlertsAPI.deleteAlert(underPriceAlert.id);
      setAlerts(prev => prev.filter(alert => alert.id !== underPriceAlert.id));
    } catch (err: unknown) {
      console.error('Error deleting alert:', err);
      const axiosError = err as { response?: { data?: { error?: string } } };
      const message = axiosError.response?.data?.error || 'Failed to delete alert.';
      alert(message);
    } finally {
      setUnderPriceAlert(null);
    }
  };

  const handleToggleActive = async (id: number, isActive: boolean) => {
    // Optimistic update - disable button immediately
    setAlerts(prev =>
      prev.map(alert =>
        alert.id === id
          ? { ...alert, is_active: isActive, triggered: false }
          : alert
      )
    );

    try {
      const response = await priceAlertsAPI.updateAlert(id, { is_active: isActive });
      // Update with server response
      setAlerts(prev => {
        const updated = prev.map(alert => (alert.id === id ? response.data : alert));
        return updated;
      });
    } catch (err: unknown) {
      console.error('Error updating alert:', err);
      const axiosError = err as { response?: { data?: { error?: string } } };
      const message = axiosError.response?.data?.error || 'Failed to update alert.';
      alert(message);
      // Revert optimistic update on error
      setAlerts(prev =>
        prev.map(alert =>
          alert.id === id
            ? { ...alert, is_active: !isActive }
            : alert
        )
      );
    }
  };

  const handleDelete = (id: number) => {
    // Find the alert to confirm deletion
    const alertToDelete = alerts.find(a => a.id === id);
    if (alertToDelete) {
      setDeleteConfirmAlert(alertToDelete);
      setAlsoRemoveTicker(false);
      setAlsoDeleteAllTickerAlerts(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteConfirmAlert) return;

    setIsDeleting(true);
    try {
      if (alsoDeleteAllTickerAlerts) {
        // Delete all alerts for this ticker
        const tickerAlerts = alerts.filter(a => a.ticker === deleteConfirmAlert.ticker);
        await Promise.all(tickerAlerts.map(a => priceAlertsAPI.deleteAlert(a.id)));
      } else {
        await priceAlertsAPI.deleteAlert(deleteConfirmAlert.id);
      }

      // If user also wants to remove ticker from monitoring
      if (alsoRemoveTicker) {
        try {
          await axios.delete(`http://localhost:8000/ticker_data/api/ticker_data/tickers/${deleteConfirmAlert.ticker}`);
        } catch (err) {
          console.warn('Could not remove ticker from monitoring (may not exist or server not running):', err);
        }
      }

      if (alsoDeleteAllTickerAlerts) {
        setAlerts(prev => prev.filter(alert => alert.ticker !== deleteConfirmAlert.ticker));
      } else {
        setAlerts(prev => prev.filter(alert => alert.id !== deleteConfirmAlert.id));
      }
      setDeleteConfirmAlert(null);
      // Refresh TickerManager to reflect ticker removal if applicable
      if (alsoRemoveTicker) {
        setTickerRefreshTrigger(prev => prev + 1);
      }
    } catch (err: unknown) {
      console.error('Error deleting alert:', err);
      const axiosError = err as { response?: { data?: { error?: string } } };
      const message = axiosError.response?.data?.error || 'Failed to delete alert.';
      alert(message);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteAll = () => {
    if (alerts.length > 0) {
      setShowDeleteAllConfirm(true);
    }
  };

  const handleConfirmDeleteAll = async () => {
    setIsDeletingAll(true);
    try {
      await priceAlertsAPI.deleteAllAlerts();
      setAlerts([]);
      setShowDeleteAllConfirm(false);
      // Refresh TickerManager
      setTickerRefreshTrigger(prev => prev + 1);
    } catch (err: unknown) {
      console.error('Error deleting all alerts:', err);
      const axiosError = err as { response?: { data?: { error?: string } } };
      const message = axiosError.response?.data?.error || 'Failed to delete all alerts.';
      alert(message);
    } finally {
      setIsDeletingAll(false);
    }
  };

  const handleUpdateSettings = async (data: Partial<AlarmSettings>) => {
    if (!alarmSettings) return;

    try {
      const response = await priceAlertsAPI.updateAlarmSettings(data);
      setAlarmSettings(response.data);
    } catch (err) {
      console.error('Error updating settings:', err);
      alert('Failed to update settings. Please try again.');
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
          <AlertComponent variant="destructive" className="mb-6">
            <AlertDescription>{error}</AlertDescription>
          </AlertComponent>
        )}

        {/* Tab Navigation */}
        <div className="flex gap-4 border-b border-border mb-6">
          <button
            onClick={() => setActiveTab('alerts')}
            className={`px-6 py-3 font-semibold transition-all ${activeTab === 'alerts'
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-500 hover:text-gray-700'
              }`}
          >
            🔔 Price Alerts
          </button>
          <button
            onClick={() => setActiveTab('telegram')}
            className={`px-6 py-3 font-semibold transition-all ${activeTab === 'telegram'
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-500 hover:text-gray-700'
              }`}
          >
            📱 Enable Phone Notifications
          </button>
          <button
            type="button"
            disabled
            title="Hosted backend integration is currently disabled"
            className="px-6 py-3 font-semibold transition-all text-gray-400 cursor-not-allowed opacity-60"
          >
            🌐 Hosted Backend (disabled)
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'alerts' && (
          <div className="space-y-6">
            {/* Fetch Error Banner - at top for visibility */}
            <FetchErrorBanner />

            {/* Data Source Selector - switch between yfinance and Webull */}
            <DataSourceSelector />

            {/* Ticker Manager - right under request stats */}
            <div className="p-3 rounded-lg border bg-muted/30">
              <TickerManager compact title="Monitored Tickers" refreshTrigger={tickerRefreshTrigger} />
            </div>

            {/* Show alerts at the top */}
            <AlertsTable
              alerts={alerts}
              priceDirections={priceDirections}
              onToggleActive={handleToggleActive}
              onDelete={handleDelete}
              onDeleteAll={handleDeleteAll}
              monitoredTickers={monitoredTickers}
            />

            <div className="grid gap-6 md:grid-cols-2">
              <AlertForm onSubmit={handleCreateAlert} isLoading={isCreating} />
              {alarmSettings && (
                <AlarmSettingsForm
                  settings={alarmSettings}
                  onUpdate={handleUpdateSettings}
                />
              )}
            </div>
          </div>
        )}

        {activeTab === 'telegram' && <TelegramSetupPage />}
      </div>

      {/* Under-price Alert Confirmation Modal */}
      <Dialog open={underPriceAlert !== null} onOpenChange={(open) => !open && setUnderPriceAlert(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-amber-600">
              <AlertTriangle className="h-5 w-5" />
              Alert Below Current Price
            </DialogTitle>
            <DialogDescription className="text-base pt-2">
              {underPriceAlert && (
                <>
                  You just created an alert for <strong>{underPriceAlert.ticker}</strong> at{' '}
                  <strong>${underPriceAlert.alert_price.toFixed(2)}</strong>, but the current price is{' '}
                  <strong>${underPriceAlert.current_price?.toFixed(2)}</strong>.
                  <br /><br />
                  <strong className="text-amber-600">⚠️ This alert will trigger when the price DROPS to ${underPriceAlert.alert_price.toFixed(2)}</strong>,
                  not when it rises above the current price.
                  <br /><br />
                  Is this what you intended?
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 sm:gap-0">
            <Button
              variant="destructive"
              onClick={handleDeleteUnderPriceAlert}
              className="flex-1 sm:flex-none"
            >
              Delete Alert
            </Button>
            <Button
              variant="default"
              onClick={handleConfirmUnderPriceAlert}
              className="flex-1 sm:flex-none"
            >
              Keep Alert
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Modal */}
      <Dialog open={deleteConfirmAlert !== null} onOpenChange={(open) => !open && setDeleteConfirmAlert(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-destructive">
              <Trash2 className="h-5 w-5" />
              Delete Alert
            </DialogTitle>
            <DialogDescription className="text-base pt-2">
              {deleteConfirmAlert && (
                <>
                  Are you sure you want to delete the alert for <strong>{deleteConfirmAlert.ticker}</strong> at{' '}
                  <strong>${deleteConfirmAlert.alert_price.toFixed(2)}</strong>?
                </>
              )}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3">
            <div className="flex items-center space-x-2 pt-2">
              <Checkbox
                id="deleteAllTickerAlerts"
                checked={alsoDeleteAllTickerAlerts}
                onCheckedChange={(checked) => setAlsoDeleteAllTickerAlerts(checked === true)}
              />
              <Label htmlFor="deleteAllTickerAlerts" className="text-sm cursor-pointer">
                Delete <strong>all</strong> alerts for <strong>{deleteConfirmAlert?.ticker}</strong>
              </Label>
            </div>
            {deleteConfirmAlert && (
              <p className="text-xs text-muted-foreground ml-6">
                This will delete {alerts.filter(a => a.ticker === deleteConfirmAlert.ticker).length} alert(s) for {deleteConfirmAlert.ticker}.
              </p>
            )}

            <div className="flex items-center space-x-2">
              <Checkbox
                id="removeTicker"
                checked={alsoRemoveTicker}
                onCheckedChange={(checked) => setAlsoRemoveTicker(checked === true)}
              />
              <Label htmlFor="removeTicker" className="text-sm cursor-pointer">
                Also remove <strong>{deleteConfirmAlert?.ticker}</strong> from Ticker Monitoring
              </Label>
            </div>
            {deleteConfirmAlert && (
              <p className="text-xs text-muted-foreground ml-6">
                This will remove the ticker and all its historical data from monitoring.
              </p>
            )}
          </div>

          <DialogFooter className="flex gap-2 sm:gap-0">
            <Button
              variant="outline"
              onClick={() => setDeleteConfirmAlert(null)}
              className="flex-1 sm:flex-none"
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDelete}
              className="flex-1 sm:flex-none"
              disabled={isDeleting}
            >
              {isDeleting ? 'Deleting...' : 'Delete Alert'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete All Confirmation Modal */}
      <Dialog open={showDeleteAllConfirm} onOpenChange={(open) => !open && setShowDeleteAllConfirm(false)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-destructive">
              <Trash2 className="h-5 w-5" />
              Delete All Alerts
            </DialogTitle>
            <DialogDescription className="text-base pt-2">
              Are you sure you want to delete <strong>all {alerts.length} alert(s)</strong>?
              <br /><br />
              <span className="text-amber-600 font-semibold">⚠️ This action cannot be undone.</span>
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 sm:gap-0">
            <Button
              variant="outline"
              onClick={() => setShowDeleteAllConfirm(false)}
              className="flex-1 sm:flex-none"
              disabled={isDeletingAll}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDeleteAll}
              className="flex-1 sm:flex-none"
              disabled={isDeletingAll}
            >
              {isDeletingAll ? 'Deleting...' : 'Delete All'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

