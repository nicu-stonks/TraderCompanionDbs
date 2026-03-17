import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { priceAlertsAPI } from '@/PriceAlertsPage/services/priceAlertsAPI';
import { Alert } from '@/PriceAlertsPage/types';

interface GlobalAlertContextType {
  triggeredAlerts: Alert[];
  dismissAllAlerts: () => void;
  stopAlert: (id: number) => Promise<void>;
  stopAllAlerts: () => Promise<void>;
  deleteAlert: (id: number) => Promise<void>;
  deleteAllAlerts: () => Promise<void>;
  isLoading: boolean;
}

const GlobalAlertContext = createContext<GlobalAlertContextType | null>(null);

export const useGlobalAlerts = () => {
  const context = useContext(GlobalAlertContext);
  if (!context) {
    throw new Error('useGlobalAlerts must be used within a GlobalAlertProvider');
  }
  return context;
};

export const GlobalAlertProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [triggeredAlerts, setTriggeredAlerts] = useState<Alert[]>([]);
  const [dismissedIds, setDismissedIds] = useState<Set<number>>(new Set());
  const [isLoading, setIsLoading] = useState(false);
  const isPollingRef = useRef(false);
  // Use refs to avoid re-creating fetchAlerts on every state change
  const triggeredAlertsRef = useRef<Alert[]>([]);
  const dismissedIdsRef = useRef<Set<number>>(new Set());

  // Keep refs in sync with state
  useEffect(() => {
    triggeredAlertsRef.current = triggeredAlerts;
  }, [triggeredAlerts]);

  useEffect(() => {
    dismissedIdsRef.current = dismissedIds;
  }, [dismissedIds]);

  // Poll for alerts - stable callback that doesn't change
  const fetchAlerts = useCallback(async () => {
    if (isPollingRef.current) return; // Prevent overlapping requests
    isPollingRef.current = true;

    try {
      const response = await priceAlertsAPI.getAlerts();
      const alerts = response.data;

      // Get alerts that are triggered OR were triggered (and we're still showing them)
      // Filter out dismissed ones - use refs to get current values
      const currentIds = new Set(triggeredAlertsRef.current.map((a) => a.id));
      const currentDismissed = dismissedIdsRef.current;

      // Keep alerts that are triggered or that we're already showing (unless dismissed)
      const newTriggered = alerts.filter(
        (alert) => (alert.triggered || currentIds.has(alert.id)) && !currentDismissed.has(alert.id)
      );

      setTriggeredAlerts(newTriggered);
    } catch (err) {
      console.error('Error polling alerts:', err);
    } finally {
      isPollingRef.current = false;
    }
  }, []); // No dependencies - uses refs instead

  // Start polling on mount - runs only once since fetchAlerts is stable
  useEffect(() => {
    fetchAlerts(); // Initial fetch

    const interval = setInterval(fetchAlerts, 5000); // Poll every 5 seconds instead of 1
    return () => clearInterval(interval);
  }, [fetchAlerts]);

  // Dismiss all alerts (close the modal)
  const dismissAllAlerts = useCallback(() => {
    setDismissedIds((prev) => {
      const newSet = new Set(prev);
      triggeredAlerts.forEach((a) => newSet.add(a.id));
      return newSet;
    });
    setTriggeredAlerts([]);
  }, [triggeredAlerts]);

  // Stop an alert (set is_active to false) - keep it in the list
  const stopAlert = useCallback(async (id: number) => {
    setIsLoading(true);
    try {
      const response = await priceAlertsAPI.updateAlert(id, { is_active: false });
      // Update the alert in our list (don't remove it)
      setTriggeredAlerts((prev) =>
        prev.map((a) => (a.id === id ? response.data : a))
      );
    } catch (err) {
      console.error('Error stopping alert:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Stop all triggered alerts - keep them in the list
  const stopAllAlerts = useCallback(async () => {
    setIsLoading(true);
    try {
      const results = await Promise.all(
        triggeredAlerts
          .filter((alert) => alert.is_active || alert.triggered)
          .map((alert) => priceAlertsAPI.updateAlert(alert.id, { is_active: false }))
      );
      // Update all alerts with their new state
      const updatedMap = new Map(results.map((r) => [r.data.id, r.data]));
      setTriggeredAlerts((prev) =>
        prev.map((a) => updatedMap.get(a.id) || a)
      );
    } catch (err) {
      console.error('Error stopping all alerts:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [triggeredAlerts]);

  // Delete an alert entirely - remove from list
  const deleteAlert = useCallback(async (id: number) => {
    setIsLoading(true);
    try {
      await priceAlertsAPI.deleteAlert(id);
      // Remove from triggered list
      setTriggeredAlerts((prev) => prev.filter((a) => a.id !== id));
    } catch (err) {
      console.error('Error deleting alert:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Delete all triggered alerts - clear the list
  const deleteAllAlerts = useCallback(async () => {
    setIsLoading(true);
    try {
      await Promise.all(triggeredAlerts.map((alert) => priceAlertsAPI.deleteAlert(alert.id)));
      // Clear the list
      setTriggeredAlerts([]);
    } catch (err) {
      console.error('Error deleting all alerts:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [triggeredAlerts]);

  return (
    <GlobalAlertContext.Provider
      value={{
        triggeredAlerts,
        dismissAllAlerts,
        stopAlert,
        stopAllAlerts,
        deleteAlert,
        deleteAllAlerts,
        isLoading,
      }}
    >
      {children}
    </GlobalAlertContext.Provider>
  );
};
