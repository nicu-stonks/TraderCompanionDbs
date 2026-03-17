import React, { useCallback, useEffect, useState } from 'react';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { AlertCircle, CheckCircle2, Loader2, RefreshCw, WifiOff } from 'lucide-react';

interface WebullStatus {
  status: 'connected' | 'disconnected' | 'login_required';
  login_in_progress: boolean;
  last_error: string | null;
  request_count: number;
  has_session: boolean;
}

interface DataSourceState {
  provider: 'yfinance' | 'webull';
  webull: WebullStatus;
  available_providers: string[];
}

const API_BASE = 'http://localhost:8000/ticker_data/api/ticker_data';

export const DataSourceSelector: React.FC = () => {
  const [state, setState] = useState<DataSourceState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSwitching, setIsSwitching] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    try {
      const [settingsResponse, webullResponse] = await Promise.all([
        axios.get<DataSourceState>(`${API_BASE}/settings`),
        axios.get<WebullStatus>(`${API_BASE}/webull/status`),
      ]);
      // Map the Django format to the requested interface format
      setState({
        provider: (settingsResponse.data as any).active_provider as 'yfinance' | 'webull',
        webull: webullResponse.data,
        available_providers: ['yfinance', 'webull']
      });
      setError(null);
    } catch (err) {
      console.error('Failed to load data source status:', err);
      setError('Failed to connect to server');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
    // Poll every 2s to update login progress
    const interval = setInterval(loadStatus, 2000);
    return () => clearInterval(interval);
  }, [loadStatus]);

  const handleSwitch = async (newProvider: 'yfinance' | 'webull') => {
    if (newProvider === state?.provider) return;

    // If switching to webull and not connected, start login first
    if (newProvider === 'webull' && state?.webull.status !== 'connected') {
      handleLogin();
      return;
    }

    setIsSwitching(true);
    try {
      await axios.put(`${API_BASE}/settings`, { active_provider: newProvider });
      await loadStatus();
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { error?: string } } };
      setError(axiosErr.response?.data?.error || 'Failed to switch provider');
    } finally {
      setIsSwitching(false);
    }
  };

  const handleLogin = async () => {
    setIsLoggingIn(true);
    setError(null);
    try {
      await axios.post(`${API_BASE}/webull/login`);
      // Login is async, will be reflected in status polling
      await loadStatus();
    } catch (err) {
      setError('Failed to start login');
    } finally {
      setIsLoggingIn(false);
    }
  };

  if (isLoading) {
    return (
      <div className="p-3 rounded-lg border bg-muted/30 flex items-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm text-muted-foreground">Loading data source...</span>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="p-3 rounded-lg border bg-destructive/10 flex items-center gap-2">
        <WifiOff className="w-4 h-4 text-destructive" />
        <span className="text-sm text-destructive">Server not connected</span>
      </div>
    );
  }

  const isWebullConnected = state.webull.status === 'connected';
  const isWebullLogging = state.webull.login_in_progress || isLoggingIn;

  return (
    <div className="p-3 rounded-lg border bg-muted/30 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Data Provider</span>
        <div className="flex gap-2">
          <Button
            variant={state.provider === 'yfinance' ? 'default' : 'outline'}
            size="sm"
            onClick={() => handleSwitch('yfinance')}
            disabled={isSwitching}
          >
            Yahoo Finance
          </Button>
          <Button
            variant={state.provider === 'webull' ? 'default' : 'outline'}
            size="sm"
            onClick={() => handleSwitch('webull')}
            disabled={isSwitching || isWebullLogging}
            className={!isWebullConnected && state.provider !== 'webull' ? 'border-yellow-500' : ''}
          >
            Webull
          </Button>
        </div>
      </div>

      {/* Webull Status Section */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2">
          {isWebullConnected ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-green-500" />
              <span className="text-green-600">Webull Connected</span>
            </>
          ) : isWebullLogging ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              <span className="text-blue-500">
                Logging in...
              </span>
            </>
          ) : (
            <>
              <AlertCircle className="w-4 h-4 text-yellow-500" />
              <span className="text-yellow-600">
                {state.webull.status === 'login_required' ? 'Login Required' : 'Disconnected'}
              </span>
            </>
          )}
          <span className="text-xs text-muted-foreground ml-2">
            (💡 Auto-fallback if yfinance fails 30s)
          </span>
        </div>

        {!isWebullConnected && !isWebullLogging && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleLogin}
            disabled={isLoggingIn}
            className="text-blue-600 hover:text-blue-700"
          >
            <RefreshCw className={`w-3 h-3 mr-1 ${isLoggingIn ? 'animate-spin' : ''}`} />
            {state.webull.has_session ? 'Retry Login' : 'Login'}
          </Button>
        )}
      </div>

      {/* Error Display - hide unhelpful messages */}
      {(error || state.webull.last_error) &&
        !state.webull.last_error?.includes('Login completed but session not saved') &&
        !state.webull.last_error?.includes('Login timeout') && (
          <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">
            {error || state.webull.last_error}
          </div>
        )}

      {/* Login Instructions */}
      {isWebullLogging && (
        <div className="text-xs text-muted-foreground bg-blue-500/10 p-2 rounded">
          <strong>Instructions:</strong> A browser window will open. Log in to your Webull account
          and complete any 2FA. The window will close automatically when done.
        </div>
      )}
    </div>
  );
};
