import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Loader2, CheckCircle, XCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { hostedBackendService, HostedAlertsResponse } from '../services/hostedBackendService';
import { priceAlertsAPI } from '../services/priceAlertsAPI';
import { telegramAPI } from '../services/telegramAPI';

export const HostedBackendConfig: React.FC = () => {
  const [username, setUsername] = useState('');
  const [hostedURL, setHostedURL] = useState('');
  const [credentialsId, setCredentialsId] = useState<number | null>(null);
  const [isConfigured, setIsConfigured] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [testResult, setTestResult] = useState<'success' | 'error' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [lastSync, setLastSync] = useState<string | null>(null);
  const [hostedAlerts, setHostedAlerts] = useState<HostedAlertsResponse | null>(null);
  const [isLoadingAlerts, setIsLoadingAlerts] = useState(false);

  useEffect(() => {
    loadCredentials();
  }, []);

  useEffect(() => {
    if (isConfigured) {
      // Start auto-sync
      hostedBackendService.startAutoSync(
        async () => {
          const response = await priceAlertsAPI.getAlerts();
          return response.data;
        },
        async () => {
          try {
            const response = await telegramAPI.getConfig();
            return response;
          } catch {
            return null;
          }
        },
        30000 // Sync every 30 seconds
      );

      // Initial sync - wrapped in try-catch to prevent crashes
      handleSync().catch(err => {
        console.error('Initial sync failed:', err);
        // Don't show error to user on initial sync - they'll see it when they manually sync
      });

      return () => {
        hostedBackendService.stopAutoSync();
      };
    }
  }, [isConfigured]);

  const loadCredentials = async () => {
    try {
      const credentials = await hostedBackendService.getCredentials();
      if (credentials) {
        setUsername(credentials.username);
        setHostedURL(credentials.hosted_url);
        setCredentialsId(credentials.id || null);
        setIsConfigured(true);
        setLastSync(credentials.last_sync || null);

        // Load hosted alerts
        loadHostedAlerts();
      }
    } catch (err) {
      console.error('Error loading credentials:', err);
    }
  };

  const loadHostedAlerts = async () => {
    try {
      setIsLoadingAlerts(true);
      const alerts = await hostedBackendService.getHostedAlerts();
      setHostedAlerts(alerts);
    } catch (err) {
      console.error('Error loading hosted alerts:', err);
      setHostedAlerts(null);
      // Don't show error to user - just fail silently for hosted backend issues
    } finally {
      setIsLoadingAlerts(false);
    }
  };

  const handleTestConnection = async () => {
    if (!hostedURL) {
      setError('Please enter a hosted backend URL');
      return;
    }

    setIsTesting(true);
    setTestResult(null);
    setError(null);

    try {
      const isOnline = await hostedBackendService.testHostedConnection(hostedURL);
      if (isOnline) {
        setTestResult('success');
        setSuccess('Connection successful! ✓');
      } else {
        setTestResult('error');
        setError('Failed to connect to hosted backend');
      }
    } catch (err) {
      setTestResult('error');
      setError('Failed to connect to hosted backend');
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = async () => {
    if (!username || !hostedURL) {
      setError('Please fill in all fields');
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      console.log('Testing hosted backend URL:', hostedURL);
      // Test connection first
      const isOnline = await hostedBackendService.testHostedConnection(hostedURL);
      if (!isOnline) {
        setError('Cannot connect to hosted backend. Please check the URL.');
        setIsSaving(false);
        return;
      }

      // Save or update credentials
      if (isConfigured && credentialsId) {
        // Update existing
        await hostedBackendService.updateCredentials(credentialsId, {
          username,
          hosted_url: hostedURL,
          is_active: true
        });
      } else {
        // Create new
        const newCreds = await hostedBackendService.saveCredentials({
          username,
          hosted_url: hostedURL,
          is_active: true
        });
        setCredentialsId(newCreds.id || null);
      }

      setIsConfigured(true);
      setSuccess('Hosted backend configured successfully!');

      // Initial sync
      await handleSync();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save credentials');
    } finally {
      setIsSaving(false);
    }
  };

  const handleSync = async () => {
    if (!isConfigured) return;

    setIsSyncing(true);
    setError(null);

    try {
      const [alertsResponse, telegramResponse] = await Promise.all([
        priceAlertsAPI.getAlerts(),
        telegramAPI.getConfig().catch(() => null)
      ]);

      await hostedBackendService.syncAlertsToHosted(
        alertsResponse.data,
        telegramResponse ?? null
      );

      setSuccess('Synced successfully!');
      setLastSync(new Date().toISOString());
      console.log('Hosted credentials stored, initiating manual sync');
      // Reload hosted alerts
      await loadHostedAlerts();
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to sync alerts';
      setError(`Sync failed: ${errorMsg}. Check if hosted backend URL is correct.`);
    } finally {
      setIsSyncing(false);
    }
  };

  const handleTestNotification = async () => {
    setError(null);
    setSuccess(null);

    try {
      await hostedBackendService.sendTestNotification();
      setSuccess('Test notification sent! Check your Telegram.');
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to send test notification';
      setError(`Test notification failed: ${errorMsg}. Verify hosted backend URL is correct and Telegram is configured.`);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>🌐 Hosted Backend Configuration</CardTitle>
          <CardDescription>
            Configure a hosted backend to keep alerts active even when your laptop is off.
            The hosted backend will continuously monitor your alerts and send Telegram notifications.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert className="border-green-500 bg-green-50">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-800">{success}</AlertDescription>
            </Alert>
          )}

          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
              />
              <p className="text-xs text-gray-500">
                Simple identifier for your alerts (no password needed)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="hostedURL">Hosted Backend URL</Label>
              <div className="flex gap-2">
                <Input
                  id="hostedURL"
                  value={hostedURL}
                  onChange={(e) => setHostedURL(e.target.value)}
                  placeholder="https://your-app.onrender.com or http://localhost:8001"
                />
                <Button
                  onClick={handleTestConnection}
                  disabled={isTesting || !hostedURL}
                  variant="outline"
                >
                  {isTesting ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    'Test'
                  )}
                </Button>
              </div>
              {testResult === 'success' && (
                <div className="flex items-center gap-2 text-green-600 text-sm">
                  <CheckCircle className="w-4 h-4" />
                  Connection successful
                </div>
              )}
              {testResult === 'error' && (
                <div className="flex items-center gap-2 text-red-600 text-sm">
                  <XCircle className="w-4 h-4" />
                  Connection failed
                </div>
              )}
            </div>

            <div className="flex gap-2">
              <Button
                onClick={handleSave}
                disabled={isSaving || !username || !hostedURL}
                className="flex-1"
              >
                {isSaving ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : isConfigured ? (
                  'Update & Reconnect'
                ) : (
                  'Save & Connect'
                )}
              </Button>
              {isConfigured && (
                <>
                  <Button
                    onClick={handleSync}
                    disabled={isSyncing}
                    variant="outline"
                  >
                    {isSyncing ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Sync
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={handleTestNotification}
                    variant="outline"
                  >
                    Test
                  </Button>
                </>
              )}
            </div>

            {isConfigured && lastSync && (
              <div className="text-sm text-gray-500">
                Last synced: {new Date(lastSync).toLocaleString()}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Hosted Alerts Display */}
      {isConfigured && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>📊 Hosted Backend Status</CardTitle>
              <Button
                onClick={loadHostedAlerts}
                disabled={isLoadingAlerts}
                variant="ghost"
                size="sm"
              >
                {isLoadingAlerts ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4" />
                )}
              </Button>
            </div>
            <CardDescription>
              View all alerts stored on the hosted backend
            </CardDescription>
          </CardHeader>
          <CardContent>
            {hostedAlerts ? (
              <div className="space-y-3">
                <div className="flex gap-4">
                  <Badge variant="outline" className="text-lg py-1 px-3">
                    Total Alerts: {hostedAlerts.total}
                  </Badge>
                </div>

                {hostedAlerts.by_user && Object.keys(hostedAlerts.by_user).length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold mb-2">Alerts by User:</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(hostedAlerts.by_user).map(([user, count]) => (
                        <Badge key={user} variant="secondary">
                          {user}: {count} alerts
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {hostedAlerts.alerts && hostedAlerts.alerts.length > 0 && (
                  <div className="border rounded-lg overflow-hidden">
                    <div className="max-h-64 overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-gray-50 sticky top-0">
                          <tr>
                            <th className="px-4 py-2 text-left">User</th>
                            <th className="px-4 py-2 text-left">Ticker</th>
                            <th className="px-4 py-2 text-right">Alert Price</th>
                            <th className="px-4 py-2 text-right">Current Price</th>
                            <th className="px-4 py-2 text-center">Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y">
                          {hostedAlerts.alerts.map((alert) => (
                            <tr key={alert.id} className="hover:bg-gray-50">
                              <td className="px-4 py-2">{alert.username}</td>
                              <td className="px-4 py-2 font-semibold">{alert.ticker}</td>
                              <td className="px-4 py-2 text-right">${alert.alert_price.toFixed(2)}</td>
                              <td className="px-4 py-2 text-right">
                                {alert.current_price ? `$${alert.current_price.toFixed(2)}` : '-'}
                              </td>
                              <td className="px-4 py-2 text-center">
                                {alert.triggered ? (
                                  <Badge variant="destructive">Triggered</Badge>
                                ) : alert.is_active ? (
                                  <Badge className="bg-green-500">Active</Badge>
                                ) : (
                                  <Badge variant="secondary">Inactive</Badge>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8">
                {isLoadingAlerts ? (
                  <span className="text-gray-500">Loading alerts...</span>
                ) : (
                  <div className="space-y-2">
                    <p className="text-gray-500">Unable to load hosted alerts</p>
                    <p className="text-sm text-gray-400">
                      This could mean the hosted backend is not accessible. Check your hosted URL.
                    </p>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Hosting Platform Recommendations */}
      {!isConfigured && (
        <Card>
          <CardHeader>
            <CardTitle>☁️ Recommended Hosting Platforms</CardTitle>
            <CardDescription>
              Deploy the hosted backend from the <code className="bg-gray-100 px-1 rounded">hosted_backend/</code> folder
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <div className="border-l-4 border-blue-500 pl-4 py-2">
                <h4 className="font-semibold">Render.com (Recommended)</h4>
                <p className="text-gray-600">Free tier with 750 hours/month. Deploy directly from GitHub.</p>
                <a href="https://render.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                  render.com →
                </a>
              </div>
              <div className="border-l-4 border-purple-500 pl-4 py-2">
                <h4 className="font-semibold">Railway.app</h4>
                <p className="text-gray-600">$5/month credit. Always-on services.</p>
                <a href="https://railway.app" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                  railway.app →
                </a>
              </div>
              <div className="border-l-4 border-green-500 pl-4 py-2">
                <h4 className="font-semibold">Fly.io</h4>
                <p className="text-gray-600">Free tier with 3 VMs. Docker-native.</p>
                <a href="https://fly.io" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                  fly.io →
                </a>
              </div>
            </div>
            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded text-sm">
              <strong>📝 Quick Start:</strong> See <code className="bg-white px-1 rounded">hosted_backend/README.md</code> for deployment instructions.
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
