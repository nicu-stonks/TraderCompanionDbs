import React, { useState, useEffect } from 'react';
import {
  getTelegramConfig,
  saveTelegramConfig,
  testTelegramConnection,
  toggleTelegramNotifications,
  TelegramConfig,
} from '../services/telegramAPI';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Loader2, CheckCircle2, XCircle, Info, ChevronDown, ChevronRight } from 'lucide-react';

export const TelegramSetupPage: React.FC = () => {
  const [config, setConfig] = useState<TelegramConfig | null>(null);
  const [botToken, setBotToken] = useState('');
  const [chatId, setChatId] = useState('');
  const [enabled, setEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [testLoading, setTestLoading] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [expandedStep, setExpandedStep] = useState<number | null>(1);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const data = await getTelegramConfig();
      setConfig(data);
      setBotToken(data.bot_token);
      setChatId(data.chat_id);
      setEnabled(data.enabled);
    } catch (error) {
      console.error('Failed to load Telegram config:', error);
    }
  };

  const handleTestConnection = async () => {
    if (!botToken.trim() || !chatId.trim()) {
      setTestResult({
        success: false,
        message: 'Please enter both Bot Token and Chat ID before testing.',
      });
      return;
    }

    setTestLoading(true);
    setTestResult(null);

    try {
      const result = await testTelegramConnection(botToken.trim(), chatId.trim());
      setTestResult(result);
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : 'Connection test failed',
      });
    } finally {
      setTestLoading(false);
    }
  };

  const handleSave = async () => {
    if (!botToken.trim() || !chatId.trim()) {
      setSaveMessage('Please enter both Bot Token and Chat ID');
      return;
    }

    setLoading(true);
    setSaveMessage(null);

    try {
      await saveTelegramConfig(botToken.trim(), chatId.trim());
      setSaveMessage('Configuration saved successfully!');
      await loadConfig();
    } catch (error) {
      setSaveMessage(`Error: ${error instanceof Error ? error.message : 'Failed to save'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = async () => {
    if (!config?.configured && !enabled) {
      setSaveMessage('Please configure and save Bot Token and Chat ID first');
      return;
    }

    setLoading(true);
    try {
      await toggleTelegramNotifications(!enabled);
      setEnabled(!enabled);
      setSaveMessage(`Notifications ${!enabled ? 'enabled' : 'disabled'}`);
      await loadConfig();
    } catch (error) {
      setSaveMessage(`Error: ${error instanceof Error ? error.message : 'Failed to toggle'}`);
    } finally {
      setLoading(false);
    }
  };

  const toggleStep = (step: number) => {
    setExpandedStep(expandedStep === step ? null : step);
  };

  const steps = [
    {
      title: 'Create Your Telegram Bot',
      content: (
        <ol className="list-decimal list-inside space-y-2 text-sm">
          <li>Open <strong>Telegram</strong> app and search for <code className="bg-muted px-1.5 py-0.5 rounded">@BotFather</code></li>
          <li>Send: <code className="bg-muted px-1.5 py-0.5 rounded">/newbot</code></li>
          <li>Enter a name (e.g., "My Price Alerts")</li>
          <li>Enter a username ending in "bot"</li>
          <li>Copy the <strong>Bot Token</strong> from @BotFather's response</li>
        </ol>
      ),
    },
    {
      title: 'Get Your Chat ID',
      content: (
        <ol className="list-decimal list-inside space-y-2 text-sm">
          <li>Search for <code className="bg-muted px-1.5 py-0.5 rounded">@userinfobot</code> in Telegram</li>
          <li>Send any message to it</li>
          <li>Copy your <strong>Id</strong> number from the response</li>
        </ol>
      ),
    },
    {
      title: 'Start Chat with Your Bot',
      content: (
        <ol className="list-decimal list-inside space-y-2 text-sm">
          <li>Search for your bot by username</li>
          <li>Click <strong>START</strong> or send <code className="bg-muted px-1.5 py-0.5 rounded">/start</code></li>
          <li>This is required before the bot can send you messages</li>
        </ol>
      ),
    },
  ];

  return (
    <div className="space-y-4 max-w-4xl mx-auto">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            📱 Telegram Notifications
          </CardTitle>
          <CardDescription>
            Receive instant price alert notifications on your phone
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Battery Warning */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          <strong>Tip:</strong> Disable battery optimization for Telegram app in your phone settings to ensure instant notifications.
        </AlertDescription>
      </Alert>

      {/* Setup Steps */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Quick Setup Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {steps.map((step, index) => (
            <div key={index} className="border rounded-lg">
              <button
                onClick={() => toggleStep(index + 1)}
                className="w-full px-4 py-2.5 flex items-center justify-between hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold">
                    {index + 1}
                  </div>
                  <span className="font-medium text-sm">{step.title}</span>
                </div>
                {expandedStep === index + 1 ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </button>
              {expandedStep === index + 1 && (
                <div className="px-4 pb-3 pt-1 border-t bg-muted/20">
                  {step.content}
                </div>
              )}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Enable/Disable Notifications */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="notifications-toggle" className="text-base cursor-pointer">
                Enable Notifications
              </Label>
              <p className="text-sm text-muted-foreground">
                {config?.configured
                  ? 'Receive alerts when prices are triggered'
                  : 'Save configuration first to enable'}
              </p>
            </div>
            <Switch
              id="notifications-toggle"
              checked={enabled}
              onCheckedChange={handleToggle}
              disabled={loading || !config?.configured}
            />
          </div>
          {/* Status Summary */}
          {config && (
            <div className="flex gap-2 pt-4">
              <Badge variant={config.configured ? 'default' : 'secondary'}>
                {config.configured ? '✓ Configured' : 'Not Configured'}
              </Badge>
              <Badge variant={config.enabled ? 'default' : 'outline'}>
                {config.enabled ? '✓ Enabled' : 'Disabled'}
              </Badge>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Configuration */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="botToken">Bot Token</Label>
              <Input
                id="botToken"
                type="text"
                placeholder="123456789:ABCdefGHIjklMNO..."
                value={botToken}
                onChange={(e) => setBotToken(e.target.value)}
                disabled={loading}
                className="font-mono text-sm"
              />
              <p className="text-xs text-muted-foreground">From @BotFather</p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="chatId">Chat ID</Label>
              <Input
                id="chatId"
                type="text"
                placeholder="123456789"
                value={chatId}
                onChange={(e) => setChatId(e.target.value)}
                disabled={loading}
                className="font-mono text-sm"
              />
              <p className="text-xs text-muted-foreground">From @userinfobot</p>
            </div>
          </div>

          {/* Test Result */}
          {testResult && (
            <Alert variant={testResult.success ? 'default' : 'destructive'}>
              {testResult.success ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <XCircle className="h-4 w-4" />
              )}
              <AlertDescription>{testResult.message}</AlertDescription>
            </Alert>
          )}

          {/* Buttons */}
          <div className="grid gap-3 md:grid-cols-2">
            <Button
              onClick={handleTestConnection}
              disabled={testLoading || loading || !botToken.trim() || !chatId.trim()}
              variant="outline"
            >
              {testLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Testing...
                </>
              ) : (
                '🧪 Test Connection'
              )}
            </Button>

            <Button
              onClick={handleSave}
              disabled={loading || !botToken.trim() || !chatId.trim()}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                '💾 Save Configuration'
              )}
            </Button>
          </div>

          {/* Save Message */}
          {saveMessage && (
            <Alert variant={saveMessage.includes('Error') ? 'destructive' : 'default'}>
              <AlertDescription>{saveMessage}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
