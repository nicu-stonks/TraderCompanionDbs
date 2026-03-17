/**
 * Telegram API Service
 * Handles all API calls related to Telegram notification configuration
 */

const API_BASE = 'http://localhost:8000/price_alerts/telegram';

export interface TelegramConfig {
  bot_token: string;
  chat_id: string;
  enabled: boolean;
  configured: boolean;
}

export interface TelegramTestResult {
  success: boolean;
  message: string;
}

/**
 * Get current Telegram configuration
 */
export async function getTelegramConfig(): Promise<TelegramConfig> {
  const response = await fetch(`${API_BASE}/config/`);

  if (!response.ok) {
    throw new Error('Failed to fetch Telegram configuration');
  }

  return response.json();
}

/**
 * Save Telegram configuration (bot token and chat ID)
 */
export async function saveTelegramConfig(
  botToken: string,
  chatId: string
): Promise<TelegramConfig> {
  const response = await fetch(`${API_BASE}/save/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      bot_token: botToken,
      chat_id: chatId,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to save Telegram configuration');
  }

  return response.json();
}

/**
 * Test Telegram bot connection
 */
export async function testTelegramConnection(
  botToken: string,
  chatId: string
): Promise<TelegramTestResult> {
  const response = await fetch(`${API_BASE}/test/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      bot_token: botToken,
      chat_id: chatId,
    }),
  });

  const result = await response.json();

  return {
    success: response.ok && result.success,
    message: result.message || 'Unknown error',
  };
}

/**
 * Toggle Telegram notifications on/off
 */
export async function toggleTelegramNotifications(
  enabled: boolean
): Promise<{ enabled: boolean; message: string }> {
  const response = await fetch(`${API_BASE}/toggle/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      enabled,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to toggle Telegram notifications');
  }

  return response.json();
}

// Export as telegramAPI object for compatibility
export const telegramAPI = {
  getConfig: getTelegramConfig,
  saveConfig: saveTelegramConfig,
  testConnection: testTelegramConnection,
  toggleNotifications: toggleTelegramNotifications
};
