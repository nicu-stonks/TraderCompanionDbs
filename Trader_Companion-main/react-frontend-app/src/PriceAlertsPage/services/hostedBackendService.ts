import axios, { AxiosInstance } from 'axios';

export interface HostedCredentials {
  id?: number;
  username: string;
  hosted_url: string;
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
  last_sync?: string;
}

export interface HostedAlert {
  id: number;
  username: string;
  ticker: string;
  alert_price: number;
  is_active: boolean;
  triggered: boolean;
  created_at: string;
  triggered_at: string | null;
  current_price: number | null;
  last_checked: string | null;
  initial_price_above_alert: boolean | null;
}

export interface HostedAlertsResponse {
  username?: string;
  alerts: HostedAlert[];
  total: number;
  by_user?: Record<string, number>;
}

class HostedBackendService {
  private localAPI: AxiosInstance;
  private hostedAPI: AxiosInstance | null = null;
  private syncInterval: NodeJS.Timeout | null = null;
  private credentials: HostedCredentials | null = null;

  constructor(localBaseURL: string) {
    this.localAPI = axios.create({
      baseURL: localBaseURL + '/hosted_backend',
      withCredentials: true,
      headers: {
        'Content-Type': 'application/json',
      }
    });

    // Get CSRF token from cookie
    this.localAPI.interceptors.request.use((config: any) => {
      const csrfToken = this.getCookie('csrftoken');
      if (csrfToken) {
        config.headers['X-CSRFToken'] = csrfToken;
      }
      return config;
    });
  }

  private getCookie(name: string): string | null {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop()?.split(';').shift() || null;
    return null;
  }

  private normalizeURL(url: string): string {
    // Replace 0.0.0.0 with localhost for browser compatibility
    return url.replace('://0.0.0.0:', '://localhost:');
  }

  private initHostedAPI(hostedURL: string) {
    const normalizedURL = this.normalizeURL(hostedURL);
    this.hostedAPI = axios.create({
      baseURL: normalizedURL,
      timeout: 10000, // 10 second timeout
      validateStatus: (status) => status < 500 // Don't throw on 4xx errors
    });
  }

  // Local API calls
  async getCredentials(): Promise<HostedCredentials | null> {
    try {
      const response = await this.localAPI.get<HostedCredentials>('/credentials/active/');
      this.credentials = response.data;
      if (this.credentials && this.credentials.hosted_url) {
        this.initHostedAPI(this.credentials.hosted_url);
      }
      return this.credentials;
    } catch (err: any) {
      if (err.response?.status === 404) {
        return null;
      }
      throw err;
    }
  }

  async saveCredentials(data: Omit<HostedCredentials, 'id' | 'created_at' | 'updated_at' | 'last_sync'>): Promise<HostedCredentials> {
    try {
      const response = await this.localAPI.post<HostedCredentials>('/credentials/', data);
      this.credentials = response.data;
      if (this.credentials && this.credentials.hosted_url) {
        this.initHostedAPI(this.credentials.hosted_url);
      }
      return this.credentials;
    } catch (err) {
      throw err;
    }
  }

  async updateCredentials(id: number, data: Partial<HostedCredentials>): Promise<HostedCredentials> {
    const response = await this.localAPI.patch<HostedCredentials>(`/credentials/${id}/`, data);
    this.credentials = response.data;
    if (this.credentials && this.credentials.hosted_url) {
      this.initHostedAPI(this.credentials.hosted_url);
    }
    return this.credentials;
  }

  async updateSyncTime(id: number): Promise<void> {
    await this.localAPI.post(`/credentials/${id}/update_sync_time/`);
  }

  // Hosted API calls
  async testHostedConnection(hostedURL: string): Promise<boolean> {
    try {
      const normalizedURL = this.normalizeURL(hostedURL);
      const testAPI = axios.create({ baseURL: normalizedURL, timeout: 5000 });
      await testAPI.get('/health');
      return true;
    } catch (err) {
      return false;
    }
  }

  async syncAlertsToHosted(localAlerts: any[], telegramConfig: any): Promise<void> {
    if (!this.hostedAPI || !this.credentials) {
      console.error('Sync failed - hostedAPI:', this.hostedAPI, 'credentials:', this.credentials);
      throw new Error('Hosted backend not configured. Please save credentials first.');
    }

    console.log('Syncing to:', this.hostedAPI.defaults.baseURL);

    // Transform alerts to hosted format
    const hostedAlerts = localAlerts.map(alert => ({
      username: this.credentials!.username,
      ticker: alert.ticker,
      alert_price: alert.alert_price,
      is_active: alert.is_active,
      triggered: alert.triggered,
      created_at: alert.created_at,
      triggered_at: alert.triggered_at,
      current_price: alert.current_price,
      last_checked: alert.last_checked,
      initial_price_above_alert: alert.initial_price_above_alert
    }));

    const payload: any = { alerts: hostedAlerts };

    if (telegramConfig && telegramConfig.bot_token && telegramConfig.chat_id) {
      payload.telegram_config = {
        username: this.credentials!.username,
        bot_token: telegramConfig.bot_token,
        chat_id: telegramConfig.chat_id,
        enabled: telegramConfig.enabled ?? true
      };
    }

    await this.hostedAPI.post('/alerts/sync', payload);

    // Update last sync time
    if (this.credentials.id) {
      await this.updateSyncTime(this.credentials.id);
    }
  }

  async getHostedAlerts(): Promise<HostedAlertsResponse> {
    if (!this.hostedAPI) {
      throw new Error('Hosted backend not configured');
    }

    const response = await this.hostedAPI.get<HostedAlertsResponse>('/alerts');
    return response.data;
  }

  async getHostedAlertsForUser(username: string): Promise<HostedAlertsResponse> {
    if (!this.hostedAPI) {
      throw new Error('Hosted backend not configured');
    }

    const response = await this.hostedAPI.get<HostedAlertsResponse>(`/alerts/${username}`);
    return response.data;
  }

  async sendTestNotification(): Promise<void> {
    if (!this.hostedAPI || !this.credentials) {
      throw new Error('Hosted backend not configured');
    }

    await this.hostedAPI.post('/telegram/test', {
      username: this.credentials.username
    });
  }

  // Auto-sync functionality
  startAutoSync(
    getLocalAlerts: () => Promise<any[]>,
    getTelegramConfig: () => Promise<any>,
    intervalMs: number = 30000
  ) {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
    }

    this.syncInterval = setInterval(async () => {
      try {
        if (this.credentials && this.credentials.is_active) {
          const [alerts, telegramConfig] = await Promise.all([
            getLocalAlerts(),
            getTelegramConfig()
          ]);
          await this.syncAlertsToHosted(alerts, telegramConfig);
          console.log('Auto-synced alerts to hosted backend');
        }
      } catch (err) {
        console.error('Auto-sync failed:', err);
      }
    }, intervalMs);
  }

  stopAutoSync() {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
    }
  }

  isConfigured(): boolean {
    return this.credentials !== null && this.hostedAPI !== null;
  }

  getUsername(): string | null {
    return this.credentials?.username || null;
  }
}

// Export singleton instance
import { API_CONFIG } from '@/config';
export const hostedBackendService = new HostedBackendService(API_CONFIG.baseURL);
