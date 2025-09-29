// Import React pro hook
import React from 'react';

/**
 * Stargate API Client - Frontend interface pro Terra Nova Stargate API
 * 
 * Poskytuje typu-bezpečné rozhraní pro komunikaci s Stargate API Gateway
 */

export interface StargatePortalStatus {
  status: 'open' | 'closed' | 'unstable';
  energy: number; // 0-100
  connections: number;
}

export interface StargateService {
  name: string;
  endpoint: string;
  status: 'active' | 'inactive' | 'maintenance';
  description: string;
  lastCheck?: string;
}

export interface StargateNetworkInfo {
  host: string;
  height?: number;
  difficulty?: string;
  hashrate?: string;
}

export interface StargateApiResponse {
  portal: StargatePortalStatus;
  services: StargateService[];
  network: StargateNetworkInfo;
  timestamp: string;
}

export interface StargateActionRequest {
  service: string;
  action: string;
  parameters?: Record<string, any>;
}

export interface StargateActionResponse {
  success: boolean;
  message?: string;
  error?: string;
  timestamp: string;
}

/**
 * Stargate API Client Class
 */
export class StargateApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = '/api/stargate') {
    this.baseUrl = baseUrl;
  }

  /**
   * Získá aktuální status Stargate portálu a všech služeb
   */
  async getStatus(): Promise<StargateApiResponse> {
    const response = await fetch(this.baseUrl, {
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Stargate API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Provede akci na konkrétní službě přes Stargate
   */
  async executeAction(request: StargateActionRequest): Promise<StargateActionResponse> {
    const response = await fetch(this.baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Stargate action failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Sleduje status portálu v reálném čase pomocí Server-Sent Events
   * TODO: Implementovat WebSocket nebo SSE pro real-time updates
   */
  subscribeToStatusUpdates(callback: (status: StargateApiResponse) => void): () => void {
    const interval = setInterval(async () => {
      try {
        const status = await this.getStatus();
        callback(status);
      } catch (error) {
        console.error('Failed to fetch Stargate status:', error);
      }
    }, 10000); // Update každých 10 sekund

    return () => clearInterval(interval);
  }

  /**
   * Zkontroluje, zda je portál připraven pro transport
   */
  async isPortalReady(): Promise<boolean> {
    try {
      const status = await this.getStatus();
      return status.portal.status === 'open' && status.portal.energy > 50;
    } catch {
      return false;
    }
  }

  /**
   * Získá seznam dostupných služeb
   */
  async getAvailableServices(): Promise<StargateService[]> {
    const status = await this.getStatus();
    return status.services.filter(service => service.status === 'active');
  }

  /**
   * Restartuje konkrétní službu (pokud je povoleno)
   */
  async restartService(serviceName: string): Promise<StargateActionResponse> {
    return this.executeAction({
      service: serviceName,
      action: 'restart'
    });
  }

  /**
   * Synchronizuje blockchain
   */
  async syncBlockchain(): Promise<StargateActionResponse> {
    return this.executeAction({
      service: 'blockchain',
      action: 'sync'
    });
  }

  /**
   * Vyčistí cache všech služeb
   */
  async clearAllCache(): Promise<StargateActionResponse> {
    return this.executeAction({
      service: 'all',
      action: 'clear-cache'
    });
  }
}

/**
 * Singleton instance Stargate API klienta
 */
export const stargateApi = new StargateApiClient();

/**
 * React hook pro Stargate status
 */
export function useStargateStatus() {
  const [status, setStatus] = React.useState<StargateApiResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let unsubscribe: (() => void) | null = null;

    const initializeStargate = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Počáteční načtení
        const initialStatus = await stargateApi.getStatus();
        setStatus(initialStatus);
        
        // Předplatit se na updates
        unsubscribe = stargateApi.subscribeToStatusUpdates((newStatus) => {
          setStatus(newStatus);
        });
        
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        console.error('Failed to initialize Stargate:', err);
      } finally {
        setLoading(false);
      }
    };

    initializeStargate();

    return () => {
      if (unsubscribe) {
        unsubscribe();
      }
    };
  }, []);

  return { status, loading, error, refresh: () => stargateApi.getStatus().then(setStatus) };
}