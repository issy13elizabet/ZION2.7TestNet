/**
 * Network Status Dashboard Component
 */
'use client';

import { useState, useEffect } from 'react';
import { ZionAPI } from '@/lib/api/client';

export default function NetworkStatus() {
  const [networkStatus, setNetworkStatus] = useState<any>(null);

  useEffect(() => {
    ZionAPI.getNetworkStatus().then(setNetworkStatus);
  }, []);

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white">
        🌐 Network Status
      </h2>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Peers Connected</p>
          <p className="text-2xl font-bold text-blue-600">{networkStatus?.peers || 0}</p>
        </div>

        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Uptime</p>
          <p className="text-2xl font-bold text-green-600">{networkStatus?.uptime || 'N/A'}</p>
        </div>
      </div>
    </div>
  );
}
