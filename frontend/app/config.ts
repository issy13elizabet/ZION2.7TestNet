// ZION 2.6.75 Backend Configuration
export const ZION_HOST = process.env.NEXT_PUBLIC_ZION_HOST || process.env.ZION_HOST || 'localhost';
export const ZION_POOL_PORT = Number(process.env.NEXT_PUBLIC_ZION_POOL_PORT || process.env.ZION_POOL_PORT || 4444);
export const ZION_RPC_PORT = Number(process.env.NEXT_PUBLIC_ZION_RPC_PORT || process.env.ZION_RPC_PORT || 18089);
export const ZION_WEB_PORT = Number(process.env.NEXT_PUBLIC_ZION_WEB_PORT || process.env.ZION_WEB_PORT || 8080);
export const ZION_MINING_PORT = Number(process.env.NEXT_PUBLIC_ZION_MINING_PORT || process.env.ZION_MINING_PORT || 8117);

// API Base URLs for ZION 2.6.75
export const ZION_API_BASE = `http://${ZION_HOST}:${ZION_RPC_PORT}`;
export const ZION_POOL_API = `http://${ZION_HOST}:${ZION_WEB_PORT}`;
export const ZION_MINING_API = `http://${ZION_HOST}:${ZION_MINING_PORT}`;

// WebSocket URLs
export const ZION_WS_URL = `ws://${ZION_HOST}:19090`;
export const ZION_POOL_WS = `ws://${ZION_HOST}:${ZION_WEB_PORT}/ws`;

// Legacy compatibility
export const ZION_SHIM_PORT = ZION_RPC_PORT;
