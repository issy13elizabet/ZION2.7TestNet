/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    // appDir is now stable in Next.js 14, removed deprecated option
  },
  
  // ZION 2.7 specific configuration
  env: {
    ZION_VERSION: '2.7.0',
    ZION_NETWORK: 'TestNet',
    ZION_AI_ENABLED: 'true',
    ZION_MINING_ENABLED: 'true',
  },

  // Note: API routes are handled by /app/api/* structure
  // No rewrites needed since we have proper API route handlers

  // Headers for CORS and security
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,POST,PUT,DELETE,OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
          { key: 'X-Powered-By', value: 'ZION 2.7 Real System' },
        ],
      },
    ];
  }
};

module.exports = nextConfig;