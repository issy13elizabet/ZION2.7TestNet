/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    appDir: true,
  },
  
  // ZION 2.7 specific configuration
  env: {
    ZION_VERSION: '2.7.0',
    ZION_NETWORK: 'TestNet',
    ZION_AI_ENABLED: 'true',
    ZION_MINING_ENABLED: 'true',
  },

  // API Routes for ZION 2.7 integration
  async rewrites() {
    return [
      {
        source: '/api/zion-core/:path*',
        destination: 'http://localhost:18088/api/:path*',
      },
      {
        source: '/api/mining/:path*', 
        destination: 'http://localhost:18089/api/:path*',
      },
      {
        source: '/api/ai/:path*',
        destination: 'http://localhost:18090/api/:path*',
      }
    ];
  },

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