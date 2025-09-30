/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  env: {
    ZION_RPC_URL: process.env.ZION_RPC_URL || 'http://localhost:18089',
    ZION_WS_URL: process.env.ZION_WS_URL || 'ws://localhost:18090',
    ZION_NETWORK: process.env.ZION_NETWORK || 'testnet',
  },
  async rewrites() {
    return [
      {
        source: '/api/rpc/:path*',
        destination: `${process.env.ZION_RPC_URL || 'http://localhost:18089'}/:path*`,
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
        ],
      },
    ];
  },
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      net: false,
      tls: false,
      crypto: require.resolve('crypto-browserify'),
      stream: require.resolve('stream-browserify'),
      buffer: require.resolve('buffer'),
    };
    return config;
  },
}

module.exports = nextConfig