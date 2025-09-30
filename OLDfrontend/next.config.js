/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production'

const securityHeaders = [
  // General hardening
  { key: 'X-DNS-Prefetch-Control', value: 'on' },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
  // HSTS (only enable when serving over HTTPS)
  { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
  // Permissions Policy (restrict powerful APIs)
  { key: 'Permissions-Policy', value: "camera=(), microphone=(), geolocation=(), payment=(), usb=(), interest-cohort=()" },
]

// CSP: v produkci bez wildcardů, allowlist z env proměnných
const FRONTEND_ORIGIN = process.env.NEXT_PUBLIC_APP_ORIGIN || ''
const API_ORIGIN = process.env.NEXT_PUBLIC_API_ORIGIN || ''
const SHIM_ORIGIN = process.env.NEXT_PUBLIC_SHIM_ORIGIN || ''
const EXTRA_CONNECT = process.env.NEXT_PUBLIC_CSP_CONNECT_EXTRA || '' // čárkou oddělené

const connectSrc = [
  "'self'",
  ...[FRONTEND_ORIGIN, API_ORIGIN, SHIM_ORIGIN]
    .filter(Boolean)
    .map(s => s.trim()),
  ...EXTRA_CONNECT.split(',').map(s => s.trim()).filter(Boolean),
]

const csp = [
  "default-src 'self'",
  // Stylování z Next/Tailwind může vyžadovat 'unsafe-inline'
  "style-src 'self' 'unsafe-inline'",
  // Next.js runtime může v dev vyžadovat 'unsafe-eval'
  `script-src 'self' ${!isProd ? "'unsafe-eval' 'unsafe-inline'" : ''}`.trim(),
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  `connect-src ${connectSrc.join(' ')}`,
  "frame-ancestors 'self'",
].join('; ')

const nextConfig = {
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          ...securityHeaders,
          // CSP pouze v produkci (v dev by omezovala HMR)
          ...(isProd ? [{ key: 'Content-Security-Policy', value: csp }] : []),
        ],
      },
    ]
  },
}

module.exports = nextConfig