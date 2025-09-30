import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'ZION Network - Decentralized Future',
  description: 'ZION 2.6.75 - Advanced blockchain with RandomX mining, cross-chain bridges, and Web3 wallet',
  keywords: 'blockchain, cryptocurrency, mining, wallet, randomx, web3',
  authors: [{ name: 'ZION Network Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#8b5cf6',
  manifest: '/manifest.json',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-900 text-white min-h-screen`}>
        <Providers>
          <div className="flex flex-col min-h-screen">
            <Header />
            <main className="flex-1 container mx-auto px-4 py-8">
              {children}
            </main>
            <Footer />
          </div>
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 5000,
              style: {
                background: '#1f2937',
                color: '#f3f4f6',
                border: '1px solid #374151',
              },
            }}
          />
        </Providers>
      </body>
    </html>
  )
}