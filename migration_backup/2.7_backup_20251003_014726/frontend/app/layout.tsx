import React from 'react'
import type { Metadata } from 'next'
import './globals.css'
import { Toaster } from 'sonner'
import ThemeShell from './components/ThemeShell'
import { LanguageProvider } from './components/LanguageContext'
import { Orbitron } from 'next/font/google'

export const metadata: Metadata = {
  title: 'ZION dApp',
  description: 'ZION – Amenti Library and ecosystem',
}

const ACCENT = '#00ff41'

const orbitron = Orbitron({
  subsets: ['latin'],
  weight: ['400', '700', '900'],
  variable: '--font-orbitron',
  display: 'swap',
})

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="cs" className={orbitron.variable}>
  <body className="min-h-screen text-slate-900 dark:text-slate-100" style={{ background:'transparent', fontFamily: 'var(--font-orbitron), system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif' }}>
        <LanguageProvider>
          <ThemeShell>
            {children}
            <Toaster position="top-right" richColors closeButton />
          </ThemeShell>
        </LanguageProvider>
      </body>
    </html>
  )
}
