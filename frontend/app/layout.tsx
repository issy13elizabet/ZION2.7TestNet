import React from 'react'
import type { Metadata } from 'next'
import './globals.css'
import { Toaster } from 'sonner'
import ThemeShell from './components/ThemeShell'
import { LanguageProvider } from './components/LanguageContext'

export const metadata: Metadata = {
  title: 'ZION dApp',
  description: 'ZION – Amenti Library and ecosystem',
}

const ACCENT = '#00ff41'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="cs">
  <body className="min-h-screen text-slate-900 dark:text-slate-100" style={{ background:'transparent' }}>
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
