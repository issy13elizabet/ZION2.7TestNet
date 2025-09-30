'use client'

import { ReactNode } from 'react'
import { ZionProvider } from '@/contexts/ZionContext'
import { WalletProvider } from '@/contexts/WalletContext'

interface ProvidersProps {
  children: ReactNode
}

export function Providers({ children }: ProvidersProps) {
  return (
    <ZionProvider>
      <WalletProvider>
        {children}
      </WalletProvider>
    </ZionProvider>
  )
}