'use client'
import React, { useEffect, useMemo, useState } from 'react'
import { usePathname } from 'next/navigation'
import DataRain from './DataRain'
import NavigationMenu from './NavigationMenu'

type Theme = 'dark' | 'light'

function getInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'dark'
  const saved = window.localStorage.getItem('zion-theme') as Theme | null
  if (saved === 'dark' || saved === 'light') return saved
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  return prefersDark ? 'dark' : 'light'
}

export default function ThemeShell({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('dark')
  const [isMobile, setIsMobile] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    setTheme(getInitialTheme())
  }, [])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('zion-theme', theme)
      const root = document.documentElement
      if (theme === 'dark') {
        root.classList.add('dark')
      } else {
        root.classList.remove('dark')
      }
    }
  }, [theme])

  // Responsive breakpoint detection
  useEffect(() => {
    if (typeof window === 'undefined') return
    const onResize = () => {
      // Treat tablets as mobile for nav (hamburger)
      const mobile = window.innerWidth < 1100
      setIsMobile(mobile)
    }
    onResize()
    window.addEventListener('resize', onResize, { passive: true })
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const fg = theme === 'dark' ? '#eaeaea' : '#0a0a0a'

  const nebulaLayerStyle = useMemo<React.CSSProperties>(() => ({
    position: 'fixed',
    inset: 0,
    zIndex: 0,
    backgroundImage: "url('/stargate/nebula.jpg')",
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    backgroundRepeat: 'no-repeat',
    backgroundAttachment: 'fixed',
  }), [])

  const gridLayerStyle = useMemo<React.CSSProperties>(() => {
    if (theme === 'dark') {
      return {
        position: 'fixed',
        inset: 0,
        zIndex: 1,
        pointerEvents: 'none',
        backgroundColor: 'transparent',
        backgroundImage:
          'radial-gradient(ellipse at 50% -20%, rgba(0,255,65,0.08), transparent 50%), linear-gradient(rgba(0,255,65,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,65,0.05) 1px, transparent 1px)',
        backgroundSize: '100% 100%, 26px 26px, 26px 26px',
        backgroundPosition: '0 0, 0 0, 0 0',
        backgroundAttachment: 'fixed, fixed, fixed',
      }
    }
    return {
      position: 'fixed',
      inset: 0,
      zIndex: 1,
      pointerEvents: 'none',
      backgroundColor: 'transparent',
      backgroundImage:
        'linear-gradient(rgba(0,107,28,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(0,107,28,0.06) 1px, transparent 1px)',
      backgroundSize: '26px 26px, 26px 26px',
      backgroundPosition: '0 0, 0 0',
      backgroundAttachment: 'fixed, fixed',
    }
  }, [theme])

  return (
    <div style={{ position: 'relative', minHeight: '100vh', color: fg }}>
      <div style={nebulaLayerStyle} />
      <div style={gridLayerStyle} />
      {theme === 'dark' && !isMobile && !pathname.startsWith('/stargate') && (
        <div style={{ position:'fixed', inset:0, zIndex:1, pointerEvents:'none' }}>
          <DataRain />
        </div>
      )}
      {/* New global navigation from Dashboard v2 */}
      <div style={{ position: 'relative', zIndex: 3 }}>
        <NavigationMenu />
      </div>
      {/* Spacer to avoid overlap with fixed nav */}
      <div style={{ height: 80 }} />
      <main style={{ maxWidth: 980, margin: '24px auto', padding: '0 16px', position: 'relative', zIndex: 2 }}>{children}</main>
    </div>
  )
}
