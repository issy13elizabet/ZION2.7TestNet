"use client"
import React, { useEffect, useMemo, useState } from 'react'

const DEFAULT_MANTRAS = [
  'All is Love • All is One',
  'I AM Presence • Pure Awareness',
  'As above, so below • As within, so without',
  'Peace • Harmony • Grace',
  'I am guided by Light • I am aligned',
  'New Jerusalem • Divine Architecture',
  'Metatron’s Cube • Sacred Geometry',
]

function rand(min: number, max: number) {
  return Math.random() * (max - min) + min
}

export default function SubliminalLayer() {
  const [enabled, setEnabled] = useState<boolean>(true)

  useEffect(() => {
    const saved = typeof window !== 'undefined' ? window.localStorage.getItem('zion-subliminal') : null
    if (saved === 'off') setEnabled(false)
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const onKey = (e: KeyboardEvent) => {
      // Toggle with Alt+S
      if (e.altKey && (e.key === 's' || e.key === 'S')) {
        setEnabled((prev) => {
          const next = !prev
          window.localStorage.setItem('zion-subliminal', next ? 'on' : 'off')
          return next
        })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const items = useMemo(() => {
    const out = [] as { top: string; left: string; rotate: number; text: string; duration: number; delay: number; size: number; }[]
    for (let i = 0; i < 8; i++) {
      out.push({
        top: `${rand(5, 90)}%`,
        left: `${rand(5, 90)}%`,
        rotate: rand(0, 360),
        text: DEFAULT_MANTRAS[Math.floor(rand(0, DEFAULT_MANTRAS.length))],
        duration: rand(14, 28),
        delay: rand(0, 10),
        size: rand(11, 14),
      })
    }
    return out
  }, [])

  if (!enabled) return null

  return (
    <div aria-hidden className="zion-subliminal-root" style={{ position: 'fixed', inset: 0, zIndex: 2, pointerEvents: 'none' }}>
      {items.map((it, idx) => (
        <div
          key={idx}
          className="zion-subliminal-item"
          style={{
            position: 'absolute',
            top: it.top,
            left: it.left,
            transform: `translate(-50%, -50%) rotate(${it.rotate}deg)`,
            color: 'rgba(255,255,255,0.05)',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.2em',
            fontSize: `${it.size}px`,
            animation: `zionFloat ${it.duration}s ease-in-out ${it.delay}s infinite`,
            whiteSpace: 'nowrap',
            userSelect: 'none',
          }}
        >
          {it.text}
        </div>
      ))}
    </div>
  )
}
