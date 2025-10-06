'use client'
import React, { useEffect, useRef } from 'react'

// Lightweight Matrix-style data rain. Designed to be subtle and low-cost.
export default function DataRain() {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const stopRef = useRef(false)

  useEffect(() => {
    const canvas = ref.current!
    const ctx = canvas.getContext('2d')!
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1))
    const fontSize = 14 // small to keep it subtle
    const stepY = 1 // pixels per frame step (scaled by dpr later)
    const chars = '01' // mostly 0/1, few control-like glyphs (not rendered)
    const headColor = 'rgba(0,255,65,0.75)'
    const tailAlpha = 0.06 // fade strength

    let width = 0
    let height = 0
    let cols = 0
    let drops: number[] = []

    const resize = () => {
      const w = window.innerWidth
      const h = window.innerHeight
      width = Math.floor(w * dpr)
      height = Math.floor(h * dpr)
      canvas.width = width
      canvas.height = height
      canvas.style.width = w + 'px'
      canvas.style.height = h + 'px'
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      cols = Math.floor(w / (fontSize + 8)) // spaced columns
      drops = new Array(cols).fill(0).map(() => Math.floor(Math.random() * -50))
      ctx.font = `${fontSize}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`
    }

    const draw = () => {
      if (stopRef.current || document.hidden) {
        rafRef.current = requestAnimationFrame(draw)
        return
      }
      // Tail fade
      ctx.fillStyle = `rgba(0, 0, 0, ${tailAlpha})`
      ctx.fillRect(0, 0, width, height)

      for (let i = 0; i < cols; i++) {
        // Light mode: skip ~50% columns randomly each frame to reduce density
        if ((i + (performance.now() / 1000)) % 2 > 1) continue
        const x = i * (fontSize + 8)
        const y = drops[i] * (fontSize + 2)
        const ch = chars[Math.floor(Math.random() * 2)] // favor 0/1
        ctx.fillStyle = headColor
        ctx.fillText(ch, x, y)

        // Advance drop; randomly reset with low probability when past bottom
        if (y > window.innerHeight + Math.random() * 200) {
          // Sparse resets create a light rainfall
          drops[i] = Math.random() < 0.03 ? 0 : drops[i]
        } else {
          drops[i] += stepY
        }
      }
      rafRef.current = requestAnimationFrame(draw)
    }

    resize()
    draw()

    const onResize = () => resize()
    window.addEventListener('resize', onResize, { passive: true })
    const onVis = () => { /* no-op: draw checks document.hidden */ }
    document.addEventListener('visibilitychange', onVis)

    return () => {
      stopRef.current = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', onResize)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [])

  return (
    <canvas
      ref={ref}
      aria-hidden
      style={{
        position: 'fixed',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 1,
        mixBlendMode: 'screen',
        opacity: 0.25, // keep it light
      }}
    />
  )
}
