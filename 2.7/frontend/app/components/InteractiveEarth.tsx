'use client'
import React, { useEffect, useRef } from 'react'

type Theme = 'dark' | 'light'

// Lightweight interactive globe: 2D canvas rendering of rotating lat/long wireframe
export default function InteractiveEarth({ theme, height = 160 }: { theme: Theme; height?: number }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const mouse = useRef({ x: 0, y: 0 })
  const rotation = useRef({ yaw: 0, pitch: 0 })
  const dprRef = useRef(1)
  const nodesRef = useRef<Array<{ lat: number; lon: number; phase: number }>>([])
  const edgesRef = useRef<Array<[number, number]>>([])

  useEffect(() => {
    const canvas = ref.current!
    const ctx = canvas.getContext('2d')!
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1))
    dprRef.current = dpr

    let w = 0, h = 0
    let disposed = false

    function resize() {
      const parent = canvas.parentElement
      const rect = parent ? parent.getBoundingClientRect() : { width: window.innerWidth, height }
      w = rect.width
      h = height
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      // init lightweight random nodes and edges
      const N = 18
      nodesRef.current = Array.from({ length: N }, () => ({
        lat: Math.random() * 140 - 70, // avoid poles
        lon: Math.random() * 360 - 180,
        phase: Math.random() * Math.PI * 2,
      }))
      const sphDist = (a: { lat: number; lon: number }, b: { lat: number; lon: number }) => {
        const toRad = (d: number) => (d * Math.PI) / 180
        const A = toRad(a.lat)
        const B = toRad(b.lat)
        const dLon = toRad(b.lon - a.lon)
        return Math.acos(
          Math.sin(A) * Math.sin(B) + Math.cos(A) * Math.cos(B) * Math.cos(dLon)
        )
      }
      const edges: Array<[number, number]> = []
      for (let i = 0; i < nodesRef.current.length; i++) {
        const dists = nodesRef.current
          .map((n, j) => ({ j, d: i === j ? Infinity : sphDist(nodesRef.current[i], n) }))
          .sort((a, b) => a.d - b.d)
        for (let k = 0; k < 2; k++) {
          const j = dists[k].j
          if (j != null) edges.push([i, j])
        }
      }
      edgesRef.current = edges
    }

    function project(x: number, y: number, z: number) {
      // simple perspective
      const scale = 220
      const fov = 300
      const s = fov / (fov - z)
      return { X: x * s * (w / scale), Y: y * s * (w / scale) }
    }

    function draw() {
      if (disposed) return
      ctx.clearRect(0, 0, w, h)
      const color = theme === 'dark' ? 'rgba(0,255,65,0.35)' : 'rgba(11,129,46,0.35)'
      const colorHead = theme === 'dark' ? 'rgba(0,255,65,0.7)' : 'rgba(11,129,46,0.7)'
  ctx.save()
  ctx.translate(w * 0.28, h * 0.55) // move left, slightly lower
      const yaw = rotation.current.yaw
      const pitch = rotation.current.pitch

      // draw latitudes
      ctx.lineWidth = 1
      for (let lat = -60; lat <= 60; lat += 15) {
        ctx.beginPath()
        let first = true
        for (let lon = -180; lon <= 180; lon += 5) {
          const lt = (lat * Math.PI) / 180
          const ln = (lon * Math.PI) / 180
          // sphere coords
          let x = Math.cos(lt) * Math.cos(ln)
          let y = Math.sin(lt)
          let z = Math.cos(lt) * Math.sin(ln)
          // rotate Y (yaw)
          let x1 = x * Math.cos(yaw) - z * Math.sin(yaw)
          let z1 = x * Math.sin(yaw) + z * Math.cos(yaw)
          // rotate X (pitch)
          let y2 = y * Math.cos(pitch) - z1 * Math.sin(pitch)
          let z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch)
          const p = project(x1, y2, z2)
          if (first) {
            ctx.moveTo(p.X, p.Y)
            first = false
          } else ctx.lineTo(p.X, p.Y)
        }
        ctx.strokeStyle = color
        ctx.stroke()
      }

      // draw longitudes
      for (let lon = -180; lon <= 180; lon += 15) {
        ctx.beginPath()
        let first = true
        for (let lat = -80; lat <= 80; lat += 5) {
          const lt = (lat * Math.PI) / 180
          const ln = (lon * Math.PI) / 180
          let x = Math.cos(lt) * Math.cos(ln)
          let y = Math.sin(lt)
          let z = Math.cos(lt) * Math.sin(ln)
          let x1 = x * Math.cos(yaw) - z * Math.sin(yaw)
          let z1 = x * Math.sin(yaw) + z * Math.cos(yaw)
          let y2 = y * Math.cos(pitch) - z1 * Math.sin(pitch)
          let z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch)
          const p = project(x1, y2, z2)
          if (first) {
            ctx.moveTo(p.X, p.Y)
            first = false
          } else ctx.lineTo(p.X, p.Y)
        }
        ctx.strokeStyle = color
        ctx.stroke()
      }

      // bright rim (equator) for subtle highlight
      ctx.beginPath()
      ctx.ellipse(0, 0, w * 0.18, w * 0.18 * 0.33, 0, 0, Math.PI * 2)
      ctx.strokeStyle = colorHead
      ctx.stroke()

      // project and draw network links/nodes
      const t = performance.now() / 1000
      const proj: Array<{ x: number; y: number; z: number }> = []
      nodesRef.current.forEach((n) => {
        const lt = (n.lat * Math.PI) / 180
        const ln = (n.lon * Math.PI) / 180
        let x = Math.cos(lt) * Math.cos(ln)
        let y = Math.sin(lt)
        let z = Math.cos(lt) * Math.sin(ln)
        let x1 = x * Math.cos(yaw) - z * Math.sin(yaw)
        let z1 = x * Math.sin(yaw) + z * Math.cos(yaw)
        let y2 = y * Math.cos(pitch) - z1 * Math.sin(pitch)
        let z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch)
        const p = project(x1, y2, z2)
        proj.push({ x: p.X, y: p.Y, z: z2 })
      })

      // links (sparse)
      ctx.lineWidth = 1
      ctx.strokeStyle = theme === 'dark' ? 'rgba(0,255,65,0.2)' : 'rgba(11,129,46,0.25)'
      edgesRef.current.forEach(([a, b]) => {
        const A = proj[a]
        const B = proj[b]
        if (!A || !B) return
        if (A.z > -0.2 && B.z > -0.2) {
          ctx.beginPath()
          ctx.moveTo(A.x, A.y)
          ctx.lineTo(B.x, B.y)
          ctx.stroke()
        }
      })

      // nodes (pulsing)
      proj.forEach((P, i) => {
        if (P.z < -0.1) return
        const ph = nodesRef.current[i].phase
        const pulse = 1 + 0.5 * Math.sin(t * 2 + ph)
        const r = Math.max(1.2, 1.8 * pulse)
        const nodeFill = theme === 'dark' ? 'rgba(0,255,65,0.8)' : 'rgba(11,129,46,0.8)'
        ctx.beginPath()
        ctx.arc(P.x, P.y, r, 0, Math.PI * 2)
        ctx.fillStyle = nodeFill
        ctx.shadowColor = nodeFill
        ctx.shadowBlur = 8
        ctx.fill()
        ctx.shadowBlur = 0
      })

      ctx.restore()

      // auto rotate
      rotation.current.yaw += 0.002
      rotation.current.pitch += 0.0004
      // gradient mask: soften top & left edges
      ctx.save()
      ctx.globalCompositeOperation = 'destination-in'
      const gTop = ctx.createLinearGradient(0, 0, 0, h)
      gTop.addColorStop(0, 'rgba(0,0,0,0)')
      gTop.addColorStop(0.25, 'rgba(0,0,0,1)')
      gTop.addColorStop(1, 'rgba(0,0,0,1)')
      ctx.fillStyle = gTop
      ctx.fillRect(0, 0, w, h)
      const gLeft = ctx.createLinearGradient(0, 0, w, 0)
      gLeft.addColorStop(0, 'rgba(0,0,0,0)')
      gLeft.addColorStop(0.18, 'rgba(0,0,0,1)')
      gLeft.addColorStop(1, 'rgba(0,0,0,1)')
      ctx.fillStyle = gLeft
      ctx.fillRect(0, 0, w, h)
      ctx.restore()
      rafRef.current = requestAnimationFrame(draw)
    }

    function onMove(e: MouseEvent) {
      const rect = canvas.getBoundingClientRect()
      mouse.current.x = (e.clientX - rect.left) / rect.width
      mouse.current.y = (e.clientY - rect.top) / rect.height
      // target rotation based on mouse (small delta)
      const targetYaw = (mouse.current.x - 0.5) * 0.6
      const targetPitch = (mouse.current.y - 0.5) * 0.2
      rotation.current.yaw += (targetYaw - rotation.current.yaw) * 0.02
      rotation.current.pitch += (targetPitch - rotation.current.pitch) * 0.02
    }

    resize()
    draw()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas.parentElement || document.body)
    window.addEventListener('mousemove', onMove, { passive: true })

    return () => {
      disposed = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      ro.disconnect()
      window.removeEventListener('mousemove', onMove)
    }
  }, [theme, height])

  return (
    <canvas
      ref={ref}
      aria-hidden
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height,
        pointerEvents: 'none',
        zIndex: 0,
        opacity: 0.8,
      }}
    />
  )
}
