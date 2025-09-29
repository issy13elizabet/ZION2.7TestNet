'use client'
import React, { useEffect, useMemo, useState } from 'react'

type Theme = 'dark' | 'light'

type PresenceState = {
  lastCheck?: string // ISO date string (YYYY-MM-DD)
  streak?: number
}

function todayIso(): string {
  const d = new Date()
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

function isYesterday(iso: string): boolean {
  const d = new Date(iso + 'T00:00:00')
  const y = new Date()
  y.setDate(y.getDate() - 1)
  return d.toDateString() === y.toDateString()
}

export default function Presence({ theme }: { theme: Theme }) {
  const [state, setState] = useState<PresenceState>({})
  const [touched, setTouched] = useState(false)

  useEffect(() => {
    try {
      const raw = localStorage.getItem('zion-presence')
      if (raw) setState(JSON.parse(raw))
    } catch {}
  }, [])

  const checkedToday = useMemo(() => state.lastCheck === todayIso(), [state.lastCheck])
  const streak = state.streak || 0

  const save = (s: PresenceState) => {
    setState(s)
    try { localStorage.setItem('zion-presence', JSON.stringify(s)) } catch {}
  }

  const doCheckIn = () => {
    if (checkedToday) return
    const last = state.lastCheck
    const next: PresenceState = {
      lastCheck: todayIso(),
      streak: last && isYesterday(last) ? (streak + 1) : Math.max(1, streak)
    }
    save(next)
    setTouched(true)
    // gentle haptic/visual cue could be added later
  }

  const styles = {
    border: theme === 'dark' ? '#072807' : '#cfe9d6',
    accent: theme === 'dark' ? '#00ff41' : '#0b812e',
    text: theme === 'dark' ? '#eaeaea' : '#0a0a0a',
    sub: theme === 'dark' ? '#8aff9a' : '#0d7a30',
    bg: theme === 'dark' ? 'transparent' : 'transparent',
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <button
        onClick={doCheckIn}
        title={checkedToday ? 'Dnešní přítomnost již potvrzena' : 'Potvrdit dnešní přítomnost'}
        style={{
          background: styles.bg,
          color: styles.text,
          border: `1px solid ${styles.border}`,
          padding: '6px 10px',
          borderRadius: 8,
          cursor: checkedToday ? 'default' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          opacity: checkedToday ? 0.9 : 1,
        }}
      >
        <span style={{ color: styles.accent }}>{checkedToday ? '✓' : '●'}</span>
        {checkedToday ? 'Přítomen' : 'Check‑in dnes'}
      </button>
      <span style={{ fontSize: 12, color: styles.sub }}>
        Streak: {streak} {streak === 1 ? 'den' : 'dnů'}
      </span>
    </div>
  )
}
