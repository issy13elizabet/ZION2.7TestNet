'use client'
import React, { useEffect, useMemo, useState } from 'react'

type Book = {
  id: string
  title: string
  description?: string
  landingPage?: string
  downloads?: { url: string; type?: string; note?: string }[]
  categories?: string[]
  tags?: string[]
  language?: string
  cover?: string
}

type AmentiImage = { id: string; title?: string; url: string }

export default function AmentiPage() {
  const [books, setBooks] = useState<Book[]>([])
  const [images, setImages] = useState<AmentiImage[]>([])
  const [q, setQ] = useState('')
  const [lang, setLang] = useState('all')
  const [cat, setCat] = useState('all')
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/amenti').then(async (r) => {
      if (!r.ok) throw new Error('Manifest load failed')
      const j = await r.json()
      setBooks(j.books || [])
      setImages(j.images || [])
    }).catch((e) => setErr(String(e)))
  }, [])

  const f = (b: Book) => {
    const hay = [b.title, b.description, ...(b.tags || []), ...(b.categories || [])]
      .join(' ')
      .toLowerCase()
    const okQ = q ? hay.includes(q.toLowerCase()) : true
    const okL = lang === 'all' ? true : ((b.language || 'multi').toLowerCase() === lang.toLowerCase())
    const okC = cat === 'all' ? true : (b.categories || []).some((c) => c.toLowerCase() === cat.toLowerCase())
    return okQ && okL && okC
  }

  const categories = useMemo(() => {
    const s = new Set<string>()
    books.forEach((b) => (b.categories || []).forEach((c) => s.add(c)))
    return Array.from(s).sort((a, b) => a.localeCompare(b))
  }, [books])

  const languages = useMemo(() => {
    const s = new Set<string>()
    books.forEach((b) => s.add((b.language || 'multi')))
    return Array.from(s).sort((a, b) => a.localeCompare(b))
  }, [books])

  const halls = useMemo(() => {
    const found = images.find((img) => img.id === 'halls-144')
    return found?.url || 'https://newearth.cz/V2/img/144Halls.jpg'
  }, [images])

  return (
    <div>
      <div style={{
        borderRadius: 16,
        overflow: 'hidden',
        border: '1px solid #072807',
        marginBottom: 16,
      }}>
        <div style={{
          height: 260,
          backgroundImage: `url(${halls})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          filter: 'saturate(110%)',
          position: 'relative',
        }} />
        <div style={{ padding: 16, background: 'rgba(0, 10, 0, 0.9)', borderTop: '1px solid #072807' }}>
          <h1 style={{ color: '#00ff41', margin: '0 0 6px 0', textShadow:'0 0 8px rgba(0,255,65,0.3)' }}>Amenti Library</h1>
          <p style={{ margin: 0, color: '#8aff9a' }}>Katalog duchovních textů a odkazů – metadata a veřejné linky z Halls of Amenti.</p>
        </div>
      </div>
      <div style={{ display:'flex', gap:12, flexWrap:'wrap', marginBottom:12 }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Hledat (název, tagy)"
          style={{ background:'#001900', color:'#eaeaea', border:'1px solid #0a3d0a', padding:'10px 12px', borderRadius:8, width:320, boxShadow:'0 0 0 2px rgba(0,255,65,0.08) inset' }}
        />
        <select value={cat} onChange={(e)=>setCat(e.target.value)} style={{ background:'#001900', color:'#eaeaea', border:'1px solid #0a3d0a', padding:'10px 12px', borderRadius:8 }}>
          <option value="all">Všechny kategorie</option>
          {categories.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select value={lang} onChange={(e)=>setLang(e.target.value)} style={{ background:'#001900', color:'#eaeaea', border:'1px solid #0a3d0a', padding:'10px 12px', borderRadius:8 }}>
          <option value="all">Všechny jazyky</option>
          {languages.map((l) => (
            <option key={l} value={l.toLowerCase()}>{l}</option>
          ))}
        </select>
      </div>
      {err && <p style={{ color:'#ff8080' }}>{err}</p>}
      <div>
        {books.filter(f).map((b) => (
          <div key={b.id} style={{ border:'1px solid #072807', borderRadius:12, margin:'12px 0', background:'rgba(0, 10, 0, 0.9)', overflow:'hidden', boxShadow:'0 0 18px rgba(0,255,65,0.06) inset' }}>
            {b.cover && (
              <div style={{ height: 160, backgroundImage:`url(${b.cover})`, backgroundSize:'cover', backgroundPosition:'center' }} />
            )}
            <div style={{ padding:16 }}>
            <h3 style={{ color:'#00ff41', margin:0, textShadow:'0 0 8px rgba(0,255,65,0.25)' }}>{b.title}</h3>
            <div style={{ display:'flex', gap:8, flexWrap:'wrap', margin:'8px 0' }}>
              {(b.categories||[]).map((c) => <span key={c} style={{ fontSize:12, border:'1px solid #0a3d0a', background:'#001900', color:'#8aff9a', padding:'2px 8px', borderRadius:999 }}>{c}</span>)}
            </div>
            {b.description && <p>{b.description}</p>}
            <div>
              {b.landingPage && <a href={b.landingPage} target="_blank" style={{ color:'#8aff9a', marginRight:12 }}>Otevřít</a>}
              {(b.downloads||[]).map((d, i) => {
                const label = 'Stáhnout' + (d.type ? ' (' + d.type + ')' : '')
                return (
                  <a key={i} href={d.url} target="_blank" style={{ color:'#8aff9a', marginRight:12 }}>{label}</a>
                )
              })}
            </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
