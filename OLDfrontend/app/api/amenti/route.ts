import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import path from 'path'

export async function GET() {
  try {
    const root = path.resolve(process.cwd(), '..')
    const p = path.join(root, 'data', 'amenti', 'library.json')
    const raw = await readFile(p, 'utf-8')
    const json = JSON.parse(raw)
    return NextResponse.json(json, { status: 200 })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || 'Failed to load manifest' }, { status: 500 })
  }
}
