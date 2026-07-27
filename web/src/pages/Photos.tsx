import { useEffect, useMemo, useState } from 'react'

type Source = 'texasmushrooms' | 'inaturalist' | 'both'

type PhotoRow = {
  id: string
  date?: string
  label_species?: string
  photo_url?: string
  page_url?: string
  latitude?: number
  longitude?: number
  h3_index?: string
  local_relpath?: string
  source?: string
  license?: string
}

function buildProxyUrl(photo: PhotoRow, baseUrl: string): string {
  const base = baseUrl.trim().replace(/\/$/, '')
  const url = (photo.photo_url ?? '').trim()
  if (!base || !url) return ''

  // Optional: the companion Python server supports `/proxy?url=...&ref=...`.
  // If the base URL is just a static file server, this will 404 and we’ll fall back.
  const ref = (photo.page_url ?? '').trim()
  const params = new URLSearchParams({ url })
  if (ref) params.set('ref', ref)
  return `${base}/proxy?${params.toString()}`
}

function buildCandidateImageUrls(photo: PhotoRow, localBaseUrl: string): string[] {
  const urls: string[] = []

  const base = localBaseUrl.trim().replace(/\/$/, '')
  const proxy = buildProxyUrl(photo, base)
  if (proxy) urls.push(proxy)

  const rel = (photo.local_relpath ?? '').trim().replace(/^\//, '')
  if (base && rel) urls.push(`${base}/${rel}`)

  const remote = (photo.photo_url ?? '').trim()
  if (remote) urls.push(remote)

  // De-dupe while preserving order.
  return urls.filter((u, i) => urls.indexOf(u) === i)
}

function PhotoCard({ photo, localBaseUrl }: { photo: PhotoRow; localBaseUrl: string }) {
  const [failedCount, setFailedCount] = useState(0)

  const candidates = buildCandidateImageUrls(photo, localBaseUrl)
  const src = candidates[Math.min(failedCount, Math.max(candidates.length - 1, 0))] ?? ''

  return (
    <div className="card">
      {src ? (
        <img
          src={src}
          loading="lazy"
          decoding="async"
          onError={() => setFailedCount((c) => c + 1)}
          alt={photo.label_species ?? 'photo'}
        />
      ) : (
        <div style={{ height: 140, display: 'grid', placeItems: 'center', fontSize: 12, color: '#666' }}>
          No image
        </div>
      )}
      <div className="card-body">
        <div style={{ fontWeight: 600, marginBottom: 4 }}>{photo.label_species ?? 'Unidentified'}</div>
        <div style={{ color: '#444' }}>{photo.date ?? ''}</div>
        {photo.source ? (
          <div style={{ marginTop: 4 }}>
            <span
              style={{
                fontSize: 11,
                padding: '1px 6px',
                borderRadius: 8,
                background: photo.source === 'inaturalist' ? '#e8f5e9' : '#e3f2fd',
                color: photo.source === 'inaturalist' ? '#1b5e20' : '#0d47a1',
              }}
            >
              {photo.source === 'inaturalist' ? 'iNaturalist' : 'texasmushrooms.org'}
            </span>
          </div>
        ) : null}
        {photo.page_url ? (
          <div style={{ marginTop: 6 }}>
            <a href={photo.page_url} target="_blank" rel="noreferrer" style={{ fontSize: 12 }}>
              Source page
            </a>
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default function PhotosPage() {
  const [txRows, setTxRows] = useState<PhotoRow[] | null>(null)
  const [inatRows, setInatRows] = useState<PhotoRow[] | null>(null)
  const [source, setSource] = useState<Source>('both')
  const [localBaseUrl, setLocalBaseUrl] = useState('')
  const [visibleCount, setVisibleCount] = useState(200)

  useEffect(() => {
    let cancelled = false
    const load = (url: string, setter: (r: PhotoRow[] | null) => void) => {
      fetch(url)
        .then((r) => (r.ok ? r.json() : null))
        .then((j) => {
          if (!cancelled) setter(j)
        })
        .catch(() => {
          if (!cancelled) setter(null)
        })
    }
    load('/data/photos_index.json', setTxRows)
    load('/data/photos_index_inat.json', setInatRows)

    return () => {
      cancelled = true
    }
  }, [])

  const rows = useMemo(() => {
    const tx = source === 'inaturalist' ? [] : txRows ?? []
    const inat = source === 'texasmushrooms' ? [] : inatRows ?? []
    if (txRows === null && inatRows === null) return null
    return [...tx, ...inat]
  }, [txRows, inatRows, source])

  const visible = useMemo(() => {
    if (!rows) return []
    return rows.slice(0, visibleCount)
  }, [rows, visibleCount])

  return (
    <div className="page">
      <div className="page-header">
        <div className="row">
          <strong>Source</strong>
          <select className="select" value={source} onChange={(e) => setSource(e.target.value as Source)}>
            <option value="texasmushrooms">texasmushrooms.org</option>
            <option value="inaturalist">iNaturalist</option>
            <option value="both">Both</option>
          </select>
          <strong style={{ marginLeft: 12 }}>Local Images Base URL</strong>
          <input
            className="input"
            style={{ width: 320 }}
            value={localBaseUrl}
            onChange={(e) => setLocalBaseUrl(e.target.value)}
            placeholder="optional fallback, e.g. http://localhost:8001"
          />
        </div>
        <div style={{ fontSize: 12, color: '#444' }}>
          {rows ? <span>Showing {visible.length} of {rows.length}</span> : <span>Export `web/public/data/photos_index*.json` to browse photos.</span>}
        </div>
        {rows && visibleCount < rows.length ? (
          <button className="button" onClick={() => setVisibleCount((c) => c + 200)}>
            Load 200 more
          </button>
        ) : null}
      </div>

      <div className="page-body" style={{ overflow: 'auto' }}>
        <div className="grid">
          {visible.map((p) => (
            <PhotoCard key={p.id} photo={p} localBaseUrl={localBaseUrl} />
          ))}
        </div>
      </div>
    </div>
  )
}
