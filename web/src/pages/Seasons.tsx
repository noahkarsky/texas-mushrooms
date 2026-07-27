import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

// --------------------------------------------------------------------------- //
// Types
// --------------------------------------------------------------------------- //
type SeasonPhoto = {
  id: string
  date: string
  doy: number
  year: number
  color: string | null
  swatches: string[]
  species: string | null
  genus: string | null
  photo_url: string | null
  page_url: string | null
}

type WeatherSeries = {
  rain7: (number | null)[]
  rain: (number | null)[]
  tmean: (number | null)[]
  soil: (number | null)[]
  anom: (number | null)[]
}

type SeasonWeather = {
  years: number[]
  doy: number[]
  series: Record<string, WeatherSeries>
  climatology: { rain30_mean: (number | null)[] }
}

type Dot = {
  photo: SeasonPhoto
  cx: number
  cy: number
  appear: number // 0..1 left-to-right entrance threshold
}

// A lane is one horizontal band: a single year, or all years combined into one.
type Lane = { key: string; label: string; years: number[] }

// Weather series aggregated (mean per day-of-year) over a lane's years.
type LaneSeries = { rain7: (number | null)[]; tmean: (number | null)[]; anom: (number | null)[] }

type RibbonMode = 'rain' | 'temp' | 'genus' | 'none'

// One genus row of the phenology ridgeline.
type Ridge = {
  genus: string
  color: string
  counts: number[] // smoothed per-day-of-year photo counts, index 0..366
  total: number
  peakDoy: number
  peakVal: number
  window: string // human "Jun–Aug" fruiting window
  samples: SeasonPhoto[] // representative photos, spread across the season
}

// Pixel geometry of the ridgeline, stashed so hit-testing can find a row.
type RidgeGeom = { topPad: number; rowH: number; ridges: Ridge[] }

type RibbonData =
  | { mode: 'none' }
  | { mode: 'rain'; vmax: number }
  | { mode: 'temp'; vmin: number; vmax: number } // degrees F
  | {
      mode: 'genus'
      filtered: boolean
      bands: { genus: string; color: string }[]
      perLane: number[][][] // [laneIndex][bandIndex][doy 0..366] smoothed counts
      vmax: number // max stacked total
    }

// --------------------------------------------------------------------------- //
// Layout constants (CSS pixels; row height is responsive to viewport)
// --------------------------------------------------------------------------- //
const MARGIN_L = 52
const MARGIN_R = 16
const MARGIN_TOP = 12
const AXIS_H = 26
const MIN_ROW_H = 72
const MAX_ROW_H = 150
const INNER_PAD = 10
const RIBBON_GAP = 3
const NULL_DOT = '#5b6470'
const WET: [number, number, number] = [58, 134, 200]
const DRY: [number, number, number] = [214, 138, 43]

// Genus stack palette — muted hues legible on the near-black background.
const GENUS_COLORS = ['#d08770', '#a3be8c', '#b48ead', '#88a0c8', '#d6b96a']
const OTHER_COLOR = '#5b6470'
const MATCH_COLOR = '#e0b155'
const N_TOP_GENERA = 5

// Genus ridgeline ("when does each genus fruit") — a taller, dedicated view
// that takes over the canvas when the Genera ribbon is selected. Each genus is
// its own labeled row of a seasonal density curve, ordered by peak fruiting day.
const RIDGE_L = 128 // left gutter for genus names
const N_RIDGE_GENERA = 14
const RIDGE_OVERLAP = 1.9 // how far a peak may rise into the row above
const RIDGE_MIN_ROW = 30
const RIDGE_MAX_ROW = 92
// Ordered, wide-gamut but muted palette; assigned in seasonal (peak-day) order
// so warm early-season genera flow into cool late-season ones.
const RIDGE_COLORS = [
  '#8fb96a', '#a3be8c', '#7fb0a0', '#6fa8c0', '#88a0c8',
  '#9d8fc4', '#b48ead', '#c98aa6', '#d08770', '#d6a45f',
  '#d8b24a', '#c2b36a', '#9bb08a', '#7fa6b8',
]

const clamp = (lo: number, v: number, hi: number) => Math.max(lo, Math.min(hi, v))

// Height of the context ribbon under a row's dots (0 when ribbon is off).
function ribbonHeight(rowH: number, on: boolean, single: boolean): number {
  if (!on) return 0
  return clamp(18, Math.round(rowH * 0.28), single ? 90 : 40)
}

const MONTHS = [
  { label: 'Jan', doy: 1 },
  { label: 'Feb', doy: 32 },
  { label: 'Mar', doy: 60 },
  { label: 'Apr', doy: 91 },
  { label: 'May', doy: 121 },
  { label: 'Jun', doy: 152 },
  { label: 'Jul', doy: 182 },
  { label: 'Aug', doy: 213 },
  { label: 'Sep', doy: 244 },
  { label: 'Oct', doy: 274 },
  { label: 'Nov', doy: 305 },
  { label: 'Dec', doy: 335 },
]
// Inclusive day-of-year bounds for each month (1-indexed month), non-leap ref.
const MONTH_BOUNDS: Record<number, [number, number]> = {
  1: [1, 31],
  2: [32, 59],
  3: [60, 90],
  4: [91, 120],
  5: [121, 151],
  6: [152, 181],
  7: [182, 212],
  8: [213, 243],
  9: [244, 273],
  10: [274, 304],
  11: [305, 334],
  12: [335, 366],
}
const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// Deterministic 0..1 jitter from a photo id, so dots stay put across renders.
function hashUnit(id: string): number {
  let h = 2166136261
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return ((h >>> 0) % 100000) / 100000
}

// Wetness anomaly -> [r, g, b, alpha]. Blue = wetter, amber = drier than normal.
function anomRGBA(a: number | null): [number, number, number, number] {
  if (a == null) return [32, 36, 44, 0.35]
  const mag = Math.min(Math.abs(a), 3) / 3
  const [r, g, b] = a >= 0 ? WET : DRY
  return [r, g, b, 0.06 + mag * 0.5]
}

function monthOf(date: string): number {
  return Number(date.slice(5, 7))
}

function fmtDate(date: string): string {
  const m = Number(date.slice(5, 7))
  const d = Number(date.slice(8, 10))
  const y = date.slice(0, 4)
  if (!m || !d) return date
  return `${MONTH_NAMES[m - 1]} ${d}, ${y}`
}

// Month name for a day-of-year (non-leap reference), e.g. 182 -> "Jul".
function monthNameOfDoy(doy: number): string {
  for (let m = 1; m <= 12; m++) {
    const [lo, hi] = MONTH_BOUNDS[m]
    if (doy >= lo && doy <= hi) return MONTH_NAMES[m - 1]
  }
  return ''
}

// Rough "fruiting window" label: the +/- span around the peak that holds the
// bulk of a genus's photos, expressed as month names.
function windowLabel(counts: number[], peakDoy: number): string {
  let total = 0
  for (let d = 1; d <= 366; d++) total += counts[d]
  if (total <= 0) return monthNameOfDoy(peakDoy)
  // Grow a window outward from the peak until it covers ~70% of the mass.
  let lo = peakDoy
  let hi = peakDoy
  let acc = counts[peakDoy] ?? 0
  while (acc < total * 0.7 && (lo > 1 || hi < 366)) {
    const left = lo > 1 ? counts[lo - 1] : -1
    const right = hi < 366 ? counts[hi + 1] : -1
    if (right >= left) {
      hi++
      acc += counts[hi] ?? 0
    } else {
      lo--
      acc += counts[lo] ?? 0
    }
  }
  const a = monthNameOfDoy(lo)
  const b = monthNameOfDoy(hi)
  return a === b ? a : `${a}–${b}`
}

// Centered 7-day moving average of a per-day-of-year count array (index 0..366).
function smooth7(counts: number[]): number[] {
  const out = new Array(counts.length).fill(0)
  for (let d = 1; d <= 366; d++) {
    let sum = 0
    let n = 0
    for (let k = -3; k <= 3; k++) {
      const j = d + k
      if (j >= 1 && j <= 366) {
        sum += counts[j]
        n++
      }
    }
    out[d] = n ? sum / n : 0
  }
  return out
}

// --------------------------------------------------------------------------- //
// Component
// --------------------------------------------------------------------------- //
export default function SeasonsPage() {
  const [photos, setPhotos] = useState<SeasonPhoto[] | null>(null)
  const [weather, setWeather] = useState<SeasonWeather | null>(null)
  const [proxyBase, setProxyBase] = useState('http://localhost:8001')
  const [filter, setFilter] = useState('')
  const [focusYear, setFocusYear] = useState<number | null>(null)
  const [combined, setCombined] = useState(false)
  const [month, setMonth] = useState<number | null>(null)
  const [ribbon, setRibbon] = useState<RibbonMode>('rain')
  const [hovered, setHovered] = useState<{ photo: SeasonPhoto; x: number; y: number } | null>(null)
  const [ridgeHover, setRidgeHover] = useState<{ ridge: Ridge; x: number; y: number } | null>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const ridgeGeomRef = useRef<RidgeGeom | null>(null)

  // ------- data -------
  useEffect(() => {
    let cancelled = false
    Promise.all([
      fetch('/data/season_photos.json').then((r) => {
        if (!r.ok) throw new Error(`season_photos.json: ${r.status}`)
        return r.json()
      }),
      fetch('/data/season_weather.json').then((r) => {
        if (!r.ok) throw new Error(`season_weather.json: ${r.status}`)
        return r.json()
      }),
    ])
      .then(([p, w]) => {
        if (!cancelled) {
          setPhotos(p)
          setWeather(w)
        }
      })
      .catch((e) => console.error(e))
    return () => {
      cancelled = true
    }
  }, [])

  // ------- responsive size -------
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const update = () =>
      setSize((prev) =>
        Math.abs(el.clientWidth - prev.w) > 0.5 || Math.abs(el.clientHeight - prev.h) > 0.5
          ? { w: el.clientWidth, h: el.clientHeight }
          : prev,
      )
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [photos])

  const allYears = useMemo(() => {
    if (weather?.years?.length) return weather.years
    if (!photos) return []
    return Array.from(new Set(photos.map((p) => p.year))).sort((a, b) => a - b)
  }, [weather, photos])

  // Which years are drawn (all, or just the zoomed one).
  const displayYears = useMemo(
    () => (focusYear != null ? [focusYear] : allYears),
    [focusYear, allYears],
  )

  // Rows to draw: one lane per year (stacked), or a single lane for every year
  // squished together (combined), or a single zoomed year.
  const lanes = useMemo<Lane[]>(() => {
    if (combined && displayYears.length > 1) {
      const a = displayYears[0]
      const b = displayYears[displayYears.length - 1]
      return [{ key: 'all', label: `${a}–${b}`, years: displayYears }]
    }
    return displayYears.map((y) => ({ key: String(y), label: String(y), years: [y] }))
  }, [combined, displayYears])

  const yearToLane = useMemo(() => {
    const m = new Map<number, number>()
    lanes.forEach((ln, i) => ln.years.forEach((y) => m.set(y, i)))
    return m
  }, [lanes])

  // Weather aggregated (mean per day-of-year) over each lane's years. For a
  // single-year lane this is just that year; for the combined lane it is the
  // climatological mean across 2018-2024.
  const laneSeries = useMemo<LaneSeries[]>(() => {
    if (!weather) return []
    const meanField = (years: number[], field: 'rain7' | 'tmean' | 'anom') => {
      const out: (number | null)[] = new Array(366).fill(null)
      for (let idx = 0; idx < 366; idx++) {
        let sum = 0
        let n = 0
        for (const y of years) {
          const v = weather.series[String(y)]?.[field]?.[idx]
          if (v != null) {
            sum += v
            n++
          }
        }
        out[idx] = n ? sum / n : null
      }
      return out
    }
    return lanes.map((ln) => ({
      rain7: meanField(ln.years, 'rain7'),
      tmean: meanField(ln.years, 'tmean'),
      anom: meanField(ln.years, 'anom'),
    }))
  }, [weather, lanes])

  // Day-of-year domain (full year, or a single zoomed month).
  const domain = useMemo<[number, number]>(
    () => (month != null ? MONTH_BOUNDS[month] : [1, 366]),
    [month],
  )

  const filterLc = filter.trim().toLowerCase()

  // ------- context ribbon data (shared scales across lanes) -------
  const ribbonData = useMemo<RibbonData>(() => {
    if (!weather || !photos || ribbon === 'none') return { mode: 'none' }

    if (ribbon === 'rain') {
      let vmax = 0
      for (const ls of laneSeries) for (const v of ls.rain7) if (v != null && v > vmax) vmax = v
      return { mode: 'rain', vmax: vmax || 1 }
    }

    if (ribbon === 'temp') {
      let vmin = Infinity
      let vmax = -Infinity
      for (const ls of laneSeries) {
        for (const c of ls.tmean) {
          if (c == null) continue
          const f = c * 1.8 + 32
          if (f < vmin) vmin = f
          if (f > vmax) vmax = f
        }
      }
      if (!isFinite(vmin)) return { mode: 'temp', vmin: 0, vmax: 1 }
      return { mode: 'temp', vmin: vmin - 2, vmax: vmax + 2 }
    }

    // genus
    if (filterLc) {
      // Species filter active: collapse to a single density band of matches.
      const perLane: number[][][] = lanes.map(() => [new Array(367).fill(0)])
      let vmax = 0
      for (const p of photos) {
        const li = yearToLane.get(p.year)
        if (li == null) continue
        if (!(p.species ?? '').toLowerCase().includes(filterLc)) continue
        perLane[li][0][p.doy]++
      }
      for (const lane of perLane) {
        lane[0] = smooth7(lane[0])
        for (const v of lane[0]) if (v > vmax) vmax = v
      }
      return {
        mode: 'genus',
        filtered: true,
        bands: [{ genus: filter.trim() || 'match', color: MATCH_COLOR }],
        perLane,
        vmax: vmax || 1,
      }
    }

    // Rank the top genera across all displayed photos; everything else -> "other".
    const totals = new Map<string, number>()
    for (const p of photos) {
      if (yearToLane.get(p.year) == null || !p.genus) continue
      totals.set(p.genus, (totals.get(p.genus) ?? 0) + 1)
    }
    const top = [...totals.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, N_TOP_GENERA)
      .map(([g]) => g)
    const bands = top.map((g, i) => ({ genus: g, color: GENUS_COLORS[i] }))
    bands.push({ genus: 'other', color: OTHER_COLOR })
    const bandIndex = new Map<string, number>()
    top.forEach((g, i) => bandIndex.set(g, i))
    const otherIdx = bands.length - 1

    const perLane: number[][][] = lanes.map(() => bands.map(() => new Array(367).fill(0)))
    for (const p of photos) {
      const li = yearToLane.get(p.year)
      if (li == null) continue
      const bi = p.genus != null && bandIndex.has(p.genus) ? bandIndex.get(p.genus)! : otherIdx
      perLane[li][bi][p.doy]++
    }
    let vmax = 0
    perLane.forEach((arr, li) => {
      const smoothed = arr.map((b) => smooth7(b))
      perLane[li] = smoothed
      for (let d = 1; d <= 366; d++) {
        let stack = 0
        for (const b of smoothed) stack += b[d]
        if (stack > vmax) vmax = stack
      }
    })
    return { mode: 'genus', filtered: false, bands, perLane, vmax: vmax || 1 }
  }, [weather, photos, lanes, laneSeries, yearToLane, ribbon, filter, filterLc])

  // ------- genus ridgeline data -------
  // One entry per top genus: its smoothed seasonal density curve, total photo
  // count, and peak day. Sorted by peak day so the ridgeline reads as a calendar
  // (earliest-fruiting genera on top, latest at the bottom).
  const phenology = useMemo<Ridge[]>(() => {
    if (ribbon !== 'genus' || !photos) return []
    const yearSet = new Set(displayYears)
    const totals = new Map<string, number>()
    for (const p of photos) {
      if (!p.genus || !yearSet.has(p.year)) continue
      if (month != null && monthOf(p.date) !== month) continue
      if (filterLc && !(p.species ?? '').toLowerCase().includes(filterLc)) continue
      totals.set(p.genus, (totals.get(p.genus) ?? 0) + 1)
    }
    const top = [...totals.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, N_RIDGE_GENERA)
      .map(([g]) => g)
    if (top.length === 0) return []
    const idx = new Map(top.map((g, i) => [g, i]))
    const raw = top.map(() => new Array(367).fill(0))
    const samplePool = top.map<SeasonPhoto[]>(() => [])
    for (const p of photos) {
      if (!p.genus || !yearSet.has(p.year)) continue
      if (filterLc && !(p.species ?? '').toLowerCase().includes(filterLc)) continue
      const bi = idx.get(p.genus)
      if (bi == null) continue
      raw[bi][p.doy]++
      if (p.photo_url) samplePool[bi].push(p)
    }
    // Pick up to 8 photos per genus, spread across the day-of-year so the
    // thumbnails show the range of what fruits in that genus over the season.
    const pickSamples = (pool: SeasonPhoto[]): SeasonPhoto[] => {
      if (pool.length <= 8) return pool
      const sorted = [...pool].sort((a, b) => a.doy - b.doy)
      const out: SeasonPhoto[] = []
      const step = sorted.length / 8
      for (let k = 0; k < 8; k++) out.push(sorted[Math.floor(k * step)])
      return out
    }
    const ridges: Ridge[] = top.map((g, i) => {
      const counts = smooth7(raw[i])
      let peakDoy = 1
      let peakVal = 0
      for (let d = 1; d <= 366; d++) {
        if (counts[d] > peakVal) {
          peakVal = counts[d]
          peakDoy = d
        }
      }
      return {
        genus: g,
        color: '',
        counts,
        total: totals.get(g) ?? 0,
        peakDoy,
        peakVal,
        window: windowLabel(counts, peakDoy),
        samples: pickSamples(samplePool[i]),
      }
    })
    ridges.sort((a, b) => a.peakDoy - b.peakDoy)
    ridges.forEach((r, i) => (r.color = RIDGE_COLORS[i % RIDGE_COLORS.length]))
    return ridges
  }, [ribbon, photos, displayYears, month, filterLc])

  // ------- layout (pixel positions) -------
  const layout = useMemo(() => {
    if (!photos || !weather || size.w <= 0 || lanes.length === 0) return null
    const single = lanes.length === 1
    const plotW = size.w - MARGIN_L - MARGIN_R
    const rowH = single
      ? Math.max(340, size.h - MARGIN_TOP - AXIS_H - 6)
      : clamp(MIN_ROW_H, Math.floor((size.h - MARGIN_TOP - AXIS_H) / lanes.length), MAX_ROW_H)
    const ribH = ribbonHeight(rowH, ribbon !== 'none', single)
    const bandH = Math.max(6, rowH - INNER_PAD - RIBBON_GAP - ribH - 4)
    const totalH = MARGIN_TOP + lanes.length * rowH + AXIS_H
    const [d0, d1] = domain
    const span = Math.max(1, d1 - d0)
    const dotR = combined ? 2.5 : single ? 3.2 : clamp(2.0, rowH / 34, 2.8)

    const xForDoy = (doy: number) => MARGIN_L + ((doy - d0) / span) * plotW
    const rowTop = (i: number) => MARGIN_TOP + i * rowH
    const bandTop = (i: number) => rowTop(i) + INNER_PAD
    const ribbonTop = (i: number) => rowTop(i) + rowH - ribH - 4

    const dotsByRow: Dot[][] = lanes.map(() => [])
    for (const p of photos) {
      const i = yearToLane.get(p.year)
      if (i == null) continue
      if (month != null && monthOf(p.date) !== month) continue
      const cx = xForDoy(p.doy)
      dotsByRow[i].push({
        photo: p,
        cx,
        cy: bandTop(i) + hashUnit(p.id) * bandH,
        appear: Math.min(1, Math.max(0, (cx - MARGIN_L) / plotW) + i * 0.02),
      })
    }
    return {
      plotW,
      rowH,
      ribH,
      bandH,
      totalH,
      dotR,
      single,
      d0,
      d1,
      span,
      xForDoy,
      rowTop,
      bandTop,
      ribbonTop,
      dotsByRow,
    }
  }, [photos, weather, size, lanes, yearToLane, combined, month, domain, ribbon])

  // ------- drawing -------
  const progressRef = useRef(1)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dprEarly = window.devicePixelRatio || 1

    // ---- Genus ridgeline: dedicated "when does each genus fruit" view ----
    if (ribbon === 'genus' && photos && phenology.length && size.w > 0) {
      const n = phenology.length
      const topPad = MARGIN_TOP + 30
      const bottomAxis = AXIS_H + 6
      const avail = Math.max(220, size.h - topPad - bottomAxis)
      const rowH = clamp(RIDGE_MIN_ROW, Math.floor(avail / n), RIDGE_MAX_ROW)
      const ridgeH = topPad + n * rowH + bottomAxis
      ridgeGeomRef.current = { topPad, rowH, ridges: phenology }

      if (
        canvas.width !== Math.round(size.w * dprEarly) ||
        canvas.height !== Math.round(ridgeH * dprEarly)
      ) {
        canvas.width = Math.round(size.w * dprEarly)
        canvas.height = Math.round(ridgeH * dprEarly)
        canvas.style.width = `${size.w}px`
        canvas.style.height = `${ridgeH}px`
      }
      ctx.setTransform(dprEarly, 0, 0, dprEarly, 0, 0)
      ctx.clearRect(0, 0, size.w, ridgeH)

      const plotW = size.w - RIDGE_L - MARGIN_R
      const [d0, d1] = domain
      const span = Math.max(1, d1 - d0)
      const xForDoy = (doy: number) => RIDGE_L + ((doy - d0) / span) * plotW
      const progress = progressRef.current
      const wipeX = RIDGE_L + plotW * progress
      const plotBottom = topPad + n * rowH

      // Month (or day) gridlines + tick labels.
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.lineWidth = 1
      const ticks: { x: number; label: string }[] = []
      if (month != null) {
        for (let d = d0; d <= d1; d++) {
          const dom = d - d0 + 1
          if (dom === 1 || dom % 5 === 0) {
            const x = Math.round(xForDoy(d)) + 0.5
            ctx.beginPath()
            ctx.moveTo(x, topPad)
            ctx.lineTo(x, plotBottom)
            ctx.stroke()
            ticks.push({ x, label: String(dom) })
          }
        }
      } else {
        for (const m of MONTHS) {
          const x = Math.round(xForDoy(m.doy)) + 0.5
          ctx.beginPath()
          ctx.moveTo(x, topPad)
          ctx.lineTo(x, plotBottom)
          ctx.stroke()
          ticks.push({ x, label: m.label })
        }
      }

      // Ridgeline curves, drawn bottom-to-top so earlier (upper) rows overlay.
      ctx.save()
      ctx.beginPath()
      ctx.rect(RIDGE_L - 1, topPad - rowH, wipeX - RIDGE_L + 1, ridgeH)
      ctx.clip()
      for (let i = n - 1; i >= 0; i--) {
        const r = phenology[i]
        const isHov = ridgeHover?.ridge.genus === r.genus
        const base = topPad + (i + 1) * rowH
        // Each genus fills its own row (normalized to its own peak) so its
        // fruiting *timing* reads clearly regardless of how common it is.
        const amp = rowH * RIDGE_OVERLAP
        const norm = r.peakVal || 1
        ctx.beginPath()
        ctx.moveTo(xForDoy(d0), base)
        for (let d = d0; d <= d1; d++) {
          ctx.lineTo(xForDoy(d), base - amp * (r.counts[d] / norm))
        }
        ctx.lineTo(xForDoy(d1), base)
        ctx.closePath()
        ctx.globalAlpha = isHov ? 0.95 : 0.82
        ctx.fillStyle = r.color
        ctx.fill()
        ctx.globalAlpha = 1
        ctx.beginPath()
        for (let d = d0; d <= d1; d++) {
          const x = xForDoy(d)
          const y = base - amp * (r.counts[d] / norm)
          if (d === d0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.strokeStyle = isHov ? '#ffffff' : 'rgba(20,22,28,0.7)'
        ctx.lineWidth = isHov ? 1.4 : 1
        ctx.stroke()
      }
      ctx.restore()

      // Per-row labels (left gutter) and peak-month callouts.
      ctx.textBaseline = 'middle'
      for (let i = 0; i < n; i++) {
        const r = phenology[i]
        const base = topPad + (i + 1) * rowH
        const amp = rowH * RIDGE_OVERLAP
        // baseline hairline
        ctx.strokeStyle = 'rgba(255,255,255,0.05)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(RIDGE_L, base + 0.5)
        ctx.lineTo(RIDGE_L + plotW, base + 0.5)
        ctx.stroke()
        // genus name + count, right-aligned in the gutter
        ctx.textAlign = 'right'
        ctx.fillStyle = r.color
        ctx.font = '700 12px system-ui, sans-serif'
        ctx.fillText(r.genus, RIDGE_L - 10, base - 8)
        ctx.fillStyle = '#7c828f'
        ctx.font = '10px system-ui, sans-serif'
        ctx.fillText(`${r.total} · ${r.window}`, RIDGE_L - 10, base + 6)
        // peak marker + month label at the crest
        if (progress > 0.55) {
          const px = xForDoy(r.peakDoy)
          if (px >= RIDGE_L && px <= RIDGE_L + plotW) {
            const py = base - amp * 1.0
            ctx.fillStyle = r.color
            ctx.beginPath()
            ctx.arc(px, py, 2.2, 0, Math.PI * 2)
            ctx.fill()
            ctx.font = '600 10px system-ui, sans-serif'
            const lbl = monthNameOfDoy(r.peakDoy)
            const w = ctx.measureText(lbl).width
            const rightSide = px > RIDGE_L + plotW - w - 8
            ctx.textAlign = rightSide ? 'right' : 'left'
            ctx.fillStyle = '#0c0e12'
            ctx.globalAlpha = 0.5
            const lx = rightSide ? px - w - 5 : px + 4
            ctx.fillRect(lx - 1, py - 12, w + 2, 11)
            ctx.globalAlpha = 1
            ctx.fillStyle = '#d7dae1'
            ctx.fillText(lbl, rightSide ? px - 4 : px + 4, py - 6)
          }
        }
      }

      // Axis tick labels.
      ctx.fillStyle = '#8b91a0'
      ctx.font = '11px system-ui, sans-serif'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'alphabetic'
      for (const t of ticks) ctx.fillText(t.label, t.x + 3, plotBottom + 16)

      // Header.
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'
      ctx.fillStyle = '#e7e9ee'
      ctx.font = '700 12px system-ui, sans-serif'
      ctx.fillText('WHEN EACH GENUS FRUITS', RIDGE_L, MARGIN_TOP - 2)
      ctx.fillStyle = '#7c828f'
      ctx.font = '11px system-ui, sans-serif'
      ctx.fillText('peak fruiting by day of year — earliest at top', RIDGE_L + 190, MARGIN_TOP - 1)
      return
    }

    if (!layout || !weather) return
    const { rowH, ribH, totalH, dotR, single, d0, d1, plotW, xForDoy, ribbonTop } = layout

    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(size.w * dpr) || canvas.height !== Math.round(totalH * dpr)) {
      canvas.width = Math.round(size.w * dpr)
      canvas.height = Math.round(totalH * dpr)
      canvas.style.width = `${size.w}px`
      canvas.style.height = `${totalH}px`
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.w, totalH)

    const progress = progressRef.current
    const stripeW = plotW / (d1 - d0) + 0.8
    const wipeX = MARGIN_L + plotW * progress

    // 1) Weather stripes (background) — per-lane aggregated anomaly
    lanes.forEach((_, i) => {
      const top = MARGIN_TOP + i * rowH
      const anom = laneSeries[i]?.anom
      if (!anom) return
      for (let d = d0; d <= d1; d++) {
        const [r, g, b, a] = anomRGBA(anom[d - 1] ?? null)
        ctx.globalAlpha = a
        ctx.fillStyle = `rgb(${r},${g},${b})`
        ctx.fillRect(xForDoy(d), top, stripeW, rowH)
      }
    })
    ctx.globalAlpha = 1

    // 2) Gridlines + tick labels
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    ctx.lineWidth = 1
    const plotBottom = MARGIN_TOP + lanes.length * rowH
    const ticks: { x: number; label: string }[] = []
    if (month != null) {
      for (let d = d0; d <= d1; d++) {
        const dom = d - d0 + 1
        if (dom === 1 || dom % 5 === 0) {
          const x = Math.round(xForDoy(d)) + 0.5
          ctx.beginPath()
          ctx.moveTo(x, MARGIN_TOP)
          ctx.lineTo(x, plotBottom)
          ctx.stroke()
          ticks.push({ x, label: String(dom) })
        }
      }
    } else {
      for (const m of MONTHS) {
        const x = Math.round(xForDoy(m.doy)) + 0.5
        ctx.beginPath()
        ctx.moveTo(x, MARGIN_TOP)
        ctx.lineTo(x, plotBottom)
        ctx.stroke()
        ticks.push({ x, label: m.label })
      }
    }

    // 3) Dots
    const dotAlpha = combined ? 0.7 : 0.92
    for (let i = 0; i < layout.dotsByRow.length; i++) {
      for (const dot of layout.dotsByRow[i]) {
        const fade = Math.max(0, Math.min(1, (progress - dot.appear) / 0.06))
        if (fade <= 0) continue
        const match = !filterLc || (dot.photo.species ?? '').toLowerCase().includes(filterLc)
        const scale = 0.4 + 0.6 * fade
        ctx.globalAlpha = fade * (match ? dotAlpha : 0.05)
        ctx.fillStyle = dot.photo.color ?? NULL_DOT
        ctx.beginPath()
        ctx.arc(dot.cx, dot.cy, (dot.photo.color ? dotR : dotR - 0.7) * scale, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    ctx.globalAlpha = 1

    // 4) Context ribbons (rain / temp / genus), wiped in with the entrance anim
    if (ribbonData.mode !== 'none' && ribH > 0) {
      lanes.forEach((_, i) => {
        const rTop = ribbonTop(i)
        const ls = laneSeries[i]
        // hairline separator above the ribbon
        ctx.strokeStyle = 'rgba(255,255,255,0.07)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(MARGIN_L, rTop - 1.5)
        ctx.lineTo(MARGIN_L + plotW, rTop - 1.5)
        ctx.stroke()

        ctx.save()
        ctx.beginPath()
        ctx.rect(MARGIN_L - 1, rTop - 2, wipeX - MARGIN_L + 1, ribH + 4)
        ctx.clip()

        if (ribbonData.mode === 'rain' && ls) {
          const { vmax } = ribbonData
          ctx.beginPath()
          ctx.moveTo(xForDoy(d0), rTop + ribH)
          for (let d = d0; d <= d1; d++) {
            const v = Math.min(ls.rain7[d - 1] ?? 0, vmax)
            ctx.lineTo(xForDoy(d), rTop + ribH * (1 - v / vmax))
          }
          ctx.lineTo(xForDoy(d1), rTop + ribH)
          ctx.closePath()
          ctx.fillStyle = 'rgba(88,140,200,0.42)'
          ctx.fill()
          ctx.beginPath()
          for (let d = d0; d <= d1; d++) {
            const v = Math.min(ls.rain7[d - 1] ?? 0, vmax)
            const x = xForDoy(d)
            const y = rTop + ribH * (1 - v / vmax)
            if (d === d0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.strokeStyle = '#7aa8d8'
          ctx.lineWidth = 1
          ctx.stroke()
        } else if (ribbonData.mode === 'temp' && ls) {
          const { vmin, vmax } = ribbonData
          const rng = Math.max(1, vmax - vmin)
          ctx.beginPath()
          let pen = false
          for (let d = d0; d <= d1; d++) {
            const c = ls.tmean[d - 1]
            if (c == null) {
              pen = false
              continue
            }
            const f = c * 1.8 + 32
            const x = xForDoy(d)
            const y = rTop + ribH * (1 - (f - vmin) / rng)
            if (!pen) {
              ctx.moveTo(x, y)
              pen = true
            } else ctx.lineTo(x, y)
          }
          ctx.strokeStyle = '#d8956a'
          ctx.lineWidth = 1.3
          ctx.stroke()
        } else if (ribbonData.mode === 'genus') {
          const arr = ribbonData.perLane[i]
          if (arr) {
            const { vmax, bands } = ribbonData
            const cum = new Array(367).fill(0)
            for (let bi = 0; bi < bands.length; bi++) {
              const b = arr[bi]
              ctx.beginPath()
              for (let d = d0; d <= d1; d++) {
                const x = xForDoy(d)
                const y = rTop + ribH * (1 - Math.min(cum[d], vmax) / vmax)
                if (d === d0) ctx.moveTo(x, y)
                else ctx.lineTo(x, y)
              }
              for (let d = d1; d >= d0; d--) {
                const x = xForDoy(d)
                const y = rTop + ribH * (1 - Math.min(cum[d] + b[d], vmax) / vmax)
                ctx.lineTo(x, y)
              }
              ctx.closePath()
              ctx.globalAlpha = 0.78
              ctx.fillStyle = bands[bi].color
              ctx.fill()
              for (let d = d0; d <= d1; d++) cum[d] += b[d]
            }
            ctx.globalAlpha = 1
          }
        }
        ctx.restore()
      })
    }

    // 5) Hovered highlight ring
    if (hovered) {
      const i = yearToLane.get(hovered.photo.year)
      if (i != null) {
        const dot = layout.dotsByRow[i].find((d) => d.photo.id === hovered.photo.id)
        if (dot) {
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth = 1.4
          ctx.beginPath()
          ctx.arc(dot.cx, dot.cy, dotR + 2.6, 0, Math.PI * 2)
          ctx.stroke()
        }
      }
    }

    // 6) Axis text: lane labels + per-lane photo counts + tick labels
    if (single && combined) {
      // Rotated label for the combined lane (too wide for the left margin).
      ctx.save()
      ctx.translate(16, MARGIN_TOP + rowH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillStyle = '#e7e9ee'
      ctx.font = '700 12px system-ui, sans-serif'
      ctx.fillText(lanes[0].label, 0, 0)
      ctx.restore()
    } else {
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      lanes.forEach((lane, i) => {
        const yMid = MARGIN_TOP + i * rowH + rowH / 2
        ctx.fillStyle = '#e7e9ee'
        ctx.font = '700 13px system-ui, sans-serif'
        ctx.fillText(lane.label, MARGIN_L - 9, yMid - 7)
        ctx.fillStyle = '#7c828f'
        ctx.font = '10px system-ui, sans-serif'
        ctx.fillText(`${layout.dotsByRow[i].length}`, MARGIN_L - 9, yMid + 8)
      })
    }
    ctx.fillStyle = '#8b91a0'
    ctx.font = '11px system-ui, sans-serif'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    for (const t of ticks) {
      ctx.fillText(t.label, t.x + 3, plotBottom + 16)
    }

    // 7) Direct labels (replace legend chips)
    // Anomaly key, top-right of the first row.
    ctx.textBaseline = 'alphabetic'
    ctx.font = '600 11px system-ui, sans-serif'
    ctx.textAlign = 'right'
    ctx.fillStyle = `rgb(${DRY.join(',')})`
    ctx.fillText('drier', MARGIN_L + plotW - 6, MARGIN_TOP + 14)
    const wetW = ctx.measureText('drier').width + 12
    ctx.fillStyle = `rgb(${WET.join(',')})`
    ctx.fillText('wetter', MARGIN_L + plotW - 6 - wetW, MARGIN_TOP + 14)

    if (ribbonData.mode !== 'none' && ribH > 0 && progress > 0.4) {
      const rTop0 = ribbonTop(0)
      // Series caption on the first ribbon.
      const caption =
        ribbonData.mode === 'rain'
          ? '7-DAY RAINFALL'
          : ribbonData.mode === 'temp'
            ? 'AVG TEMPERATURE °F'
            : ribbonData.filtered
              ? 'MATCHING PHOTOS'
              : 'PHOTOS BY GENUS'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'
      ctx.font = '600 9px system-ui, sans-serif'
      ctx.fillStyle = 'rgba(178,184,196,0.9)'
      ctx.fillText(caption.toUpperCase(), MARGIN_L + 4, rTop0 + 3)

      if (ribbonData.mode === 'rain') {
        // Label the single wettest 7-day peak across displayed rows.
        let bestV = 0
        let bestX = 0
        let bestY = 0
        lanes.forEach((_, i) => {
          const ls = laneSeries[i]
          if (!ls) return
          const rTop = ribbonTop(i)
          for (let d = d0; d <= d1; d++) {
            const v = ls.rain7[d - 1]
            if (v != null && v > bestV) {
              bestV = v
              bestX = xForDoy(d)
              bestY = rTop + ribH * (1 - Math.min(v, ribbonData.vmax) / ribbonData.vmax)
            }
          }
        })
        if (bestV > 0) {
          ctx.font = '600 10px system-ui, sans-serif'
          ctx.fillStyle = '#a9c8e8'
          ctx.textAlign = bestX > MARGIN_L + plotW - 70 ? 'right' : 'left'
          ctx.textBaseline = 'bottom'
          ctx.fillText(`${Math.round(bestV)} mm / 7d`, bestX + (ctx.textAlign === 'right' ? -3 : 3), bestY - 2)
        }
      } else if (ribbonData.mode === 'temp') {
        // Min/max °F markers on the first row.
        const rTop = ribbonTop(0)
        ctx.font = '600 10px system-ui, sans-serif'
        ctx.fillStyle = '#e0ac86'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'top'
        ctx.fillText(`${Math.round(ribbonData.vmax)}°`, MARGIN_L + 4, rTop + 14)
        ctx.textBaseline = 'bottom'
        ctx.fillText(`${Math.round(ribbonData.vmin)}°`, MARGIN_L + 4, rTop + ribH - 2)
      } else if (ribbonData.mode === 'genus' && !ribbonData.filtered) {
        // Label each band once, at its thickest point across all rows.
        const { bands, perLane, vmax } = ribbonData
        const occupied: Array<Array<[number, number]>> = lanes.map(() => [])
        ctx.font = '600 9px system-ui, sans-serif'
        ctx.textBaseline = 'middle'
        for (let bi = 0; bi < bands.length; bi++) {
          if (bands[bi].genus === 'other') continue
          let bestPx = 0
          let bestI = -1
          let bestD = 0
          lanes.forEach((_, i) => {
            const a = perLane[i]
            if (!a) return
            for (let d = d0; d <= d1; d++) {
              const px = (ribH * a[bi][d]) / vmax
              if (px > bestPx) {
                bestPx = px
                bestI = i
                bestD = d
              }
            }
          })
          if (bestPx < 9 || bestI < 0) continue
          const a = perLane[bestI]
          let below = 0
          for (let k = 0; k < bi; k++) below += a[k][bestD]
          const midCount = below + a[bi][bestD] / 2
          const rTop = ribbonTop(bestI)
          const x = clamp(MARGIN_L + 4, xForDoy(bestD), MARGIN_L + plotW - 60)
          const y = rTop + ribH * (1 - Math.min(midCount, vmax) / vmax)
          const w = ctx.measureText(bands[bi].genus).width + 6
          const overlaps = occupied[bestI].some(([x0, x1]) => x < x1 + 4 && x + w > x0 - 4)
          if (overlaps) continue
          occupied[bestI].push([x, x + w])
          ctx.fillStyle = '#0c0e12'
          ctx.globalAlpha = 0.55
          ctx.fillRect(x - 2, y - 6, w, 12)
          ctx.globalAlpha = 1
          ctx.textAlign = 'left'
          ctx.fillStyle = bands[bi].color
          ctx.fillText(bands[bi].genus, x, y)
        }
      }
    }
  }, [layout, weather, lanes, laneSeries, yearToLane, size, month, filterLc, hovered, ribbonData, ribbon, combined, phenology, ridgeHover, domain, photos])

  // keep a live ref so the animation loop always calls the latest draw
  const drawRef = useRef(draw)
  drawRef.current = draw

  // Entrance animation: runs on first load and whenever the view changes
  // (year zoom / combine / month filter / ribbon series) — discrete user
  // actions, never per-frame, so it always terminates.
  useEffect(() => {
    if (!photos || !weather) return
    let raf = 0
    progressRef.current = 0
    const start = performance.now()
    const dur = 1000
    const tick = (now: number) => {
      progressRef.current = Math.min(1, (now - start) / dur)
      drawRef.current()
      if (progressRef.current < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [photos, weather, focusYear, combined, month, ribbon])

  // redraw on size/filter/hover changes (after animation has settled)
  useEffect(() => {
    draw()
  }, [draw])

  // ------- hit testing -------
  const handleMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top

    // Ridgeline mode: hit-test which genus row the cursor is over.
    if (ribbon === 'genus' && ridgeGeomRef.current) {
      const { topPad, rowH, ridges } = ridgeGeomRef.current
      const i = Math.floor((py - topPad) / rowH)
      if (i >= 0 && i < ridges.length && px >= RIDGE_L) {
        setRidgeHover({ ridge: ridges[i], x: e.clientX, y: e.clientY })
      } else if (ridgeHover) {
        setRidgeHover(null)
      }
      return
    }

    if (!layout) return
    const i = Math.floor((py - MARGIN_TOP) / layout.rowH)
    if (i < 0 || i >= layout.dotsByRow.length) {
      if (hovered) setHovered(null)
      return
    }
    // Only the dot band is interactive; the ribbon below it is not.
    const bandTop = layout.bandTop(i)
    if (py < bandTop - 3 || py > bandTop + layout.bandH + 3) {
      if (hovered) setHovered(null)
      return
    }
    let best: Dot | null = null
    const reach = layout.dotR + 3
    let bestD = reach * reach // px^2 threshold, scales with dot size
    for (const dot of layout.dotsByRow[i]) {
      const dx = dot.cx - px
      const dy = dot.cy - py
      const d = dx * dx + dy * dy
      if (d < bestD) {
        bestD = d
        best = dot
      }
    }
    if (best) setHovered({ photo: best.photo, x: e.clientX, y: e.clientY })
    else if (hovered) setHovered(null)
  }

  const handleClick = () => {
    const url = hovered?.photo.page_url
    if (url) window.open(url, '_blank', 'noopener')
  }

  const buildProxySrc = useCallback(
    (photo: SeasonPhoto, maxWidth?: number) => {
      const base = proxyBase.trim().replace(/\/$/, '')
      const url = (photo.photo_url ?? '').trim()
      if (!url) return ''
      if (!base) return url
      const params = new URLSearchParams({ url })
      if (photo.page_url) params.set('ref', photo.page_url)
      if (maxWidth) params.set('w', String(maxWidth))
      return `${base}/proxy?${params.toString()}`
    },
    [proxyBase],
  )

  const proxyImgSrc = useMemo(
    () => (hovered ? buildProxySrc(hovered.photo, 480) : ''),
    [hovered, buildProxySrc],
  )

  const shownCount = useMemo(
    () => layout?.dotsByRow.reduce((n, row) => n + row.length, 0) ?? 0,
    [layout],
  )

  const ribbonOptions: { key: RibbonMode; label: string }[] = [
    { key: 'rain', label: 'Rain' },
    { key: 'temp', label: 'Temp' },
    { key: 'genus', label: 'Genera' },
    { key: 'none', label: 'Off' },
  ]

  // Year control value: '' = stacked, 'combined' = one squished lane, or a year.
  const yearValue = combined ? 'combined' : focusYear != null ? String(focusYear) : ''
  const onYearChange = (v: string) => {
    if (v === 'combined') {
      setCombined(true)
      setFocusYear(null)
    } else if (v === '') {
      setCombined(false)
      setFocusYear(null)
    } else {
      setCombined(false)
      setFocusYear(Number(v))
    }
  }

  return (
    <div className="page seasons">
      <div className="page-header seasons-header">
        <div className="seasons-masthead">
          <h1 className="seasons-title">Mushrooms follow the rain</h1>
          <p className="seasons-intro">
            Every dot is a single photograph, placed by the day it was taken and painted in the
            mushroom&rsquo;s own colors. The background runs blue in wetter-than-normal stretches and
            amber in drier ones, across the 2018&ndash;2024 seasons near Houston.
          </p>
        </div>

        <div className="seasons-controls">
          <div className="seg" role="group" aria-label="Context ribbon">
            {ribbonOptions.map((o) => (
              <button
                key={o.key}
                type="button"
                className="seg-btn"
                aria-pressed={ribbon === o.key}
                onClick={() => setRibbon(o.key)}
              >
                {o.label}
              </button>
            ))}
          </div>

          <label className="ctl">
            <span>Years</span>
            <select className="select" value={yearValue} onChange={(e) => onYearChange(e.target.value)}>
              <option value="">Stacked by year</option>
              <option value="combined">All years combined</option>
              {allYears.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </label>

          <label className="ctl">
            <span>Month</span>
            <select
              className="select"
              value={month ?? ''}
              onChange={(e) => setMonth(e.target.value ? Number(e.target.value) : null)}
            >
              <option value="">All year</option>
              {MONTHS.map((m, i) => (
                <option key={m.label} value={i + 1}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          <label className="ctl">
            <span>Species</span>
            <input
              className="input"
              style={{ width: 150 }}
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="e.g. Amanita"
            />
          </label>

          <div className="seasons-controls-right">
            <span className="season-count">
              {photos ? `${shownCount.toLocaleString()} photos` : 'Run the export script.'}
            </span>
            <input
              className="input proxy-input"
              value={proxyBase}
              onChange={(e) => setProxyBase(e.target.value)}
              placeholder="proxy"
              title="Image proxy base URL (run web_proxy.py for hover previews)"
            />
          </div>
        </div>
      </div>

      <div className="page-body seasons-body" ref={containerRef}>
        {photos && weather ? (
          <canvas
            ref={canvasRef}
            className="seasons-canvas"
            onMouseMove={handleMove}
            onMouseLeave={() => {
              setHovered(null)
              setRidgeHover(null)
            }}
            onClick={handleClick}
            style={{ cursor: hovered ? 'pointer' : 'default' }}
          />
        ) : (
          <div className="seasons-empty">{photos === null ? 'Loading…' : 'No data — run the export script.'}</div>
        )}
      </div>

      {hovered ? (
        <div
          className="season-tooltip"
          style={{
            left: Math.min(hovered.x + 16, window.innerWidth - 256),
            top: Math.min(hovered.y + 16, window.innerHeight - 260),
          }}
        >
          {proxyImgSrc ? (
            <img className="season-tooltip-img" src={proxyImgSrc} alt={hovered.photo.species ?? 'mushroom'} />
          ) : null}
          <div
            className="swatch-chip"
            style={{
              background:
                hovered.photo.swatches.length >= 2
                  ? `linear-gradient(90deg, ${hovered.photo.swatches.join(', ')})`
                  : hovered.photo.color ?? NULL_DOT,
            }}
          />
          <div className="season-tooltip-meta">
            <div className="season-tooltip-species">{hovered.photo.species ?? 'Unidentified'}</div>
            {hovered.photo.genus ? (
              <div className="season-tooltip-genus">{hovered.photo.genus}</div>
            ) : null}
            <div className="season-tooltip-date">{fmtDate(hovered.photo.date)}</div>
            <div className="season-tooltip-hint">click to open source page</div>
          </div>
        </div>
      ) : null}

      {ridgeHover ? (
        <div
          className="season-tooltip ridge-tooltip"
          style={{
            left: Math.min(ridgeHover.x + 16, window.innerWidth - 288),
            top: Math.min(ridgeHover.y + 16, window.innerHeight - 240),
          }}
        >
          <div className="ridge-tooltip-head">
            <div className="swatch-chip" style={{ background: ridgeHover.ridge.color }} />
            <div className="season-tooltip-meta">
              <div className="season-tooltip-species">{ridgeHover.ridge.genus}</div>
              <div className="season-tooltip-genus">
                peaks in {monthNameOfDoy(ridgeHover.ridge.peakDoy)} · mostly{' '}
                {ridgeHover.ridge.window}
              </div>
              <div className="season-tooltip-date">
                {ridgeHover.ridge.total.toLocaleString()} photos
              </div>
            </div>
          </div>
          {ridgeHover.ridge.samples.length ? (
            <div className="ridge-tooltip-grid">
              {ridgeHover.ridge.samples.map((s) => {
                const src = buildProxySrc(s, 200)
                return src ? (
                  <img
                    key={s.id}
                    className="ridge-tooltip-thumb"
                    src={src}
                    alt={s.species ?? ridgeHover.ridge.genus}
                    title={s.species ?? undefined}
                  />
                ) : null
              })}
            </div>
          ) : (
            <div className="season-tooltip-hint">
              run web_proxy.py for photo previews
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}
