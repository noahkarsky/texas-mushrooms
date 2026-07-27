import { useEffect, useMemo, useState } from 'react'
import { GeoJSON, MapContainer, TileLayer, useMap } from 'react-leaflet'
import * as L from 'leaflet'
import { scaleSequential } from 'd3-scale'
import { interpolateYlGnBu, interpolateYlOrRd, interpolatePuRd } from 'd3-scale-chromatic'

type Metric = 'total_count' | 'elevation'
type Source = 'texasmushrooms' | 'inaturalist' | 'both'

type FeatureCollection = GeoJSON.FeatureCollection<GeoJSON.Geometry, Record<string, unknown>>

// Distinct outline colors so overlapping layers stay legible in "both" mode.
const OUTLINE = {
  texasmushrooms: '#444',
  inaturalist: '#1b5e20',
} as const

function FitBounds({ collections }: { collections: (FeatureCollection | null)[] }) {
  const map = useMap()

  useEffect(() => {
    const bounds = L.latLngBounds([])
    for (const c of collections) {
      if (!c) continue
      const layer = L.geoJSON(c as any)
      const b = layer.getBounds()
      if (b.isValid()) bounds.extend(b)
    }
    if (bounds.isValid()) map.fitBounds(bounds.pad(0.05))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collections, map])

  return null
}

function useColorScale(geojson: FeatureCollection | null, metric: Metric, kind: 'count' | 'elevation') {
  return useMemo(() => {
    const values: number[] = []
    if (geojson) {
      for (const f of geojson.features) {
        const v = Number((f.properties as any)?.[metric])
        if (Number.isFinite(v)) values.push(v)
      }
    }
    const min = values.length ? Math.min(...values) : 0
    const max = values.length ? Math.max(...values) : 1
    const interp =
      metric === 'elevation' ? interpolateYlGnBu : kind === 'count' ? interpolateYlOrRd : interpolatePuRd
    return { min, max, scale: scaleSequential(interp).domain([min, max]) }
  }, [geojson, metric, kind])
}

function SourceLayer({
  geojson,
  metric,
  source,
}: {
  geojson: FeatureCollection | null
  metric: Metric
  source: 'texasmushrooms' | 'inaturalist'
}) {
  // iNaturalist cells have no elevation, so always color them by count.
  const kind = source === 'inaturalist' ? 'count' : metric === 'elevation' ? 'elevation' : 'count'
  const colorScale = useColorScale(geojson, source === 'inaturalist' ? 'total_count' : metric, kind)

  const styleFn = useMemo(() => {
    const metricKey = source === 'inaturalist' ? 'total_count' : metric
    return (feature: any) => {
      const v = Number(feature?.properties?.[metricKey])
      const fillColor = Number.isFinite(v) ? colorScale.scale(v) : '#cccccc'
      return {
        color: OUTLINE[source],
        weight: source === 'inaturalist' ? 0.6 : 0.25,
        opacity: 0.8,
        fillColor,
        fillOpacity: 0.6,
      } as L.PathOptions
    }
  }, [colorScale, metric, source])

  if (!geojson) return null

  return (
    <GeoJSON
      key={`${source}-${metric}`}
      data={geojson as any}
      style={styleFn as any}
      onEachFeature={(feature, layer) => {
        const p: any = feature.properties || {}
        const h3 = String(p.h3_index ?? '')
        const total = p.total_count
        const elev = p.elevation
        const src = String(p.source ?? source)
        layer.bindTooltip(
          `<div style="font-size:12px"><div><strong>Source</strong> ${src}</div><div><strong>H3</strong> ${h3}</div><div><strong>Total</strong> ${total ?? ''}</div>${
            elev != null ? `<div><strong>Elevation</strong> ${elev}</div>` : ''
          }</div>`,
          { sticky: true },
        )
      }}
    />
  )
}

export default function MapPage() {
  const [txGeojson, setTxGeojson] = useState<FeatureCollection | null>(null)
  const [inatGeojson, setInatGeojson] = useState<FeatureCollection | null>(null)
  const [metric, setMetric] = useState<Metric>('total_count')
  const [source, setSource] = useState<Source>('both')

  useEffect(() => {
    let cancelled = false
    const load = (url: string, setter: (fc: FeatureCollection | null) => void) => {
      fetch(url)
        .then((r) => (r.ok ? r.json() : null))
        .then((j) => {
          if (!cancelled) setter(j)
        })
        .catch(() => {
          if (!cancelled) setter(null)
        })
    }
    load('/data/h3_cells.geojson', setTxGeojson)
    load('/data/h3_cells_inat.geojson', setInatGeojson)
    return () => {
      cancelled = true
    }
  }, [])

  const showTx = source === 'texasmushrooms' || source === 'both'
  const showInat = source === 'inaturalist' || source === 'both'

  const activeCollections = useMemo(
    () => [showTx ? txGeojson : null, showInat ? inatGeojson : null],
    [showTx, showInat, txGeojson, inatGeojson],
  )

  const anyData = (showTx && txGeojson) || (showInat && inatGeojson)

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
          <strong style={{ marginLeft: 12 }}>Metric</strong>
          <select className="select" value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
            <option value="total_count">Total photos</option>
            <option value="elevation">Elevation (m)</option>
          </select>
        </div>
        <div style={{ fontSize: 12, color: '#444' }}>
          {anyData ? (
            <span>
              {showTx && txGeojson ? `texasmushrooms: ${txGeojson.features.length} cells` : ''}
              {showTx && txGeojson && showInat && inatGeojson ? ' · ' : ''}
              {showInat && inatGeojson ? `iNaturalist: ${inatGeojson.features.length} cells` : ''}
            </span>
          ) : (
            <span>Export `web/public/data/h3_cells*.geojson` to view the map.</span>
          )}
        </div>
      </div>

      <div className="page-body">
        <div className="map-wrap">
          <MapContainer center={[31.9686, -99.9018]} zoom={6} style={{ height: '100%', width: '100%' }}>
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <FitBounds collections={activeCollections} />

            {showTx ? <SourceLayer geojson={txGeojson} metric={metric} source="texasmushrooms" /> : null}
            {showInat ? <SourceLayer geojson={inatGeojson} metric={metric} source="inaturalist" /> : null}
          </MapContainer>
        </div>
      </div>
    </div>
  )
}
