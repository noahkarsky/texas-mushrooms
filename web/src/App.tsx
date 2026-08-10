import { Suspense, lazy } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import HomePage from './pages/Home'

// `/` is the entry route and needs neither Leaflet nor the Seasons canvas, so
// both are split out. Home is imported eagerly — it is what first paint renders.
const MapPage = lazy(() => import('./pages/Map'))
const SeasonsPage = lazy(() => import('./pages/Seasons'))

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <NavLink to="/" end className="nav-brand">
          Texas Mushrooms
        </NavLink>
        <NavLink to="/map">Map</NavLink>
        <NavLink to="/seasons">Seasons</NavLink>
      </nav>

      <Suspense fallback={<div className="route-loading">Loading&hellip;</div>}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/seasons" element={<SeasonsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </div>
  )
}
