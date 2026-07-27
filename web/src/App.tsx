import { NavLink, Route, Routes } from 'react-router-dom'
import MapPage from './pages/Map'
import PhotosPage from './pages/Photos'
import SeasonsPage from './pages/Seasons'

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <NavLink to="/" end>
          Map
        </NavLink>
        <NavLink to="/seasons">Seasons</NavLink>
        <NavLink to="/photos">Photos</NavLink>
      </nav>

      <Routes>
        <Route path="/" element={<MapPage />} />
        <Route path="/seasons" element={<SeasonsPage />} />
        <Route path="/photos" element={<PhotosPage />} />
      </Routes>
    </div>
  )
}
