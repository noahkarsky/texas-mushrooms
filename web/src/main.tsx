import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

// Leaflet's CSS is imported by `pages/Map.tsx`, not here: `/` is now the landing
// page, and importing it at the entry point would put Leaflet's stylesheet in the
// initial chunk for every visitor, including those who never open the map.
import './styles.css'

import App from './App'

// HashRouter, not BrowserRouter: the site is published to GitHub Pages, which
// serves static files with no rewrite rule, so a deep link to or a refresh on
// /map or /seasons would 404. Hash routes are resolved entirely in the browser.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>,
)
