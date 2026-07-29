import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

import 'leaflet/dist/leaflet.css'
import './styles.css'

import App from './App'

// HashRouter, not BrowserRouter: the site is published to GitHub Pages, which
// serves static files with no rewrite rule, so a deep link to or a refresh on
// /seasons would 404. Hash routes are resolved entirely in the browser.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>,
)
