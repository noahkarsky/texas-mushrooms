import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Helps when serving as static files (e.g., GitHub Pages)
  base: './',
})
