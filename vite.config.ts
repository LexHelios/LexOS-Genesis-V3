import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.warn('Proxy error - backend not available:', err.message)
            // Send a mock response for development
            if (!res.headersSent) {
              res.writeHead(503, { 'Content-Type': 'application/json' })
              res.end(JSON.stringify({ 
                error: 'Backend not available', 
                message: 'Using mock data for development' 
              }))
            }
          })
        }
      },
      '/ws': {
        target: 'ws://localhost:8081',
        ws: true,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.warn('WebSocket proxy error - backend not available:', err.message)
          })
        }
      },
    },
  },
})