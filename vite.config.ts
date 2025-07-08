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
        timeout: 5000,
        proxyTimeout: 5000,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.warn('Backend API not available - using development mode')
            // Send a mock response for development
            if (!res.headersSent) {
              // Provide mock responses based on the endpoint
              const mockResponse = getMockResponse(req.url)
              res.writeHead(200, { 'Content-Type': 'application/json' })
              res.end(JSON.stringify(mockResponse))
            }
          })
          
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // Set a timeout for proxy requests
            proxyReq.setTimeout(5000, () => {
              proxyReq.destroy()
            })
          })
        }
      },
      '/ws': {
        target: 'ws://localhost:8081',
        ws: true,
        timeout: 5000,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.warn('WebSocket backend not available - using development mode')
          })
        }
      },
    },
  },
})

function getMockResponse(url: string) {
  if (url?.includes('/api/models/active')) {
    return [
      { name: 'llama2:7b', status: 'loaded', size: '3.8GB' },
      { name: 'codellama:13b', status: 'available', size: '7.3GB' },
      { name: 'mistral:7b', status: 'available', size: '4.1GB' }
    ]
  }
  
  if (url?.includes('/api/system/status')) {
    return {
      status: 'development',
      uptime: Date.now() - (Date.now() % 1000000),
      memory_usage: Math.random() * 50 + 25,
      cpu_usage: Math.random() * 30 + 10,
      active_models: ['llama2:7b'],
      last_updated: new Date().toISOString()
    }
  }
  
  if (url?.includes('/api/chat')) {
    return {
      message: 'This is a mock response from the development server. The backend is not running.',
      timestamp: new Date().toISOString(),
      model: 'development-mock'
    }
  }
  
  return {
    message: 'Backend service not available - running in development mode',
    timestamp: new Date().toISOString()
  }
}