import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Mock API plugin to handle requests when backend is not available
const mockApiPlugin = () => {
  return {
    name: 'mock-api',
    configureServer(server: any) {
      server.middlewares.use('/api', (req: any, res: any, next: any) => {
        // Only handle if no backend URL is configured
        if (!process.env.VITE_LEXOS_API_URL) {
          console.log(`[Mock API] Handling ${req.method} ${req.url}`)
          
          const mockResponse = getMockResponse(req.url)
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Access-Control-Allow-Origin', '*')
          res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
          res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization')
          
          if (req.method === 'OPTIONS') {
            res.statusCode = 200
            res.end()
            return
          }
          
          res.statusCode = 200
          res.end(JSON.stringify(mockResponse))
        } else {
          next()
        }
      })
    }
  }
}

function getMockResponse(url: string) {
  console.log(`[Mock API] Generating mock response for: ${url}`)
  
  if (url?.includes('/api/models/active')) {
    return [
      { name: 'llama3.2:3b', status: 'loaded', size: '2.0GB' },
      { name: 'qwen2.5-coder:32b', status: 'available', size: '19GB' },
      { name: 'llava:7b', status: 'available', size: '4.7GB' }
    ]
  }
  
  if (url?.includes('/api/models')) {
    return [
      {
        name: 'llama3.2:3b',
        size_gb: 2.0,
        capabilities: ['balanced_tasks', 'general_qa', 'moderate_reasoning'],
        best_for: ['general_chat', 'moderate_complexity'],
        speed: 'fast',
        context_window: 4096,
        status: 'active'
      },
      {
        name: 'qwen2.5-coder:32b',
        size_gb: 19,
        capabilities: ['coding', 'debugging', 'code_review'],
        best_for: ['code_generation', 'bug_fixing'],
        speed: 'slow',
        context_window: 32768,
        status: 'inactive'
      }
    ]
  }
  
  if (url?.includes('/api/system/status') || url?.includes('/api/stats')) {
    return {
      status: 'development',
      uptime: Date.now() - (Date.now() % 1000000),
      cpu_percent: Math.random() * 50 + 25,
      memory_percent: Math.random() * 50 + 25,
      memory_available_gb: 16 + Math.random() * 8,
      active_models: ['llama3.2:3b'],
      gpus: [{
        id: 0,
        name: 'Development GPU',
        memory_used: Math.random() * 8000 + 2000,
        memory_total: 24000,
        memory_percent: Math.random() * 50 + 25,
        temperature: Math.random() * 20 + 60,
        load: Math.random() * 0.5 + 0.2
      }],
      last_updated: new Date().toISOString()
    }
  }
  
  if (url?.includes('/api/memory/stats')) {
    return {
      total_memories: Math.floor(Math.random() * 1000 + 500),
      working_memory_size: Math.floor(Math.random() * 50 + 10),
      episodic_memory_size: Math.floor(Math.random() * 200 + 100),
      conversation_history_size: Math.floor(Math.random() * 100 + 20),
      semantic_categories: ['coding', 'general', 'reasoning'],
      vector_store_size: Math.floor(Math.random() * 1000 + 500),
      graph_nodes: Math.floor(Math.random() * 800 + 400),
      graph_edges: Math.floor(Math.random() * 1500 + 800)
    }
  }
  
  if (url?.includes('/api/performance')) {
    return {
      operations: {
        'chat_processing': {
          count: Math.floor(Math.random() * 100 + 50),
          avg_time: Math.random() * 2 + 0.5,
          error_rate: Math.random() * 0.05
        },
        'memory_retrieval': {
          count: Math.floor(Math.random() * 200 + 100),
          avg_time: Math.random() * 0.5 + 0.1,
          error_rate: Math.random() * 0.02
        }
      },
      resource_usage: {
        cpu_avg: Math.random() * 40 + 20,
        memory_avg: Math.random() * 50 + 30,
        gpu_memory_avg: Math.random() * 60 + 20
      }
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
    timestamp: new Date().toISOString(),
    mode: 'development'
  }
}

export default defineConfig({
  plugins: [
    react(),
    mockApiPlugin()
  ],
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    port: 3000,
    // Only configure proxy if backend URLs are explicitly provided
    ...(process.env.VITE_LEXOS_API_URL && {
      proxy: {
        '/api': {
          target: process.env.VITE_LEXOS_API_URL,
          changeOrigin: true,
          timeout: 5000,
          proxyTimeout: 5000,
        },
        ...(process.env.VITE_LEXOS_WS_URL && {
          '/ws': {
            target: process.env.VITE_LEXOS_WS_URL,
            ws: true,
            timeout: 5000,
          }
        })
      }
    })
  },
})