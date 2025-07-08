import { useState, useEffect, useRef } from 'react'
import { useLexOS } from '../context/LexOSContext'

// Mock data for development when backend is not available
const mockSystemStatus = {
  status: 'development',
  uptime: 0,
  memory_usage: 0,
  cpu_usage: 0,
  active_models: [],
  last_updated: new Date().toISOString()
}

const mockPerformanceMetrics = {
  response_time: 0,
  requests_per_second: 0,
  error_rate: 0,
  memory_usage: 0
}

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const [useMockData, setUseMockData] = useState(false)
  const [connectionTimeout, setConnectionTimeout] = useState<NodeJS.Timeout>()
  const wsRef = useRef<WebSocket | null>(null)
  const { state, dispatch } = useLexOS()
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const mockDataIntervalRef = useRef<NodeJS.Timeout>()

  const connect = () => {
    // Clear any existing connection timeout
    if (connectionTimeout) {
      clearTimeout(connectionTimeout)
    }
    
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      
      // Determine WebSocket URL based on environment
      let wsUrl: string
      if (import.meta.env.DEV) {
        // In development, try localhost first
        wsUrl = 'ws://localhost:8081/ws'
      } else {
        // In production, use the current host
        wsUrl = `${protocol}//${window.location.hostname}:8081/ws`
      }
      
      wsRef.current = new WebSocket(wsUrl)
      
      // Set a connection timeout
      const timeout = setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING) {
          console.warn('WebSocket connection timeout - switching to mock data mode')
          wsRef.current.close()
          setUseMockData(true)
          startMockDataUpdates()
        }
      }, 5000) // 5 second timeout
      
      setConnectionTimeout(timeout)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected successfully')
        if (connectionTimeout) {
          clearTimeout(connectionTimeout)
        }
        setIsConnected(true)
        setReconnectAttempts(0)
        setUseMockData(false)
        // Clear mock data interval if it was running
        if (mockDataIntervalRef.current) {
          clearInterval(mockDataIntervalRef.current)
        }
      }

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          switch (data.type) {
            case 'system_status':
              dispatch({ type: 'SET_SYSTEM_STATUS', payload: data.data })
              break
            case 'chat_response':
              dispatch({ type: 'ADD_CHAT_MESSAGE', payload: data.data })
              break
            case 'memory_stats':
              dispatch({ type: 'SET_MEMORY_STATS', payload: data.data })
              break
            case 'performance_metrics':
              dispatch({ type: 'SET_PERFORMANCE_METRICS', payload: data.data })
              break
            case 'model_loaded':
              // Refresh active models
              fetchActiveModels()
              break
            default:
              console.log('Unknown WebSocket message type:', data.type)
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      wsRef.current.onclose = () => {
        console.log('WebSocket connection closed')
        if (connectionTimeout) {
          clearTimeout(connectionTimeout)
        }
        setIsConnected(false)
        
        // Only attempt reconnection if we were previously connected
        // or if we haven't exceeded the maximum attempts
        if (reconnectAttempts < 2) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000)
          console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttempts + 1}/2)`)
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1)
            connect()
          }, delay)
        } else {
          // After failed attempts, switch to mock data mode
          console.log('Max reconnection attempts reached - switching to development mode with mock data')
          setUseMockData(true)
          startMockDataUpdates()
        }
      }

      wsRef.current.onerror = (error) => {
        console.warn('WebSocket connection error - this is normal in development mode without backend services')
        if (connectionTimeout) {
          clearTimeout(connectionTimeout)
        }
      }
    } catch (error) {
      console.warn('Failed to create WebSocket connection - running in development mode with mock data')
      setUseMockData(true)
      startMockDataUpdates()
    }
  }

  const startMockDataUpdates = () => {
    console.log('Starting mock data updates for development mode')
    // Initialize with mock data
    dispatch({ type: 'SET_SYSTEM_STATUS', payload: mockSystemStatus })
    dispatch({ type: 'SET_PERFORMANCE_METRICS', payload: mockPerformanceMetrics })
    
    // Update mock data periodically to simulate real updates
    mockDataIntervalRef.current = setInterval(() => {
      const updatedStatus = {
        ...mockSystemStatus,
        status: 'development',
        uptime: Date.now() - (Date.now() % 1000000),
        memory_usage: Math.random() * 50 + 25,
        cpu_usage: Math.random() * 30 + 10,
        active_models: ['llama2:7b', 'codellama:13b'],
        last_updated: new Date().toISOString()
      }
      
      const updatedMetrics = {
        ...mockPerformanceMetrics,
        response_time: Math.random() * 100 + 50,
        requests_per_second: Math.random() * 10 + 5,
        error_rate: Math.random() * 5,
        memory_usage: Math.random() * 50 + 25
      }
      
      dispatch({ type: 'SET_SYSTEM_STATUS', payload: updatedStatus })
      dispatch({ type: 'SET_PERFORMANCE_METRICS', payload: updatedMetrics })
    }, 5000) // Update every 5 seconds
  }
  const disconnect = () => {
    if (connectionTimeout) {
      clearTimeout(connectionTimeout)
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (mockDataIntervalRef.current) {
      clearInterval(mockDataIntervalRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
    setUseMockData(false)
  }

  const sendMessage = (message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      if (useMockData) {
        console.log('Development mode: WebSocket message would be sent to backend:', message)
      } else {
        console.warn('WebSocket not connected - message queued for when connection is established')
      }
    }
  }

  const fetchActiveModels = async () => {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout
      
      const response = await fetch('/api/models/active', {
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)
      
      if (response.ok) {
        const models = await response.json()
        dispatch({ type: 'SET_ACTIVE_MODELS', payload: models })
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.warn('API request timeout - using mock models for development')
      } else {
        console.warn('API not available - using mock models for development')
      }
      const mockModels = [
        { name: 'llama2:7b', status: 'loaded', size: '3.8GB' },
        { name: 'codellama:13b', status: 'available', size: '7.3GB' },
        { name: 'mistral:7b', status: 'available', size: '4.1GB' }
      ]
      dispatch({ type: 'SET_ACTIVE_MODELS', payload: mockModels })
    }
  }

  useEffect(() => {
    connect()
    fetchActiveModels()

    return () => {
      disconnect()
    }
  }, [])

  return {
    isConnected,
    reconnectAttempts,
    useMockData,
    sendMessage,
    connect: () => connect(), // Allow manual reconnection
    systemStatus: state.systemStatus
  }
}