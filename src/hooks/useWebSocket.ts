import { useState, useEffect, useRef } from 'react'
import { useLexOS } from '../context/LexOSContext'

// Mock data for development when backend is not available
const mockSystemStatus = {
  status: 'offline',
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
  const wsRef = useRef<WebSocket | null>(null)
  const { state, dispatch } = useLexOS()
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const mockDataIntervalRef = useRef<NodeJS.Timeout>()

  const connect = () => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      // Use the correct WebSocket URL for the current environment
      const wsUrl = import.meta.env.DEV 
        ? 'ws://localhost:8081/ws'
        : `${protocol}//${window.location.hostname}:8081/ws`
      
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
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
        console.log('WebSocket disconnected')
        setIsConnected(false)
        
        // Attempt to reconnect with exponential backoff, but limit attempts
        if (reconnectAttempts < 3) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000)
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1)
            connect()
          }, delay)
        } else {
          // After 3 failed attempts, switch to mock data mode
          console.log('Switching to mock data mode - backend appears to be offline')
          setUseMockData(true)
          startMockDataUpdates()
        }
      }

      wsRef.current.onerror = (error) => {
        console.warn('WebSocket connection failed - this is expected in development without backend')
      }
    } catch (error) {
      console.warn('Failed to create WebSocket connection - using mock data for development')
      setUseMockData(true)
      startMockDataUpdates()
    }
  }

  const startMockDataUpdates = () => {
    // Initialize with mock data
    dispatch({ type: 'SET_SYSTEM_STATUS', payload: mockSystemStatus })
    dispatch({ type: 'SET_PERFORMANCE_METRICS', payload: mockPerformanceMetrics })
    
    // Update mock data periodically to simulate real updates
    mockDataIntervalRef.current = setInterval(() => {
      const updatedStatus = {
        ...mockSystemStatus,
        uptime: Date.now() - (Date.now() % 1000000),
        memory_usage: Math.random() * 50 + 25,
        cpu_usage: Math.random() * 30 + 10,
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
      console.warn('WebSocket not connected - message would be sent in production:', message)
    }
  }

  const fetchActiveModels = async () => {
    try {
      const response = await fetch('/api/models/active')
      if (response.ok) {
        const models = await response.json()
        dispatch({ type: 'SET_ACTIVE_MODELS', payload: models })
      } else {
        // Use mock models when API is not available
        const mockModels = [
          { name: 'llama2:7b', status: 'loaded', size: '3.8GB' },
          { name: 'codellama:13b', status: 'available', size: '7.3GB' }
        ]
        dispatch({ type: 'SET_ACTIVE_MODELS', payload: mockModels })
      }
    } catch (error) {
      console.warn('API not available - using mock models for development')
      const mockModels = [
        { name: 'llama2:7b', status: 'loaded', size: '3.8GB' },
        { name: 'codellama:13b', status: 'available', size: '7.3GB' }
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
    systemStatus: state.systemStatus
  }
}