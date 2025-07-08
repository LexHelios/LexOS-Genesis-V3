import { useState, useEffect, useRef } from 'react'
import { useLexOS } from '../context/LexOSContext'

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const wsRef = useRef<WebSocket | null>(null)
  const { state, dispatch } = useLexOS()
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  const connect = () => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.hostname}:8081/ws`
      
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setReconnectAttempts(0)
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
        
        // Attempt to reconnect with exponential backoff
        if (reconnectAttempts < 10) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000)
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1)
            connect()
          }, delay)
        }
      }

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
    }
  }

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
  }

  const sendMessage = (message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected, cannot send message')
    }
  }

  const fetchActiveModels = async () => {
    try {
      const response = await fetch('/api/models/active')
      if (response.ok) {
        const models = await response.json()
        dispatch({ type: 'SET_ACTIVE_MODELS', payload: models })
      }
    } catch (error) {
      console.error('Failed to fetch active models:', error)
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
    sendMessage,
    systemStatus: state.systemStatus
  }
}