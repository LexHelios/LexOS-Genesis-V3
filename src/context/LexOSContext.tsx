import React, { createContext, useContext, useReducer, ReactNode } from 'react'

interface SystemStatus {
  cpu_percent: number
  memory_percent: number
  memory_available_gb: number
  gpus: Array<{
    id: number
    name: string
    memory_used: number
    memory_total: number
    memory_percent: number
    temperature?: number
    load: number
  }>
  active_models: string[]
  model_gpu_assignments: Record<string, number>
}

interface ChatMessage {
  id: string
  type: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  model?: string
  attachments?: Array<{
    type: 'image' | 'file' | 'video' | 'audio'
    url: string
    name: string
    size?: number
  }>
  metadata?: Record<string, any>
}

interface LexOSState {
  systemStatus: SystemStatus | null
  chatMessages: ChatMessage[]
  activeModels: string[]
  memoryStats: any
  performanceMetrics: any
  isProcessing: boolean
  currentModel: string
}

type LexOSAction =
  | { type: 'SET_SYSTEM_STATUS'; payload: SystemStatus }
  | { type: 'ADD_CHAT_MESSAGE'; payload: ChatMessage }
  | { type: 'SET_CHAT_MESSAGES'; payload: ChatMessage[] }
  | { type: 'SET_ACTIVE_MODELS'; payload: string[] }
  | { type: 'SET_MEMORY_STATS'; payload: any }
  | { type: 'SET_PERFORMANCE_METRICS'; payload: any }
  | { type: 'SET_PROCESSING'; payload: boolean }
  | { type: 'SET_CURRENT_MODEL'; payload: string }

const initialState: LexOSState = {
  systemStatus: null,
  chatMessages: [],
  activeModels: [],
  memoryStats: null,
  performanceMetrics: null,
  isProcessing: false,
  currentModel: 'llama3.2:3b'
}

function lexosReducer(state: LexOSState, action: LexOSAction): LexOSState {
  switch (action.type) {
    case 'SET_SYSTEM_STATUS':
      return { ...state, systemStatus: action.payload }
    case 'ADD_CHAT_MESSAGE':
      return { 
        ...state, 
        chatMessages: [...state.chatMessages, action.payload]
      }
    case 'SET_CHAT_MESSAGES':
      return { ...state, chatMessages: action.payload }
    case 'SET_ACTIVE_MODELS':
      return { ...state, activeModels: action.payload }
    case 'SET_MEMORY_STATS':
      return { ...state, memoryStats: action.payload }
    case 'SET_PERFORMANCE_METRICS':
      return { ...state, performanceMetrics: action.payload }
    case 'SET_PROCESSING':
      return { ...state, isProcessing: action.payload }
    case 'SET_CURRENT_MODEL':
      return { ...state, currentModel: action.payload }
    default:
      return state
  }
}

const LexOSContext = createContext<{
  state: LexOSState
  dispatch: React.Dispatch<LexOSAction>
} | null>(null)

export function LexOSProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(lexosReducer, initialState)

  return (
    <LexOSContext.Provider value={{ state, dispatch }}>
      {children}
    </LexOSContext.Provider>
  )
}

export function useLexOS() {
  const context = useContext(LexOSContext)
  if (!context) {
    throw new Error('useLexOS must be used within a LexOSProvider')
  }
  return context
}