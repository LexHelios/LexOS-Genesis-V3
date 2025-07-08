export interface Agent {
  id: string
  name: string
  type: 'orchestrator' | 'chat' | 'coding' | 'vision' | 'reasoning' | 'research' | 'memory'
  model: string
  capabilities: string[]
  status: 'idle' | 'busy' | 'error' | 'offline'
  currentTask?: string
  performance: {
    avgResponseTime: number
    successRate: number
    tasksCompleted: number
    lastUsed?: string
  }
  cost: {
    tokensPerSecond: number
    costPerToken: number
  }
  specializations: string[]
}

export interface Task {
  id: string
  type: string
  content: string
  priority: number
  complexity: 'simple' | 'medium' | 'complex'
  requiresVision: boolean
  requiresMemory: boolean
  estimatedTokens: number
  maxResponseTime: number
  context?: any
  assignedAgent?: string
  assignedModel?: string
  status: 'pending' | 'analyzing' | 'processing' | 'completed' | 'failed'
  startTime?: number
  endTime?: number
  reasoning?: string
}

export interface LLMModel {
  id: string
  name: string
  provider: string
  capabilities: string[]
  contextWindow: number
  speed: 'very_fast' | 'fast' | 'medium' | 'slow'
  cost: number
  quality: number
  specialties: string[]
  availability: 'available' | 'busy' | 'offline'
  currentLoad: number
}

export interface ConversationContext {
  id: string
  messages: ConversationMessage[]
  activeAgents: string[]
  currentTask?: Task
  metadata: {
    startTime: string
    lastActivity: string
    totalTokens: number
    totalCost: number
  }
}

export interface ConversationMessage {
  id: string
  type: 'user' | 'agent' | 'system' | 'orchestrator'
  content: string
  timestamp: string
  agentId?: string
  modelUsed?: string
  metadata?: {
    reasoning?: string
    confidence?: number
    processingTime?: number
    tokensUsed?: number
  }
}

export interface OrchestratorDecision {
  selectedAgent: Agent
  selectedModel: LLMModel
  reasoning: string
  confidence: number
  estimatedCost: number
  estimatedTime: number
  fallbackOptions: Array<{
    agent: Agent
    model: LLMModel
    reason: string
  }>
}