import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  PaperAirplaneIcon,
  CpuChipIcon,
  UserIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  LightBulbIcon,
  EyeIcon,
  CodeBracketIcon,
  ChatBubbleLeftRightIcon,
  MagnifyingGlassIcon,
  CircleStackIcon
} from '@heroicons/react/24/outline'
import { OrchestratorService } from '../services/OrchestratorService'
import { Agent, Task, ConversationMessage, ConversationContext, OrchestratorDecision } from '../types/agents'

export default function MultiAgentChat() {
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [input, setInput] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [orchestratorDecision, setOrchestratorDecision] = useState<OrchestratorDecision | null>(null)
  const [conversationContext, setConversationContext] = useState<ConversationContext>({
    id: `conv_${Date.now()}`,
    messages: [],
    activeAgents: [],
    metadata: {
      startTime: new Date().toISOString(),
      lastActivity: new Date().toISOString(),
      totalTokens: 0,
      totalCost: 0
    }
  })

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const orchestratorRef = useRef<OrchestratorService>()

  useEffect(() => {
    // Initialize orchestrator with logging
    orchestratorRef.current = new OrchestratorService((level, message, data) => {
      console.log(`[${level.toUpperCase()}] ${message}`, data)
    })

    // Add welcome message
    const welcomeMessage: ConversationMessage = {
      id: 'welcome',
      type: 'system',
      content: 'Welcome to the Multi-Agent Conversation System! I\'ll analyze your requests and coordinate with specialized agents to provide the best responses.',
      timestamp: new Date().toISOString()
    }
    setMessages([welcomeMessage])
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isProcessing || !orchestratorRef.current) return

    const userMessage: ConversationMessage = {
      id: `msg_${Date.now()}`,
      type: 'user',
      content: input,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsProcessing(true)

    try {
      // Step 1: Analyze the task
      const task = await orchestratorRef.current.analyzeTask(input, conversationContext)
      setCurrentTask(task)

      // Add orchestrator analysis message
      const analysisMessage: ConversationMessage = {
        id: `analysis_${Date.now()}`,
        type: 'orchestrator',
        content: `🔍 **Task Analysis Complete**\n\n**Type:** ${task.type}\n**Complexity:** ${task.complexity}\n**Priority:** ${task.priority}/10\n**Estimated Tokens:** ${task.estimatedTokens}\n**Requires Vision:** ${task.requiresVision ? 'Yes' : 'No'}\n**Requires Memory:** ${task.requiresMemory ? 'Yes' : 'No'}`,
        timestamp: new Date().toISOString(),
        agentId: 'orchestrator',
        metadata: {
          reasoning: 'Analyzing user request to determine optimal processing strategy',
          confidence: 0.95
        }
      }
      setMessages(prev => [...prev, analysisMessage])

      // Step 2: Select optimal agent and model
      const decision = await orchestratorRef.current.selectOptimalAgentAndModel(task, conversationContext)
      setOrchestratorDecision(decision)

      // Add selection reasoning message
      const selectionMessage: ConversationMessage = {
        id: `selection_${Date.now()}`,
        type: 'orchestrator',
        content: `🎯 **Agent & Model Selection**\n\n**Selected Agent:** ${decision.selectedAgent.name}\n**Selected Model:** ${decision.selectedModel.name}\n**Confidence:** ${(decision.confidence * 100).toFixed(1)}%\n**Estimated Cost:** $${decision.estimatedCost.toFixed(4)}\n**Estimated Time:** ${(decision.estimatedTime / 1000).toFixed(1)}s\n\n**Reasoning:** ${decision.reasoning}`,
        timestamp: new Date().toISOString(),
        agentId: 'orchestrator',
        metadata: {
          reasoning: decision.reasoning,
          confidence: decision.confidence
        }
      }
      setMessages(prev => [...prev, selectionMessage])

      // Step 3: Simulate agent processing
      await simulateAgentProcessing(decision, task, userMessage.content)

    } catch (error) {
      console.error('Error processing request:', error)
      const errorMessage: ConversationMessage = {
        id: `error_${Date.now()}`,
        type: 'system',
        content: `❌ **Error:** ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsProcessing(false)
      setCurrentTask(null)
      setOrchestratorDecision(null)
    }
  }

  const simulateAgentProcessing = async (decision: OrchestratorDecision, task: Task, userInput: string) => {
    const { selectedAgent, selectedModel } = decision
    const startTime = Date.now()

    // Update agent status
    if (orchestratorRef.current) {
      const agent = orchestratorRef.current.getAgentById(selectedAgent.id)
      if (agent) {
        agent.status = 'busy'
        agent.currentTask = userInput.substring(0, 50) + '...'
      }
    }

    // Add processing message
    const processingMessage: ConversationMessage = {
      id: `processing_${Date.now()}`,
      type: 'agent',
      content: `⚡ **${selectedAgent.name} is processing your request...**\n\nUsing ${selectedModel.name} for optimal results.`,
      timestamp: new Date().toISOString(),
      agentId: selectedAgent.id,
      modelUsed: selectedModel.id
    }
    setMessages(prev => [...prev, processingMessage])

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, Math.random() * 3000 + 1000))

    // Generate response based on agent type
    const response = generateAgentResponse(selectedAgent, userInput, task)
    const endTime = Date.now()
    const processingTime = endTime - startTime

    // Add agent response
    const responseMessage: ConversationMessage = {
      id: `response_${Date.now()}`,
      type: 'agent',
      content: response,
      timestamp: new Date().toISOString(),
      agentId: selectedAgent.id,
      modelUsed: selectedModel.id,
      metadata: {
        processingTime,
        tokensUsed: Math.floor(Math.random() * 500 + 100),
        confidence: 0.85 + Math.random() * 0.1
      }
    }
    setMessages(prev => [...prev, responseMessage])

    // Update agent performance
    if (orchestratorRef.current) {
      orchestratorRef.current.updateAgentPerformance(
        selectedAgent.id,
        true, // Assume success for demo
        processingTime / 1000
      )

      // Reset agent status
      const agent = orchestratorRef.current.getAgentById(selectedAgent.id)
      if (agent) {
        agent.status = 'idle'
        agent.currentTask = undefined
      }
    }

    // Update conversation context
    setConversationContext(prev => ({
      ...prev,
      messages: [...prev.messages, responseMessage],
      metadata: {
        ...prev.metadata,
        lastActivity: new Date().toISOString(),
        totalTokens: prev.metadata.totalTokens + (responseMessage.metadata?.tokensUsed || 0),
        totalCost: prev.metadata.totalCost + decision.estimatedCost
      }
    }))
  }

  const generateAgentResponse = (agent: Agent, userInput: string, task: Task): string => {
    const responses = {
      'chat-assistant': `I understand you're asking about "${userInput}". As your general assistant, I'm here to help with a wide range of topics. Let me provide you with a comprehensive response that addresses your question while considering the context of our conversation.`,
      
      'code-specialist': `Looking at your coding request: "${userInput}"\n\n\`\`\`javascript\n// Here's a solution approach:\nfunction exampleSolution() {\n  // Implementation would go here\n  return "This is a code example";\n}\n\`\`\`\n\nI've analyzed your requirements and provided a structured approach. Would you like me to elaborate on any specific part of the implementation?`,
      
      'vision-analyst': `I've analyzed the visual content you've shared. Based on my computer vision capabilities, I can identify key elements, patterns, and provide detailed insights about what I observe. The image appears to contain [detailed analysis would go here based on actual image content].`,
      
      'reasoning-engine': `Let me break down this problem systematically:\n\n1. **Problem Analysis**: ${userInput}\n2. **Logical Approach**: I'll apply step-by-step reasoning\n3. **Solution Path**: [Mathematical/logical solution would be provided]\n4. **Verification**: Double-checking the reasoning\n\nThe solution demonstrates clear logical progression and mathematical rigor.`,
      
      'research-assistant': `I've conducted a comprehensive analysis of your research query: "${userInput}"\n\n**Key Findings:**\n- Primary insights and data points\n- Relevant background information\n- Current trends and developments\n- Implications and recommendations\n\nThis research synthesis draws from multiple sources and provides you with actionable insights.`,
      
      'memory-manager': `I've searched through our conversation history and relevant stored information. Based on previous interactions and context, I can provide you with relevant details that connect to your current request. Here's what I found that might be helpful...`
    }

    return responses[agent.id as keyof typeof responses] || 
           `As ${agent.name}, I'm processing your request: "${userInput}". I'll provide a specialized response based on my capabilities in ${agent.specializations.join(', ')}.`
  }

  const getAgentIcon = (agentId: string) => {
    const icons = {
      'orchestrator': LightBulbIcon,
      'chat-assistant': ChatBubbleLeftRightIcon,
      'code-specialist': CodeBracketIcon,
      'vision-analyst': EyeIcon,
      'reasoning-engine': LightBulbIcon,
      'research-assistant': MagnifyingGlassIcon,
      'memory-manager': CircleStackIcon
    }
    return icons[agentId as keyof typeof icons] || CpuChipIcon
  }

  const getMessageTypeColor = (type: string, agentId?: string) => {
    if (type === 'user') return 'bg-primary-500 text-white'
    if (type === 'system') return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200'
    if (type === 'orchestrator') return 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200'
    
    // Agent-specific colors
    const agentColors = {
      'chat-assistant': 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200',
      'code-specialist': 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200',
      'vision-analyst': 'bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-200',
      'reasoning-engine': 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-800 dark:text-indigo-200',
      'research-assistant': 'bg-pink-100 dark:bg-pink-900/30 text-pink-800 dark:text-pink-200',
      'memory-manager': 'bg-teal-100 dark:bg-teal-900/30 text-teal-800 dark:text-teal-200'
    }
    
    return agentColors[agentId as keyof typeof agentColors] || 'bg-gray-100 dark:bg-dark-700 text-gray-800 dark:text-gray-200'
  }

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-dark-900">
      {/* Header */}
      <div className="bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
              Multi-Agent Conversation System
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Intelligent orchestration with specialized AI agents
            </p>
          </div>
          
          {/* Current Task Status */}
          {currentTask && (
            <div className="flex items-center space-x-2 text-sm">
              <ClockIcon className="w-4 h-4 text-blue-500" />
              <span className="text-blue-600 dark:text-blue-400">
                Processing {currentTask.type} task...
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {messages.map((message) => {
            const Icon = message.type === 'user' ? UserIcon : getAgentIcon(message.agentId || 'orchestrator')
            
            return (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex space-x-3 max-w-4xl ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.type === 'user' 
                      ? 'bg-primary-500 text-white' 
                      : getMessageTypeColor(message.type, message.agentId)
                  }`}>
                    <Icon className="w-5 h-5" />
                  </div>

                  {/* Message Content */}
                  <div className={`flex flex-col space-y-2 ${message.type === 'user' ? 'items-end' : 'items-start'}`}>
                    {/* Agent/User Label */}
                    {message.agentId && (
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {orchestratorRef.current?.getAgentById(message.agentId)?.name || message.agentId}
                        {message.modelUsed && ` • ${message.modelUsed}`}
                      </div>
                    )}

                    {/* Message Bubble */}
                    <div className={`rounded-2xl px-4 py-3 max-w-2xl ${
                      message.type === 'user'
                        ? 'bg-primary-500 text-white'
                        : getMessageTypeColor(message.type, message.agentId)
                    }`}>
                      <div className="prose prose-sm max-w-none">
                        {message.content.split('\n').map((line, index) => (
                          <div key={index}>
                            {line.startsWith('**') && line.endsWith('**') ? (
                              <strong>{line.slice(2, -2)}</strong>
                            ) : line.startsWith('```') ? (
                              <code className="block bg-black/10 p-2 rounded mt-2 font-mono text-xs">
                                {line.slice(3)}
                              </code>
                            ) : (
                              line
                            )}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Metadata */}
                    {message.metadata && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                        {message.metadata.processingTime && (
                          <div>Processing time: {(message.metadata.processingTime / 1000).toFixed(2)}s</div>
                        )}
                        {message.metadata.confidence && (
                          <div>Confidence: {(message.metadata.confidence * 100).toFixed(1)}%</div>
                        )}
                        {message.metadata.tokensUsed && (
                          <div>Tokens used: {message.metadata.tokensUsed}</div>
                        )}
                      </div>
                    )}

                    {/* Timestamp */}
                    <div className={`text-xs text-gray-500 dark:text-gray-400 ${
                      message.type === 'user' ? 'text-right' : 'text-left'
                    }`}>
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>

        {/* Processing Indicator */}
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center space-x-2 text-gray-500 dark:text-gray-400"
          >
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
            </div>
            <span>Orchestrator is coordinating agents...</span>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white dark:bg-dark-800 border-t border-gray-200 dark:border-dark-700 p-4">
        <form onSubmit={handleSubmit} className="flex items-end space-x-2">
          <div className="flex-1">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e)
                }
              }}
              placeholder="Ask me anything... I'll coordinate with the best agents to help you!"
              className="w-full resize-none rounded-lg border border-gray-300 dark:border-dark-600 bg-white dark:bg-dark-700 px-4 py-3 focus:ring-2 focus:ring-primary-500 focus:border-transparent max-h-32"
              rows={1}
              style={{
                minHeight: '48px',
                height: Math.min(Math.max(48, input.split('\n').length * 24), 128)
              }}
            />
          </div>
          
          <button
            type="submit"
            disabled={!input.trim() || isProcessing}
            className="p-3 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </form>

        {/* Conversation Stats */}
        <div className="mt-2 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center space-x-4">
            <span>Messages: {messages.length}</span>
            <span>Tokens: {conversationContext.metadata.totalTokens}</span>
            <span>Cost: ${conversationContext.metadata.totalCost.toFixed(4)}</span>
          </div>
          <div>
            Last activity: {new Date(conversationContext.metadata.lastActivity).toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  )
}