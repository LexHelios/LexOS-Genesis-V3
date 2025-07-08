import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CpuChipIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ChartBarIcon,
  LightBulbIcon,
  EyeIcon,
  CodeBracketIcon,
  ChatBubbleLeftRightIcon,
  MagnifyingGlassIcon,
  CircleStackIcon,
  BoltIcon
} from '@heroicons/react/24/outline'
import { OrchestratorService } from '../services/OrchestratorService'
import { Agent, LLMModel } from '../types/agents'

export default function AgentDashboard() {
  const [orchestrator, setOrchestrator] = useState<OrchestratorService | null>(null)
  const [agents, setAgents] = useState<Agent[]>([])
  const [models, setModels] = useState<LLMModel[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [selectedModel, setSelectedModel] = useState<LLMModel | null>(null)

  useEffect(() => {
    const orchestratorService = new OrchestratorService((level, message, data) => {
      console.log(`[${level.toUpperCase()}] ${message}`, data)
    })
    
    setOrchestrator(orchestratorService)
    setAgents(orchestratorService.getAgents())
    setModels(orchestratorService.getModels())

    // Simulate some activity for demo
    const interval = setInterval(() => {
      setAgents(orchestratorService.getAgents())
      setModels(orchestratorService.getModels())
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const getAgentIcon = (type: string) => {
    const icons = {
      'orchestrator': LightBulbIcon,
      'chat': ChatBubbleLeftRightIcon,
      'coding': CodeBracketIcon,
      'vision': EyeIcon,
      'reasoning': LightBulbIcon,
      'research': MagnifyingGlassIcon,
      'memory': CircleStackIcon
    }
    return icons[type as keyof typeof icons] || CpuChipIcon
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'idle': return 'text-green-600 dark:text-green-400'
      case 'busy': return 'text-blue-600 dark:text-blue-400'
      case 'error': return 'text-red-600 dark:text-red-400'
      case 'offline': return 'text-gray-600 dark:text-gray-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'idle': return CheckCircleIcon
      case 'busy': return ClockIcon
      case 'error': return ExclamationTriangleIcon
      case 'offline': return ExclamationTriangleIcon
      default: return ClockIcon
    }
  }

  const getSpeedColor = (speed: string) => {
    switch (speed) {
      case 'very_fast': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
      case 'fast': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
      case 'slow': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300'
    }
  }

  const getAvailabilityColor = (availability: string) => {
    switch (availability) {
      case 'available': return 'text-green-600 dark:text-green-400'
      case 'busy': return 'text-yellow-600 dark:text-yellow-400'
      case 'offline': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-dark-900 min-h-full">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Agent Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Monitor and manage the multi-agent orchestration system
        </p>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Agents</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{agents.length}</p>
            </div>
            <CpuChipIcon className="w-8 h-8 text-primary-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Agents</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {agents.filter(a => a.status === 'busy').length}
              </p>
            </div>
            <BoltIcon className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Available Models</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {models.filter(m => m.availability === 'available').length}
              </p>
            </div>
            <ChartBarIcon className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Tasks</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {agents.reduce((sum, agent) => sum + agent.performance.tasksCompleted, 0)}
              </p>
            </div>
            <CheckCircleIcon className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Agents and Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Agents */}
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Agents ({agents.length})
          </h2>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            <AnimatePresence>
              {agents.map((agent) => {
                const Icon = getAgentIcon(agent.type)
                const StatusIcon = getStatusIcon(agent.status)
                
                return (
                  <motion.div
                    key={agent.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`bg-white dark:bg-dark-800 rounded-lg border p-4 cursor-pointer transition-all hover:shadow-md ${
                      selectedAgent?.id === agent.id 
                        ? 'border-primary-500 ring-2 ring-primary-200 dark:ring-primary-800' 
                        : 'border-gray-200 dark:border-dark-700'
                    }`}
                    onClick={() => setSelectedAgent(selectedAgent?.id === agent.id ? null : agent)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        <Icon className="w-6 h-6 text-primary-500" />
                        <div>
                          <h3 className="font-medium text-gray-900 dark:text-white">
                            {agent.name}
                          </h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {agent.type} • {agent.model}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        <StatusIcon className={`w-4 h-4 ${getStatusColor(agent.status)}`} />
                        <span className={`text-xs font-medium ${getStatusColor(agent.status)}`}>
                          {agent.status}
                        </span>
                      </div>
                    </div>

                    {agent.currentTask && (
                      <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-xs">
                        <span className="text-blue-700 dark:text-blue-300">
                          Current: {agent.currentTask}
                        </span>
                      </div>
                    )}

                    <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Success Rate:</span>
                        <div className="font-medium">{(agent.performance.successRate * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Avg Time:</span>
                        <div className="font-medium">{agent.performance.avgResponseTime.toFixed(1)}s</div>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Tasks:</span>
                        <div className="font-medium">{agent.performance.tasksCompleted}</div>
                      </div>
                    </div>

                    {/* Expanded Details */}
                    <AnimatePresence>
                      {selectedAgent?.id === agent.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-4 pt-4 border-t border-gray-200 dark:border-dark-700"
                        >
                          <div className="space-y-3">
                            <div>
                              <span className="text-sm font-medium text-gray-900 dark:text-white">Capabilities:</span>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {agent.capabilities.map((cap) => (
                                  <span
                                    key={cap}
                                    className="px-2 py-0.5 bg-gray-100 dark:bg-dark-700 text-gray-700 dark:text-gray-300 rounded text-xs"
                                  >
                                    {cap.replace('_', ' ')}
                                  </span>
                                ))}
                              </div>
                            </div>
                            
                            <div>
                              <span className="text-sm font-medium text-gray-900 dark:text-white">Specializations:</span>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {agent.specializations.map((spec) => (
                                  <span
                                    key={spec}
                                    className="px-2 py-0.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded text-xs"
                                  >
                                    {spec.replace('_', ' ')}
                                  </span>
                                ))}
                              </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="text-gray-600 dark:text-gray-400">Tokens/sec:</span>
                                <div className="font-medium">{agent.cost.tokensPerSecond}</div>
                              </div>
                              <div>
                                <span className="text-gray-600 dark:text-gray-400">Cost/token:</span>
                                <div className="font-medium">${agent.cost.costPerToken.toFixed(6)}</div>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                )
              })}
            </AnimatePresence>
          </div>
        </div>

        {/* Models */}
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Models ({models.length})
          </h2>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            <AnimatePresence>
              {models.map((model) => (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`bg-white dark:bg-dark-800 rounded-lg border p-4 cursor-pointer transition-all hover:shadow-md ${
                    selectedModel?.id === model.id 
                      ? 'border-primary-500 ring-2 ring-primary-200 dark:ring-primary-800' 
                      : 'border-gray-200 dark:border-dark-700'
                  }`}
                  onClick={() => setSelectedModel(selectedModel?.id === model.id ? null : model)}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">
                        {model.name}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {model.provider} • {model.contextWindow.toLocaleString()} tokens
                      </p>
                    </div>
                    
                    <div className="flex flex-col items-end space-y-1">
                      <span className={`text-xs font-medium ${getAvailabilityColor(model.availability)}`}>
                        {model.availability}
                      </span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${getSpeedColor(model.speed)}`}>
                        {model.speed.replace('_', ' ')}
                      </span>
                    </div>
                  </div>

                  <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Quality:</span>
                      <div className="font-medium">{model.quality}/10</div>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Cost:</span>
                      <div className="font-medium">{model.cost}/10</div>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Load:</span>
                      <div className="font-medium">{model.currentLoad}%</div>
                    </div>
                  </div>

                  {/* Load Bar */}
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
                      <div
                        className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${model.currentLoad}%` }}
                      />
                    </div>
                  </div>

                  {/* Expanded Details */}
                  <AnimatePresence>
                    {selectedModel?.id === model.id && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 pt-4 border-t border-gray-200 dark:border-dark-700"
                      >
                        <div className="space-y-3">
                          <div>
                            <span className="text-sm font-medium text-gray-900 dark:text-white">Capabilities:</span>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {model.capabilities.map((cap) => (
                                <span
                                  key={cap}
                                  className="px-2 py-0.5 bg-gray-100 dark:bg-dark-700 text-gray-700 dark:text-gray-300 rounded text-xs"
                                >
                                  {cap.replace('_', ' ')}
                                </span>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <span className="text-sm font-medium text-gray-900 dark:text-white">Specialties:</span>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {model.specialties.map((spec) => (
                                <span
                                  key={spec}
                                  className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-xs"
                                >
                                  {spec.replace('_', ' ')}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}