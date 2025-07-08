import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CpuChipIcon,
  PlayIcon,
  StopIcon,
  TrashIcon,
  CloudArrowDownIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { useLexOS } from '../context/LexOSContext'

interface ModelInfo {
  name: string
  size_gb: number
  capabilities: string[]
  best_for: string[]
  speed: string
  context_window: number
  quantization?: string
  updated_at?: string
  status?: 'active' | 'inactive' | 'loading' | 'error'
}

export default function ModelManager() {
  const { state, dispatch } = useLexOS()
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  const [loadingModels, setLoadingModels] = useState<Set<string>>(new Set())

  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    try {
      const response = await fetch('/api/models')
      if (response.ok) {
        const modelsData = await response.json()
        setModels(modelsData.map((model: ModelInfo) => ({
          ...model,
          status: state.activeModels.includes(model.name) ? 'active' : 'inactive'
        })))
      }
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadModel = async (modelName: string) => {
    setLoadingModels(prev => new Set(prev).add(modelName))
    try {
      const response = await fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelName })
      })
      
      if (response.ok) {
        setModels(prev => prev.map(model => 
          model.name === modelName 
            ? { ...model, status: 'active' as const }
            : model
        ))
        dispatch({ type: 'SET_ACTIVE_MODELS', payload: [...state.activeModels, modelName] })
      }
    } catch (error) {
      console.error('Failed to load model:', error)
      setModels(prev => prev.map(model => 
        model.name === modelName 
          ? { ...model, status: 'error' as const }
          : model
      ))
    } finally {
      setLoadingModels(prev => {
        const newSet = new Set(prev)
        newSet.delete(modelName)
        return newSet
      })
    }
  }

  const unloadModel = async (modelName: string) => {
    try {
      const response = await fetch('/api/models/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelName })
      })
      
      if (response.ok) {
        setModels(prev => prev.map(model => 
          model.name === modelName 
            ? { ...model, status: 'inactive' as const }
            : model
        ))
        dispatch({ 
          type: 'SET_ACTIVE_MODELS', 
          payload: state.activeModels.filter(m => m !== modelName) 
        })
      }
    } catch (error) {
      console.error('Failed to unload model:', error)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 dark:text-green-400'
      case 'loading': return 'text-blue-600 dark:text-blue-400'
      case 'error': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return CheckCircleIcon
      case 'loading': return CpuChipIcon
      case 'error': return ExclamationTriangleIcon
      default: return StopIcon
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

  const getModelIcon = (capabilities: string[]) => {
    if (capabilities.includes('vision')) return '👁️'
    if (capabilities.includes('coding')) return '💻'
    if (capabilities.includes('reasoning')) return '🧠'
    return '🤖'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    )
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-dark-900 min-h-full">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Manager</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Manage and monitor your AI models
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Models</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{models.length}</p>
            </div>
            <CpuChipIcon className="w-8 h-8 text-primary-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Models</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {models.filter(m => m.status === 'active').length}
              </p>
            </div>
            <PlayIcon className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Size</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {models.reduce((acc, model) => acc + model.size_gb, 0).toFixed(1)}GB
              </p>
            </div>
            <CloudArrowDownIcon className="w-8 h-8 text-blue-500" />
          </div>
        </div>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        <AnimatePresence>
          {models.map((model) => {
            const StatusIcon = getStatusIcon(model.status || 'inactive')
            const isLoading = loadingModels.has(model.name)
            
            return (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="bg-white dark:bg-dark-800 rounded-lg border border-gray-200 dark:border-dark-700 p-6 hover:shadow-lg transition-shadow"
              >
                {/* Model Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{getModelIcon(model.capabilities)}</span>
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {model.name}
                      </h3>
                      <div className="flex items-center space-x-2 mt-1">
                        <StatusIcon className={`w-4 h-4 ${getStatusColor(model.status || 'inactive')}`} />
                        <span className={`text-sm ${getStatusColor(model.status || 'inactive')}`}>
                          {isLoading ? 'Loading...' : (model.status || 'inactive')}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => setSelectedModel(selectedModel?.name === model.name ? null : model)}
                    className="p-1 hover:bg-gray-100 dark:hover:bg-dark-700 rounded"
                  >
                    <InformationCircleIcon className="w-5 h-5 text-gray-500" />
                  </button>
                </div>

                {/* Model Info */}
                <div className="space-y-3 mb-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Size:</span>
                    <span className="font-medium">{model.size_gb.toFixed(1)}GB</span>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Speed:</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSpeedColor(model.speed)}`}>
                      {model.speed.replace('_', ' ')}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Context:</span>
                    <span className="font-medium">{model.context_window.toLocaleString()}</span>
                  </div>
                </div>

                {/* Capabilities */}
                <div className="mb-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Capabilities:</p>
                  <div className="flex flex-wrap gap-1">
                    {model.capabilities.slice(0, 3).map((cap) => (
                      <span
                        key={cap}
                        className="px-2 py-1 bg-gray-100 dark:bg-dark-700 text-gray-700 dark:text-gray-300 rounded text-xs"
                      >
                        {cap.replace('_', ' ')}
                      </span>
                    ))}
                    {model.capabilities.length > 3 && (
                      <span className="text-xs text-gray-500">
                        +{model.capabilities.length - 3}
                      </span>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex space-x-2">
                  {model.status === 'active' ? (
                    <button
                      onClick={() => unloadModel(model.name)}
                      className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors"
                    >
                      <StopIcon className="w-4 h-4" />
                      <span>Unload</span>
                    </button>
                  ) : (
                    <button
                      onClick={() => loadModel(model.name)}
                      disabled={isLoading}
                      className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-lg hover:bg-primary-200 dark:hover:bg-primary-900/50 transition-colors disabled:opacity-50"
                    >
                      {isLoading ? (
                        <div className="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <PlayIcon className="w-4 h-4" />
                      )}
                      <span>{isLoading ? 'Loading...' : 'Load'}</span>
                    </button>
                  )}
                </div>

                {/* Expanded Details */}
                <AnimatePresence>
                  {selectedModel?.name === model.name && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-4 pt-4 border-t border-gray-200 dark:border-dark-700"
                    >
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="font-medium text-gray-900 dark:text-white">Best for:</span>
                          <p className="text-gray-600 dark:text-gray-400 mt-1">
                            {model.best_for.join(', ')}
                          </p>
                        </div>
                        
                        {model.quantization && (
                          <div>
                            <span className="font-medium text-gray-900 dark:text-white">Quantization:</span>
                            <p className="text-gray-600 dark:text-gray-400">{model.quantization}</p>
                          </div>
                        )}
                        
                        {model.updated_at && (
                          <div>
                            <span className="font-medium text-gray-900 dark:text-white">Updated:</span>
                            <p className="text-gray-600 dark:text-gray-400">
                              {new Date(model.updated_at).toLocaleDateString()}
                            </p>
                          </div>
                        )}
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
  )
}