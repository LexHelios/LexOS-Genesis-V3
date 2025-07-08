import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDownIcon, CpuChipIcon, CheckIcon } from '@heroicons/react/24/outline'
import { useLexOS } from '../context/LexOSContext'

interface ModelInfo {
  name: string
  size_gb: number
  capabilities: string[]
  best_for: string[]
  speed: string
  context_window: number
}

export default function ModelSelector() {
  const { state, dispatch } = useLexOS()
  const [isOpen, setIsOpen] = useState(false)
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchAvailableModels()
  }, [])

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('/api/models')
      if (response.ok) {
        const models = await response.json()
        setAvailableModels(models)
      }
    } catch (error) {
      console.error('Failed to fetch models:', error)
    }
  }

  const selectModel = async (modelName: string) => {
    setLoading(true)
    try {
      const response = await fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelName })
      })
      
      if (response.ok) {
        dispatch({ type: 'SET_CURRENT_MODEL', payload: modelName })
        setIsOpen(false)
      }
    } catch (error) {
      console.error('Failed to load model:', error)
    } finally {
      setLoading(false)
    }
  }

  const getModelIcon = (capabilities: string[]) => {
    if (capabilities.includes('vision')) return '👁️'
    if (capabilities.includes('coding')) return '💻'
    if (capabilities.includes('reasoning')) return '🧠'
    return '🤖'
  }

  const getSpeedColor = (speed: string) => {
    switch (speed) {
      case 'very_fast': return 'text-green-600 dark:text-green-400'
      case 'fast': return 'text-blue-600 dark:text-blue-400'
      case 'medium': return 'text-yellow-600 dark:text-yellow-400'
      case 'slow': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={loading}
        className="flex items-center space-x-2 px-3 py-2 bg-white dark:bg-dark-700 border border-gray-300 dark:border-dark-600 rounded-lg hover:bg-gray-50 dark:hover:bg-dark-600 transition-colors disabled:opacity-50"
      >
        <CpuChipIcon className="w-4 h-4" />
        <span className="text-sm font-medium">{state.currentModel}</span>
        <ChevronDownIcon className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full right-0 mt-2 w-96 bg-white dark:bg-dark-800 border border-gray-200 dark:border-dark-700 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto"
          >
            <div className="p-3 border-b border-gray-200 dark:border-dark-700">
              <h3 className="font-semibold text-sm">Select Model</h3>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Choose the best model for your task
              </p>
            </div>

            <div className="p-2 space-y-1">
              {availableModels.map((model) => (
                <motion.button
                  key={model.name}
                  onClick={() => selectModel(model.name)}
                  disabled={loading}
                  className={`w-full text-left p-3 rounded-lg transition-colors hover:bg-gray-100 dark:hover:bg-dark-700 disabled:opacity-50 ${
                    state.currentModel === model.name 
                      ? 'bg-primary-100 dark:bg-primary-900/30 border border-primary-200 dark:border-primary-800' 
                      : ''
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{getModelIcon(model.capabilities)}</span>
                        <span className="font-medium text-sm">{model.name}</span>
                        {state.currentModel === model.name && (
                          <CheckIcon className="w-4 h-4 text-primary-600" />
                        )}
                      </div>
                      
                      <div className="mt-1 flex items-center space-x-3 text-xs">
                        <span className="text-gray-600 dark:text-gray-400">
                          {model.size_gb.toFixed(1)}GB
                        </span>
                        <span className={getSpeedColor(model.speed)}>
                          {model.speed.replace('_', ' ')}
                        </span>
                        <span className="text-gray-600 dark:text-gray-400">
                          {model.context_window.toLocaleString()} ctx
                        </span>
                      </div>

                      <div className="mt-2 flex flex-wrap gap-1">
                        {model.capabilities.slice(0, 3).map((cap) => (
                          <span
                            key={cap}
                            className="px-2 py-0.5 bg-gray-100 dark:bg-dark-700 text-gray-700 dark:text-gray-300 rounded text-xs"
                          >
                            {cap.replace('_', ' ')}
                          </span>
                        ))}
                        {model.capabilities.length > 3 && (
                          <span className="text-xs text-gray-500">
                            +{model.capabilities.length - 3} more
                          </span>
                        )}
                      </div>

                      <div className="mt-1 text-xs text-gray-600 dark:text-gray-400">
                        Best for: {model.best_for.slice(0, 2).join(', ')}
                        {model.best_for.length > 2 && '...'}
                      </div>
                    </div>
                  </div>
                </motion.button>
              ))}
            </div>

            {availableModels.length === 0 && (
              <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                <CpuChipIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No models available</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}