import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  MagnifyingGlassIcon,
  TrashIcon,
  EyeIcon,
  ClockIcon,
  TagIcon,
  CircleStackIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline'
import { useLexOS } from '../context/LexOSContext'

interface Memory {
  id: string
  content: string
  metadata: Record<string, any>
  timestamp: string
  distance?: number
  importance?: number
}

export default function MemoryExplorer() {
  const { state } = useLexOS()
  const [memories, setMemories] = useState<Memory[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null)
  const [loading, setLoading] = useState(false)
  const [memoryStats, setMemoryStats] = useState<any>(null)
  const [searchStrategy, setSearchStrategy] = useState<'semantic' | 'keyword' | 'importance'>('semantic')

  useEffect(() => {
    fetchMemoryStats()
    fetchRecentMemories()
  }, [])

  const fetchMemoryStats = async () => {
    try {
      const response = await fetch('/api/memory/stats')
      if (response.ok) {
        const stats = await response.json()
        setMemoryStats(stats)
      }
    } catch (error) {
      console.error('Failed to fetch memory stats:', error)
    }
  }

  const fetchRecentMemories = async () => {
    try {
      const response = await fetch('/api/memory/recent')
      if (response.ok) {
        const recentMemories = await response.json()
        setMemories(recentMemories)
      }
    } catch (error) {
      console.error('Failed to fetch recent memories:', error)
    }
  }

  const searchMemories = async () => {
    if (!searchQuery.trim()) {
      fetchRecentMemories()
      return
    }

    setLoading(true)
    try {
      const response = await fetch('/api/memory/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          strategy: searchStrategy,
          n_results: 20
        })
      })
      
      if (response.ok) {
        const results = await response.json()
        setMemories(results)
      }
    } catch (error) {
      console.error('Failed to search memories:', error)
    } finally {
      setLoading(false)
    }
  }

  const deleteMemory = async (memoryId: string) => {
    try {
      const response = await fetch(`/api/memory/${memoryId}`, {
        method: 'DELETE'
      })
      
      if (response.ok) {
        setMemories(prev => prev.filter(m => m.id !== memoryId))
        if (selectedMemory?.id === memoryId) {
          setSelectedMemory(null)
        }
        fetchMemoryStats() // Refresh stats
      }
    } catch (error) {
      console.error('Failed to delete memory:', error)
    }
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  const getImportanceColor = (importance: number) => {
    if (importance >= 0.8) return 'text-red-600 dark:text-red-400'
    if (importance >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
    if (importance >= 0.4) return 'text-blue-600 dark:text-blue-400'
    return 'text-gray-600 dark:text-gray-400'
  }

  const getImportanceLabel = (importance: number) => {
    if (importance >= 0.8) return 'Critical'
    if (importance >= 0.6) return 'High'
    if (importance >= 0.4) return 'Medium'
    return 'Low'
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-dark-900 min-h-full">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Memory Explorer</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Browse and search through LexOS memory system
        </p>
      </div>

      {/* Stats */}
      {memoryStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Memories</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {memoryStats.total_memories}
                </p>
              </div>
              <CircleStackIcon className="w-8 h-8 text-primary-500" />
            </div>
          </div>

          <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Working Memory</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {memoryStats.working_memory_size}
                </p>
              </div>
              <ClockIcon className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Graph Nodes</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {memoryStats.graph_nodes}
                </p>
              </div>
              <ChartBarIcon className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Categories</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {memoryStats.semantic_categories?.length || 0}
                </p>
              </div>
              <TagIcon className="w-8 h-8 text-purple-500" />
            </div>
          </div>
        </div>
      )}

      {/* Search */}
      <div className="bg-white dark:bg-dark-800 rounded-lg border border-gray-200 dark:border-dark-700 p-6 mb-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && searchMemories()}
                placeholder="Search memories..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>
          
          <div className="flex space-x-2">
            <select
              value={searchStrategy}
              onChange={(e) => setSearchStrategy(e.target.value as any)}
              className="px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="semantic">Semantic</option>
              <option value="keyword">Keyword</option>
              <option value="importance">Importance</option>
            </select>
            
            <button
              onClick={searchMemories}
              disabled={loading}
              className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 transition-colors"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>
        </div>
      </div>

      {/* Memory List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Memory Cards */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            {searchQuery ? 'Search Results' : 'Recent Memories'}
          </h2>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            <AnimatePresence>
              {memories.map((memory) => (
                <motion.div
                  key={memory.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`bg-white dark:bg-dark-800 rounded-lg border p-4 cursor-pointer transition-all hover:shadow-md ${
                    selectedMemory?.id === memory.id 
                      ? 'border-primary-500 ring-2 ring-primary-200 dark:ring-primary-800' 
                      : 'border-gray-200 dark:border-dark-700'
                  }`}
                  onClick={() => setSelectedMemory(memory)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-gray-900 dark:text-white line-clamp-2">
                        {memory.content.substring(0, 150)}
                        {memory.content.length > 150 && '...'}
                      </p>
                      
                      <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                        <span>{formatTimestamp(memory.timestamp)}</span>
                        
                        {memory.importance !== undefined && (
                          <span className={getImportanceColor(memory.importance)}>
                            {getImportanceLabel(memory.importance)}
                          </span>
                        )}
                        
                        {memory.distance !== undefined && (
                          <span>
                            Relevance: {((1 - memory.distance) * 100).toFixed(0)}%
                          </span>
                        )}
                        
                        {memory.metadata?.task_type && (
                          <span className="px-2 py-0.5 bg-gray-100 dark:bg-dark-700 rounded">
                            {memory.metadata.task_type}
                          </span>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex space-x-1 ml-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setSelectedMemory(memory)
                        }}
                        className="p-1 hover:bg-gray-100 dark:hover:bg-dark-700 rounded"
                      >
                        <EyeIcon className="w-4 h-4" />
                      </button>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          deleteMemory(memory.id)
                        }}
                        className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 text-red-600 dark:text-red-400 rounded"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            
            {memories.length === 0 && !loading && (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                {searchQuery ? 'No memories found' : 'No memories yet'}
              </div>
            )}
          </div>
        </div>

        {/* Memory Detail */}
        <div className="bg-white dark:bg-dark-800 rounded-lg border border-gray-200 dark:border-dark-700">
          {selectedMemory ? (
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Memory Details
                </h3>
                <button
                  onClick={() => deleteMemory(selectedMemory.id)}
                  className="p-2 hover:bg-red-100 dark:hover:bg-red-900/30 text-red-600 dark:text-red-400 rounded-lg"
                >
                  <TrashIcon className="w-5 h-5" />
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Content:</label>
                  <div className="mt-1 p-3 bg-gray-50 dark:bg-dark-700 rounded-lg">
                    <p className="text-sm text-gray-900 dark:text-white whitespace-pre-wrap">
                      {selectedMemory.content}
                    </p>
                  </div>
                </div>
                
                <div>
                  <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Timestamp:</label>
                  <p className="text-sm text-gray-900 dark:text-white">
                    {new Date(selectedMemory.timestamp).toLocaleString()}
                  </p>
                </div>
                
                {selectedMemory.importance !== undefined && (
                  <div>
                    <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Importance:</label>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-200 dark:bg-dark-700 rounded-full h-2">
                        <div
                          className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${selectedMemory.importance * 100}%` }}
                        />
                      </div>
                      <span className={`text-sm ${getImportanceColor(selectedMemory.importance)}`}>
                        {getImportanceLabel(selectedMemory.importance)}
                      </span>
                    </div>
                  </div>
                )}
                
                {selectedMemory.metadata && Object.keys(selectedMemory.metadata).length > 0 && (
                  <div>
                    <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Metadata:</label>
                    <div className="mt-1 p-3 bg-gray-50 dark:bg-dark-700 rounded-lg">
                      <pre className="text-xs text-gray-900 dark:text-white overflow-x-auto">
                        {JSON.stringify(selectedMemory.metadata, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="p-6 text-center text-gray-500 dark:text-gray-400">
              <CircleStackIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Select a memory to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}