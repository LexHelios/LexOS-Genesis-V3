import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Cog6ToothIcon,
  BellIcon,
  ShieldCheckIcon,
  DocumentArrowDownIcon,
  TrashIcon,
  CloudArrowUpIcon,
  ServerIcon
} from '@heroicons/react/24/outline'

interface Settings {
  performance: {
    max_concurrent_tasks: number
    max_memory_gb: number
    gpu_memory_fraction: number
  }
  models: {
    default_chat_model: string
    default_code_model: string
    default_vision_model: string
    default_reasoning_model: string
  }
  memory: {
    consolidation_interval: number
    max_memories: number
    working_memory_size: number
  }
  features: {
    enable_learning: boolean
    enable_consciousness: boolean
    enable_web_interface: boolean
    enable_cli: boolean
  }
  alerts: {
    enable_alerts: boolean
    cpu_threshold: number
    memory_threshold: number
    gpu_threshold: number
  }
}

export default function Settings() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [activeTab, setActiveTab] = useState('performance')

  useEffect(() => {
    fetchSettings()
  }, [])

  const fetchSettings = async () => {
    try {
      const response = await fetch('/api/config')
      if (response.ok) {
        const config = await response.json()
        setSettings(config)
      }
    } catch (error) {
      console.error('Failed to fetch settings:', error)
    } finally {
      setLoading(false)
    }
  }

  const saveSettings = async () => {
    if (!settings) return
    
    setSaving(true)
    try {
      const response = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      })
      
      if (response.ok) {
        // Show success message
        console.log('Settings saved successfully')
      }
    } catch (error) {
      console.error('Failed to save settings:', error)
    } finally {
      setSaving(false)
    }
  }

  const updateSetting = (section: keyof Settings, key: string, value: any) => {
    if (!settings) return
    
    setSettings(prev => ({
      ...prev!,
      [section]: {
        ...prev![section],
        [key]: value
      }
    }))
  }

  const createBackup = async () => {
    try {
      const response = await fetch('/api/backup', { method: 'POST' })
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `lexos-backup-${new Date().toISOString().split('T')[0]}.tar.gz`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (error) {
      console.error('Failed to create backup:', error)
    }
  }

  const clearMemories = async () => {
    if (confirm('Are you sure you want to clear all memories? This action cannot be undone.')) {
      try {
        const response = await fetch('/api/memory/clear', { method: 'POST' })
        if (response.ok) {
          console.log('Memories cleared successfully')
        }
      } catch (error) {
        console.error('Failed to clear memories:', error)
      }
    }
  }

  const tabs = [
    { id: 'performance', label: 'Performance', icon: Cog6ToothIcon },
    { id: 'models', label: 'Models', icon: ServerIcon },
    { id: 'memory', label: 'Memory', icon: DocumentArrowDownIcon },
    { id: 'features', label: 'Features', icon: ShieldCheckIcon },
    { id: 'alerts', label: 'Alerts', icon: BellIcon },
    { id: 'backup', label: 'Backup', icon: CloudArrowUpIcon },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    )
  }

  if (!settings) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-gray-500 dark:text-gray-400">Failed to load settings</p>
      </div>
    )
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-dark-900 min-h-full">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Configure LexOS system preferences and behavior
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar */}
        <div className="lg:w-64">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-700'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1">
          <div className="bg-white dark:bg-dark-800 rounded-lg border border-gray-200 dark:border-dark-700 p-6">
            {/* Performance Settings */}
            {activeTab === 'performance' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Performance Settings</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Max Concurrent Tasks
                    </label>
                    <input
                      type="number"
                      value={settings.performance.max_concurrent_tasks}
                      onChange={(e) => updateSetting('performance', 'max_concurrent_tasks', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Max Memory (GB)
                    </label>
                    <input
                      type="number"
                      value={settings.performance.max_memory_gb}
                      onChange={(e) => updateSetting('performance', 'max_memory_gb', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      GPU Memory Fraction
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={settings.performance.gpu_memory_fraction}
                      onChange={(e) => updateSetting('performance', 'gpu_memory_fraction', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {/* Model Settings */}
            {activeTab === 'models' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Default Models</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Chat Model
                    </label>
                    <input
                      type="text"
                      value={settings.models.default_chat_model}
                      onChange={(e) => updateSetting('models', 'default_chat_model', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Code Model
                    </label>
                    <input
                      type="text"
                      value={settings.models.default_code_model}
                      onChange={(e) => updateSetting('models', 'default_code_model', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Vision Model
                    </label>
                    <input
                      type="text"
                      value={settings.models.default_vision_model}
                      onChange={(e) => updateSetting('models', 'default_vision_model', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Reasoning Model
                    </label>
                    <input
                      type="text"
                      value={settings.models.default_reasoning_model}
                      onChange={(e) => updateSetting('models', 'default_reasoning_model', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {/* Memory Settings */}
            {activeTab === 'memory' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Memory Settings</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Consolidation Interval (seconds)
                    </label>
                    <input
                      type="number"
                      value={settings.memory.consolidation_interval}
                      onChange={(e) => updateSetting('memory', 'consolidation_interval', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Max Memories
                    </label>
                    <input
                      type="number"
                      value={settings.memory.max_memories}
                      onChange={(e) => updateSetting('memory', 'max_memories', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Working Memory Size
                    </label>
                    <input
                      type="number"
                      value={settings.memory.working_memory_size}
                      onChange={(e) => updateSetting('memory', 'working_memory_size', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {/* Feature Settings */}
            {activeTab === 'features' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Feature Settings</h2>
                
                <div className="space-y-4">
                  {Object.entries(settings.features).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between">
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </label>
                      <button
                        onClick={() => updateSetting('features', key, !value)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                          value ? 'bg-primary-500' : 'bg-gray-300 dark:bg-dark-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            value ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Alert Settings */}
            {activeTab === 'alerts' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Alert Settings</h2>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Enable Alerts
                    </label>
                    <button
                      onClick={() => updateSetting('alerts', 'enable_alerts', !settings.alerts.enable_alerts)}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                        settings.alerts.enable_alerts ? 'bg-primary-500' : 'bg-gray-300 dark:bg-dark-600'
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          settings.alerts.enable_alerts ? 'translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        CPU Threshold (%)
                      </label>
                      <input
                        type="number"
                        value={settings.alerts.cpu_threshold}
                        onChange={(e) => updateSetting('alerts', 'cpu_threshold', parseInt(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Memory Threshold (%)
                      </label>
                      <input
                        type="number"
                        value={settings.alerts.memory_threshold}
                        onChange={(e) => updateSetting('alerts', 'memory_threshold', parseInt(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        GPU Threshold (%)
                      </label>
                      <input
                        type="number"
                        value={settings.alerts.gpu_threshold}
                        onChange={(e) => updateSetting('alerts', 'gpu_threshold', parseInt(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-dark-600 rounded-lg bg-white dark:bg-dark-700 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Backup Settings */}
            {activeTab === 'backup' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Backup & Maintenance</h2>
                
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <h3 className="font-medium text-blue-900 dark:text-blue-300 mb-2">Create Backup</h3>
                    <p className="text-sm text-blue-700 dark:text-blue-400 mb-3">
                      Download a complete backup of your LexOS data including memories, configurations, and logs.
                    </p>
                    <button
                      onClick={createBackup}
                      className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      <CloudArrowUpIcon className="w-4 h-4" />
                      <span>Create Backup</span>
                    </button>
                  </div>
                  
                  <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                    <h3 className="font-medium text-red-900 dark:text-red-300 mb-2">Clear All Memories</h3>
                    <p className="text-sm text-red-700 dark:text-red-400 mb-3">
                      Permanently delete all stored memories. This action cannot be undone.
                    </p>
                    <button
                      onClick={clearMemories}
                      className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                    >
                      <TrashIcon className="w-4 h-4" />
                      <span>Clear Memories</span>
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Save Button */}
            <div className="mt-8 pt-6 border-t border-gray-200 dark:border-dark-700">
              <button
                onClick={saveSettings}
                disabled={saving}
                className="px-6 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 transition-colors"
              >
                {saving ? 'Saving...' : 'Save Settings'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}