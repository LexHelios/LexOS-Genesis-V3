import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  CpuChipIcon,
  CircleStackIcon,
  ComputerDesktopIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  FireIcon
} from '@heroicons/react/24/outline'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { useLexOS } from '../context/LexOSContext'

interface PerformanceData {
  timestamp: string
  cpu: number
  memory: number
  gpu: number
  temperature?: number
}

interface GPUInfo {
  id: number
  name: string
  memory_used: number
  memory_total: number
  memory_percent: number
  temperature?: number
  load: number
}

export default function SystemMonitor() {
  const { state } = useLexOS()
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceData[]>([])
  const [alerts, setAlerts] = useState<any[]>([])
  const [processes, setProcesses] = useState<any[]>([])

  useEffect(() => {
    const interval = setInterval(() => {
      if (state.systemStatus) {
        const newDataPoint: PerformanceData = {
          timestamp: new Date().toLocaleTimeString(),
          cpu: state.systemStatus.cpu_percent,
          memory: state.systemStatus.memory_percent,
          gpu: state.systemStatus.gpus?.[0]?.load * 100 || 0,
          temperature: state.systemStatus.gpus?.[0]?.temperature
        }
        
        setPerformanceHistory(prev => [...prev.slice(-29), newDataPoint])
        
        // Check for alerts
        checkAlerts(newDataPoint)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [state.systemStatus])

  const checkAlerts = (data: PerformanceData) => {
    const newAlerts = []
    
    if (data.cpu > 90) {
      newAlerts.push({
        id: Date.now(),
        type: 'warning',
        message: `High CPU usage: ${data.cpu.toFixed(1)}%`,
        timestamp: new Date().toISOString()
      })
    }
    
    if (data.memory > 85) {
      newAlerts.push({
        id: Date.now() + 1,
        type: 'warning',
        message: `High memory usage: ${data.memory.toFixed(1)}%`,
        timestamp: new Date().toISOString()
      })
    }
    
    if (data.temperature && data.temperature > 80) {
      newAlerts.push({
        id: Date.now() + 2,
        type: 'error',
        message: `High GPU temperature: ${data.temperature}°C`,
        timestamp: new Date().toISOString()
      })
    }
    
    if (newAlerts.length > 0) {
      setAlerts(prev => [...newAlerts, ...prev.slice(0, 9)])
    }
  }

  const getStatusColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'text-red-600 dark:text-red-400'
    if (value >= thresholds.warning) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-green-600 dark:text-green-400'
  }

  const getProgressColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'bg-red-500'
    if (value >= thresholds.warning) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400)
    const hours = Math.floor((seconds % 86400) / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`
    if (hours > 0) return `${hours}h ${minutes}m`
    return `${minutes}m`
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-dark-900 min-h-full">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">System Monitor</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Real-time system performance and resource monitoring
        </p>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        {/* CPU */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <CpuChipIcon className="w-6 h-6 text-blue-500" />
              <span className="font-medium text-gray-900 dark:text-white">CPU</span>
            </div>
            <span className={`text-2xl font-bold ${getStatusColor(
              state.systemStatus?.cpu_percent || 0,
              { warning: 70, critical: 90 }
            )}`}>
              {state.systemStatus?.cpu_percent?.toFixed(1) || 0}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(
                state.systemStatus?.cpu_percent || 0,
                { warning: 70, critical: 90 }
              )}`}
              style={{ width: `${state.systemStatus?.cpu_percent || 0}%` }}
            />
          </div>
        </motion.div>

        {/* Memory */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <CircleStackIcon className="w-6 h-6 text-purple-500" />
              <span className="font-medium text-gray-900 dark:text-white">Memory</span>
            </div>
            <span className={`text-2xl font-bold ${getStatusColor(
              state.systemStatus?.memory_percent || 0,
              { warning: 80, critical: 95 }
            )}`}>
              {state.systemStatus?.memory_percent?.toFixed(1) || 0}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(
                state.systemStatus?.memory_percent || 0,
                { warning: 80, critical: 95 }
              )}`}
              style={{ width: `${state.systemStatus?.memory_percent || 0}%` }}
            />
          </div>
          <div className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            {formatBytes((state.systemStatus?.memory_available_gb || 0) * 1024 * 1024 * 1024)} available
          </div>
        </motion.div>

        {/* GPU */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <ComputerDesktopIcon className="w-6 h-6 text-green-500" />
              <span className="font-medium text-gray-900 dark:text-white">GPU</span>
            </div>
            <span className={`text-2xl font-bold ${getStatusColor(
              (state.systemStatus?.gpus?.[0]?.load || 0) * 100,
              { warning: 80, critical: 95 }
            )}`}>
              {((state.systemStatus?.gpus?.[0]?.load || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(
                (state.systemStatus?.gpus?.[0]?.load || 0) * 100,
                { warning: 80, critical: 95 }
              )}`}
              style={{ width: `${(state.systemStatus?.gpus?.[0]?.load || 0) * 100}%` }}
            />
          </div>
          {state.systemStatus?.gpus?.[0] && (
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              {state.systemStatus.gpus[0].name}
            </div>
          )}
        </motion.div>

        {/* Temperature */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <FireIcon className="w-6 h-6 text-orange-500" />
              <span className="font-medium text-gray-900 dark:text-white">Temperature</span>
            </div>
            <span className={`text-2xl font-bold ${getStatusColor(
              state.systemStatus?.gpus?.[0]?.temperature || 0,
              { warning: 70, critical: 85 }
            )}`}>
              {state.systemStatus?.gpus?.[0]?.temperature || '--'}°C
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(
                state.systemStatus?.gpus?.[0]?.temperature || 0,
                { warning: 70, critical: 85 }
              )}`}
              style={{ width: `${Math.min((state.systemStatus?.gpus?.[0]?.temperature || 0) / 100 * 100, 100)}%` }}
            />
          </div>
        </motion.div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Resource Usage Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Resource Usage History
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={performanceHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="timestamp" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1f2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Area type="monotone" dataKey="cpu" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              <Area type="monotone" dataKey="memory" stackId="2" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
              <Area type="monotone" dataKey="gpu" stackId="3" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* GPU Details */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            GPU Information
          </h3>
          <div className="space-y-4">
            {state.systemStatus?.gpus?.map((gpu: GPUInfo) => (
              <div key={gpu.id} className="p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900 dark:text-white">
                    GPU {gpu.id}: {gpu.name}
                  </span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {gpu.temperature}°C
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Memory Usage:</span>
                    <span>{gpu.memory_percent.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-dark-600 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${gpu.memory_percent}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400">
                    <span>{formatBytes(gpu.memory_used * 1024 * 1024)} used</span>
                    <span>{formatBytes(gpu.memory_total * 1024 * 1024)} total</span>
                  </div>
                </div>
              </div>
            )) || (
              <div className="text-center text-gray-500 dark:text-gray-400 py-8">
                No GPU information available
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Alerts and Active Models */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Alerts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-white dark:bg-dark-800 rounded-lg border border-gray-200 dark:border-dark-700"
        >
          <div className="p-6 border-b border-gray-200 dark:border-dark-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">System Alerts</h3>
          </div>
          <div className="p-6">
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {alerts.length > 0 ? (
                alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={`flex items-start space-x-3 p-3 rounded-lg ${
                      alert.type === 'error'
                        ? 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
                        : 'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300'
                    }`}
                  >
                    <ExclamationTriangleIcon className="w-5 h-5 mt-0.5" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">{alert.message}</p>
                      <p className="text-xs opacity-75">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-gray-500 dark:text-gray-400 py-8">
                  <CheckCircleIcon className="w-8 h-8 mx-auto mb-2" />
                  <p>No alerts - system running normally</p>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Active Models */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-white dark:bg-dark-800 rounded-lg border border-gray-200 dark:border-dark-700"
        >
          <div className="p-6 border-b border-gray-200 dark:border-dark-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Active Models</h3>
          </div>
          <div className="p-6">
            <div className="space-y-3">
              {state.activeModels.length > 0 ? (
                state.activeModels.map((model) => (
                  <div
                    key={model}
                    className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded-lg"
                  >
                    <div className="flex items-center space-x-3">
                      <CpuChipIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
                      <span className="font-medium text-gray-900 dark:text-white">{model}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                      <span className="text-sm text-green-600 dark:text-green-400">Active</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-gray-500 dark:text-gray-400 py-8">
                  <CpuChipIcon className="w-8 h-8 mx-auto mb-2" />
                  <p>No models currently loaded</p>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}