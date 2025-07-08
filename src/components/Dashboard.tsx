import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  CpuChipIcon,
  CircleStackIcon,
  ClockIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { useLexOS } from '../context/LexOSContext'

export default function Dashboard() {
  const { state } = useLexOS()
  const [performanceData, setPerformanceData] = useState<any[]>([])
  const [memoryStats, setMemoryStats] = useState<any>(null)
  const [recentTasks, setRecentTasks] = useState<any[]>([])

  useEffect(() => {
    fetchDashboardData()
    const interval = setInterval(fetchDashboardData, 30000) // Update every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchDashboardData = async () => {
    try {
      // Fetch performance metrics
      const perfResponse = await fetch('/api/performance')
      if (perfResponse.ok) {
        const perfData = await perfResponse.json()
        setPerformanceData(prev => [...prev.slice(-20), {
          timestamp: new Date().toLocaleTimeString(),
          cpu: state.systemStatus?.cpu_percent || 0,
          memory: state.systemStatus?.memory_percent || 0,
          gpu: state.systemStatus?.gpus?.[0]?.load * 100 || 0
        }])
      }

      // Fetch memory stats
      const memResponse = await fetch('/api/memory/stats')
      if (memResponse.ok) {
        const memData = await memResponse.json()
        setMemoryStats(memData)
      }

      // Fetch recent tasks (mock data for now)
      setRecentTasks([
        { id: 1, type: 'chat', status: 'completed', duration: 1.2, model: 'llama3.2:3b' },
        { id: 2, type: 'vision', status: 'completed', duration: 3.5, model: 'llava:7b' },
        { id: 3, type: 'coding', status: 'running', duration: 0, model: 'qwen2.5-coder:32b' },
        { id: 4, type: 'reasoning', status: 'completed', duration: 2.8, model: 'deepseek-r1:7b' },
      ])
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 dark:text-green-400'
      case 'running': return 'text-blue-600 dark:text-blue-400'
      case 'failed': return 'text-red-600 dark:text-red-400'
      default: return 'text-gray-600 dark:text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return CheckCircleIcon
      case 'running': return ClockIcon
      case 'failed': return ExclamationTriangleIcon
      default: return ClockIcon
    }
  }

  const pieData = [
    { name: 'Chat', value: 45, color: '#3b82f6' },
    { name: 'Coding', value: 25, color: '#10b981' },
    { name: 'Vision', value: 20, color: '#f59e0b' },
    { name: 'Reasoning', value: 10, color: '#ef4444' },
  ]

  return (
    <div className="p-6 space-y-6 bg-gray-50 dark:bg-dark-900 min-h-full">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-400">System overview and performance metrics</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Models</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {state.activeModels.length}
              </p>
            </div>
            <CpuChipIcon className="w-8 h-8 text-primary-500" />
          </div>
          <div className="mt-2">
            <span className="text-sm text-green-600 dark:text-green-400">
              {state.activeModels.join(', ') || 'None loaded'}
            </span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Memories</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {memoryStats?.total_memories || 0}
              </p>
            </div>
            <CircleStackIcon className="w-8 h-8 text-green-500" />
          </div>
          <div className="mt-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {memoryStats?.working_memory_size || 0} in working memory
            </span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">CPU Usage</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {state.systemStatus?.cpu_percent?.toFixed(1) || 0}%
              </p>
            </div>
            <ChartBarIcon className="w-8 h-8 text-yellow-500" />
          </div>
          <div className="mt-2">
            <div className="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
              <div
                className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${state.systemStatus?.cpu_percent || 0}%` }}
              />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Memory Usage</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {state.systemStatus?.memory_percent?.toFixed(1) || 0}%
              </p>
            </div>
            <CircleStackIcon className="w-8 h-8 text-purple-500" />
          </div>
          <div className="mt-2">
            <div className="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
              <div
                className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${state.systemStatus?.memory_percent || 0}%` }}
              />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            System Performance
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
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
              <Line type="monotone" dataKey="cpu" stroke="#f59e0b" strokeWidth={2} name="CPU %" />
              <Line type="monotone" dataKey="memory" stroke="#8b5cf6" strokeWidth={2} name="Memory %" />
              <Line type="monotone" dataKey="gpu" stroke="#10b981" strokeWidth={2} name="GPU %" />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Task Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white dark:bg-dark-800 rounded-lg p-6 border border-gray-200 dark:border-dark-700"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Task Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Recent Tasks */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-white dark:bg-dark-800 rounded-lg border border-gray-200 dark:border-dark-700"
      >
        <div className="p-6 border-b border-gray-200 dark:border-dark-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Tasks</h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {recentTasks.map((task) => {
              const StatusIcon = getStatusIcon(task.status)
              return (
                <div key={task.id} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <StatusIcon className={`w-5 h-5 ${getStatusColor(task.status)}`} />
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white capitalize">
                        {task.type} Task
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Model: {task.model}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`text-sm font-medium ${getStatusColor(task.status)}`}>
                      {task.status}
                    </p>
                    {task.duration > 0 && (
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {task.duration}s
                      </p>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </motion.div>
    </div>
  )
}