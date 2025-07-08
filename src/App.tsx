import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './components/Sidebar'
import ChatInterface from './components/ChatInterface'
import Dashboard from './components/Dashboard'
import ModelManager from './components/ModelManager'
import MemoryExplorer from './components/MemoryExplorer'
import SystemMonitor from './components/SystemMonitor'
import Settings from './components/Settings'
import { useWebSocket } from './hooks/useWebSocket'
import { useTheme } from './hooks/useTheme'
import { LexOSProvider } from './context/LexOSContext'

export type ViewType = 'chat' | 'dashboard' | 'models' | 'memory' | 'monitor' | 'settings'

function AppContent() {
  const [currentView, setCurrentView] = useState<ViewType>('chat')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const { theme, toggleTheme } = useTheme()
  const { isConnected, systemStatus } = useWebSocket()

  useEffect(() => {
    // Apply theme to document
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme])

  const renderView = () => {
    switch (currentView) {
      case 'chat':
        return <ChatInterface />
      case 'dashboard':
        return <Dashboard />
      case 'models':
        return <ModelManager />
      case 'memory':
        return <MemoryExplorer />
      case 'monitor':
        return <SystemMonitor />
      case 'settings':
        return <Settings />
      default:
        return <ChatInterface />
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-dark-900 text-gray-900 dark:text-gray-100">
      <Sidebar
        currentView={currentView}
        onViewChange={setCurrentView}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        isConnected={isConnected}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      
      <main className={`flex-1 flex flex-col transition-all duration-300 ${
        sidebarCollapsed ? 'ml-16' : 'ml-64'
      }`}>
        {/* Status Bar */}
        <div className="h-8 bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 flex items-center justify-between px-4 text-xs">
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-1 ${
              isConnected ? 'text-green-600' : 'text-red-600'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`} />
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            {systemStatus && (
              <div className="text-gray-600 dark:text-gray-400">
                CPU: {systemStatus.cpu_percent?.toFixed(1)}% | 
                Memory: {systemStatus.memory_percent?.toFixed(1)}% |
                Active Models: {systemStatus.active_models?.length || 0}
              </div>
            )}
          </div>
          <div className="text-gray-500 dark:text-gray-400">
            LexOS v3.0 - Personal AI Operating System
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentView}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              {renderView()}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}

function App() {
  return (
    <LexOSProvider>
      <AppContent />
    </LexOSProvider>
  )
}

export default App