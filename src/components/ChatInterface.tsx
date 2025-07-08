import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import {
  PaperAirplaneIcon,
  PaperClipIcon,
  MicrophoneIcon,
  StopIcon,
  PhotoIcon,
  DocumentIcon,
  VideoCameraIcon,
  XMarkIcon,
  EyeIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline'
import { useLexOS } from '../context/LexOSContext'
import { useWebSocket } from '../hooks/useWebSocket'
import MessageBubble from './MessageBubble'
import ModelSelector from './ModelSelector'

interface Attachment {
  id: string
  type: 'image' | 'file' | 'video' | 'audio'
  file: File
  url: string
  name: string
  size: number
}

export default function ChatInterface() {
  const { state, dispatch } = useLexOS()
  const { sendMessage } = useWebSocket()
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [state.chatMessages])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      handleFileUpload(acceptedFiles)
    },
    onDragEnter: () => setIsDragOver(true),
    onDragLeave: () => setIsDragOver(false),
    noClick: true,
    multiple: true,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
      'video/*': ['.mp4', '.webm', '.mov'],
      'audio/*': ['.mp3', '.wav', '.ogg'],
      'application/pdf': ['.pdf'],
      'text/*': ['.txt', '.md', '.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    }
  })

  const handleFileUpload = (files: File[]) => {
    const newAttachments: Attachment[] = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      type: getFileType(file),
      file,
      url: URL.createObjectURL(file),
      name: file.name,
      size: file.size
    }))
    
    setAttachments(prev => [...prev, ...newAttachments])
    setIsDragOver(false)
  }

  const getFileType = (file: File): 'image' | 'file' | 'video' | 'audio' => {
    if (file.type.startsWith('image/')) return 'image'
    if (file.type.startsWith('video/')) return 'video'
    if (file.type.startsWith('audio/')) return 'audio'
    return 'file'
  }

  const removeAttachment = (id: string) => {
    setAttachments(prev => {
      const attachment = prev.find(a => a.id === id)
      if (attachment) {
        URL.revokeObjectURL(attachment.url)
      }
      return prev.filter(a => a.id !== id)
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!input.trim() && attachments.length === 0) return
    if (state.isProcessing) return

    const messageId = Math.random().toString(36).substr(2, 9)
    const userMessage = {
      id: messageId,
      type: 'user' as const,
      content: input,
      timestamp: new Date().toISOString(),
      attachments: attachments.map(att => ({
        type: att.type,
        url: att.url,
        name: att.name,
        size: att.size
      }))
    }

    dispatch({ type: 'ADD_CHAT_MESSAGE', payload: userMessage })
    dispatch({ type: 'SET_PROCESSING', payload: true })

    // Upload files if any
    const uploadedAttachments = []
    for (const attachment of attachments) {
      try {
        const formData = new FormData()
        formData.append('file', attachment.file)
        
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        })
        
        if (response.ok) {
          const result = await response.json()
          uploadedAttachments.push({
            ...attachment,
            url: result.url
          })
        }
      } catch (error) {
        console.error('Failed to upload file:', error)
      }
    }

    // Send message via WebSocket
    sendMessage({
      type: 'chat_message',
      data: {
        content: input,
        model: state.currentModel,
        attachments: uploadedAttachments,
        requires_vision: attachments.some(a => a.type === 'image'),
        complexity: input.length > 500 ? 'complex' : 'medium'
      }
    })

    // Clear input and attachments
    setInput('')
    setAttachments([])
    
    // Clean up object URLs
    attachments.forEach(att => URL.revokeObjectURL(att.url))
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      setIsRecording(true)
      // Implement audio recording logic here
    } catch (error) {
      console.error('Failed to start recording:', error)
    }
  }

  const stopRecording = () => {
    setIsRecording(false)
    // Implement stop recording logic here
  }

  const getAttachmentIcon = (type: string) => {
    switch (type) {
      case 'image': return PhotoIcon
      case 'video': return VideoCameraIcon
      case 'audio': return MicrophoneIcon
      default: return DocumentIcon
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div 
      {...getRootProps()}
      className={`h-full flex flex-col relative ${
        isDragActive || isDragOver ? 'bg-primary-50 dark:bg-primary-900/20' : ''
      }`}
    >
      <input {...getInputProps()} />
      
      {/* Drag Overlay */}
      <AnimatePresence>
        {(isDragActive || isDragOver) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-primary-500/20 backdrop-blur-sm z-50 flex items-center justify-center"
          >
            <div className="text-center">
              <PhotoIcon className="w-16 h-16 mx-auto text-primary-500 mb-4" />
              <p className="text-xl font-semibold text-primary-700 dark:text-primary-300">
                Drop files here to upload
              </p>
              <p className="text-primary-600 dark:text-primary-400 mt-2">
                Images, videos, documents, and more
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div className="bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">Chat Interface</h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Multimodal AI conversation with {state.currentModel}
            </p>
          </div>
          <ModelSelector />
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {state.chatMessages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </AnimatePresence>
        
        {state.isProcessing && (
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
            <span>LexOS is thinking...</span>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Attachments Preview */}
      <AnimatePresence>
        {attachments.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-gray-200 dark:border-dark-700 p-4 bg-gray-50 dark:bg-dark-800"
          >
            <div className="flex flex-wrap gap-2">
              {attachments.map((attachment) => {
                const Icon = getAttachmentIcon(attachment.type)
                return (
                  <motion.div
                    key={attachment.id}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="flex items-center space-x-2 bg-white dark:bg-dark-700 rounded-lg p-2 border border-gray-200 dark:border-dark-600"
                  >
                    {attachment.type === 'image' ? (
                      <img
                        src={attachment.url}
                        alt={attachment.name}
                        className="w-8 h-8 object-cover rounded"
                      />
                    ) : (
                      <Icon className="w-6 h-6 text-gray-500" />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{attachment.name}</p>
                      <p className="text-xs text-gray-500">{formatFileSize(attachment.size)}</p>
                    </div>
                    <button
                      onClick={() => removeAttachment(attachment.id)}
                      className="p-1 hover:bg-gray-100 dark:hover:bg-dark-600 rounded"
                    >
                      <XMarkIcon className="w-4 h-4" />
                    </button>
                  </motion.div>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input Area */}
      <div className="bg-white dark:bg-dark-800 border-t border-gray-200 dark:border-dark-700 p-4">
        <form onSubmit={handleSubmit} className="flex items-end space-x-2">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message... (Shift+Enter for new line)"
              className="w-full resize-none rounded-lg border border-gray-300 dark:border-dark-600 bg-white dark:bg-dark-700 px-4 py-3 pr-12 focus:ring-2 focus:ring-primary-500 focus:border-transparent max-h-32"
              rows={1}
              style={{
                minHeight: '48px',
                height: Math.min(Math.max(48, input.split('\n').length * 24), 128)
              }}
            />
          </div>
          
          <div className="flex space-x-1">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,video/*,audio/*,.pdf,.txt,.md,.csv,.xlsx,.xls"
              onChange={(e) => {
                if (e.target.files) {
                  handleFileUpload(Array.from(e.target.files))
                }
              }}
              className="hidden"
            />
            
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-3 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-dark-700 rounded-lg transition-colors"
            >
              <PaperClipIcon className="w-5 h-5" />
            </button>
            
            <button
              type="button"
              onClick={isRecording ? stopRecording : startRecording}
              className={`p-3 rounded-lg transition-colors ${
                isRecording
                  ? 'text-red-500 hover:text-red-700 bg-red-50 dark:bg-red-900/20'
                  : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-dark-700'
              }`}
            >
              {isRecording ? (
                <StopIcon className="w-5 h-5" />
              ) : (
                <MicrophoneIcon className="w-5 h-5" />
              )}
            </button>
            
            <button
              type="submit"
              disabled={(!input.trim() && attachments.length === 0) || state.isProcessing}
              className="p-3 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <PaperAirplaneIcon className="w-5 h-5" />
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}