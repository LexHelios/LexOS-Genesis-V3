import React, { useState } from 'react'
import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  UserIcon,
  CpuChipIcon,
  EyeIcon,
  DownloadIcon,
  PlayIcon,
  PauseIcon,
  DocumentIcon,
  PhotoIcon,
  VideoCameraIcon,
  SpeakerWaveIcon
} from '@heroicons/react/24/outline'

interface ChatMessage {
  id: string
  type: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  model?: string
  attachments?: Array<{
    type: 'image' | 'file' | 'video' | 'audio'
    url: string
    name: string
    size?: number
  }>
  metadata?: Record<string, any>
}

interface MessageBubbleProps {
  message: ChatMessage
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const [imageExpanded, setImageExpanded] = useState<string | null>(null)
  const [videoPlaying, setVideoPlaying] = useState<string | null>(null)
  const [audioPlaying, setAudioPlaying] = useState<string | null>(null)

  const isUser = message.type === 'user'
  const isSystem = message.type === 'system'

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  const downloadFile = (url: string, name: string) => {
    const a = document.createElement('a')
    a.href = url
    a.download = name
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const renderAttachment = (attachment: any, index: number) => {
    const { type, url, name, size } = attachment

    switch (type) {
      case 'image':
        return (
          <div key={index} className="relative group">
            <motion.img
              src={url}
              alt={name}
              className={`rounded-lg cursor-pointer transition-all duration-200 ${
                imageExpanded === url ? 'max-w-full' : 'max-w-xs hover:scale-105'
              }`}
              onClick={() => setImageExpanded(imageExpanded === url ? null : url)}
              whileHover={{ scale: imageExpanded === url ? 1 : 1.05 }}
            />
            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  downloadFile(url, name)
                }}
                className="p-1.5 bg-black/50 text-white rounded-lg hover:bg-black/70"
              >
                <DownloadIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        )

      case 'video':
        return (
          <div key={index} className="relative max-w-md">
            <video
              src={url}
              controls
              className="w-full rounded-lg"
              onPlay={() => setVideoPlaying(url)}
              onPause={() => setVideoPlaying(null)}
            >
              Your browser does not support the video tag.
            </video>
            <div className="mt-2 flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
              <span>{name}</span>
              <button
                onClick={() => downloadFile(url, name)}
                className="p-1 hover:bg-gray-100 dark:hover:bg-dark-700 rounded"
              >
                <DownloadIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        )

      case 'audio':
        return (
          <div key={index} className="flex items-center space-x-3 bg-gray-100 dark:bg-dark-700 rounded-lg p-3 max-w-sm">
            <SpeakerWaveIcon className="w-6 h-6 text-gray-500" />
            <div className="flex-1">
              <audio
                src={url}
                controls
                className="w-full"
                onPlay={() => setAudioPlaying(url)}
                onPause={() => setAudioPlaying(null)}
              >
                Your browser does not support the audio tag.
              </audio>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{name}</p>
            </div>
            <button
              onClick={() => downloadFile(url, name)}
              className="p-1 hover:bg-gray-200 dark:hover:bg-dark-600 rounded"
            >
              <DownloadIcon className="w-4 h-4" />
            </button>
          </div>
        )

      default:
        return (
          <div key={index} className="flex items-center space-x-3 bg-gray-100 dark:bg-dark-700 rounded-lg p-3 max-w-sm">
            <DocumentIcon className="w-6 h-6 text-gray-500" />
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{name}</p>
              {size && (
                <p className="text-sm text-gray-500">
                  {(size / 1024 / 1024).toFixed(2)} MB
                </p>
              )}
            </div>
            <div className="flex space-x-1">
              <button
                onClick={() => window.open(url, '_blank')}
                className="p-1 hover:bg-gray-200 dark:hover:bg-dark-600 rounded"
              >
                <EyeIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => downloadFile(url, name)}
                className="p-1 hover:bg-gray-200 dark:hover:bg-dark-600 rounded"
              >
                <DownloadIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        )
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} ${
        isSystem ? 'justify-center' : ''
      }`}
    >
      <div className={`flex space-x-3 max-w-4xl ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        {/* Avatar */}
        {!isSystem && (
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
            isUser 
              ? 'bg-primary-500 text-white' 
              : 'bg-gray-200 dark:bg-dark-700 text-gray-600 dark:text-gray-300'
          }`}>
            {isUser ? (
              <UserIcon className="w-5 h-5" />
            ) : (
              <CpuChipIcon className="w-5 h-5" />
            )}
          </div>
        )}

        {/* Message Content */}
        <div className={`flex flex-col space-y-2 ${isUser ? 'items-end' : 'items-start'}`}>
          {/* Message Bubble */}
          <div className={`rounded-2xl px-4 py-3 max-w-2xl ${
            isSystem
              ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200 text-center'
              : isUser
              ? 'bg-primary-500 text-white'
              : 'bg-white dark:bg-dark-700 border border-gray-200 dark:border-dark-600'
          }`}>
            {/* Model info for assistant messages */}
            {!isUser && !isSystem && message.model && (
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-2 flex items-center space-x-1">
                <CpuChipIcon className="w-3 h-3" />
                <span>{message.model}</span>
              </div>
            )}

            {/* Message content */}
            <div className={`prose prose-sm max-w-none ${
              isUser 
                ? 'prose-invert' 
                : 'prose-gray dark:prose-invert'
            }`}>
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '')
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={oneDark}
                        language={match[1]}
                        PreTag="div"
                        className="rounded-lg"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    )
                  }
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {/* Attachments */}
            {message.attachments && message.attachments.length > 0 && (
              <div className="mt-3 space-y-2">
                {message.attachments.map((attachment, index) => 
                  renderAttachment(attachment, index)
                )}
              </div>
            )}
          </div>

          {/* Timestamp */}
          <div className={`text-xs text-gray-500 dark:text-gray-400 ${
            isUser ? 'text-right' : 'text-left'
          }`}>
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    </motion.div>
  )
}