'use client'

import { cn } from '@/lib/utils'
import { ChatMessageItem } from '@/components/chat-message'
import { useChatScroll } from '@/hooks/use-chat-scroll'
import {
  type ChatMessage,
  useRealtimeChat,
} from '@/hooks/use-realtime-chat'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Send, Users, TrendingUp, Brain } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSentiment } from '@/hooks/use-sentiment-analysis'

interface RealtimeChatProps {
  roomName: string
  username: string
  onMessage?: (messages: ChatMessage[]) => void
  messages?: ChatMessage[]
  onSendMessageRef?: (sendMessage: (content: string) => void) => void
}

/**
 * Realtime chat component with sentiment analysis
 */
export const RealtimeChat = ({
  roomName,
  username,
  onMessage,
  messages: initialMessages = [],
  onSendMessageRef,
}: RealtimeChatProps) => {
  const { containerRef, scrollToBottom } = useChatScroll()
  
  // Use sentiment hook
  const { 
    sentimentData, 
    updateMessagesData, 
    getDominantEmotion, 
    getSentimentTrend,
    getSentimentScore 
  } = useSentiment()

  const {
    messages: realtimeMessages,
    sendMessage,
    isConnected,
  } = useRealtimeChat({
    roomName,
    username,
  })
  const [newMessage, setNewMessage] = useState('')

  // Local mode detection (set NEXT_PUBLIC_NODE_ENV=local in .env.local for local dev)
  const isLocal = process.env.NEXT_PUBLIC_NODE_ENV === 'local'

  // Local message state for local mode
  const [localMessages, setLocalMessages] = useState<ChatMessage[]>([])

  // Merge messages: use localMessages in local mode, otherwise merge initial and realtime
  const allMessages = useMemo(() => {
    if (isLocal) return localMessages
    const mergedMessages = [...initialMessages, ...realtimeMessages]
    // Remove duplicates based on message id
    const uniqueMessages = mergedMessages.filter(
      (message, index, self) => index === self.findIndex((m) => m.id === message.id)
    )
    // Sort by creation date
    const sortedMessages = uniqueMessages.sort((a, b) => a.createdAt.localeCompare(b.createdAt))

    return sortedMessages
  }, [isLocal, localMessages, initialMessages, realtimeMessages])

  // In local mode, update localMessages when parent messages prop changes
  useEffect(() => {
    if (isLocal && initialMessages.length > 0) {
      setLocalMessages(initialMessages)
    }
  }, [isLocal, initialMessages])

  // Update sentiment analysis ONLY when messages actually change
  // This is separate from the message sending flow
  useEffect(() => {
    if (allMessages.length > 0) {
      updateMessagesData(allMessages)
    }
  }, [allMessages.length, allMessages[allMessages.length - 1]?.id]) // Only when count or last message ID changes

  // Calculate room statistics
  const roomStats = useMemo(() => {
    const uniqueUsers = new Set(allMessages.map(msg => msg.user.name))
    const totalMessages = allMessages.length
    
    const dominantEmotion = getDominantEmotion()
    const sentimentTrend = getSentimentTrend()
    const dominantScore = getSentimentScore(dominantEmotion)
    
    return {
      userCount: uniqueUsers.size,
      totalMessages,
      dominantEmotion,
      sentimentTrend,
      dominantScore
    }
  }, [allMessages, getDominantEmotion, getSentimentTrend, getSentimentScore])

  // Call parent onMessage callback when messages change
  useEffect(() => {
    if (onMessage) {
      onMessage(allMessages)
    }
  }, [allMessages, onMessage])

  // Auto-scroll when messages change
  useEffect(() => {
    scrollToBottom()
  }, [allMessages, scrollToBottom])

  // Expose sendMessage to parent via ref/callback
  useEffect(() => {
    if (onSendMessageRef) {
      if (isLocal) {
        // Local sendMessage: add to localMessages
        onSendMessageRef((content: string) => {
          const newMsg: ChatMessage = {
            id: Date.now().toString(),
            content,
            user: { name: username },
            createdAt: new Date().toISOString(),
          }
          setLocalMessages((msgs) => [...msgs, newMsg])
        })
      } else {
        onSendMessageRef(sendMessage)
      }
    }
  }, [onSendMessageRef, sendMessage, isLocal, username])

  // Handle sending messages - keep this simple and don't interfere with realtime flow
  const handleSendMessage = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      if (!newMessage.trim() || (!isConnected && !isLocal)) return
      if (isLocal) {
        const newMsg: ChatMessage = {
          id: Date.now().toString(),
          content: newMessage,
          user: { name: username },
          createdAt: new Date().toISOString(),
        }
        setLocalMessages((msgs) => [...msgs, newMsg])
        setNewMessage('')
      } else {
        sendMessage(newMessage)
        setNewMessage('')
      }
    },
    [newMessage, isConnected, isLocal, sendMessage, username]
  )

  // Get emotion display info
  const getEmotionInfo = (emotion: string) => {
    const emotionMap = {
      joy: { label: 'Joyful', color: 'text-yellow-600', emoji: '😊' },
      love: { label: 'Loving', color: 'text-pink-600', emoji: '💖' },
      sadness: { label: 'Sad', color: 'text-blue-600', emoji: '😢' },
      anger: { label: 'Angry', color: 'text-red-600', emoji: '😠' },
      fear: { label: 'Fearful', color: 'text-purple-600', emoji: '😰' },
      unknown: { label: 'Neutral', color: 'text-gray-600', emoji: '🤔' }
    }
    return emotionMap[emotion as keyof typeof emotionMap] || emotionMap.unknown
  }

  const emotionInfo = getEmotionInfo(roomStats.dominantEmotion)

  return (
    <div className="flex flex-col h-full w-full bg-gradient-to-br from-gray-50 to-blue-50/30 text-foreground antialiased">
      {/* Enhanced Header with Room Stats */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200/50 p-4 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-gray-500" />
              <span className="text-sm font-medium">{roomStats.userCount} users</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-gray-500" />
              <span className="text-sm">{roomStats.totalMessages} messages</span>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500">Room mood:</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-sm">{emotionInfo.emoji}</span>
              <span className={`text-sm font-medium ${emotionInfo.color}`}>
                {emotionInfo.label}
              </span>
              <span className="text-xs text-gray-500">
                ({Math.round(roomStats.dominantScore * 100)}%)
              </span>
            </div>
            {roomStats.sentimentTrend !== 'stable' && (
              <div className={`text-xs px-2 py-1 rounded-full ${
                roomStats.sentimentTrend === 'improving' 
                  ? 'bg-green-100 text-green-700' 
                  : 'bg-orange-100 text-orange-700'
              }`}>
                {roomStats.sentimentTrend}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div ref={containerRef} className="flex-1 overflow-y-auto p-4 space-y-2">
        {allMessages.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-gray-400 mb-2">
              <Users className="w-8 h-8 mx-auto mb-2 opacity-50" />
            </div>
            <div className="text-sm text-gray-500">
              No messages yet. Start the conversation!
            </div>
            <div className="text-xs text-gray-400 mt-2">
              AI sentiment analysis will begin once messages are sent
            </div>
          </div>
        ) : (
          <div className="space-y-1">
            {allMessages.map((message, index) => {
              const prevMessage = index > 0 ? allMessages[index - 1] : null
              const showHeader = !prevMessage || prevMessage.user.name !== message.user.name

              return (
                <div
                  key={message.id}
                  className="animate-in fade-in slide-in-from-bottom-4 duration-300"
                >
                  <ChatMessageItem
                    message={message}
                    isOwnMessage={message.user.name === username}
                    showHeader={showHeader}
                  />
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Enhanced Input Form */}
      <form onSubmit={handleSendMessage} className="bg-white/80 backdrop-blur-sm border-t border-gray-200/50 p-4 shadow-sm">
        <div className="flex w-full gap-3">
          <div className="flex-1 relative">
            <Input
              className={cn(
                'rounded-full bg-white/90 border-gray-200/50 text-sm transition-all duration-300',
                'focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400',
                'placeholder:text-gray-400',
                (isConnected || isLocal) && newMessage.trim() ? 'pr-12' : ''
              )}
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder={isConnected || isLocal ? "Type a message..." : "Connecting..."}
              disabled={!(isConnected || isLocal)}
            />
            
            {/* Connection status indicator */}
            <div className={`absolute right-3 top-1/2 transform -translate-y-1/2 w-2 h-2 rounded-full ${
              (isConnected || isLocal) ? 'bg-green-400' : 'bg-red-400'
            }`} />
          </div>
          {(isConnected || isLocal) && newMessage.trim() && (
            <Button
              className="aspect-square rounded-full bg-gradient-to-br from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 shadow-md hover:shadow-lg transition-all duration-200 animate-in fade-in slide-in-from-right-4"
              type="submit"
              disabled={!(isConnected || isLocal)}
            >
              <Send className="size-4" />
            </Button>
          )}
        </div>
        
        {/* Status indicator area */}
        <div className="mt-2 h-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {!isConnected && !isLocal && (
              <span className="text-xs text-gray-400 flex items-center gap-1">
                <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse"></div>
                Connecting to {roomName}...
              </span>
            )}
          </div>
          
          {(isConnected || isLocal) && allMessages.length > 0 && (
            <div className="text-xs text-gray-400 flex items-center gap-1">
              <Brain className="w-3 h-3" />
              AI analyzing conversation sentiment
            </div>
          )}
        </div>
      </form>
    </div>
  )
}
