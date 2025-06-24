import React, { useState } from 'react'
import { cn } from '@/lib/utils'
import type { ChatMessage } from '@/hooks/use-realtime-chat'
import { Heart, Smile, Frown, Zap, ShieldAlert, HelpCircle } from 'lucide-react'
import { useSentiment } from '@/hooks/use-sentiment-analysis'

interface ChatMessageItemProps {
  message: ChatMessage
  isOwnMessage: boolean
  showHeader: boolean
}

// Emotion type definition
type EmotionType = 'sadness' | 'joy' | 'love' | 'anger' | 'fear' | 'surprise';

// Sentiment Badge Component for the 6 specified emotions
const SentimentBadge = ({ 
  emotion = 'surprise', 
  score = 0.5, 
  size = 'sm',
  type = 'message',
  tooltipSide = 'right' // 'left' or 'right'
}: {
  emotion: EmotionType;
  score: number;
  size?: 'xs' | 'sm' | 'md';
  type?: 'message' | 'user';
  tooltipSide?: 'left' | 'right';
}) => {
  const [isHovered, setIsHovered] = useState(false)

  const emotionConfig = {
    joy: {
      icon: Smile,
      color: 'from-yellow-400 to-orange-500',
      bgColor: 'bg-yellow-50',
      textColor: 'text-yellow-700',
      label: 'Joy',
      emoji: '😊'
    },
    love: {
      icon: Heart,
      color: 'from-pink-500 to-rose-500',
      bgColor: 'bg-pink-50',
      textColor: 'text-pink-700',
      label: 'Love',
      emoji: '💖'
    },
    sadness: {
      icon: Frown,
      color: 'from-blue-500 to-indigo-500',
      bgColor: 'bg-blue-50',
      textColor: 'text-blue-700',
      label: 'Sadness',
      emoji: '😢'
    },
    anger: {
      icon: Zap,
      color: 'from-red-500 to-red-600',
      bgColor: 'bg-red-50',
      textColor: 'text-red-700',
      label: 'Anger',
      emoji: '😠'
    },
    fear: {
      icon: ShieldAlert,
      color: 'from-purple-500 to-violet-500',
      bgColor: 'bg-purple-50',
      textColor: 'text-purple-700',
      label: 'Fear',
      emoji: '😰'
    },
    surprise: {
      icon: HelpCircle,
      color: 'from-gray-500 to-slate-500',
      bgColor: 'bg-gray-50',
      textColor: 'text-gray-700',
      label: 'Surprise',
      emoji: '🤔'
    }
  }

  const sizeConfig = {
    xs: { container: 'w-4 h-4', icon: 'w-2 h-2', tooltip: 'text-xs px-2 py-1' },
    sm: { container: 'w-5 h-5', icon: 'w-3 h-3', tooltip: 'text-xs px-2 py-1' },
    md: { container: 'w-6 h-6', icon: 'w-3 h-3', tooltip: 'text-sm px-3 py-2' }
  }

  const config = emotionConfig[emotion]
  const sizes = sizeConfig[size]
  const IconComponent = config.icon

  return (
    <div className="relative inline-block overflow-visible z-50">
      <div
        className={`
          ${sizes.container}
          bg-gradient-to-br ${config.color}
          rounded-full
          flex items-center justify-center
          shadow-sm
          cursor-pointer
          transition-all duration-300 ease-out
          hover:scale-110 hover:shadow-md
          ${isHovered ? 'ring-1 ring-white ring-opacity-50' : ''}
          ${type === 'user' ? 'ring-2 ring-white' : ''}
        `}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{ zIndex: 100 }}
      >
        <IconComponent 
          className={`${sizes.icon} text-white drop-shadow-sm`}
          strokeWidth={2.5}
        />
        {/* Special effect for love emotion */}
        {emotion === 'love' && (
          <div className="absolute inset-0 rounded-full bg-pink-400 animate-ping opacity-20"></div>
        )}
      </div>

      {/* Tooltip */}
      {isHovered && (
        <div className={
          `absolute ${type === 'message' ? 'bottom-full' : 'top-full'} ${tooltipSide === 'left' ? 'left-0' : 'right-0'}
          ${type === 'message' ? 'mb-2' : 'mt-2'}
          ${config.bgColor} ${config.textColor}
          ${sizes.tooltip}
          rounded-lg
          shadow-lg
          whitespace-pre-line min-w-fit max-w-xs break-words
          z-[9999]
          border border-gray-200
          animate-in fade-in-0 zoom-in-95 duration-200`
        }>
          <div className="flex items-center gap-2">
            <span>{config.emoji}</span>
            <span className="font-medium">
              {type === 'message' ? 'Message' : 'User'} {config.label}
            </span>
            <span className="text-xs opacity-75">
              ({Math.round(score * 100)}%)
            </span>
          </div>
          {/* Tooltip arrow */}
          <div className={`
            absolute ${type === 'message' ? 'top-full' : 'bottom-full'} ${tooltipSide === 'left' ? 'left-2' : 'right-2'}
            w-0 h-0
            border-l-4 border-r-4 ${type === 'message' ? 'border-t-4' : 'border-b-4'}
            border-l-transparent border-r-transparent
            ${type === 'message' 
              ? config.bgColor.replace('bg-', 'border-t-') 
              : config.bgColor.replace('bg-', 'border-b-')
            }
          `}></div>
        </div>
      )}
    </div>
  )
}

export const ChatMessageItem = ({ 
  message, 
  isOwnMessage, 
  showHeader
}: ChatMessageItemProps) => {
  const { getMessageSentiment } = useSentiment();
  // Get sentiment data from the hook
  const messageSentiment = getMessageSentiment(message.id);

  return (
    <div className={`flex mt-3 ${isOwnMessage ? 'justify-end' : 'justify-start'}`}>
      <div
        className={cn('max-w-[75%] w-fit flex flex-col gap-2', {
          'items-end': isOwnMessage,
        })}
        style={{ overflow: 'visible' }}
      >
        {showHeader && (
          <div
            className={cn('flex items-center gap-2 text-xs px-3', {
              'justify-end flex-row-reverse': isOwnMessage,
            })}
            style={{ overflow: 'visible' }}
          >
            <div className="flex items-center gap-2" style={{ overflow: 'visible' }}>
              {/* User Avatar without Sentiment Badge */}
              <div className="relative" style={{ overflow: 'visible' }}>
                <div className="w-6 h-6 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white text-xs font-semibold">
                  {message.user.name.charAt(0).toUpperCase()}
                </div>
                {/* Sentiment badge removed from user avatar */}
              </div>
              <span className="font-medium text-gray-800">{message.user.name}</span>
            </div>
            <span className="text-foreground/50 text-xs">
              {new Date(message.createdAt).toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: true,
              })}
            </span>
          </div>
        )}
        <div className="relative" style={{ overflow: 'visible' }}>
          <div
            className={cn(
              'py-3 px-4 rounded-2xl text-sm w-fit relative overflow-visible',
              'backdrop-blur-sm border shadow-sm',
              isOwnMessage 
                ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white border-blue-400/20' 
                : 'bg-white/80 text-gray-800 border-gray-200/50'
            )}
            style={{ overflow: 'visible' }}
          >
            {/* Subtle background pattern for own messages */}
            {isOwnMessage && (
              <div className="absolute inset-0 opacity-10">
                <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent"></div>
              </div>
            )}
            <div className="relative z-10">
              {message.content}
            </div>
            {/* Message Sentiment Badge */}
            {messageSentiment && (
              <div className={`absolute -top-2 ${isOwnMessage ? '-left-2' : '-right-2'} z-50`} style={{ pointerEvents: 'auto' }}>
                <SentimentBadge 
                  emotion={messageSentiment.emotion}
                  score={messageSentiment.confidence}
                  size="xs"
                  type="message"
                  tooltipSide={!isOwnMessage ? 'left' : 'right'}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
