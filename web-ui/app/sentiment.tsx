import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BarChart3, Brain, TrendingUp, Smile, Frown, Meh, Zap, CloudRain, Flame, LucideIcon, MessageCircle, Heart } from 'lucide-react';
import { useSentiment } from "@hooks/use-sentiment-analysis";

// Type definitions
export interface ChatMessage {
  id: string;
  content: string;
  user: {
    name: string;
  };
  createdAt: string;
}

interface EmotionConfig {
  color: string;
  bgColor: string;
  icon: LucideIcon;
  label: string;
}

interface SentimentSidebarProps {
  chatId: string;
  messages?: ChatMessage[];
}

interface ProgressBarProps {
  value: number;
  emotion: string;
  config: EmotionConfig;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ value, emotion, config }) => {
  const Icon = config.icon;
  
  return (
    <div className="mb-6 last:mb-0">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={`p-2 rounded-lg ${config.bgColor}`}>
            <Icon 
              size={18} 
              style={{ color: config.color }}
              className="transition-all duration-300"
            />
          </div>
          <span className="font-medium text-gray-700 dark:text-gray-200">
            {config.label}
          </span>
        </div>
        <span className="text-sm font-semibold" style={{ color: config.color }}>
          {Math.round(value * 100)}%
        </span>
      </div>
      
      <div className="relative h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-1000 ease-out"
          style={{
            width: `${value * 100}%`,
            backgroundColor: config.color,
            opacity: 0.9
          }}
        >
          <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
        </div>
      </div>
    </div>
  );
};

const SentimentSidebar: React.FC<SentimentSidebarProps> = ({ 
  chatId, 
  messages = []
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const isAnalyzingRef = useRef<boolean>(false);
  const lastAnalyzedCountRef = useRef<number>(0);

  // Use sentiment context
  const {
    sentimentData,
    updateSentimentFromAPI,
    getDominantEmotion,
    getSentimentTrend,
    isPositiveSentiment
  } = useSentiment();

  // Emotion configuration mapping API emotions to UI
  const emotionConfig: Record<string, EmotionConfig> = {
    joy: {
      color: '#10B981',
      bgColor: 'bg-emerald-500/10',
      icon: Smile,
      label: 'Joy'
    },
    sadness: {
      color: '#3B82F6',
      bgColor: 'bg-blue-500/10',
      icon: CloudRain,
      label: 'Sadness'
    },
    anger: {
      color: '#EF4444',
      bgColor: 'bg-red-500/10',
      icon: Flame,
      label: 'Anger'
    },
    fear: {
      color: '#8B5CF6',
      bgColor: 'bg-purple-500/10',
      icon: Frown,
      label: 'Fear'
    },
    love: {
      color: '#EC4899',
      bgColor: 'bg-pink-500/10',
      icon: Heart,
      label: 'Love'
    },
    unknown: {
      color: '#6B7280',
      bgColor: 'bg-gray-500/10',
      icon: Meh,
      label: 'Neutral'
    }
  };

  // Analyze sentiments via API
  const analyzeSentiments = useCallback(async (): Promise<void> => {
    if (isAnalyzingRef.current || messages.length === 0) return;
    
    try {
      isAnalyzingRef.current = true;
      setLoading(true);
      setError(null);

      const response = await fetch('/api/analyze-sentiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chatId: chatId,
          messages: messages.map(msg => ({
            id: msg.id,
            content: msg.content,
            user: msg.user.name
          })),
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        throw new Error('Failed to analyze sentiments');
      }

      const data = await response.json();
      
      // Update context with API response
      updateSentimentFromAPI(data);
      lastAnalyzedCountRef.current = messages.length;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
    } finally {
      setLoading(false);
      isAnalyzingRef.current = false;
    }
  }, [chatId, messages, updateSentimentFromAPI]);

  // Auto-analyze when messages change
  useEffect(() => {
    if (!chatId || messages.length === 0) return;
    if (messages.length === lastAnalyzedCountRef.current) return;

    const timeoutId = setTimeout(() => {
      analyzeSentiments();
    }, 1000);

    return () => clearTimeout(timeoutId);
  }, [messages.length, chatId, analyzeSentiments]);

  if (messages.length === 0) {
    return (
      <div className="w-80 h-full bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-lg">
        <div className="p-6 flex-1 flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
            <MessageCircle className="w-8 h-8 text-blue-600" />
          </div>
          
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Messages Yet</h3>
          
          <p className="text-sm text-gray-600 mb-6 leading-relaxed">
            Start analyzing sentiment by sending your first message. Our AI will automatically 
            detect emotions and provide insights.
          </p>

          <div className="space-y-3 w-full">
            <div className="flex items-center space-x-3 text-sm text-gray-500">
              <BarChart3 className="w-4 h-4" />
              <span>Real-time sentiment tracking</span>
            </div>
            <div className="flex items-center space-x-3 text-sm text-gray-500">
              <TrendingUp className="w-4 h-4" />
              <span>Emotional trend analysis</span>
            </div>
          </div>

          <div className="mt-8 p-4 bg-blue-50 rounded-lg">
            <p className="text-xs text-blue-700">
              <strong>Tip:</strong> Sentiment analysis works best with complete sentences.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 h-full bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-lg">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl shadow-lg">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-800 dark:text-white">
              Sentiment Analysis
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {getDominantEmotion()} · {getSentimentTrend()}
            </p>
          </div>
        </div>

        {/* Overall sentiment indicator */}
        <div className="mb-6 p-4 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Last Message Emotion
            </span>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
              isPositiveSentiment() ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
            }`}>
              {sentimentData.overallSentiment.emotion_last_message || 'unknown'}
            </span>
          </div>
        </div>

        {loading && (
          <div className="space-y-6">
            {Object.keys(emotionConfig).map((key) => (
              <div key={key} className="animate-pulse">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className="w-10 h-10 bg-gray-300 dark:bg-gray-600 rounded-lg"></div>
                    <div className="w-20 h-4 bg-gray-300 dark:bg-gray-600 rounded"></div>
                  </div>
                  <div className="w-10 h-4 bg-gray-300 dark:bg-gray-600 rounded"></div>
                </div>
                <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded-full"></div>
              </div>
            ))}
          </div>
        )}

        {!loading && !error && (
          <div className="space-y-1">
            {Object.entries(sentimentData.overallSentiment.emotional_scores).map(([emotion, score]) => {
              const config = emotionConfig[emotion];
              if (!config) return null;
              
              return (
                <ProgressBar 
                  key={emotion} 
                  value={score} 
                  emotion={emotion}
                  config={config}
                />
              );
            })}
          </div>
        )}

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <p className="text-sm text-red-600 dark:text-red-400">
              API error: {error}
            </p>
          </div>
        )}

        {/* Message-level sentiments */}
        {sentimentData.messageSentiments.size > 0 && (
          <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
              Recent Messages
            </h3>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {Array.from(sentimentData.messageSentiments.entries()).slice(-3).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center text-xs">
                  <span className="truncate max-w-[180px] text-gray-600">
                    "{typeof value === 'object' ? value.text : key}"
                  </span>
                  <span className={`px-2 py-1 rounded-full ${
                    ['joy', 'love'].includes(typeof value === 'object' ? value.emotion : value) 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {typeof value === 'object' ? value.emotion : value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500 dark:text-gray-400">
              Last updated
            </span>
            <span className="text-gray-700 dark:text-gray-300">
              {new Date().toLocaleTimeString()}
            </span>
          </div>
          
          <button
            onClick={analyzeSentiments}
            disabled={loading || isAnalyzingRef.current}
            className="mt-4 w-full py-2 px-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-medium hover:from-purple-600 hover:to-pink-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <TrendingUp size={16} />
            {loading ? 'Analyzing...' : 'Refresh Analysis'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SentimentSidebar;
