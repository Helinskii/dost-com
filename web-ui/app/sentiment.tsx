import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BarChart3, Brain, TrendingUp, Smile, Frown, Meh, Zap, CloudRain, Flame, LucideIcon } from 'lucide-react';

// Type definitions
export interface ChatMessage {
  id: string;
  content: string;
  user: {
    name: string;
  };
  createdAt: string;
}

interface SentimentValues {
  positive: number;
  negative: number;
  neutral: number;
  excited: number;
  sad: number;
  angry: number;
}

interface SentimentConfig {
  color: string;
  bgColor: string;
  icon: LucideIcon;
  label: string;
}

interface SentimentSidebarProps {
  chatId: string;
  messages?: ChatMessage[];
  onSentimentsUpdate: (sentiments:SentimentValues) => void;
}

interface ProgressBarProps {
  value: number;
  sentimentKey: keyof SentimentValues;
  sentimentConfig: Record<keyof SentimentValues, SentimentConfig>;
}

interface AnalyzeSentimentRequest {
  chatId: string;
  messages: ChatMessage[];
  timestamp: string;
}

interface AnalyzeSentimentResponse {
  success: boolean;
  sentiments: SentimentValues;
  metadata: {
    totalMessages: number;
    analyzedAt: string;
    confidence: number;
  };
}

const ProgressBar: React.FC<ProgressBarProps> = ({ value, sentimentKey, sentimentConfig }) => {
  const config = sentimentConfig[sentimentKey];
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
          {value}%
        </span>
      </div>
      
      <div className="relative h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-1000 ease-out"
          style={{
            width: `${value}%`,
            background: `linear-gradient(90deg, #EF4444 0%, #F59E0B 25%, #FCD34D 50%, #84CC16 75%, #10B981 100%)`,
            opacity: 0.9
          }}
        >
          <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
        </div>
      </div>
    </div>
  );
};

const SentimentSidebar: React.FC<SentimentSidebarProps> = ({ chatId, messages = [], onSentimentsUpdate}) => {
  const [sentiments, setSentiments] = useState<SentimentValues>({
    positive: 0,
    negative: 0,
    neutral: 0,
    excited: 0,
    sad: 0,
    angry: 0
  });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const isAnalyzingRef = useRef<boolean>(false);
  const lastAnalyzedCountRef = useRef<number>(0);

  // Sentiment configuration with colors and icons
  const sentimentConfig: Record<keyof SentimentValues, SentimentConfig> = {
    positive: {
      color: '#10B981',
      bgColor: 'bg-emerald-500/10',
      icon: Smile,
      label: 'Positive'
    },
    negative: {
      color: '#EF4444',
      bgColor: 'bg-red-500/10',
      icon: Frown,
      label: 'Negative'
    },
    neutral: {
      color: '#6B7280',
      bgColor: 'bg-gray-500/10',
      icon: Meh,
      label: 'Neutral'
    },
    excited: {
      color: '#F59E0B',
      bgColor: 'bg-amber-500/10',
      icon: Zap,
      label: 'Excited'
    },
    sad: {
      color: '#3B82F6',
      bgColor: 'bg-blue-500/10',
      icon: CloudRain,
      label: 'Sad'
    },
    angry: {
      color: '#DC2626',
      bgColor: 'bg-red-600/10',
      icon: Flame,
      label: 'Angry'
    }
  };

  // Fetch sentiment analysis from API
  const analyzeSentiments = useCallback(async (): Promise<void> => {
    // Prevent concurrent calls
    if (isAnalyzingRef.current) return;
    
    try {
      isAnalyzingRef.current = true;
      setLoading(true);
      setError(null);

      const requestBody: AnalyzeSentimentRequest = {
        chatId: chatId,
        messages: messages || [],
        timestamp: new Date().toISOString()
      };

      const response = await fetch('/api/analyze-sentiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error('Failed to analyze sentiments');
      }

      const data: AnalyzeSentimentResponse = await response.json();
      
      // Animate the progress bars
      setTimeout(() => {
        onSentimentsUpdate(data.sentiments);
        setSentiments(data.sentiments);
        lastAnalyzedCountRef.current = messages.length;
      }, 100);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      // Set demo values if API fails
      setSentiments({
        positive: 75,
        negative: 15,
        neutral: 45,
        excited: 60,
        sad: 20,
        angry: 10
      });
    } finally {
      setLoading(false);
      isAnalyzingRef.current = false;
    }
  }, [chatId, messages]);

  // Initial load
  useEffect(() => {
    if (chatId) {
      analyzeSentiments();
    }
  }, [chatId, analyzeSentiments]);

  // Update when messages change (with debouncing)
  useEffect(() => {
    if (!chatId || messages.length === 0) return;
    
    // Skip if message count hasn't changed
    if (messages.length === lastAnalyzedCountRef.current) return;

    const timeoutId = setTimeout(() => {
      analyzeSentiments();
    }, 1000);

    return () => clearTimeout(timeoutId);
  }, [messages.length, chatId, analyzeSentiments]);

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
              Real-time chat emotions
            </p>
          </div>
        </div>

        {loading && (
          <div className="space-y-6">
            {(Object.keys(sentimentConfig) as Array<keyof SentimentValues>).map((key) => (
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
            {(Object.entries(sentiments) as Array<[keyof SentimentValues, number]>).map(([key, value]) => (
              <ProgressBar 
                key={key} 
                value={value} 
                sentimentKey={key} 
                sentimentConfig={sentimentConfig}
              />
            ))}
          </div>
        )}

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <p className="text-sm text-red-600 dark:text-red-400">
              Using demo data. API error: {error}
            </p>
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
