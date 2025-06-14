import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Brain, TrendingUp, MessageCircle, BarChart3, Settings, Zap, ZapOff, RefreshCw, Activity } from 'lucide-react';
import { useSentiment } from '@/hooks/use-sentiment-analysis';

interface ChatMessage {
  id: string;
  content: string;
  user: {
    name: string;
  };
  createdAt: string;
}

interface SentimentSidebarProps {
  chatId: string;
  messages: ChatMessage[];
}

// Emotion configuration with enhanced visuals
const emotionConfig = {
  joy: { 
    icon: '😊', 
    color: 'from-yellow-400 to-orange-500', 
    bgColor: 'bg-yellow-50',
    label: 'Joy',
    description: 'Happiness and positivity'
  },
  love: { 
    icon: '💖', 
    color: 'from-pink-400 to-rose-500', 
    bgColor: 'bg-pink-50',
    label: 'Love',
    description: 'Affection and care'
  },
  sadness: { 
    icon: '😢', 
    color: 'from-blue-400 to-indigo-500', 
    bgColor: 'bg-blue-50',
    label: 'Sadness',
    description: 'Melancholy and sorrow'
  },
  anger: { 
    icon: '😠', 
    color: 'from-red-400 to-red-600', 
    bgColor: 'bg-red-50',
    label: 'Anger',
    description: 'Frustration and irritation'
  },
  fear: { 
    icon: '😰', 
    color: 'from-purple-400 to-violet-500', 
    bgColor: 'bg-purple-50',
    label: 'Fear',
    description: 'Anxiety and concern'
  },
  unknown: { 
    icon: '🤔', 
    color: 'from-gray-400 to-slate-500', 
    bgColor: 'bg-gray-50',
    label: 'Unknown',
    description: 'Neutral or unclear'
  }
};

// Enhanced Progress Bar Component
const ProgressBar: React.FC<{
  value: number;
  emotion: string;
  config: any;
  isActive?: boolean;
}> = ({ value, emotion, config, isActive = false }) => {
  const percentage = Math.round(value * 100);
  
  return (
    <div className={`p-4 rounded-xl border transition-all duration-300 ${
      isActive 
        ? 'border-purple-300 bg-purple-50 shadow-md' 
        : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${config.color} flex items-center justify-center text-white text-lg shadow-sm`}>
            {config.icon}
          </div>
          <div>
            <p className="font-semibold text-gray-800">{config.label}</p>
            <p className="text-xs text-gray-500">{config.description}</p>
          </div>
        </div>
        <div className="text-right">
          <span className="text-lg font-bold text-gray-800">{percentage}%</span>
        </div>
      </div>
      
      <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className={`absolute left-0 top-0 h-full bg-gradient-to-r ${config.color} rounded-full transition-all duration-700 ease-out`}
          style={{ width: `${percentage}%` }}
        />
        {isActive && (
          <div className="absolute inset-0 bg-white/20 animate-pulse rounded-full" />
        )}
      </div>
    </div>
  );
};

// Generate random sentiment values for mock mode
const generateRandomSentiments = () => {
  const emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'unknown'];
  const scores: Record<string, number> = {};
  const values = Array.from({ length: 6 }, () => Math.random());
  const sum = values.reduce((a, b) => a + b, 0);
  
  emotions.forEach((emotion, index) => {
    scores[emotion] = values[index] / sum; // Normalize to sum to 1
  });
  
  const dominantEmotion = emotions[values.indexOf(Math.max(...values))];
  
  return {
    emotion_last_message: dominantEmotion,
    emotional_scores: scores,
    emotion_per_text: []
  };
};

const SentimentSidebar: React.FC<SentimentSidebarProps> = ({ chatId, messages }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isApiEnabled, setIsApiEnabled] = useState(true);
  const [lastAnalyzedCount, setLastAnalyzedCount] = useState(0);
  
  const isAnalyzingRef = useRef(false);
  const randomUpdateInterval = useRef<NodeJS.Timeout | null>(null);

  const { 
    sentimentData, 
    updateSentimentFromAPI, 
    getDominantEmotion, 
    getSentimentTrend, 
    isPositiveSentiment 
  } = useSentiment();

  // Generate random sentiments in mock mode
  const generateRandomSentimentData = useCallback(() => {
    if (!isApiEnabled && messages.length > 0) {
      const randomData = generateRandomSentiments();
      updateSentimentFromAPI(randomData);
    }
  }, [isApiEnabled, messages.length, updateSentimentFromAPI]);

  // API-based sentiment analysis
  const analyzeSentiments = useCallback(async () => {
    if (isAnalyzingRef.current || messages.length === 0) return;

    isAnalyzingRef.current = true;
    setLoading(true);
    setError(null);

    try {
      if (!isApiEnabled) {
        // Mock mode: generate random sentiments
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API delay
        generateRandomSentimentData();
      } else {
        // Real API call
        const response = await fetch('/api/analyze-sentiment', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            chatId,
            messages: messages.slice(-10), // Analyze last 10 messages
            timestamp: new Date().toISOString()
          })
        });

        if (!response.ok) {
          throw new Error(`API Error: ${response.status}`);
        }

        const data = await response.json();
        
        if (!data.success) {
          throw new Error('Sentiment analysis failed');
        }

        updateSentimentFromAPI(data);
      }
      
      setLastAnalyzedCount(messages.length);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      console.error('Sentiment analysis failed:', err);
    } finally {
      setLoading(false);
      isAnalyzingRef.current = false;
    }
  }, [chatId, messages, isApiEnabled, updateSentimentFromAPI, generateRandomSentimentData]);

  // Auto-analyze when messages change
  useEffect(() => {
    if (!chatId || messages.length === 0) return;
    if (messages.length === lastAnalyzedCount) return;

    const timeoutId = setTimeout(() => {
      analyzeSentiments();
    }, 1000);

    return () => clearTimeout(timeoutId);
  }, [messages.length, chatId, analyzeSentiments, lastAnalyzedCount]);

  // Setup random updates in mock mode
  useEffect(() => {
    if (!isApiEnabled && messages.length > 0) {
      randomUpdateInterval.current = setInterval(() => {
        generateRandomSentimentData();
      }, 5000); // Update every 5 seconds in mock mode

      return () => {
        if (randomUpdateInterval.current) {
          clearInterval(randomUpdateInterval.current);
        }
      };
    } else {
      if (randomUpdateInterval.current) {
        clearInterval(randomUpdateInterval.current);
        randomUpdateInterval.current = null;
      }
    }
  }, [isApiEnabled, messages.length, generateRandomSentimentData]);

  // Toggle API mode
  const handleToggleApi = useCallback(() => {
    setIsApiEnabled(prev => {
      const newValue = !prev;
      if (!newValue) {
        // Switching to mock mode
        generateRandomSentimentData();
      }
      return newValue;
    });
  }, [generateRandomSentimentData]);

  // Get the dominant emotion for highlighting
  const dominantEmotion = getDominantEmotion();

  // Empty state
  if (messages.length === 0) {
    return (
      <div className="w-80 h-full bg-gradient-to-br from-gray-50 to-blue-50/30 border-l border-gray-200/60 shadow-lg">
        <div className="p-6 flex-1 flex flex-col items-center justify-center text-center">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 shadow-lg">
            <MessageCircle className="w-10 h-10 text-white" />
          </div>
          
          <h3 className="text-xl font-bold text-gray-800 mb-3">No Messages Yet</h3>
          
          <p className="text-sm text-gray-600 mb-8 leading-relaxed max-w-xs">
            Start analyzing sentiment by sending your first message. Our AI will automatically 
            detect emotions and provide insights.
          </p>

          <div className="space-y-4 w-full max-w-xs">
            <div className="flex items-center space-x-3 text-sm text-gray-500 p-3 bg-white/60 rounded-lg">
              <BarChart3 className="w-5 h-5 text-blue-500" />
              <span>Real-time sentiment tracking</span>
            </div>
            <div className="flex items-center space-x-3 text-sm text-gray-500 p-3 bg-white/60 rounded-lg">
              <TrendingUp className="w-5 h-5 text-green-500" />
              <span>Emotional trend analysis</span>
            </div>
            <div className="flex items-center space-x-3 text-sm text-gray-500 p-3 bg-white/60 rounded-lg">
              <Brain className="w-5 h-5 text-purple-500" />
              <span>AI-powered insights</span>
            </div>
          </div>

          {/* API Toggle in empty state */}
          <div className="mt-8 w-full max-w-xs">
            <div className="flex items-center justify-between p-3 bg-white/80 rounded-lg border border-gray-200">
              <div className="flex items-center gap-2">
                {isApiEnabled ? (
                  <Zap className="w-4 h-4 text-yellow-500" />
                ) : (
                  <ZapOff className="w-4 h-4 text-gray-400" />
                )}
                <span className="text-sm font-medium text-gray-700">
                  {isApiEnabled ? 'Live API' : 'Demo Mode'}
                </span>
              </div>
              <button
                onClick={handleToggleApi}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  isApiEnabled ? 'bg-purple-600' : 'bg-gray-300'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    isApiEnabled ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>

          <div className="mt-6 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-xs text-blue-700">
              <strong>Tip:</strong> Sentiment analysis works best with complete sentences and expressive language.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 h-full bg-gradient-to-br from-gray-50 to-blue-50/30 border-l border-gray-200/60 shadow-lg flex flex-col">
      <div className="flex-1 p-6 overflow-y-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <div className="relative">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl shadow-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            {loading && (
              <div className="absolute -inset-1 rounded-xl bg-purple-200 animate-ping opacity-30"></div>
            )}
            {!isApiEnabled && (
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-orange-500 rounded-full flex items-center justify-center">
                <span className="text-xs text-white font-bold">M</span>
              </div>
            )}
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-800">
              Sentiment Analysis
            </h2>
            <p className="text-sm text-gray-600">
              {loading ? 'Analyzing...' : `${dominantEmotion} · ${getSentimentTrend()}`}
            </p>
          </div>
        </div>

        {/* Status Card */}
        <div className="mb-6 p-4 rounded-xl bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">
              Last Message Emotion
            </span>
            <div className="flex items-center gap-2">
              {!isApiEnabled && (
                <span className="text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded-full font-medium">
                  DEMO
                </span>
              )}
              {loading && <Activity className="w-4 h-4 text-purple-500 animate-pulse" />}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-2xl">
              {emotionConfig[sentimentData.overallSentiment.emotion_last_message || 'unknown']?.icon}
            </span>
            <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
              isPositiveSentiment() 
                ? 'bg-green-100 text-green-800' 
                : 'bg-gray-100 text-gray-800'
            }`}>
              {sentimentData.overallSentiment.emotion_last_message || 'unknown'}
            </span>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="space-y-4 mb-6">
            {Object.keys(emotionConfig).map((key) => (
              <div key={key} className="animate-pulse p-4 rounded-xl bg-white">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gray-300 rounded-lg"></div>
                    <div>
                      <div className="w-16 h-4 bg-gray-300 rounded mb-1"></div>
                      <div className="w-24 h-3 bg-gray-200 rounded"></div>
                    </div>
                  </div>
                  <div className="w-8 h-6 bg-gray-300 rounded"></div>
                </div>
                <div className="h-3 bg-gray-200 rounded-full"></div>
              </div>
            ))}
          </div>
        )}

        {/* Emotion Progress Bars */}
        {!loading && !error && (
          <div className="space-y-3 mb-6">
            {Object.entries(sentimentData.overallSentiment.emotional_scores).map(([emotion, score]) => {
              const config = emotionConfig[emotion];
              if (!config) return null;
              
              return (
                <ProgressBar 
                  key={emotion} 
                  value={score} 
                  emotion={emotion}
                  config={config}
                  isActive={emotion === dominantEmotion}
                />
              );
            })}
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl">
            <div className="flex items-start gap-3">
              <div className="w-5 h-5 text-red-500 mt-0.5">⚠️</div>
              <div>
                <p className="text-sm font-medium text-red-800">Analysis Failed</p>
                <p className="text-xs text-red-600 mt-1">{error}</p>
                <button
                  onClick={analyzeSentiments}
                  className="mt-2 text-xs font-medium text-red-700 hover:text-red-800 underline"
                >
                  Try again
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Recent Messages */}
        {sentimentData.messageSentiments.size > 0 && (
          <div className="pt-6 border-t border-gray-200">
            <h3 className="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2">
              <MessageCircle className="w-4 h-4" />
              Recent Messages
            </h3>
            <div className="space-y-3 max-h-40 overflow-y-auto">
              {Array.from(sentimentData.messageSentiments.entries()).slice(-3).map(([key, value]) => (
                <div key={key} className="p-3 bg-white rounded-lg border border-gray-200 hover:border-gray-300 transition-colors">
                  <div className="flex justify-between items-start gap-2">
                    <p className="text-xs text-gray-600 leading-relaxed truncate flex-1">
                      "{typeof value === 'object' ? value.text : key}"
                    </p>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium whitespace-nowrap ${
                      ['joy', 'love'].includes(typeof value === 'object' ? value.emotion : value) 
                        ? 'bg-green-100 text-green-700' 
                        : 'bg-gray-100 text-gray-700'
                    }`}>
                      {typeof value === 'object' ? value.emotion : value}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="mt-6 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div className="p-3 bg-white rounded-lg border border-gray-200">
              <div className="text-2xl font-bold text-purple-600">{messages.length}</div>
              <div className="text-xs text-gray-500">Messages</div>
            </div>
            <div className="p-3 bg-white rounded-lg border border-gray-200">
              <div className="text-2xl font-bold text-blue-600">
                {Math.round((sentimentData.overallSentiment.emotional_scores[dominantEmotion] || 0) * 100)}%
              </div>
              <div className="text-xs text-gray-500">Confidence</div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="p-6 bg-white/80 backdrop-blur border-t border-gray-200">
        {/* API Toggle */}
        <div className="mb-4">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${isApiEnabled ? 'bg-yellow-100' : 'bg-gray-100'}`}>
                {isApiEnabled ? (
                  <Zap className="w-4 h-4 text-yellow-600" />
                ) : (
                  <ZapOff className="w-4 h-4 text-gray-500" />
                )}
              </div>
              <div>
                <p className="text-sm font-medium text-gray-800">
                  {isApiEnabled ? 'Live API Analysis' : 'Demo Mode'}
                </p>
                <p className="text-xs text-gray-500">
                  {isApiEnabled ? 'Real sentiment analysis' : 'Random sentiment values'}
                </p>
              </div>
            </div>
            <button
              onClick={handleToggleApi}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 ${
                isApiEnabled ? 'bg-purple-600' : 'bg-gray-300'
              }`}
              aria-label={`Switch to ${isApiEnabled ? 'demo' : 'live'} mode`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  isApiEnabled ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>

        {/* Refresh Button */}
        <div className="flex items-center justify-between text-sm mb-4">
          <span className="text-gray-500">
            Last updated: {new Date().toLocaleTimeString()}
          </span>
        </div>
        
        <button
          onClick={analyzeSentiments}
          disabled={loading || isAnalyzingRef.current}
          className="w-full py-3 px-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg hover:shadow-xl"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          {loading ? 'Analyzing...' : isApiEnabled ? 'Refresh Analysis' : 'Generate Random'}
        </button>
      </div>
    </div>
  );
};

export default SentimentSidebar;
