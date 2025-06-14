import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Brain, TrendingUp, BarChart3, Settings, Zap, ZapOff, RefreshCw, Activity } from 'lucide-react';
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

  // Analyze sentiments via API
  const analyzeSentiments = useCallback(async () => {
    if (!chatId || messages.length === 0 || isAnalyzingRef.current || !isApiEnabled) return;

    isAnalyzingRef.current = true;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chatId,
          messages: messages.map(msg => ({
            id: msg.id,
            content: msg.content,
            user: { name: msg.user.name },
            createdAt: msg.createdAt
          })),
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data = await response.json();
      updateSentimentFromAPI(data);
      setLastAnalyzedCount(messages.length);
    } catch (err) {
      const errorMessage = err instanceof Error ? 
        err.message : 'Unknown error occurred';
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
            <Brain className="w-10 h-10 text-white" />
          </div>
          
          <h3 className="text-xl font-bold text-gray-800 mb-3">No Messages Yet</h3>
          
          <p className="text-sm text-gray-600 mb-8 leading-relaxed max-w-xs">
            Start analyzing sentiment by sending your first message. Our AI will automatically 
            detect emotions and provide insights.
          </p>

          <div className="w-full max-w-xs">
            <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200 shadow-sm">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${isApiEnabled ? 'bg-green-100' : 'bg-gray-100'}`}>
                  {isApiEnabled ? (
                    <Zap className="w-4 h-4 text-green-600" />
                  ) : (
                    <ZapOff className="w-4 h-4 text-gray-600" />
                  )}
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-800">
                    {isApiEnabled ? 'API Mode' : 'Demo Mode'}
                  </p>
                  <p className="text-xs text-gray-500">
                    {isApiEnabled ? 'Real-time analysis' : 'Mock data'}
                  </p>
                </div>
              </div>
              <button
                onClick={handleToggleApi}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  isApiEnabled ? 'bg-green-500' : 'bg-gray-300'
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

        {/* Emotional Scores */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Emotional Breakdown
            </h3>
            <button
              onClick={analyzeSentiments}
              disabled={loading}
              className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-all duration-200 disabled:opacity-50"
              title="Refresh analysis"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {Object.entries(sentimentData.overallSentiment.emotional_scores)
              .sort(([,a], [,b]) => b - a)
              .map(([emotion, score]) => (
                <ProgressBar
                  key={emotion}
                  value={score}
                  emotion={emotion}
                  config={emotionConfig[emotion as keyof typeof emotionConfig]}
                  isActive={emotion === dominantEmotion}
                />
              ))}
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl">
            <div className="flex items-start gap-3">
              <div className="w-5 h-5 text-red-500 mt-0.5">⚠️</div>
              <div className="flex-1">
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
                  <ZapOff className="w-4 h-4 text-gray-600" />
                )}
              </div>
              <div>
                <p className="text-sm font-medium text-gray-800">
                  {isApiEnabled ? 'API Analysis' : 'Demo Mode'}
                </p>
                <p className="text-xs text-gray-500">
                  {isApiEnabled ? 'Live sentiment detection' : 'Simulated responses'}
                </p>
              </div>
            </div>
            <button
              onClick={handleToggleApi}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                isApiEnabled ? 'bg-yellow-500' : 'bg-gray-300'
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

        {/* Manual Analysis */}
        <button
          onClick={analyzeSentiments}
          disabled={loading || !isApiEnabled}
          className="w-full py-3 px-4 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Activity className="w-4 h-4 animate-pulse" />
              Analyzing...
            </>
          ) : (
            <>
              <TrendingUp className="w-4 h-4" />
              Analyze Now
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default SentimentSidebar;
