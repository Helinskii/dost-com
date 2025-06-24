import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
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
  surprise: { 
    icon: '😮', 
    color: 'from-gray-400 to-slate-500', 
    bgColor: 'bg-gray-50',
    label: 'Surprise',
    description: 'Unexpected or unclear'
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
  // Defensive: fallback config if undefined
  const safeConfig = config || {
    icon: '❓',
    color: 'from-gray-400 to-slate-500',
    bgColor: 'bg-gray-50',
    label: emotion,
    description: 'No data'
  };
  return (
    <div className={`p-4 rounded-xl border transition-all duration-300 ${
      isActive 
        ? 'border-purple-300 bg-purple-50 shadow-md' 
        : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
    }`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${safeConfig.color} flex items-center justify-center text-white text-lg shadow-sm`}>
            {safeConfig.icon}
          </div>
          <div>
            <p className="font-semibold text-gray-800">{safeConfig.label}</p>
            <p className="text-xs text-gray-500">{safeConfig.description}</p>
          </div>
        </div>
        <div className="text-right">
          <span className="text-lg font-bold text-gray-800">{percentage}%</span>
        </div>
      </div>
      
      <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className={`absolute left-0 top-0 h-full bg-gradient-to-r ${safeConfig.color} rounded-full transition-all duration-700 ease-out`}
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
  const emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];
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

const TABS = [
  { key: 'breakdown', label: 'Emotional Breakdown' },
  { key: 'group', label: 'User Sentiment' },
];

const sentimentOrder = [
  'sadness',
  'joy',
  'love',
  'anger',
  'fear',
  'surprise',
] as const;

const vibrantBarColors = [
  'from-blue-400 to-indigo-500',
  'from-yellow-400 to-orange-500',
  'from-pink-400 to-rose-500',
  'from-red-400 to-red-600',
  'from-purple-400 to-violet-500',
  'from-gray-400 to-slate-500',
];

const SentimentSidebar: React.FC<SentimentSidebarProps> = ({ chatId, messages }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isApiEnabled, setIsApiEnabled] = useState(true);
  const [lastAnalyzedCount, setLastAnalyzedCount] = useState(0);
  const [activeTab, setActiveTab] = useState<'breakdown' | 'group'>('breakdown');
  const [analyzeResponses, setAnalyzeResponses] = useState<any[]>([]); // Store all analyze API responses
  const [analyzeLoading, setAnalyzeLoading] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  
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
    if (!chatId || messages.length === 0 || isAnalyzingRef.current) return;

    // In mock mode, just generate random data
    if (!isApiEnabled) {
      generateRandomSentimentData();
      setLastAnalyzedCount(messages.length);
      return;
    }

    isAnalyzingRef.current = true;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('https://lynx-divine-lovely.ngrok-free.app/predict', {
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
          })).slice(-1),
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

  // Analyze group/user sentiment via new API
  const analyzeGroupSentiment = useCallback(async () => {
    if (!chatId || messages.length === 0) return;
    setAnalyzeLoading(true);
    setAnalyzeError(null);
    try {
      const response = await fetch('https://lynx-divine-lovely.ngrok-free.app/analyse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chatId,
          messages: messages.slice(-1), // Only last message
          timestamp: new Date().toISOString(),
        }),
      });
      if (!response.ok) throw new Error(`Analyze API failed: ${response.status}`);
      const data = await response.json();
      setAnalyzeResponses((prev) => [...prev, data]);
    } catch (err) {
      setAnalyzeError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setAnalyzeLoading(false);
    }
  }, [chatId, messages]);

  // Call analyzeGroupSentiment on message change
  useEffect(() => {
    if (!chatId || messages.length === 0) return;
    analyzeGroupSentiment();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length, chatId]);

  // Auto-analyze when messages change
  useEffect(() => {
    if (!chatId || messages.length === 0) return;
    if (messages.length === lastAnalyzedCount) return;

    const timeoutId = setTimeout(() => {
      analyzeSentiments();
    }, 1000);

    return () => clearTimeout(timeoutId);
  }, [messages.length, chatId, analyzeSentiments, lastAnalyzedCount]);

  // Toggle API mode
  const handleToggleApi = useCallback(() => {
    setIsApiEnabled(prev => !prev);
  }, []);

  // Get the dominant emotion for highlighting
  const dominantEmotion = getDominantEmotion();

  // Get latest group sentiment (mu_10 from __GrOuP__)
  const latestGroup = useMemo(() => {
    if (analyzeResponses.length === 0) return null;
    const last = analyzeResponses[analyzeResponses.length - 1];
    return last['__GrOuP__']?.mu_10 || null;
  }, [analyzeResponses]);

  // Get all usernames from latest response
  const userList = useMemo(() => {
    if (analyzeResponses.length === 0) return [];
    const last = analyzeResponses[analyzeResponses.length - 1];
    return Object.keys(last).filter((k) => k !== '__GrOuP__');
  }, [analyzeResponses]);

  // Get latest user sentiment for selected user
  const latestUserSentiment = useMemo(() => {
    if (!selectedUser || analyzeResponses.length === 0) return null;
    const last = analyzeResponses[analyzeResponses.length - 1];
    return last[selectedUser]?.mu_10 || null;
  }, [selectedUser, analyzeResponses]);

  // Get user sentiment history for line chart
  const userSentimentHistory = useMemo(() => {
    if (!selectedUser) return [];
    return analyzeResponses.map((resp) => resp[selectedUser]?.mu_10 || null).filter(Boolean);
  }, [selectedUser, analyzeResponses]);

  // Get last message sentiment from predict API
  const lastMessageSentiment = useMemo(() => {
    if (!sentimentData?.overallSentiment?.emotional_scores) return null;
    // Map to array in sentimentOrder
    return sentimentOrder.map((emo) => sentimentData.overallSentiment.emotional_scores[emo as keyof typeof sentimentData.overallSentiment.emotional_scores] || 0);
  }, [sentimentData]);

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
        {/* Tab Switcher */}
        <div className="flex gap-2 mb-6">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              className={`px-4 py-2 rounded-xl font-semibold transition-all duration-200 text-sm shadow-sm border
                ${activeTab === tab.key ? 'bg-purple-600 text-white border-purple-600 scale-105' : 'bg-white text-gray-700 border-gray-200 hover:bg-purple-50'}`}
              onClick={() => setActiveTab(tab.key as any)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'breakdown' ? (
          <>
            {/* Group Sentiment Breakdown */}
            <div className="mb-8">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-5 h-5 text-purple-600" />
                <span className="font-semibold text-purple-700">Group Sentiment (Latest)</span>
              </div>
              <div className="space-y-3">
                {latestGroup ? sentimentOrder.map((emo, i) => (
                  <ProgressBar
                    key={emo}
                    value={latestGroup[i]}
                    emotion={emo}
                    config={emotionConfig[emo as keyof typeof emotionConfig]}
                  />
                )) : <div className="text-gray-400 text-sm">No group sentiment yet.</div>}
              </div>
            </div>
            {/* Last Message Sentiment Breakdown */}
            <div className="mb-8">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                <span className="font-semibold text-blue-700">Last Message Sentiment</span>
              </div>
              <div className="space-y-3">
                {lastMessageSentiment ? sentimentOrder.map((emo, i) => (
                  <ProgressBar
                    key={emo}
                    value={lastMessageSentiment[i]}
                    emotion={emo}
                    config={emotionConfig[emo as keyof typeof emotionConfig]}
                  />
                )) : <div className="text-gray-400 text-sm">No sentiment data yet.</div>}
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
                    {latestGroup ? Math.round(Math.max(...latestGroup) * 100) : 0}%
                  </div>
                  <div className="text-xs text-gray-500">Group Confidence</div>
                </div>
              </div>
            </div>
          </>
        ) : (
          // Group/User Sentiment Tab
          <div>
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-5 h-5 text-purple-600" />
              <span className="font-semibold text-purple-700">User Sentiment</span>
              {analyzeLoading && <RefreshCw className="w-4 h-4 animate-spin text-purple-400 ml-2" />}
            </div>
            {/* User Dropdown */}
            <div className="mb-4">
              <label className="block text-xs font-semibold text-gray-600 mb-1">Select User</label>
              <select
                className="w-full px-3 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-purple-400"
                value={selectedUser || ''}
                onChange={e => setSelectedUser(e.target.value)}
              >
                <option value="" disabled>Select a user</option>
                {userList.map(u => <option key={u} value={u}>{u}</option>)}
              </select>
            </div>
            {/* User Sentiment Bar Chart */}
            {selectedUser && latestUserSentiment && (
              <div className="mb-8">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-5 h-5 text-pink-600" />
                  <span className="font-semibold text-pink-700">Latest Sentiment for {selectedUser}</span>
                </div>
                <div className="space-y-3">
                  {sentimentOrder.map((emo, i) => (
                    <ProgressBar
                      key={emo}
                      value={latestUserSentiment[i]}
                      emotion={emo}
                      config={emotionConfig[emo as keyof typeof emotionConfig]}
                    />
                  ))}
                </div>
              </div>
            )}
            {/* User Sentiment Line Graph */}
            {selectedUser && userSentimentHistory.length > 1 && (
              <div className="mb-8">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-5 h-5 text-indigo-600" />
                  <span className="font-semibold text-indigo-700">Sentiment Trend for {selectedUser}</span>
                </div>
                <UserSentimentLineChart history={userSentimentHistory} />
              </div>
            )}
            {analyzeError && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
                {analyzeError}
              </div>
            )}
            {userList.length === 0 && <div className="text-center text-gray-500 py-8">No user sentiment data yet.</div>}
          </div>
        )}
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

// UserSentimentLineChart: simple SVG line chart for each emotion
const UserSentimentLineChart: React.FC<{ history: number[][] }> = ({ history }) => {
  // history: array of [sadness, joy, love, anger, fear, surprise] arrays
  if (!history.length) return null;
  const width = 220;
  const height = 80;
  const pad = 18;
  const n = history.length;
  // For each emotion, build a line
  const lines = sentimentOrder.map((emo, emoIdx) => {
    const points = history.map((h, i) => {
      const x = pad + (width - 2 * pad) * (i / (n - 1 || 1));
      const y = height - pad - (height - 2 * pad) * (h[emoIdx] || 0);
      return `${x},${y}`;
    }).join(' ');
    return { emo, color: vibrantBarColors[emoIdx].split(' ')[0].replace('from-', ''), points };
  });
  return (
    <svg width={width} height={height} className="bg-white rounded-lg border border-gray-200 w-full">
      {/* Axes */}
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#ddd" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#ddd" strokeWidth={1} />
      {/* Lines */}
      {lines.map((l, i) => (
        <polyline
          key={l.emo}
          fill="none"
          stroke={`var(--tw-${l.color})`}
          strokeWidth={2}
          points={l.points}
          style={{ filter: 'drop-shadow(0 1px 2px #0001)' }}
        />
      ))}
      {/* Dots */}
      {lines.map((l, emoIdx) => history.map((h, i) => {
        const x = pad + (width - 2 * pad) * (i / (n - 1 || 1));
        const y = height - pad - (height - 2 * pad) * (h[emoIdx] || 0);
        return <circle key={i} cx={x} cy={y} r={2.5} fill={`var(--tw-${l.color})`} />;
      }))}
      {/* Legend */}
      {lines.map((l, i) => (
        <g key={l.emo}>
          <rect x={pad + i * 32} y={4} width={10} height={10} fill={`var(--tw-${l.color})`} rx={2} />
          <text x={pad + i * 32 + 14} y={13} fontSize={10} fill="#444">{emotionConfig[l.emo as keyof typeof emotionConfig].icon}</text>
        </g>
      ))}
    </svg>
  );
};

export default SentimentSidebar;
