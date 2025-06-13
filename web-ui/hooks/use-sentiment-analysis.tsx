import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';

// Type definitions
interface Message {
  id: string;
  content: string;
  timestamp: string;
  userId: string;
}

interface SentimentValues {
  positive: number;
  negative: number;
  neutral: number;
  excited: number;
  sad: number;
  angry: number;
}

interface SentimentMetadata {
  totalMessages: number;
  analyzedAt: string;
  confidence: number;
}

interface AnalyzeSentimentRequest {
  chatId: string;
  messages: Message[];
  timestamp: string;
}

interface AnalyzeSentimentResponse {
  success: boolean;
  sentiments: SentimentValues;
  metadata: SentimentMetadata;
}

interface UseSentimentAnalysisOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  enableCache?: boolean;
  cacheTimeout?: number;
  onError?: (error: Error) => void;
  onSuccess?: (data: AnalyzeSentimentResponse) => void;
}

interface UseSentimentAnalysisReturn {
  sentiments: SentimentValues;
  metadata: SentimentMetadata | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  clearCache: () => void;
  lastUpdated: Date | null;
}

interface CacheEntry {
  data: AnalyzeSentimentResponse;
  timestamp: number;
}

// Cache storage
const sentimentCache = new Map<string, CacheEntry>();

/**
 * Custom hook for sentiment analysis
 * @param chatId - The ID of the chat to analyze
 * @param messages - Array of messages to analyze
 * @param options - Configuration options
 * @returns Sentiment analysis data and control functions
 */
export const useSentimentAnalysis = (
  chatId: string,
  messages: Message[] = [],
  options: UseSentimentAnalysisOptions = {}
): UseSentimentAnalysisReturn => {
  const {
    autoRefresh = false,
    refreshInterval = 30000, // 30 seconds
    enableCache = true,
    cacheTimeout = 60000, // 1 minute
    onError,
    onSuccess
  } = options;

  // State management
  const [sentiments, setSentiments] = useState<SentimentValues>({
    positive: 0,
    negative: 0,
    neutral: 0,
    excited: 0,
    sad: 0,
    angry: 0
  });
  const [metadata, setMetadata] = useState<SentimentMetadata | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Refs for cleanup and preventing memory leaks
  const abortControllerRef = useRef<AbortController | null>(null);
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Generate cache key
  const getCacheKey = useCallback((): string => {
    return `${chatId}-${messages.length}`;
  }, [chatId, messages.length]);

  // Check if cache is valid
  const isCacheValid = useCallback((entry: CacheEntry): boolean => {
    return Date.now() - entry.timestamp < cacheTimeout;
  }, [cacheTimeout]);

  // Clear cache for current chat
  const clearCache = useCallback(() => {
    const cacheKey = getCacheKey();
    sentimentCache.delete(cacheKey);
  }, [getCacheKey]);

  // Main fetch function
  const fetchSentiments = useCallback(async () => {
    // Cancel any ongoing requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller
    abortControllerRef.current = new AbortController();

    try {
      setLoading(true);
      setError(null);

      const cacheKey = getCacheKey();

      // Check cache first
      if (enableCache && sentimentCache.has(cacheKey)) {
        const cachedEntry = sentimentCache.get(cacheKey)!;
        if (isCacheValid(cachedEntry)) {
          setSentiments(cachedEntry.data.sentiments);
          setMetadata(cachedEntry.data.metadata);
          setLastUpdated(new Date(cachedEntry.timestamp));
          setLoading(false);
          return;
        }
      }

      // Prepare request body
      const requestBody: AnalyzeSentimentRequest = {
        chatId,
        messages,
        timestamp: new Date().toISOString()
      };

      // Make API call
      const response = await fetch('/api/analyze-sentiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data: AnalyzeSentimentResponse = await response.json();

      if (!data.success) {
        throw new Error('Sentiment analysis failed');
      }

      // Update cache
      if (enableCache) {
        sentimentCache.set(cacheKey, {
          data,
          timestamp: Date.now()
        });
      }

      // Update state
      setSentiments(data.sentiments);
      setMetadata(data.metadata);
      setLastUpdated(new Date());
      
      // Call success callback
      if (onSuccess) {
        onSuccess(data);
      }
    } catch (err) {
      // Ignore abort errors
      if (err instanceof Error && err.name === 'AbortError') {
        return;
      }

      const error = err instanceof Error ? err : new Error('Unknown error occurred');
      setError(error);
      
      // Set demo/fallback data on error
      setSentiments({
        positive: 75,
        negative: 15,
        neutral: 45,
        excited: 60,
        sad: 20,
        angry: 10
      });
      setMetadata({
        totalMessages: messages.length,
        analyzedAt: new Date().toISOString(),
        confidence: 0.5
      });
      
      // Call error callback
      if (onError) {
        onError(error);
      }
    } finally {
      setLoading(false);
    }
  }, [chatId, messages, enableCache, getCacheKey, isCacheValid, onError, onSuccess]);

  // Refetch function (force refresh, ignoring cache)
  const refetch = useCallback(async () => {
    clearCache();
    await fetchSentiments();
  }, [clearCache, fetchSentiments]);

  // Set up auto-refresh
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      refreshIntervalRef.current = setInterval(() => {
        fetchSentiments();
      }, refreshInterval);

      return () => {
        if (refreshIntervalRef.current) {
          clearInterval(refreshIntervalRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, fetchSentiments]);

  // Initial fetch and refetch on dependencies change
  useEffect(() => {
    fetchSentiments();

    // Cleanup function
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchSentiments]);

  return {
    sentiments,
    metadata,
    loading,
    error,
    refetch,
    clearCache,
    lastUpdated
  };
};

/**
 * Hook for polling sentiment changes
 * Useful for real-time updates with custom logic
 */
export const useSentimentPolling = (
  chatId: string,
  messages: Message[],
  {
    interval = 5000,
    threshold = 10, // Minimum change percentage to trigger callback
    onChange
  }: {
    interval?: number;
    threshold?: number;
    onChange?: (oldSentiments: SentimentValues, newSentiments: SentimentValues) => void;
  } = {}
) => {
  const previousSentimentsRef = useRef<SentimentValues | null>(null);
  
  const { sentiments, ...rest } = useSentimentAnalysis(chatId, messages, {
    autoRefresh: true,
    refreshInterval: interval,
    onSuccess: (data) => {
      if (previousSentimentsRef.current && onChange) {
        // Check if any sentiment changed by more than threshold
        const hasSignificantChange = Object.keys(data.sentiments).some((key) => {
          const sentimentKey = key as keyof SentimentValues;
          const oldValue = previousSentimentsRef.current![sentimentKey];
          const newValue = data.sentiments[sentimentKey];
          return Math.abs(oldValue - newValue) >= threshold;
        });

        if (hasSignificantChange) {
          onChange(previousSentimentsRef.current, data.sentiments);
        }
      }
      previousSentimentsRef.current = data.sentiments;
    }
  });

  return { sentiments, ...rest };
};

/**
 * Hook for batch sentiment analysis
 * Useful for analyzing multiple chats at once
 */
export const useBatchSentimentAnalysis = (
  chatData: Array<{ chatId: string; messages: Message[] }>,
  options: UseSentimentAnalysisOptions = {}
) => {
  const [results, setResults] = useState<Map<string, SentimentValues>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchBatch = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const promises = chatData.map(async ({ chatId, messages }) => {
        const response = await fetch('/api/analyze-sentiment', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chatId,
            messages,
            timestamp: new Date().toISOString()
          })
        });

        const data: AnalyzeSentimentResponse = await response.json();
        return { chatId, sentiments: data.sentiments };
      });

      const batchResults = await Promise.all(promises);
      const resultsMap = new Map<string, SentimentValues>();
      
      batchResults.forEach(({ chatId, sentiments }) => {
        resultsMap.set(chatId, sentiments);
      });

      setResults(resultsMap);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Batch analysis failed'));
    } finally {
      setLoading(false);
    }
  }, [chatData]);

  useEffect(() => {
    fetchBatch();
  }, [fetchBatch]);

  return { results, loading, error, refetch: fetchBatch };
};

// Create context
const SentimentContext = createContext();

// Custom hook to use sentiment context
export const useSentiment = () => {
  const context = useContext(SentimentContext);
  if (!context) {
    throw new Error('useSentiment must be used within a SentimentProvider');
  }
  return context;
};

// Sentiment provider component
export const SentimentProvider = ({ children }) => {
  const [sentimentData, setSentimentData] = useState({
    overallSentiment: {
      emotion_last_message: null,
      emotional_scores: {
        sadness: 0,
        joy: 0,
        love: 0,
        anger: 0,
        fear: 0,
        unknown: 0
      }
    },
    messageSentiments: new Map(), // Store per-message sentiment
    conversationHistory: []
  });

  // Update sentiment from API response
  const updateSentimentFromAPI = (apiResponse) => {
    const { emotion_last_message, emotional_scores, emotion_per_text } = apiResponse;
    
    // Update overall sentiment
    setSentimentData(prev => ({
      ...prev,
      overallSentiment: {
        emotion_last_message,
        emotional_scores
      }
    }));

    // Update per-message sentiment
    if (emotion_per_text) {
      const newMessageSentiments = new Map(prev.messageSentiments);
      
      emotion_per_text.forEach(item => {
        const messageText = Object.keys(item)[0];
        const emotion = item[messageText];
        newMessageSentiments.set(messageText, emotion);
      });

      setSentimentData(prev => ({
        ...prev,
        messageSentiments: newMessageSentiments
      }));
    }
  };

  // Add new message sentiment
  const addMessageSentiment = (messageId, messageText, emotion) => {
    setSentimentData(prev => ({
      ...prev,
      messageSentiments: new Map(prev.messageSentiments).set(messageId, {
        text: messageText,
        emotion,
        timestamp: new Date().toISOString()
      })
    }));
  };

  // Get sentiment for specific message
  const getMessageSentiment = (messageId) => {
    return sentimentData.messageSentiments.get(messageId);
  };

  // Get dominant emotion from scores
  const getDominantEmotion = () => {
    const scores = sentimentData.overallSentiment.emotional_scores;
    return Object.entries(scores).reduce((a, b) => 
      scores[a[0]] > scores[b[0]] ? a : b
    )[0];
  };

  // Calculate sentiment trend over conversation
  const getSentimentTrend = () => {
    const messages = Array.from(sentimentData.messageSentiments.values());
    if (messages.length < 2) return 'stable';

    const recent = messages.slice(-3);
    const older = messages.slice(-6, -3);

    const getPositiveScore = (msgs) => {
      return msgs.reduce((sum, msg) => {
        return sum + (['joy', 'love'].includes(msg.emotion) ? 1 : 0);
      }, 0) / msgs.length;
    };

    const recentPositive = getPositiveScore(recent);
    const olderPositive = getPositiveScore(older);

    if (recentPositive > olderPositive + 0.2) return 'improving';
    if (recentPositive < olderPositive - 0.2) return 'declining';
    return 'stable';
  };

  const contextValue = {
    sentimentData,
    updateSentimentFromAPI,
    addMessageSentiment,
    getMessageSentiment,
    getDominantEmotion,
    getSentimentTrend,
    // Utility functions
    isPositiveSentiment: () => ['joy', 'love'].includes(sentimentData.overallSentiment.emotion_last_message),
    isNegativeSentiment: () => ['sadness', 'anger', 'fear'].includes(sentimentData.overallSentiment.emotion_last_message),
    getSentimentScore: (emotion) => sentimentData.overallSentiment.emotional_scores[emotion] || 0
  };

  return (
    <SentimentContext.Provider value={contextValue}>
      {children}
    </SentimentContext.Provider>
  );
};
