import { useState, useEffect, useCallback, useRef } from 'react';

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
