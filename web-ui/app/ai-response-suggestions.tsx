import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Sparkles, Copy, Send, RefreshCw, ChevronUp, ChevronDown, Loader2 } from 'lucide-react';
import { useSentiment } from '@/hooks/use-sentiment-analysis';

export interface ChatMessage {
  id: string;
  content: string;
  user: {
    name: string;
  };
  createdAt: string;
}

interface AISuggestion {
  id: string;
  content: string;
  tone: 'professional' | 'friendly' | 'empathetic';
  confidence: number;
}

interface AIResponseSuggestionsProps {
  messages: ChatMessage[];
  sentiments?: any; // Legacy prop
  onSendMessage: (message: string) => void;
  isExpanded?: boolean;
  username: string;
}

const AIResponseSuggestions: React.FC<AIResponseSuggestionsProps> = ({
  username,
  messages,
  onSendMessage,
  isExpanded: initialExpanded = true
}) => {
  const [suggestions, setSuggestions] = useState<AISuggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(initialExpanded);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [waitingForSentiment, setWaitingForSentiment] = useState(false);

  // Use sentiment context
  const { sentimentData, getDominantEmotion, getSentimentTrend, isPositiveSentiment } = useSentiment();

  // Track when sentiment analysis is complete
  const lastProcessedMessageCount = useRef(0);
  const lastSentimentUpdateTime = useRef(0);
  const requestInProgress = useRef(false);

  // Detect when sentiment analysis is complete
  const isSentimentAnalysisComplete = useMemo(() => {
    // Check if we have sentiment data for recent messages
    const recentMessages = messages.slice(-5);
    const analyzedCount = recentMessages.filter(msg => 
      sentimentData.messageSentiments.has(msg.id)
    ).length;
    
    // Consider complete if we have analysis for most recent messages
    return recentMessages.length === 0 || analyzedCount >= Math.min(3, recentMessages.length);
  }, [messages, sentimentData.messageSentiments]);

  // Track sentiment data changes
  const sentimentDataSignature = useMemo(() => {
    return {
      messageCount: sentimentData.messageSentiments.size,
      lastEmotion: sentimentData.overallSentiment.emotion_last_message,
      timestamp: Date.now()
    };
  }, [sentimentData.messageSentiments.size, sentimentData.overallSentiment.emotion_last_message]);

  // Update sentiment tracking when data changes
  useEffect(() => {
    if (sentimentDataSignature.messageCount > 0) {
      lastSentimentUpdateTime.current = sentimentDataSignature.timestamp;
    }
  }, [sentimentDataSignature]);

  const fetchSuggestions = useCallback(async (forceRefresh = false) => {
    // Don't fetch if no messages or already in progress
    if (messages.length === 0 || (requestInProgress.current && !forceRefresh)) {
      return;
    }

    // Don't fetch if we haven't processed new messages
    if (!forceRefresh && messages.length <= lastProcessedMessageCount.current) {
      return;
    }

    // Don't fetch if sentiment analysis is not complete yet
    if (!forceRefresh && !isSentimentAnalysisComplete) {
      setWaitingForSentiment(true);
      return;
    }

    requestInProgress.current = true;
    setLoading(true);
    setError(null);
    setWaitingForSentiment(false);

    try {
      const sentimentPayload = {
        overallSentiment: sentimentData.overallSentiment,
        dominantEmotion: getDominantEmotion(),
        trend: getSentimentTrend(),
        isPositive: isPositiveSentiment(),
        messageSentiments: Array.from(sentimentData.messageSentiments.entries()).slice(-5)
      };

      const response = await fetch('/api/llm/suggestions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username,
          chatHistory: {
            chatId: "",
            messages: messages.slice(-10), // Only send last 10 messages
            timestamp: new Date().toISOString()
          },
          sentiment: sentimentPayload,
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate responses');
      }

      const data = await response.json();
      setSuggestions(data.data || []);
      lastProcessedMessageCount.current = messages.length;
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Failed to fetch suggestions:', err);
    } finally {
      setLoading(false);
      requestInProgress.current = false;
    }
  }, [username, messages, sentimentData, getDominantEmotion, getSentimentTrend, isPositiveSentiment, isSentimentAnalysisComplete]);

  // Trigger suggestions when sentiment analysis is complete
  useEffect(() => {
    // Only proceed if we have new messages
    if (messages.length <= lastProcessedMessageCount.current) {
      return;
    }

    // If sentiment analysis is complete, fetch suggestions
    if (isSentimentAnalysisComplete && !requestInProgress.current) {
      const timeoutId = setTimeout(() => {
        fetchSuggestions();
      }, 500); // Small delay to ensure sentiment data is fully processed

      return () => clearTimeout(timeoutId);
    }
    
    // If sentiment analysis is not complete, wait for it
    if (!isSentimentAnalysisComplete) {
      setWaitingForSentiment(true);
    }
  }, [isSentimentAnalysisComplete, messages.length, fetchSuggestions]);

  // Handle manual refresh
  const handleRefresh = useCallback(() => {
    fetchSuggestions(true);
  }, [fetchSuggestions]);

  const handleCopy = useCallback(async (suggestion: AISuggestion) => {
    try {
      await navigator.clipboard.writeText(suggestion.content);
      setCopiedId(suggestion.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, []);

  const handleSend = useCallback((suggestion: AISuggestion) => {
    onSendMessage(suggestion.content);
    setSelectedId(suggestion.id);
    setTimeout(() => setSelectedId(null), 1000);
  }, [onSendMessage]);

  const getToneColor = (tone: string) => {
    switch (tone) {
      case 'professional': return 'bg-blue-100 text-blue-800';
      case 'friendly': return 'bg-green-100 text-green-800';
      case 'empathetic': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getToneIcon = (tone: string) => {
    switch (tone) {
      case 'professional': return '💼';
      case 'friendly': return '😊';
      case 'empathetic': return '🤗';
      default: return '💭';
    }
  };

  if (messages.length === 0) {
    return null;
  }

  return (
    <div className="bg-white border-t border-gray-200 shadow-sm">
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-purple-600" />
            <h3 className="text-sm font-medium text-gray-800">AI Response Suggestions</h3>
            {(loading || waitingForSentiment) && (
              <Loader2 className="w-3 h-3 animate-spin text-gray-400" />
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handleRefresh}
              disabled={loading || waitingForSentiment}
              className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors disabled:opacity-50"
              title="Refresh suggestions"
            >
              <RefreshCw className={`w-3 h-3 ${(loading || waitingForSentiment) ? 'animate-spin' : ''}`} />
            </button>
            
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors"
              title={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
            </button>
          </div>
        </div>

        {isExpanded && (
          <div className="space-y-2">
            {error ? (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-600">{error}</p>
                <button
                  onClick={handleRefresh}
                  className="mt-2 text-xs text-red-700 hover:text-red-800 underline"
                >
                  Try again
                </button>
              </div>
            ) : waitingForSentiment ? (
              <div className="p-4 text-center">
                <div className="flex items-center justify-center mb-2">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-500 mr-2" />
                  <span className="text-sm text-blue-600">Analyzing sentiment...</span>
                </div>
                <p className="text-xs text-gray-500">
                  Waiting for sentiment analysis to complete before generating suggestions
                </p>
              </div>
            ) : suggestions.length === 0 && !loading ? (
              <div className="p-4 text-center">
                <p className="text-sm text-gray-500">Start chatting to get AI suggestions!</p>
              </div>
            ) : (
              <div className="space-y-2">
                {suggestions.map((suggestion) => (
                  <div
                    key={suggestion.id}
                    className={`p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-all duration-200 ${
                      selectedId === suggestion.id ? 'ring-2 ring-blue-500 bg-blue-50' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-2 py-1 rounded-full ${getToneColor(suggestion.tone)}`}>
                          {getToneIcon(suggestion.tone)} {suggestion.tone}
                        </span>
                        <span className="text-xs text-gray-500">
                          {Math.round(suggestion.confidence * 100)}% confidence
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleCopy(suggestion)}
                          className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors"
                          title="Copy to clipboard"
                        >
                          {copiedId === suggestion.id ? (
                            <span className="text-xs text-green-600">✓</span>
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                        </button>
                        
                        <button
                          onClick={() => handleSend(suggestion)}
                          className="p-1.5 text-purple-600 hover:text-purple-700 transition-colors"
                          title="Send this message"
                        >
                          <Send className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-700 leading-relaxed">
                      {suggestion.content}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {loading && !waitingForSentiment && (
              <div className="flex items-center justify-center p-4">
                <Loader2 className="w-4 h-4 animate-spin text-gray-400 mr-2" />
                <span className="text-sm text-gray-500">Generating suggestions...</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AIResponseSuggestions;
