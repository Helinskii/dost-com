import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Sparkles, Copy, Send, RefreshCw, ChevronUp, ChevronDown, Loader2, CheckCircle2 } from 'lucide-react';
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
}

interface AIResponseSuggestionsProps {
  roomName: string;
  messages: ChatMessage[];
  sentiments?: any; // Legacy prop
  onSendMessage: (message: string) => void;
  isExpanded?: boolean;
  username: string;
}

const AIResponseSuggestions: React.FC<AIResponseSuggestionsProps> = ({
  roomName,
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

  // Refs for tracking state
  const lastProcessedMessageCount = useRef(0);
  const lastSentimentUpdateTime = useRef(0);
  const requestInProgress = useRef(false);

  // Detect when sentiment analysis is complete
  const isSentimentAnalysisComplete = useMemo(() => {
    // Check if sentiment analysis has been done and is marked as complete
    return sentimentData.isAnalysisComplete && sentimentData.lastAnalysisTimestamp > 0;
  }, [sentimentData.isAnalysisComplete, sentimentData.lastAnalysisTimestamp]);

  // Track sentiment data changes with timestamp
  const sentimentDataSignature = useMemo(() => ({
    messageCount: sentimentData.messageSentiments.size,
    lastEmotion: sentimentData.overallSentiment.emotion_last_message,
    timestamp: sentimentData.lastAnalysisTimestamp,
    isComplete: sentimentData.isAnalysisComplete
  }), [
    sentimentData.messageSentiments.size, 
    sentimentData.overallSentiment.emotion_last_message,
    sentimentData.lastAnalysisTimestamp,
    sentimentData.isAnalysisComplete
  ]);

  // Update sentiment tracking when data changes
  useEffect(() => {
    if (sentimentDataSignature.messageCount > 0 && sentimentDataSignature.isComplete) {
      lastSentimentUpdateTime.current = sentimentDataSignature.timestamp;
    }
  }, [sentimentDataSignature]);

  const fetchSuggestions = useCallback(async (forceRefresh = false) => {
    // Early returns for invalid states
    if (messages.length === 0 || (requestInProgress.current && !forceRefresh)) {
      return;
    }

    if (!forceRefresh && messages.length <= lastProcessedMessageCount.current) {
      return;
    }

    // Start the request
    requestInProgress.current = true;
    setLoading(true);
    setError(null);
    setWaitingForSentiment(false);

    try {
      const response = await fetch('https://lynx-divine-lovely.ngrok-free.app/rag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username,
          chatId: roomName,
          messages: messages.slice(-1),
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to generate responses: ${response.status}`);
      }

      const data = await response.json();
      // Expecting: { Response: [ ... ] }
      const suggestionsArray = Array.isArray(data.Response) ? data.Response : [];
      setSuggestions(
        suggestionsArray.map((content: string, idx: number) => ({
          id: `suggestion-${idx}`,
          content,
        }))
      );
      lastProcessedMessageCount.current = messages.length;
      
    } catch (err) {
      const errorMessage = err instanceof Error ? 
        err.message : 'Unknown error occurred';
      setError(errorMessage);
      console.error('Failed to fetch suggestions:', err);
    } finally {
      setLoading(false);
      requestInProgress.current = false;
    }
  }, [username, messages, sentimentData, getDominantEmotion, getSentimentTrend, isPositiveSentiment, isSentimentAnalysisComplete]);

  // Only trigger suggestions when a new message arrives
  useEffect(() => {
    if (messages.length <= lastProcessedMessageCount.current) {
      return;
    }
    if (!requestInProgress.current) {
      const timeoutId = setTimeout(() => {
        fetchSuggestions();
      }, 500);
      return () => clearTimeout(timeoutId);
    }
  }, [messages.length, fetchSuggestions, messages]);

  const handleRefresh = useCallback(() => {
    fetchSuggestions(true);
  }, [fetchSuggestions]);

  const handleCopy = useCallback(async (suggestion: AISuggestion) => {
    try {
      await navigator.clipboard.writeText(suggestion.content);
      setCopiedId(suggestion.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text to clipboard:', err);
      // Fallback for browsers that don't support clipboard API
      const textArea = document.createElement('textarea');
      textArea.value = suggestion.content;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopiedId(suggestion.id);
      setTimeout(() => setCopiedId(null), 2000);
    }
  }, []);

  const handleSend = useCallback((suggestion: AISuggestion) => {
    console.log('Sending message:', suggestion.content);
    console.log('onSendMessage function:', onSendMessage);
    
    if (typeof onSendMessage === 'function') {
      onSendMessage(suggestion.content);
      setSelectedId(suggestion.id);
      setTimeout(() => setSelectedId(null), 1000);
    } else {
      console.error('onSendMessage is not a function:', onSendMessage);
    }
  }, [onSendMessage]);

  // Don't render if no messages
  if (messages.length === 0) {
    return null;
  }

  const showLoader = loading && !waitingForSentiment;
  const showSuggestions = !loading && !waitingForSentiment && suggestions.length > 0;

  return (
    <div className="bg-gradient-to-r from-white to-blue-50/30 border-t border-gray-200/60 shadow-sm backdrop-blur-sm">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Sparkles className="w-5 h-5 text-purple-600" />
              {(loading || waitingForSentiment) && (
                <div className="absolute -inset-1 rounded-full bg-purple-100 animate-ping opacity-30"></div>
              )}
            </div>
            <div>
              <h3 className="text-base font-semibold text-gray-800">AI Response Suggestions</h3>
              <p className="text-xs text-gray-500">
                {waitingForSentiment 
                  ? 'Analyzing conversation...' 
                  : 'Smart responses based on sentiment'}
              </p>
            </div>
            {(loading || waitingForSentiment) && (
              <Loader2 className="w-4 h-4 animate-spin text-purple-500" />
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handleRefresh}
              disabled={loading || waitingForSentiment}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Refresh suggestions"
              aria-label="Refresh AI suggestions"
            >
              <RefreshCw className={`w-4 h-4 ${(loading || waitingForSentiment) ? 'animate-spin' : ''}`} />
            </button>
            
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-all duration-200"
              title={isExpanded ? 'Collapse suggestions' : 'Expand suggestions'}
              aria-label={isExpanded ? 'Collapse suggestions' : 'Expand suggestions'}
            >
              {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Content - Fixed height container to prevent layout shifts */}
        {isExpanded && (
          <div className="min-h-[120px]"> {/* Fixed minimum height */}
            {/* Error State */}
            {error && !loading && !waitingForSentiment && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                <div className="flex items-start gap-3">
                  <div className="w-5 h-5 text-red-500 mt-0.5">⚠️</div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-red-800">Unable to generate suggestions</p>
                    <p className="text-xs text-red-600 mt-1">{error}</p>
                    <button
                      onClick={handleRefresh}
                      className="mt-3 text-xs font-medium text-red-700 hover:text-red-800 underline underline-offset-2 transition-colors"
                    >
                      Try again
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Suggestions */}
            {showSuggestions && (
              <div className="space-y-3">
                {suggestions.map((suggestion, index) => (
                  <div 
                    key={suggestion.id}
                    className={`
                      relative p-4 bg-white border border-gray-200 rounded-xl shadow-sm 
                      hover:shadow-md hover:border-gray-300 transition-all duration-200 group
                      ${selectedId === suggestion.id ? 
                        'ring-2 ring-purple-500 bg-purple-50 border-purple-300' 
                        : ''
                    }`}
                  >
                    {/* Suggestion Number */}
                    <div className="absolute -top-2 -left-2 w-6 h-6 bg-purple-500 text-white text-xs font-bold rounded-full flex items-center justify-center shadow-sm">
                      {index + 1}
                    </div>

                    {/* Suggestion Content */}
                    <div className="pr-16"> {/* Reduced padding to accommodate smaller buttons */}
                      <p className="text-sm text-gray-800 leading-relaxed">
                        {suggestion.content}
                      </p>
                    </div>
                    
                    {/* Action Buttons - Made smaller */}
                    <div className="absolute top-3 right-3 flex items-center gap-1">
                      <button
                        onClick={() => handleCopy(suggestion)}
                        className="flex items-center justify-center w-7 h-7 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-all duration-200 group-hover:opacity-100 opacity-70"
                        title="Copy to clipboard"
                        aria-label={`Copy suggestion ${index + 1} to clipboard`}
                      >
                        {copiedId === suggestion.id ? (
                          <CheckCircle2 className="w-4 h-4 text-green-500" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </button>
                      
                      <button
                        onClick={() => handleSend(suggestion)}
                        className="flex items-center justify-center w-7 h-7 text-white bg-purple-500 hover:bg-purple-600 rounded-md transition-all duration-200 hover:scale-105 shadow-sm hover:shadow-md"
                        title="Send this message"
                        aria-label={`Send suggestion ${index + 1}`}
                      >
                        <Send className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Success feedback for sent message */}
                    {selectedId === suggestion.id && (
                      <div className="absolute inset-0 bg-purple-100 rounded-xl flex items-center justify-center opacity-80">
                        <div className="text-purple-700 font-medium text-sm">Message sent! 🎉</div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Loading State for Generation - Fixed position */}
            {showLoader && (
              <div className="flex items-center justify-center h-24"> {/* Fixed height to prevent shifts */}
                <div className="flex flex-col items-center gap-3">
                  <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
                  <div className="text-center">
                    <p className="text-sm font-medium text-purple-700">Generating suggestions...</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Creating personalized responses for you
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Waiting for sentiment state */}
            {waitingForSentiment && !loading && (
              <div className="flex items-center justify-center h-24">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-6 h-6 border-2 border-purple-200 border-t-purple-500 rounded-full animate-spin"></div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-purple-700">Waiting for sentiment analysis...</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Analyzing conversation tone first
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Empty state when no suggestions and not loading */}
            {!loading && !waitingForSentiment && !error && suggestions.length === 0 && (
              <div className="flex items-center justify-center h-24">
                <div className="text-center">
                  <p className="text-sm text-gray-500">No suggestions available</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Try refreshing or send more messages
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Collapsed State Indicator */}
        {!isExpanded && suggestions.length > 0 && (
          <div className="flex items-center justify-center py-2">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Sparkles className="w-3 h-3" />
              <span>{suggestions.length} suggestion{suggestions.length !== 1 ? 's' : ''} available</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIResponseSuggestions;
