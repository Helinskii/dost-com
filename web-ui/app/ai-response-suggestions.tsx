import React, { useState, useEffect, useCallback } from 'react';
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

  // Use sentiment context
  const { sentimentData, getDominantEmotion, getSentimentTrend, isPositiveSentiment } = useSentiment();

  const fetchSuggestions = useCallback(async () => {
    if (messages.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/llm/suggestions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username,
          chatHistory: {
            chatId: "",
            messages: messages.slice(-10),
            timestamp: new Date().toISOString()
          },
          sentiment: {
            overallSentiment: sentimentData.overallSentiment,
            dominantEmotion: getDominantEmotion(),
            trend: getSentimentTrend(),
            isPositive: isPositiveSentiment(),
            messageSentiments: Array.from(sentimentData.messageSentiments.entries()).slice(-5)
          },
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate responses');
      }

      const data = await response.json();
      setSuggestions(data.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate suggestions');
      
      // Generate contextual fallback suggestions based on sentiment
      const emotion = sentimentData.overallSentiment.emotion_last_message;
      const fallbackSuggestions = generateFallbackSuggestions(emotion);
      setSuggestions(fallbackSuggestions);
    } finally {
      setLoading(false);
    }
  }, [sentimentData, getDominantEmotion, getSentimentTrend, isPositiveSentiment, username]);

  // Generate contextual fallback suggestions
  const generateFallbackSuggestions = (emotion: string | null): AISuggestion[] => {
    const baseId = Date.now().toString();
    
    switch (emotion) {
      case 'sadness':
        return [
          {
            id: `${baseId}_1`,
            content: "I can see this is difficult for you. I'm here to help and support you through this.",
            tone: 'empathetic',
            confidence: 0.85
          },
          {
            id: `${baseId}_2`, 
            content: "Thank you for sharing that with me. Let's work together to find a way forward.",
            tone: 'professional',
            confidence: 0.8
          },
          {
            id: `${baseId}_3`,
            content: "I understand how you're feeling. Take your time, and know that I'm here whenever you need me.",
            tone: 'empathetic',
            confidence: 0.9
          }
        ];
      
      case 'anger':
        return [
          {
            id: `${baseId}_1`,
            content: "I understand your frustration. Let me help resolve this issue as quickly as possible.",
            tone: 'professional',
            confidence: 0.9
          },
          {
            id: `${baseId}_2`,
            content: "I hear you, and your concerns are completely valid. Let's work on a solution together.",
            tone: 'empathetic',
            confidence: 0.85
          },
          {
            id: `${baseId}_3`,
            content: "Thank you for bringing this to my attention. I'll make sure we address this properly.",
            tone: 'professional',
            confidence: 0.8
          }
        ];
      
      case 'joy':
        return [
          {
            id: `${baseId}_1`,
            content: "That's fantastic news! I'm so happy to hear that. 😊",
            tone: 'friendly',
            confidence: 0.9
          },
          {
            id: `${baseId}_2`,
            content: "Wonderful! It's great to see things working out well for you.",
            tone: 'friendly',
            confidence: 0.85
          },
          {
            id: `${baseId}_3`,
            content: "Excellent! Thanks for sharing this positive update with me.",
            tone: 'professional',
            confidence: 0.8
          }
        ];
      
      default:
        return [
          {
            id: `${baseId}_1`,
            content: "Thank you for your message. How can I best assist you today?",
            tone: 'professional',
            confidence: 0.75
          },
          {
            id: `${baseId}_2`,
            content: "I appreciate you reaching out. Let me know if there's anything specific I can help with!",
            tone: 'friendly',
            confidence: 0.8
          },
          {
            id: `${baseId}_3`,
            content: "Thanks for connecting. I'm here to help with whatever you need.",
            tone: 'friendly',
            confidence: 0.75
          }
        ];
    }
  };

  useEffect(() => {
    if (messages.length > 0) {
      fetchSuggestions();
    }
  }, [fetchSuggestions]);

  const handleCopy = async (suggestion: AISuggestion) => {
    try {
      await navigator.clipboard.writeText(suggestion.content);
      setCopiedId(suggestion.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleSend = (suggestion: AISuggestion) => {
    setSelectedId(suggestion.id);
    setTimeout(() => {
      onSendMessage(suggestion.content);
      setSelectedId(null);
    }, 300);
  };

  if (messages.length === 0) {
    return null;
  }

  return (
    <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 transition-all duration-300">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-gradient-to-r from-violet-500 to-purple-600 rounded-lg">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
            AI Suggested Responses
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            ({getDominantEmotion()})
          </span>
          {loading && <Loader2 className="w-4 h-4 animate-spin text-gray-400" />}
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronUp className="w-4 h-4 text-gray-400" />
        )}
      </button>

      <div
        className={`overflow-hidden transition-all duration-300 ${
          isExpanded ? 'max-h-96' : 'max-h-0'
        }`}
      >
        <div className="px-4 pb-4 space-y-3">
          {loading && suggestions.length === 0 ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
            </div>
          ) : error && suggestions.length === 0 ? (
            <div className="text-center py-4 text-sm text-gray-500 dark:text-gray-400">
              {error}
            </div>
          ) : (
            <>
              {suggestions.map((suggestion) => {
                const isSelected = selectedId === suggestion.id;
                const isCopied = copiedId === suggestion.id;

                return (
                  <div
                    key={suggestion.id}
                    className={`relative p-4 rounded-lg border transition-all duration-300 bg-blue-50 dark:bg-blue-900/2 
                      border-blue-200 dark:border-blue-800 ${
                      isSelected ? 'scale-[0.98] opacity-70' : 'hover:shadow-md'
                    }`}
                  >
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 leading-relaxed">
                      {suggestion.content}
                    </p>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleCopy(suggestion)}
                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                          isCopied
                            ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
                        }`}
                      >
                        <Copy className="w-3.5 h-3.5" />
                        {isCopied ? 'Copied!' : 'Copy'}
                      </button>
                      <button
                        onClick={() => handleSend(suggestion)}
                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:shadow-md active:scale-95`}
                      >
                        <Send className="w-3.5 h-3.5" />
                        Send
                      </button>
                    </div>
                  </div>
                );
              })}

              <button
                onClick={fetchSuggestions}
                disabled={loading}
                className="w-full py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                Generate New Suggestions
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIResponseSuggestions;
