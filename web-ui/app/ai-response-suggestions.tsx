import React, { useState, useEffect, useCallback } from 'react';
import { Sparkles, Copy, Send, RefreshCw, ChevronUp, ChevronDown, Loader2 } from 'lucide-react';

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

interface AISuggestion {
  id: string;
  content: string;
  // tone: 'professional' | 'friendly' | 'empathetic';
  // confidence: number;
}

interface AIResponseSuggestionsProps {
  messages: ChatMessage[];
  sentiments: SentimentValues;
  onSendMessage: (message: string) => void;
  isExpanded?: boolean;
}

const AIResponseSuggestions: React.FC<AIResponseSuggestionsProps> = ({
  messages,
  sentiments,
  onSendMessage,
  isExpanded: initialExpanded = true
}) => {
  const [suggestions, setSuggestions] = useState<AISuggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(initialExpanded);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Tone colors and labels
  const toneConfig = {
    professional: {
      color: 'from-blue-500 to-indigo-600',
      label: 'Professional',
      bgColor: 'bg-blue-50 dark:bg-blue-900/20',
      borderColor: 'border-blue-200 dark:border-blue-800'
    },
    friendly: {
      color: 'from-emerald-500 to-teal-600',
      label: 'Friendly',
      bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
      borderColor: 'border-emerald-200 dark:border-emerald-800'
    },
    empathetic: {
      color: 'from-purple-500 to-pink-600',
      label: 'Empathetic',
      bgColor: 'bg-purple-50 dark:bg-purple-900/20',
      borderColor: 'border-purple-200 dark:border-purple-800'
    }
  };

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
          chatHistory: {
            chatId: "",
            messages: messages.slice(-10), // Last 10 messages for context
            timestamp: new Date().toISOString()
          },
          sentiment: {
            sentiments,
          },
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate responses');
      }

      const data = await response.json();
      setSuggestions(data.suggestions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate suggestions');
      // Fallback mock suggestions
      setSuggestions([
        {
          id: '1',
          content: "I understand your concern. Let me help you resolve this issue as quickly as possible."
        },
        {
          id: '2',
          content: "Thanks for bringing this up! I'd be happy to help you out with that. 😊"
        },
        {
          id: '3',
          content: "I can see how that might be frustrating. Let's work together to find a solution that works for you."
        }
      ]);
    } finally {
      setLoading(false);
    }
  }, [messages, sentiments]);

  useEffect(() => {
    if (messages.length > 0) {
      fetchSuggestions();
    }
  }, [messages.length, fetchSuggestions]);

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
      {/* Header */}
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
          {loading && <Loader2 className="w-4 h-4 animate-spin text-gray-400" />}
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronUp className="w-4 h-4 text-gray-400" />
        )}
      </button>

      {/* Suggestions */}
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
                // const config = toneConfig[suggestion.tone];
                const isSelected = selectedId === suggestion.id;
                const isCopied = copiedId === suggestion.id;

                return (
                  <div
                    key={suggestion.id}
                    className={`relative p-4 rounded-lg border transition-all duration-300 ${
                      "bg-blue-50 dark:bg-blue-900/20" //config.bgColor
                    } ${
                      "border-blue-200 dark:border-blue-800" //config.borderColor
                      } 
                    ${
                      isSelected ? 'scale-[0.98] opacity-70' : 'hover:shadow-md'
                    }`}
                  >
                    {/* Tone Badge */}
                    {/* <div className="flex items-center justify-between mb-2">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-white bg-gradient-to-r ${config.color}`}
                      >
                        {config.label}
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {Math.round(suggestion.confidence * 100)}% match
                      </span>
                    </div> */}

                    {/* Content */}
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 leading-relaxed">
                      {suggestion.content}
                    </p>

                    {/* Actions */}
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
                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all bg-gradient-to-r ${
                          "from-blue-500 to-indigo-600" // config.color
                        } text-white hover:shadow-md active:scale-95`}
                      >
                        <Send className="w-3.5 h-3.5" />
                        Send
                      </button>
                    </div>
                  </div>
                );
              })}

              {/* Refresh Button */}
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
