import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';

// Type definitions
interface Message {
  id: string;
  content: string;
  createdAt: string;
  user: {
    name: string;
  };
}

// Emotion types based on your requirement
type EmotionType = 'sadness' | 'joy' | 'love' | 'anger' | 'fear' | 'unknown';

interface EmotionalScores {
  sadness: number;
  joy: number;
  love: number;
  anger: number;
  fear: number;
  unknown: number;
}

interface MessageSentiment {
  messageId: string;
  emotion: EmotionType;
  confidence: number;
  text: string;
  timestamp: string;
}

interface UserSentiment {
  username: string;
  averageEmotion: EmotionType;
  averageScore: number;
  emotionalScores: EmotionalScores;
  messageCount: number;
  lastUpdated: string;
}

interface SentimentData {
  overallSentiment: {
    emotion_last_message: EmotionType | null;
    emotional_scores: EmotionalScores;
  };
  messageSentiments: Map<string, MessageSentiment>;
  userSentiments: Map<string, UserSentiment>;
  conversationHistory: Message[];
}

interface SentimentContextValue {
  sentimentData: SentimentData;
  updateSentimentFromAPI: (apiResponse: any) => void;
  addMessageSentiment: (messageId: string, messageText: string, emotion: EmotionType, score?: number) => void;
  getMessageSentiment: (messageId: string) => MessageSentiment | undefined;
  getUserSentiment: (username: string) => UserSentiment | undefined;
  getDominantEmotion: () => EmotionType;
  getSentimentTrend: () => 'improving' | 'declining' | 'stable';
  isPositiveSentiment: () => boolean;
  isNegativeSentiment: () => boolean;
  getSentimentScore: (emotion: EmotionType) => number;
  updateMessagesData: (messages: Message[]) => void;
}

// Mock sentiment analysis function
const mockAnalyzeSentiment = (text: string): { emotion: EmotionType; confidence: number } => {
  const words = text.toLowerCase().split(' ');
  
  const emotionKeywords = {
    joy: ['happy', 'great', 'awesome', 'wonderful', 'amazing', 'fantastic', 'excellent', 'good', 'nice', 'perfect', 'love', 'beautiful'],
    love: ['love', 'adore', 'cherish', 'heart', 'dear', 'sweetheart', 'honey', 'darling', 'kiss', 'hug', 'romantic'],
    sadness: ['sad', 'depressed', 'cry', 'tears', 'grief', 'sorrow', 'lonely', 'miserable', 'heartbroken', 'devastated'],
    anger: ['angry', 'furious', 'mad', 'hate', 'rage', 'pissed', 'annoyed', 'frustrated', 'irritated', 'outraged'],
    fear: ['scared', 'afraid', 'terrified', 'anxious', 'worried', 'nervous', 'panic', 'frightened', 'concerned', 'stress'],
    unknown: ['maybe', 'perhaps', 'not sure', 'unclear', 'confused', 'uncertain', 'dunno', 'whatever']
  };

  let bestEmotion: EmotionType = 'unknown';
  let maxScore = 0;
  let totalMatches = 0;

  Object.entries(emotionKeywords).forEach(([emotion, keywords]) => {
    const matches = words.filter(word => keywords.some(keyword => word.includes(keyword))).length;
    if (matches > maxScore) {
      maxScore = matches;
      bestEmotion = emotion as EmotionType;
    }
    totalMatches += matches;
  });

  // Calculate confidence
  const confidence = totalMatches > 0 ? Math.min(0.95, 0.3 + (totalMatches / words.length)) : 0.2;

  // Add some randomness to make it more realistic
  const randomFactor = Math.random() * 0.2 - 0.1; // -0.1 to +0.1
  const finalConfidence = Math.min(0.95, Math.max(0.05, confidence + randomFactor));

  return { 
    emotion: bestEmotion, 
    confidence: finalConfidence
  };
};

// Generate mock user sentiment based on their messages
const calculateUserSentiment = (messages: Message[], username: string): UserSentiment => {
  const userMessages = messages.filter(msg => msg.user.name === username);
  
  if (userMessages.length === 0) {
    return {
      username,
      averageEmotion: 'unknown',
      averageScore: 0.5,
      emotionalScores: { sadness: 0, joy: 0, love: 0, anger: 0, fear: 0, unknown: 1 },
      messageCount: 0,
      lastUpdated: new Date().toISOString()
    };
  }

  const sentiments = userMessages.map(msg => mockAnalyzeSentiment(msg.content));
  
  // Calculate emotional scores
  const emotionalScores: EmotionalScores = {
    sadness: 0, joy: 0, love: 0, anger: 0, fear: 0, unknown: 0
  };

  sentiments.forEach(sentiment => {
    emotionalScores[sentiment.emotion] += sentiment.score;
  });

  // Normalize scores
  const total = Object.values(emotionalScores).reduce((sum, score) => sum + score, 0);
  if (total > 0) {
    Object.keys(emotionalScores).forEach(emotion => {
      emotionalScores[emotion as EmotionType] /= total;
    });
  }

  // Find dominant emotion
  const dominantEmotion = Object.entries(emotionalScores).reduce((a, b) => 
    emotionalScores[a[0] as EmotionType] > emotionalScores[b[0] as EmotionType] ? a : b
  )[0] as EmotionType;

  const averageScore = emotionalScores[dominantEmotion];

  return {
    username,
    averageEmotion: dominantEmotion,
    averageScore,
    emotionalScores,
    messageCount: userMessages.length,
    lastUpdated: new Date().toISOString()
  };
};

// Create context
const SentimentContext = createContext<SentimentContextValue | undefined>(undefined);

// Custom hook to use sentiment context
export const useSentiment = (): SentimentContextValue => {
  const context = useContext(SentimentContext);
  if (!context) {
    throw new Error('useSentiment must be used within a SentimentProvider');
  }
  return context;
};

// Sentiment provider component
export const SentimentProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sentimentData, setSentimentData] = useState<SentimentData>({
    overallSentiment: {
      emotion_last_message: null,
      emotional_scores: {
        sadness: 0.1,
        joy: 0.3,
        love: 0.2,
        anger: 0.1,
        fear: 0.1,
        unknown: 0.2
      }
    },
    messageSentiments: new Map(),
    userSentiments: new Map(),
    conversationHistory: []
  });

  // Update sentiment from API response
  const updateSentimentFromAPI = useCallback((apiResponse: any) => {
    const { emotion_last_message, emotional_scores, emotion_per_text } = apiResponse;
    
    setSentimentData(prev => ({
      ...prev,
      overallSentiment: {
        emotion_last_message: emotion_last_message || prev.overallSentiment.emotion_last_message,
        emotional_scores: emotional_scores || prev.overallSentiment.emotional_scores
      }
    }));

    // Update per-message sentiment if provided
    if (emotion_per_text) {
      setSentimentData(prev => ({
        ...prev,
        messageSentiments: new Map(prev.messageSentiments)
      }));
    }
  }, []);

  // Add new message sentiment
  const addMessageSentiment = useCallback((messageId: string, messageText: string, emotion: EmotionType, confidence: number = 0.5) => {
    const sentiment: MessageSentiment = {
      messageId,
      emotion,
      confidence,
      text: messageText,
      timestamp: new Date().toISOString()
    };

    setSentimentData(prev => ({
      ...prev,
      messageSentiments: new Map(prev.messageSentiments).set(messageId, sentiment)
    }));
  }, []);

  // Get sentiment for specific message
  const getMessageSentiment = useCallback((messageId: string): MessageSentiment | undefined => {
    return sentimentData.messageSentiments.get(messageId);
  }, [sentimentData.messageSentiments]);

  // Get sentiment for specific user
  const getUserSentiment = useCallback((username: string): UserSentiment | undefined => {
    return sentimentData.userSentiments.get(username);
  }, [sentimentData.userSentiments]);

  // Update messages and recalculate sentiments
  const updateMessagesData = useCallback((messages: Message[]) => {
    setSentimentData(prev => {
      const newMessageSentiments = new Map(prev.messageSentiments);
      const newUserSentiments = new Map(prev.userSentiments);

      // Analyze new messages
      messages.forEach(message => {
        if (!newMessageSentiments.has(message.id)) {
          const analysis = mockAnalyzeSentiment(message.content);
          newMessageSentiments.set(message.id, {
            messageId: message.id,
            emotion: analysis.emotion,
            confidence: analysis.confidence,
            text: message.content,
            timestamp: message.createdAt
          });
        }
      });

      // Recalculate user sentiments
      const uniqueUsers = [...new Set(messages.map(msg => msg.user.name))];
      uniqueUsers.forEach(username => {
        const userSentiment = calculateUserSentiment(messages, username);
        newUserSentiments.set(username, userSentiment);
      });

      // Update overall sentiment based on recent messages
      const recentMessages = messages.slice(-5); // Last 5 messages
      let overallScores: EmotionalScores = {
        sadness: 0, joy: 0, love: 0, anger: 0, fear: 0, unknown: 0
      };

      if (recentMessages.length > 0) {
        recentMessages.forEach(msg => {
          const sentiment = newMessageSentiments.get(msg.id);
          if (sentiment) {
            // Use confidence as a weight for the emotion
            overallScores[sentiment.emotion] += sentiment.confidence;
          }
        });

        // Normalize
        const total = Object.values(overallScores).reduce((sum, score) => sum + score, 0);
        if (total > 0) {
          Object.keys(overallScores).forEach(emotion => {
            overallScores[emotion as EmotionType] /= total;
          });
        }
      }

      const lastMessage = messages[messages.length - 1];
      const lastMessageSentiment = lastMessage ? newMessageSentiments.get(lastMessage.id) : null;

      return {
        ...prev,
        messageSentiments: newMessageSentiments,
        userSentiments: newUserSentiments,
        conversationHistory: messages,
        overallSentiment: {
          emotion_last_message: lastMessageSentiment?.emotion || prev.overallSentiment.emotion_last_message,
          emotional_scores: Object.values(overallScores).some(score => score > 0) ? overallScores : prev.overallSentiment.emotional_scores
        }
      };
    });
  }, []);

  // Get dominant emotion from scores
  const getDominantEmotion = useCallback((): EmotionType => {
    const scores = sentimentData.overallSentiment.emotional_scores;
    return Object.entries(scores).reduce((a, b) => 
      scores[a[0] as EmotionType] > scores[b[0] as EmotionType] ? a : b
    )[0] as EmotionType;
  }, [sentimentData.overallSentiment.emotional_scores]);

  // Calculate sentiment trend over conversation
  const getSentimentTrend = useCallback((): 'improving' | 'declining' | 'stable' => {
    const messages = Array.from(sentimentData.messageSentiments.values());
    if (messages.length < 4) return 'stable';

    const recent = messages.slice(-3);
    const older = messages.slice(-6, -3);

    const getPositiveScore = (msgs: MessageSentiment[]) => {
      return msgs.reduce((sum, msg) => {
        return sum + (['joy', 'love'].includes(msg.emotion) ? msg.confidence : 0);
      }, 0) / msgs.length;
    };

    const recentPositive = getPositiveScore(recent);
    const olderPositive = getPositiveScore(older);

    if (recentPositive > olderPositive + 0.2) return 'improving';
    if (recentPositive < olderPositive - 0.2) return 'declining';
    return 'stable';
  }, [sentimentData.messageSentiments]);

  // Utility functions
  const isPositiveSentiment = useCallback((): boolean => {
    return ['joy', 'love'].includes(sentimentData.overallSentiment.emotion_last_message || 'unknown');
  }, [sentimentData.overallSentiment.emotion_last_message]);

  const isNegativeSentiment = useCallback((): boolean => {
    return ['sadness', 'anger', 'fear'].includes(sentimentData.overallSentiment.emotion_last_message || 'unknown');
  }, [sentimentData.overallSentiment.emotion_last_message]);

  const getSentimentScore = useCallback((emotion: EmotionType): number => {
    return sentimentData.overallSentiment.emotional_scores[emotion] || 0;
  }, [sentimentData.overallSentiment.emotional_scores]);

  const contextValue: SentimentContextValue = {
    sentimentData,
    updateSentimentFromAPI,
    addMessageSentiment,
    getMessageSentiment,
    getUserSentiment,
    getDominantEmotion,
    getSentimentTrend,
    isPositiveSentiment,
    isNegativeSentiment,
    getSentimentScore,
    updateMessagesData
  };

  return (
    <SentimentContext.Provider value={contextValue}>
      {children}
    </SentimentContext.Provider>
  );
};
