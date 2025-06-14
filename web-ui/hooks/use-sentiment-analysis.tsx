import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';


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
      setSentimentData(prev => ({
        ...prev,
        messageSentiments: prev.messageSentiments
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
