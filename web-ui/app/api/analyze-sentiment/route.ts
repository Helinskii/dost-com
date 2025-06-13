import { NextRequest, NextResponse } from 'next/server';

export interface ChatMessage {
  id: string;
  content: string;
  user: {
    name: string;
  };
  createdAt: string;
}

interface AnalyzeSentimentRequest {
  chatId: string;
  messages: ChatMessage[];
  timestamp: string;
}

interface EmotionalScores {
  sadness: number;
  joy: number;
  love: number;
  anger: number;
  fear: number;
  unknown: number;
}

interface AnalyzeSentimentResponse {
  success: boolean;
  emotion_last_message: string;
  emotional_scores: EmotionalScores;
  emotion_per_text: Array<Record<string, string>>;
  metadata: {
    totalMessages: number;
    analyzedAt: string;
    confidence: number;
  };
}

export async function POST(request: NextRequest) {
  try {
    const data: AnalyzeSentimentRequest = await request.json();
    
    if (!data.messages || data.messages.length === 0) {
      return NextResponse.json(
        { success: false, error: 'No messages provided' },
        { status: 400 }
      );
    }

    // Mock emotion detection for each message
    const emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'unknown'];
    const emotionPerText = data.messages.map(msg => ({
      [msg.content]: emotions[Math.floor(Math.random() * emotions.length)]
    }));

    // Get last message emotion
    const lastMessage = data.messages[data.messages.length - 1];
    const lastMessageEmotion = emotions[Math.floor(Math.random() * emotions.length)];

    // Generate emotional scores (normalized to sum ≤ 1.0)
    const rawScores = {
      sadness: Math.random() * 0.5,
      joy: Math.random() * 0.5,
      love: Math.random() * 0.3,
      anger: Math.random() * 0.4,
      fear: Math.random() * 0.3,
      unknown: Math.random() * 0.2
    };

    // Normalize scores
    const total = Object.values(rawScores).reduce((sum, val) => sum + val, 0);
    const normalizedScores: EmotionalScores = {
      sadness: Number((rawScores.sadness / total).toFixed(2)),
      joy: Number((rawScores.joy / total).toFixed(2)),
      love: Number((rawScores.love / total).toFixed(2)),
      anger: Number((rawScores.anger / total).toFixed(2)),
      fear: Number((rawScores.fear / total).toFixed(2)),
      unknown: Number((rawScores.unknown / total).toFixed(2))
    };

    const response: AnalyzeSentimentResponse = {
      success: true,
      emotion_last_message: lastMessageEmotion,
      emotional_scores: normalizedScores,
      emotion_per_text: emotionPerText,
      metadata: {
        totalMessages: data.messages.length,
        analyzedAt: new Date().toISOString(),
        confidence: 0.85 + Math.random() * 0.1 // 0.85-0.95
      }
    };
    
    return NextResponse.json(response);
  } catch (error) {
    console.error('Sentiment analysis error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to analyze sentiments' },
      { status: 500 }
    );
  }
}
