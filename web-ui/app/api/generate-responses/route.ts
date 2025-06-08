// app/api/generate-responses/route.ts
import { NextRequest, NextResponse } from 'next/server';

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

interface GenerateResponsesRequest {
  messages: ChatMessage[];
  sentiments: SentimentValues;
  timestamp: string;
}

interface AISuggestion {
  id: string;
  content: string;
  tone: 'professional' | 'friendly' | 'empathetic';
  confidence: number;
}

interface GenerateResponsesResponse {
  success: boolean;
  suggestions: AISuggestion[];
}

// Example using OpenAI (you can replace with any LLM API)
export async function POST(request: NextRequest) {
  try {
    const data: GenerateResponsesRequest = await request.json();
    const { messages, sentiments } = data;

    // Analyze sentiment context
    const dominantSentiment = Object.entries(sentiments).reduce((a, b) => 
      sentiments[a[0] as keyof SentimentValues] > sentiments[b[0] as keyof SentimentValues] ? a : b
    )[0];

    // Get conversation context
    const recentMessages = messages.slice(-5).map(m => 
      `${m.user.name}: ${m.content}`
    ).join('\n');

    // For production, replace this with actual LLM API call
    // Example with OpenAI:
    /*
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: `Generate 3 different response suggestions for a chat conversation.
          Current sentiment analysis: ${JSON.stringify(sentiments)}
          Dominant sentiment: ${dominantSentiment}
          
          Generate responses in 3 tones:
          1. Professional - formal and business-like
          2. Friendly - warm and approachable
          3. Empathetic - understanding and supportive
          
          Each response should be appropriate for the conversation context and sentiment.
          Keep responses concise (under 50 words each).
          Return as JSON array with fields: content, tone`
        },
        {
          role: "user",
          content: `Recent conversation:\n${recentMessages}\n\nGenerate 3 appropriate responses.`
        }
      ],
      temperature: 0.7,
    });
    
    const suggestions = JSON.parse(completion.choices[0].message.content);
    */

    // Mock responses based on sentiment
    let suggestions: AISuggestion[] = [];

    if (sentiments.positive > 60) {
      suggestions = [
        {
          id: '1',
          content: "That's wonderful to hear! I'm delighted that everything is working well for you. Is there anything else I can help you with today?",
          tone: 'professional',
          confidence: 0.92
        },
        {
          id: '2',
          content: "Awesome! So glad you're having a great experience! 🎉 Feel free to reach out if you need anything else!",
          tone: 'friendly',
          confidence: 0.88
        },
        {
          id: '3',
          content: "It makes me so happy to see you're satisfied! Your positive feedback really brightens our day. Thank you for sharing!",
          tone: 'empathetic',
          confidence: 0.90
        }
      ];
    } else if (sentiments.negative > 60 || sentiments.angry > 40) {
      suggestions = [
        {
          id: '1',
          content: "I sincerely apologize for this experience. Let me escalate this to our team immediately to resolve it for you.",
          tone: 'professional',
          confidence: 0.95
        },
        {
          id: '2',
          content: "I'm really sorry you're dealing with this! That's definitely not okay. Let me help fix this right away!",
          tone: 'friendly',
          confidence: 0.87
        },
        {
          id: '3',
          content: "I completely understand your frustration - this situation would upset me too. You have every right to feel this way. Let's work together to make this right.",
          tone: 'empathetic',
          confidence: 0.93
        }
      ];
    } else if (sentiments.sad > 40) {
      suggestions = [
        {
          id: '1',
          content: "I understand this can be challenging. Please know that we're here to support you through this process.",
          tone: 'professional',
          confidence: 0.89
        },
        {
          id: '2',
          content: "I hear you, and I'm here to help however I can. Sometimes these things take time, but we'll get through it together!",
          tone: 'friendly',
          confidence: 0.86
        },
        {
          id: '3',
          content: "I can sense this has been difficult for you, and that's completely valid. Your feelings matter, and I want to help make this easier for you.",
          tone: 'empathetic',
          confidence: 0.94
        }
      ];
    } else {
      // Neutral or mixed sentiments
      suggestions = [
        {
          id: '1',
          content: "Thank you for your message. I'd be happy to assist you with any questions or concerns you may have.",
          tone: 'professional',
          confidence: 0.88
        },
        {
          id: '2',
          content: "Hey there! Thanks for reaching out! What can I help you with today? 😊",
          tone: 'friendly',
          confidence: 0.85
        },
        {
          id: '3',
          content: "I appreciate you taking the time to connect with us. I'm here to listen and help in whatever way I can.",
          tone: 'empathetic',
          confidence: 0.87
        }
      ];
    }

    const response: GenerateResponsesResponse = {
      success: true,
      suggestions
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Generate responses error:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to generate responses',
        suggestions: []
      },
      { status: 500 }
    );
  }
}
