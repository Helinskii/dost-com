"use client"

import type React from "react"

import { RealtimeChat } from "@/components/realtime-chat"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useRef, useState } from "react"
import SentimentSidebar from "./sentiment"
import { ChatMessage } from "@/hooks/use-realtime-chat"
import AIResponseSuggestions from "./ai-response-suggestions"
import { SentimentProvider, useSentiment } from "@/hooks/use-sentiment-analysis"

// Main Chat Component (wrapped with context)
function ChatPage() {
  const [username, setUsername] = useState("")
  const [roomName, setRoomName] = useState("general")
  const [hasJoined, setHasJoined] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [showSidebar, setShowSidebar] = useState(true);
  const messagesRef = useRef([]);

  // Use sentiment context
  const {
    sentimentData,
    updateSentimentFromAPI,
    addMessageSentiment,
    getDominantEmotion,
    getSentimentTrend,
    isPositiveSentiment
  } = useSentiment();

  const handleJoinChat = (e: React.FormEvent) => {
    e.preventDefault()
    if (username.trim()) {
      setHasJoined(true)
    }
  }

  // Update messages and sentiment analysis
  const handleMessageUpdate = (updatedMessages: any) => {
    const hasChanged = JSON.stringify(messagesRef.current) !== JSON.stringify(updatedMessages);
    if (hasChanged) {
      messagesRef.current = updatedMessages;
      setMessages(updatedMessages);
      
      console.log("Messages updated:", updatedMessages.length);
    }
  };

  // Send message with sentiment tracking
  const handleSendMessage = (message: any) => {
    console.log('Sending message:', message);
    // Your actual send message implementation here
    
    // Optionally add optimistic sentiment update
    // addMessageSentiment(generateId(), message, 'unknown');
  };

  if (!hasJoined) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="text-center">Join Chat</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleJoinChat} className="space-y-4">
              <div>
                <label htmlFor="username" className="block text-sm font-medium mb-2">
                  Username
                </label>
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter your username"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label htmlFor="room" className="block text-sm font-medium mb-2">
                  Room
                </label>
                <select
                  id="room"
                  value={roomName}
                  onChange={(e) => setRoomName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="general">General</option>
                  <option value="random">Random</option>
                  <option value="tech">Tech Talk</option>
                </select>
              </div>
              <button
                type="submit"
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Join Chat
              </button>
            </form>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col">
      <header className="bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-semibold">Chat Room: {roomName}</h1>
            {/* Show sentiment indicator in header */}
            <div className="flex items-center gap-2 text-sm">
              <span className={`px-2 py-1 rounded-full text-xs ${
                isPositiveSentiment() ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
              }`}>
                {getDominantEmotion()} · {getSentimentTrend()}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="text-sm text-gray-600 hover:text-gray-800 flex items-center gap-2"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              {showSidebar ? 'Hide' : 'Show'} Sentiment Analysis
            </button>
            <span className="text-sm text-gray-600">
              Logged in as: <strong>{username}</strong>
            </span>
            <button 
              onClick={() => setHasJoined(false)} 
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              Leave Room
            </button>
          </div>
        </div>
      </header>
      
      <div className="flex-1 flex overflow-hidden">
        <main className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-hidden">
            <RealtimeChat
              roomName={roomName}
              username={username}
              onMessage={handleMessageUpdate}
            />
          </div>
          
          {/* Pass sentiment context data to AI suggestions */}
          <AIResponseSuggestions
            messages={messages}
            username={username}
            onSendMessage={handleSendMessage}
          />
        </main>
        
        <div
          className={`transition-all duration-300 ease-in-out ${
            showSidebar ? 'w-80' : 'w-0'
          } overflow-hidden`}
        >
          {showSidebar && (
            <SentimentSidebar
              chatId={roomName}
              messages={messages}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// Export wrapped component
export default function Page() {
  return (
    <SentimentProvider>
      <ChatPage />
    </SentimentProvider>
  );
}