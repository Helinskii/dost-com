"use client"

import type React from "react"
import { RealtimeChat } from "@/components/realtime-chat"
import { useRef, useState, useCallback } from "react"
import SentimentSidebar from "./sentiment"
import { ChatMessage } from "@/hooks/use-realtime-chat"
import AIResponseSuggestions from "./ai-response-suggestions"
import { SentimentProvider, useSentiment } from "@/hooks/use-sentiment-analysis"

// Main Chat Component (wrapped with context)
function ChatPage() {
  const [username, setUsername] = useState("")
  const [roomName, setRoomName] = useState("general")
  const [hasJoined, setHasJoined] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [showSidebar, setShowSidebar] = useState(true)
  
  // Refs for tracking
  const messagesRef = useRef<ChatMessage[]>([])
  const sendMessageRef = useRef<((content: string) => void) | null>(null)

  // Use sentiment context
  const {
    sentimentData,
    getDominantEmotion,
    getSentimentTrend,
    isPositiveSentiment
  } = useSentiment()

  const handleJoinChat = (e: React.FormEvent) => {
    e.preventDefault()
    if (username.trim()) {
      setHasJoined(true)
    }
  }

  // Update messages when they change from the chat component
  const handleMessageUpdate = useCallback((updatedMessages: ChatMessage[]) => {
    const hasChanged = JSON.stringify(messagesRef.current) !== JSON.stringify(updatedMessages)
    if (hasChanged) {
      messagesRef.current = updatedMessages
      setMessages(updatedMessages)
      console.log("Messages updated:", updatedMessages.length)
    }
  }, [])

  // Store the sendMessage function from RealtimeChat
  const handleSendMessageRef = useCallback((sendMessageFn: (content: string) => void) => {
    sendMessageRef.current = sendMessageFn
  }, [])

  // Handle sending messages from AI suggestions
  const handleSendMessage = useCallback((messageContent: string) => {
    console.log('Sending AI suggestion:', messageContent)
    
    if (sendMessageRef.current) {
      sendMessageRef.current(messageContent)
    } else {
      console.warn('Send message function not available')
    }
  }, [])

  if (!hasJoined) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                <span className="text-2xl">💬</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-800 mb-2">Join Chat Room</h1>
              <p className="text-gray-600">Enter your username to start chatting with AI sentiment analysis</p>
            </div>
            
            <form onSubmit={handleJoinChat} className="space-y-6">
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                  Username
                </label>
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter your username"
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  required
                />
              </div>
              
              <div>
                <label htmlFor="roomName" className="block text-sm font-medium text-gray-700 mb-2">
                  Room Name
                </label>
                <input
                  id="roomName"
                  type="text"
                  value={roomName}
                  onChange={(e) => setRoomName(e.target.value)}
                  placeholder="general"
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                />
              </div>
              
              <button
                type="submit"
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-xl font-semibold hover:from-blue-600 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                Join Chat Room
              </button>
            </form>
            
            <div className="mt-6 p-4 bg-blue-50 rounded-xl border border-blue-200">
              <p className="text-sm text-blue-700">
                <strong>Features:</strong> Real-time chat with AI sentiment analysis and response suggestions
              </p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      <header className="bg-white shadow-sm border-b border-gray-200 z-10">
        <div className="px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold">💬</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">Room: {roomName}</h1>
                <p className="text-sm text-gray-600">AI-Powered Chat Analysis</p>
              </div>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              isPositiveSentiment() 
                ? 'bg-green-100 text-green-800' 
                : 'bg-gray-100 text-gray-600'
            }`}>
              {getDominantEmotion()} · {getSentimentTrend()}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="text-sm text-gray-600 hover:text-gray-800 flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
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
              className="text-sm text-blue-600 hover:text-blue-800 px-3 py-2 rounded-lg hover:bg-blue-50 transition-colors"
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
              onSendMessageRef={handleSendMessageRef}
            />
          </div>
          
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
  )
}

// Export wrapped component
export default function Page() {
  return (
    <SentimentProvider>
      <ChatPage />
    </SentimentProvider>
  )
}
