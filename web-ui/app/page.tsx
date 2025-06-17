"use client"

import type React from "react"
import { RealtimeChat } from "@/components/realtime-chat"
import { useRef, useState, useCallback } from "react"
import SentimentSidebar from "./sentiment"
import { ChatMessage, useRealtimeChat } from "@/hooks/use-realtime-chat"
import AIResponseSuggestions from "./ai-response-suggestions"
import ArchitectureDiagram from './architecture'
import { SentimentProvider, useSentiment } from "@/hooks/use-sentiment-analysis"
import { Info, X, ChevronRight, Sparkles } from "lucide-react"


// Main Chat Component (wrapped with context)
function ChatPage() {
  const [username, setUsername] = useState("")
  const [roomName, setRoomName] = useState("general")
  const [hasJoined, setHasJoined] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [showSidebar, setShowSidebar] = useState(true)
  const [showArchitecture, setShowArchitecture] = useState(false)
  
  // Refs for tracking
  const messagesRef = useRef<ChatMessage[]>([])

  // Use sentiment context
  const {
    sentimentData,
    getDominantEmotion,
    getSentimentTrend,
    isPositiveSentiment
  } = useSentiment()

  const {
    messages: realtimeMessages,
    sendMessage,
    isConnected,
  } = useRealtimeChat({
    roomName,
    username,
  })

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

  // Handle sending messages from AI suggestions
  const handleSendMessage = useCallback((messageContent: string) => {
    console.log('Sending AI suggestion:', messageContent)
    sendMessage(messageContent);
  }, [])

  if (showArchitecture) {
    return <ArchitectureDiagram onClose={() => setShowArchitecture(false)} />
  }

  if (!hasJoined) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          {/* Architecture Button - Prominent at the top */}
          <button
            onClick={() => setShowArchitecture(true)}
            className="mb-4 w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-700 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center justify-center gap-3 group"
          >
            <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center group-hover:rotate-12 transition-transform">
              <Info className="w-5 h-5" />
            </div>
            <span className="text-lg">Explore Our Architecture</span>
            <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>

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
            
            <div className="mt-6 space-y-4">
              <div className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                <p className="text-sm text-blue-700">
                  <strong>Features:</strong> Real-time chat with AI sentiment analysis and response suggestions
                </p>
              </div>
              
              {/* Tech Stack Preview */}
              <div className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl border border-purple-200">
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="w-4 h-4 text-purple-600" />
                  <p className="text-sm font-semibold text-purple-700">Powered by cutting-edge tech:</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="text-xs px-2 py-1 bg-white rounded-full text-gray-600">React</span>
                  <span className="text-xs px-2 py-1 bg-white rounded-full text-gray-600">WebSockets</span>
                  <span className="text-xs px-2 py-1 bg-white rounded-full text-gray-600">AI/ML</span>
                  <span className="text-xs px-2 py-1 bg-white rounded-full text-gray-600">LLM API</span>
                </div>
              </div>
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
                : 'bg-orange-100 text-orange-800'
            }`}>
              {getDominantEmotion().toUpperCase()} • {getSentimentTrend()}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowArchitecture(true)}
              className="text-gray-600 hover:text-gray-800 p-2 rounded-lg hover:bg-gray-100 transition-colors"
              title="View Architecture"
            >
              <Info className="w-5 h-5" />
            </button>
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
              Hello, <span className="font-medium">{username}</span>
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

// Main App Component with Provider
export default function App() {
  return (
    <SentimentProvider>
      <ChatPage />
    </SentimentProvider>
  )
}
