"use client"

import type React from "react"

import { RealtimeChat } from "@/components/realtime-chat"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useRef, useState } from "react"
import SentimentSidebar from "./sentiment"
import { ChatMessage } from "@/hooks/use-realtime-chat"
import AIResponseSuggestions from "./ai-response-suggestions"

export default function Page() {
  const [username, setUsername] = useState("")
  const [roomName, setRoomName] = useState("general")
  const [hasJoined, setHasJoined] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [showSidebar, setShowSidebar] = useState(true);
  const [sentiments, setSentiments] = useState({
    positive: 0,
    negative: 0,
    neutral: 0,
    excited: 0,
    sad: 0,
    angry: 0
  });
  const messagesRef = useRef([]);

  const handleJoinChat = (e: React.FormEvent) => {
    e.preventDefault()
    if (username.trim()) {
      setHasJoined(true)
    }
  }

  // Only update messages if they actually changed
  const handleMessageUpdate = (updatedMessages:any) => {
    const hasChanged = JSON.stringify(messagesRef.current) !== JSON.stringify(updatedMessages);
    if (hasChanged) {
      messagesRef.current = updatedMessages;
      setMessages(updatedMessages);
      console.log("Messages updated:", updatedMessages.length);
    }
  };

  // Function to send message (you'll need to implement this with your chat logic)
  const handleSendMessage = (message:any) => {
    // Send message through your RealtimeChat component
    console.log('Sending message:', message);
    // You'll need to call your actual send message function here
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
          <h1 className="text-xl font-semibold">Chat Room: {roomName}</h1>
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
          {/* Chat messages area */}
          <div className="flex-1 overflow-hidden">
            <RealtimeChat
              roomName={roomName}
              username={username}
              onMessage={handleMessageUpdate}
            />
          </div>
          
          {/* AI Suggestions - positioned above the message input */}
          <AIResponseSuggestions
            messages={messages}
            sentiments={sentiments}
            username={username}
            onSendMessage={handleSendMessage}
          />
        </main>
        
        {/* Sentiment Sidebar */}
        <div
          className={`transition-all duration-300 ease-in-out ${
            showSidebar ? 'w-80' : 'w-0'
          } overflow-hidden`}
        >
          {showSidebar && (
            <SentimentSidebar
              chatId={roomName}
              messages={messages}
              onSentimentsUpdate={(newSentiments) => setSentiments(newSentiments)}
            />
          )}
        </div>
      </div>
    </div>
  );
}
