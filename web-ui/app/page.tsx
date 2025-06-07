"use client"

import type React from "react"

import { RealtimeChat } from "@/components/realtime-chat"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useState } from "react"

export default function Page() {
  const [username, setUsername] = useState("")
  const [roomName, setRoomName] = useState("general")
  const [hasJoined, setHasJoined] = useState(false)

  const handleJoinChat = (e: React.FormEvent) => {
    e.preventDefault()
    if (username.trim()) {
      setHasJoined(true)
    }
  }

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
            <span className="text-sm text-gray-600">
              Logged in as: <strong>{username}</strong>
            </span>
            <button onClick={() => setHasJoined(false)} className="text-sm text-blue-600 hover:text-blue-800">
              Leave Room
            </button>
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-hidden">
        <RealtimeChat
          roomName={roomName}
          username={username}
          onMessage={(messages) => {
            // You can add database persistence here
            console.log("Messages updated:", messages.length)
          }}
        />
      </main>
    </div>
  )
}
