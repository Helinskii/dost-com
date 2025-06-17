import { Info, X, ChevronRight, Sparkles } from "lucide-react"

// Architecture Component
export default function ArchitectureDiagram({ onClose }: { onClose: () => void }) {
    return (
        <div className="fixed inset-0 z-50 overflow-auto bg-gradient-to-br from-purple-600 to-blue-600">
            <div className="min-h-screen p-4">
                <div className="max-w-6xl mx-auto">
                    <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl p-8 relative">
                        <button
                            onClick={onClose}
                            className="absolute top-4 right-4 p-2 hover:bg-gray-100 rounded-full transition-colors"
                        >
                            <X className="w-6 h-6 text-gray-600" />
                        </button>

                        <h1 className="text-4xl font-bold text-center mb-2 bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                            🚀 Dost-Com Architecture
                        </h1>
                        <p className="text-center text-gray-600 mb-10 text-lg">
                            Real-time Chat with AI-Powered Sentiment Analysis
                        </p>

                        <div className="space-y-8">
                            {/* Flow Description */}
                            <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl p-6 text-white">
                                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                                    <Sparkles className="w-6 h-6" />
                                    How It Works:
                                </h3>
                                <div className="space-y-2">
                                    <div className="flex items-start gap-3">
                                        <span className="font-bold">1️⃣</span>
                                        <p>Users type messages in the chat interface</p>
                                    </div>
                                    <div className="flex items-start gap-3">
                                        <span className="font-bold">2️⃣</span>
                                        <p>Messages are broadcast in real-time via WebSockets</p>
                                    </div>
                                    <div className="flex items-start gap-3">
                                        <span className="font-bold">3️⃣</span>
                                        <p>Each message triggers sentiment analysis</p>
                                    </div>
                                    <div className="flex items-start gap-3">
                                        <span className="font-bold">4️⃣</span>
                                        <p>AI analyzes emotions and generates response suggestions</p>
                                    </div>
                                    <div className="flex items-start gap-3">
                                        <span className="font-bold">5️⃣</span>
                                        <p>Results are displayed instantly in the UI</p>
                                    </div>
                                </div>
                            </div>

                            {/* User Layer */}
                            <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-6 border border-blue-200">
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="text-2xl">👤</span>
                                    <h2 className="text-xl font-semibold text-gray-800">User Interaction</h2>
                                </div>
                                <div className="grid md:grid-cols-3 gap-4">
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">💬</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Chat Interface</h3>
                                        <p className="text-sm text-gray-600">Type and send messages in real-time</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">😊</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Sentiment Display</h3>
                                        <p className="text-sm text-gray-600">View live emotion analysis</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">💡</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">AI Suggestions</h3>
                                        <p className="text-sm text-gray-600">Get smart response recommendations</p>
                                    </div>
                                </div>
                            </div>

                            <div className="text-center text-3xl text-gray-400 animate-bounce">⬇️</div>

                            {/* Frontend Layer */}
                            <div className="bg-gradient-to-r from-pink-50 to-orange-50 rounded-xl p-6 border border-pink-200">
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="text-2xl">🎨</span>
                                    <h2 className="text-xl font-semibold text-gray-800">Frontend Application</h2>
                                </div>
                                <div className="grid md:grid-cols-3 gap-4">
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">⚛️</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">React Components</h3>
                                        <p className="text-sm text-gray-600">RealtimeChat, SentimentSidebar, AIResponseSuggestions</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">🪝</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Custom Hooks</h3>
                                        <p className="text-sm text-gray-600">useRealtimeChat, useSentiment, useChatScroll</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">🌐</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Context API</h3>
                                        <p className="text-sm text-gray-600">SentimentProvider for global state</p>
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-2 mt-4">
                                    <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">Next.js</span>
                                    <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">React</span>
                                    <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">TypeScript</span>
                                    <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">Tailwind CSS</span>
                                </div>
                            </div>

                            <div className="text-center text-3xl text-gray-400 animate-bounce">⬇️</div>

                            {/* Real-time Layer */}
                            <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-xl p-6 border border-purple-200">
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="text-2xl">🔌</span>
                                    <h2 className="text-xl font-semibold text-gray-800">Real-time Communication</h2>
                                </div>
                                <div className="grid md:grid-cols-2 gap-4">
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">📡</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">WebSocket Connection</h3>
                                        <p className="text-sm text-gray-600">Supabase Realtime channels for instant messaging</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">📢</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Message Broadcasting</h3>
                                        <p className="text-sm text-gray-600">Distribute messages to all connected users</p>
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-2 mt-4">
                                    <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">Supabase</span>
                                    <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">WebSockets</span>
                                </div>
                            </div>

                            <div className="text-center text-3xl text-gray-400 animate-bounce">⬇️</div>

                            {/* Backend API Layer */}
                            <div className="bg-gradient-to-r from-green-50 to-teal-50 rounded-xl p-6 border border-green-200">
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="text-2xl">⚙️</span>
                                    <h2 className="text-xl font-semibold text-gray-800">Backend APIs</h2>
                                </div>
                                <div className="grid md:grid-cols-3 gap-4">
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">🧠</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Sentiment Analysis API</h3>
                                        <p className="text-sm text-gray-600">/api/analyze-sentiment - Processes messages for emotions</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">🤖</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Response Generator API</h3>
                                        <p className="text-sm text-gray-600">/api/generate-responses - Creates AI suggestions</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">🐍</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Python Functions</h3>
                                        <p className="text-sm text-gray-600">/api/llm/suggestions.py - LLM integration</p>
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-2 mt-4">
                                    <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">Next.js API Routes</span>
                                    <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">Python</span>
                                    <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">Vercel Functions</span>
                                </div>
                            </div>

                            <div className="text-center text-3xl text-gray-400 animate-bounce">⬇️</div>

                            {/* AI Services Layer */}
                            <div className="bg-gradient-to-r from-red-50 to-pink-50 rounded-xl p-6 border border-red-200">
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="text-2xl">🤖</span>
                                    <h2 className="text-xl font-semibold text-gray-800">AI & ML Services</h2>
                                </div>
                                <div className="grid md:grid-cols-3 gap-4">
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">🎭</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Sentiment Model</h3>
                                        <p className="text-sm text-gray-600">FastAPI service analyzing emotions (joy, sadness, anger, etc.)</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">✨</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">LLM API</h3>
                                        <p className="text-sm text-gray-600">LLM generating contextual response suggestions</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
                                        <div className="text-3xl mb-2 text-center">📊</div>
                                        <h3 className="font-semibold text-gray-800 mb-1">Emotion Scoring</h3>
                                        <p className="text-sm text-gray-600">Calculates sentiment percentages and trends</p>
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-2 mt-4">
                                    <span className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm font-medium">FastAPI</span>
                                    <span className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm font-medium">LLM API</span>
                                    <span className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm font-medium">ML Models</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
