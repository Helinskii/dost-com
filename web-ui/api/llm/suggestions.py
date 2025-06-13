from http.server import BaseHTTPRequestHandler
import json
import os
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

# Provider classes
class BaseProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.model = config.get('model')
        self.base_url = config.get('base_url')
        
        if not self.api_key:
            raise ValueError(f"API key required for {self.__class__.__name__}")
    
    @abstractmethod
    async def generate_suggestions(self, username, chat_history: Dict, sentiment: Dict, max_responses: int) -> List[str]:
        pass
    
    def build_prompt(self, username, chat_history: Dict, sentiment: Dict) -> str:
        messages = chat_history.get('messages', [])
        recent_messages = messages[-5:]
        
        context = '\n'.join([
            f"{msg.get('user', {}).get('name', 'User')}: {msg.get('content', '')}"
            for msg in recent_messages
        ])
        
        sentiments = sentiment.get('sentiments', {})
        dominant = ', '.join([
            f"{k}: {v}%" 
            for k, v in sorted(sentiments.items(), key=lambda x: x[1], reverse=True)[:3]
        ])
        
        return f"""You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: {username}

Your task is to generate response suggestions for {username}, based only on messages from other participants in the chat history.  
Use {username}'s previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CURRENT SENTIMENT (0-100): {dominant}  
The sentiment reflects the emotional tone of the entire conversation and should be used to guide de-escalation and promote a positive, relationship-preserving response.

CONTEXT:
{context}

Generate 1-3 short response suggestions (max 150 characters each) from {username}'s perspective that:
- Respond directly and appropriately to other participants' most recent messages
- De-escalate tension and promote a positive tone
- Show empathy, understanding, or warmth
- Help preserve or improve the relationship
- Make the other person feel heard and better

Provide only the suggestions, one per line, without numbering.
"""


class GeminiProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')
        self.model = config.get('model', 'gemini-2.0-flash')
    
    async def generate_suggestions(self, username, chat_history: Dict, sentiment: Dict, max_responses: int = 3) -> List[str]:
        prompt = self.build_prompt(username, chat_history, sentiment)
        
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.api_key}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 200,
                "topP": 0.9,
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, params=params, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error: {response.status} {error_text}")
                
                data = await response.json()
                
                if 'candidates' not in data or not data['candidates']:
                    raise Exception("No response from Gemini API")
                
                generated_text = data['candidates'][0]['content']['parts'][0].get('text', '')
                
                if not generated_text:
                    raise Exception("Empty response from Gemini API")
                
                responses = [
                    line.strip() 
                    for line in generated_text.split('\n') 
                    if line.strip()
                ][:max_responses]
                
                return responses


class ChatResponseService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.providers = {}
        config = config or {}
        self.default_provider = config.get('default_provider', 'gemini')
        self.max_responses = config.get('max_responses', 3)
        
        self._initialize_providers(config)
    
    def _initialize_providers(self, config: Dict[str, Any]):
        if 'gemini' in config:
            gemini_config = config['gemini'].copy()
            if 'api_key' not in gemini_config or not gemini_config['api_key']:
                gemini_config['api_key'] = os.getenv('GEMINI_API_KEY')
            self.providers['gemini'] = GeminiProvider(gemini_config)
    
    async def generate_suggestions(self, username, chat_history: Dict, sentiment: Dict, options: Optional[Dict] = None) -> List[Dict[str, str]]:
        options = options or {}
        provider_name = options.get('provider', self.default_provider)
        max_responses = options.get('maxResponses', self.max_responses)
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not configured")
        
        provider = self.providers[provider_name]
        responses = await provider.generate_suggestions(username, chat_history, sentiment, max_responses)
        
        return [
            {"id": str(i + 1), "content": response.strip()}
            for i, response in enumerate(responses)
            if response.strip()
        ]


# Service factory
def create_service() -> ChatResponseService:
    config = {
        'default_provider': 'gemini',
        'max_responses': int(os.getenv('MAX_RESPONSES', '3')),
        'gemini': {
            'api_key': os.getenv('GEMINI_API_KEY'),
            'model': os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
        }
    }
    return ChatResponseService(config)


# Handler class
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "name": "Chat Suggestions API",
            "status": "running",
            "version": "1.0.0",
            "gemini_configured": bool(os.getenv('GEMINI_API_KEY'))
        }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            if 'chatHistory' not in data or 'sentiment' not in data or 'username' not in data:
                self.send_error_response(400, "Missing chatHistory or sentimen or username")
                return
            
            service = create_service()
            suggestions = asyncio.run(service.generate_suggestions(
                data['username'],
                data['chatHistory'], 
                data['sentiment'], 
                data.get('options', {})
            ))
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "success": True,
                "data": suggestions,
                "message": "Suggestions generated successfully"
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error_response(500, str(e))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_error_response(self, status_code: int, message: str):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {"success": False, "error": message}
        self.wfile.write(json.dumps(error_response).encode())

def test_local():
    test_data = {
        "username": "Alice",
        "chatHistory": {
            "chatId": "general",
            "messages": [
                {
                    "id": "1",
                    "content": "I'm feeling a bit overwhelmed with work lately",
                    "user": {"name": "Alice"},
                    "createdAt": "2025-06-07T11:29:07.095Z"
                },
                {
                    "id": "2",
                    "content": "What happened?",
                    "user": {"name": "John"},
                    "createdAt": "2025-06-07T11:35:07.095Z"
                }
            ],
            "timestamp": "2025-06-07T11:29:18.624Z"
        },
        "sentiment": {
            "sentiments": {
                "positive": 20,
                "negative": 60,
                "neutral": 30,
                "excited": 10,
                "sad": 70,
                "angry": 15
            }
        },
        "options": {
            "maxResponses": 3
        }
    }
   
    # Test the service
    service = create_service()
    suggestions = asyncio.run(service.generate_suggestions(
        test_data['chatHistory'],
        test_data['sentiment'],
        test_data['options']
    ))
   
    print("Generated suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion['content']}")

# if __name__ == "__main__":
    # Sample input data
    # test_local()
