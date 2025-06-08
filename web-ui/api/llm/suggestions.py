from http.server import BaseHTTPRequestHandler
import json
import os
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional

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
            
            # Validate required fields
            if 'chatHistory' not in data or 'sentiment' not in data:
                self.send_error_response(400, "Missing chatHistory or sentiment")
                return
            
            # Generate suggestions using Gemini
            suggestions = asyncio.run(self.generate_suggestions(
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
    
    async def generate_suggestions(self, chat_history: Dict, sentiment: Dict, options: Dict) -> List[Dict[str, str]]:
        """Generate suggestions using Gemini API"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
        max_responses = options.get('maxResponses', 3)
        
        # Build prompt
        prompt = self.build_prompt(chat_history, sentiment)
        
        # Call Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {'Content-Type': 'application/json'}
        params = {'key': api_key}
        
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
                
                # Parse responses
                responses = [
                    line.strip() 
                    for line in generated_text.split('\n') 
                    if line.strip()
                ][:max_responses]
                
                return [
                    {"id": str(i + 1), "content": response}
                    for i, response in enumerate(responses)
                ]
    
    def build_prompt(self, chat_history: Dict, sentiment: Dict) -> str:
        """Build prompt for Gemini"""
        messages = chat_history.get('messages', [])
        recent_messages = messages[-5:]  # Last 5 messages
        
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

CONTEXT:
{context}

CURRENT SENTIMENT: {dominant}

Generate 1-3 short, contextually appropriate response suggestions (max 150 characters each).
Be helpful, engaging, and match the conversation tone.

Provide only the suggestions, one per line, without numbering."""