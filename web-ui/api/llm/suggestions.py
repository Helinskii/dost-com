"""
Vercel API endpoint for chat response suggestions
Endpoint: /api/llm/suggestions
"""

import json
import os
import asyncio
from typing import Dict, Any, List, Optional
from http.server import BaseHTTPRequestHandler

# Import our chat response service
try:
    from .provider import ChatResponseService, GeminiProvider
except ImportError:
    # Fallback for local development
    from provider import ChatResponseService, GeminiProvider


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler"""
    
    def do_POST(self):
        """Handle POST requests for chat suggestions"""
        try:
            # Set CORS headers
            self.send_cors_headers()
            
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error_response(400, "Request body is required")
                return
            
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode('utf-8'))
            
            # Validate request structure
            validation_error = self.validate_request(request_data)
            if validation_error:
                self.send_error_response(400, validation_error)
                return
            
            # Extract data from request
            chat_history = request_data.get('chatHistory')
            sentiment = request_data.get('sentiment')
            options = request_data.get('options', {})
            
            # Generate suggestions
            suggestions = asyncio.run(self.generate_suggestions_async(
                chat_history, sentiment, options
            ))
            
            # Send successful response
            self.send_json_response(200, {
                "success": True,
                "data": suggestions,
                "message": "Suggestions generated successfully"
            })
            
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON in request body")
        except Exception as e:
            print(f"Error in suggestions API: {str(e)}")
            self.send_error_response(500, f"Internal server error: {str(e)}")
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self.send_cors_headers()
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests - return API information"""
        try:
            self.send_cors_headers()
            self.send_json_response(200, {
                "name": "Chat Response Suggestions API",
                "version": "1.0.0",
                "description": "Generate AI-powered chat response suggestions",
                "endpoints": {
                    "POST /api/llm/suggestions": "Generate chat response suggestions"
                },
                "usage": {
                    "method": "POST",
                    "content_type": "application/json",
                    "body": {
                        "chatHistory": {
                            "chatId": "string",
                            "messages": [
                                {
                                    "id": "string",
                                    "content": "string",
                                    "user": {"name": "string"},
                                    "createdAt": "string"
                                }
                            ],
                            "timestamp": "string"
                        },
                        "sentiment": {
                            "sentiments": {
                                "positive": "number",
                                "negative": "number",
                                "neutral": "number",
                                "excited": "number",
                                "sad": "number",
                                "angry": "number"
                            }
                        },
                        "options": {
                            "provider": "string (optional, default: gemini)",
                            "maxResponses": "number (optional, default: 3)"
                        }
                    }
                }
            })
        except Exception as e:
            self.send_error_response(500, f"Error: {str(e)}")
    
    def send_cors_headers(self):
        """Send CORS headers for cross-origin requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Content-Type', 'application/json')
    
    def send_json_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_json = json.dumps(data, ensure_ascii=False, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def send_error_response(self, status_code: int, message: str):
        """Send error response"""
        self.send_json_response(status_code, {
            "success": False,
            "error": message,
            "data": None
        })
    
    def validate_request(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Validate request structure and return error message if invalid"""
        
        # Check required fields
        if 'chatHistory' not in request_data:
            return "Missing required field: chatHistory"
        
        if 'sentiment' not in request_data:
            return "Missing required field: sentiment"
        
        # Validate chatHistory structure
        chat_history = request_data['chatHistory']
        if not isinstance(chat_history, dict):
            return "chatHistory must be an object"
        
        if 'messages' not in chat_history:
            return "chatHistory.messages is required"
        
        if not isinstance(chat_history['messages'], list):
            return "chatHistory.messages must be an array"
        
        # Validate at least one message exists
        if len(chat_history['messages']) == 0:
            return "At least one message is required in chatHistory.messages"
        
        # Validate message structure
        for i, message in enumerate(chat_history['messages']):
            if not isinstance(message, dict):
                return f"Message at index {i} must be an object"
            
            required_fields = ['content', 'user']
            for field in required_fields:
                if field not in message:
                    return f"Message at index {i} missing required field: {field}"
            
            if 'name' not in message['user']:
                return f"Message at index {i} user.name is required"
        
        # Validate sentiment structure
        sentiment = request_data['sentiment']
        if not isinstance(sentiment, dict):
            return "sentiment must be an object"
        
        if 'sentiments' not in sentiment:
            return "sentiment.sentiments is required"
        
        if not isinstance(sentiment['sentiments'], dict):
            return "sentiment.sentiments must be an object"
        
        # Validate options if provided
        options = request_data.get('options', {})
        if not isinstance(options, dict):
            return "options must be an object"
        
        # Validate maxResponses if provided
        max_responses = options.get('maxResponses')
        if max_responses is not None:
            if not isinstance(max_responses, int) or max_responses < 1 or max_responses > 10:
                return "options.maxResponses must be an integer between 1 and 10"
        
        return None  # No validation errors
    
    async def generate_suggestions_async(
        self, 
        chat_history: Dict[str, Any], 
        sentiment: Dict[str, Any], 
        options: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate suggestions using the chat response service"""
        
        # Create service configuration
        config = self.create_service_config()
        
        # Initialize service
        service = ChatResponseService(config)
        
        try:
            # Generate suggestions
            suggestions = await service.generate_suggestions(
                chat_history, sentiment, options
            )
            return suggestions
            
        finally:
            # Clean up sessions
            for provider in service.providers.values():
                if hasattr(provider, 'session') and provider.session:
                    await provider.session.close()
    
    def create_service_config(self) -> Dict[str, Any]:
        """Create service configuration from environment variables"""
        config = {
            'default_provider': 'gemini',
            'max_responses': int(os.getenv('MAX_RESPONSES', '3'))
        }
        
        # Add Gemini config if API key is available
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Please set it in your Vercel environment variables."
            )
        
        config['gemini'] = {
            'api_key': gemini_api_key,
            'model': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        }
        
        return config


# Alternative function-based handler for Vercel (if needed)
async def suggestions_handler(request_body: str) -> Dict[str, Any]:
    """
    Alternative async function handler for Vercel
    Can be used if the class-based approach doesn't work
    """
    try:
        # Parse request
        request_data = json.loads(request_body)
        
        # Basic validation
        if 'chatHistory' not in request_data or 'sentiment' not in request_data:
            return {
                "success": False,
                "error": "Missing required fields: chatHistory and sentiment",
                "data": None
            }
        
        # Create service
        config = {
            'default_provider': 'gemini',
            'max_responses': int(os.getenv('MAX_RESPONSES', '3')),
            'gemini': {
                'api_key': os.getenv('GEMINI_API_KEY'),
                'model': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
            }
        }
        
        service = ChatResponseService(config)
        
        # Generate suggestions
        suggestions = await service.generate_suggestions(
            request_data['chatHistory'],
            request_data['sentiment'],
            request_data.get('options', {})
        )
        
        return {
            "success": True,
            "data": suggestions,
            "message": "Suggestions generated successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": None
        }


# For testing locally
if __name__ == "__main__":
    import asyncio
    
    # Sample test data
    test_data = {
        "chatHistory": {
            "chatId": "general",
            "messages": [
                {
                    "id": "353fc390-3afe-49a4-a88a-64a32aed0c85",
                    "content": "I can't believe you did that!",
                    "user": {
                        "name": "god"
                    },
                    "createdAt": "2025-06-07T11:29:07.095Z"
                },
                {
                    "id": "353fc390-3afe-49a4-a88a-64a32aed0c85",
                    "content": "I'm so mad at you right now!",
                    "user": {
                        "name": "god"
                    },
                    "createdAt": "2025-06-07T11:29:07.095Z"
                },
                {
                    "id": "353fc390-3afe-49a4-a88a-64a32aed0c85",
                    "content": "Don't message me anymore!",
                    "user": {
                        "name": "god"
                    },
                    "createdAt": "2025-06-07T11:29:07.095Z"
                }
            ],
            "timestamp": "2025-06-07T11:29:18.624Z"
        },
        "sentiment": {
            "sentiments": {
                "positive": 1,
                "negative": 80,
                "neutral": 2,
                "excited": 10,
                "sad": 50,
                "angry": 99
            }
        },
        "options": {
            "maxResponses": 3
        }
    }
    
    async def test():
        result = await suggestions_handler(json.dumps(test_data))
        print(json.dumps(result, indent=2))
    
    # Uncomment to test locally
    asyncio.run(test())