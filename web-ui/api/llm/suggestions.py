"""
Vercel API endpoint for chat response suggestions
Endpoint: /api/llm/suggestions
"""

import json
import os
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from http.server import BaseHTTPRequestHandler
from abc import ABC, abstractmethod
from datetime import datetime

# Optional: Load from .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    # Load from .env.local by default, fallback to .env if not found
    load_dotenv('.env.local')
    load_dotenv('.env')  # This won't override existing variables
except ImportError:
    pass


class ChatResponseService:
    """Main service class for generating chat response suggestions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.providers = {}
        config = config or {}  # Handle None case
        self.default_provider = config.get('default_provider', 'gemini')
        self.max_responses = config.get('max_responses', 3)
        
        # Initialize providers
        self._initialize_providers(config)
    
    def _initialize_providers(self, config: Dict[str, Any]):
        """Initialize AI providers based on configuration"""
        if 'gemini' in config:
            gemini_config = config['gemini'].copy()
            # Get API key from env if not provided in config
            if 'api_key' not in gemini_config or not gemini_config['api_key']:
                gemini_config['api_key'] = os.getenv('GEMINI_API_KEY')
            self.providers['gemini'] = GeminiProvider(gemini_config)
    
    async def generate_suggestions(
        self, 
        chat_history: Dict[str, Any], 
        sentiment: Dict[str, Any], 
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Generate response suggestions using specified or default provider"""
        options = options or {}
        provider_name = options.get('provider', self.default_provider)
        max_responses = options.get('max_responses', self.max_responses)
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not configured")
        
        try:
            provider = self.providers[provider_name]
            responses = await provider.generate_suggestions(
                chat_history, sentiment, max_responses
            )
            return self._format_responses(responses, max_responses)
        
        except Exception as e:
            print(f"Error generating suggestions with {provider_name}: {e}")
            raise


    def _format_responses(self, responses: List[str], max_count: int) -> List[Dict[str, str]]:
        """Format responses to expected structure"""
        return [
            {
                "id": str(i + 1),
                "content": response.strip()
            }
            for i, response in enumerate(responses[:max_count])
            if response.strip()
        ]


class BaseProvider(ABC):
    """Base class for AI providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.model = config.get('model')
        self.session = None
        
        # Validate API key
        if not self.api_key:
            provider_name = self.__class__.__name__.replace('Provider', '').lower()
            raise ValueError(f"API key is required for {provider_name}. Set it in config or environment variable.")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def generate_suggestions(
        self, 
        chat_history: Dict[str, Any], 
        sentiment: Dict[str, Any], 
        max_responses: int = 3
    ) -> List[str]:
        """Generate response suggestions - to be implemented by each provider"""
        pass
    
    def build_prompt(self, chat_history: Dict[str, Any], sentiment: Dict[str, Any]) -> str:
        """Build prompt for AI model"""
        conversation_context = self._build_conversation_context(chat_history)
        sentiment_context = self._build_sentiment_context(sentiment)
        
        return f"""You are a helpful assistant providing response suggestions for a chat application.

CONTEXT:
{conversation_context}

CURRENT SENTIMENT ANALYSIS:
{sentiment_context}

TASK: Generate 1-3 short, contextually appropriate response suggestions (each should be a single line, max 150 characters).

REQUIREMENTS:
- Match the conversation tone and sentiment
- Be helpful and engaging
- Keep responses concise and natural
- Consider the emotional context from sentiment analysis

Provide only the response suggestions, one per line, without numbering or formatting."""
    
    def _build_conversation_context(self, chat_history: Dict[str, Any]) -> str:
        """Build conversation context from recent messages"""
        messages = chat_history.get('messages', [])
        recent_messages = messages[-5:]  # Last 5 messages for context
        
        context_lines = []
        for msg in recent_messages:
            user_name = msg.get('user', {}).get('name', 'User')
            content = msg.get('content', '')
            context_lines.append(f"{user_name}: {content}")
        
        return '\n'.join(context_lines)
    
    def _build_sentiment_context(self, sentiment: Dict[str, Any]) -> str:
        """Build sentiment context from sentiment analysis"""
        sentiments = sentiment.get('sentiments', {})
        
        # Sort sentiments by value and get top 3
        sorted_sentiments = sorted(
            sentiments.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        dominant = ', '.join([f"{key}: {value}%" for key, value in sorted_sentiments])
        return f"Current mood indicators: {dominant}"
    
    def parse_responses(self, text: str, max_responses: int) -> List[str]:
        """Parse generated text into individual responses"""
        responses = [
            line.strip() 
            for line in text.split('\n') 
            if line.strip() and len(line.strip()) > 0
        ]
        return responses[:max_responses]


class GeminiProvider(BaseProvider):
    """Google Gemini API Provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')
        # Use the correct model name for Gemini API
        self.model = config.get('model', 'gemini-1.5-flash')
    
    async def generate_suggestions(
        self, 
        chat_history: Dict[str, Any], 
        sentiment: Dict[str, Any], 
        max_responses: int = 3
    ) -> List[str]:
        """Generate suggestions using Gemini API"""
        prompt = self.build_prompt(chat_history, sentiment)
        
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.api_key}
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 200,
                "topP": 0.9,
            }
        }
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(url, headers=headers, params=params, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error: {response.status} {error_text}")
                
                data = await response.json()
                
                # Check if we got a valid response
                if 'candidates' not in data or not data['candidates']:
                    raise Exception("No candidates returned from Gemini API")
                
                candidate = data['candidates'][0]
                if 'content' not in candidate or 'parts' not in candidate['content']:
                    raise Exception("Invalid response structure from Gemini API")
                
                generated_text = candidate['content']['parts'][0].get('text', '')
                
                if not generated_text:
                    raise Exception("Empty response from Gemini API")
                
                return self.parse_responses(generated_text, max_responses)
                
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP error when calling Gemini API: {e}")
        except Exception as e:
            raise Exception(f"Error processing Gemini API response: {e}")


# Configuration utilities
def load_env_config(env_file: Optional[str] = None):
    """Load environment variables from specified file or default locations"""
    try:
        from dotenv import load_dotenv
        
        if env_file:
            # Load specific file
            load_dotenv(env_file)
        else:
            # Load in priority order: .env.local -> .env.development -> .env
            env_files = ['.env.local', '.env.development', '.env']
            for file in env_files:
                if os.path.exists(file):
                    load_dotenv(file)
                    print(f"Loaded environment from: {file}")
                    break
    except ImportError:
        print("python-dotenv not installed. Using system environment variables only.")


# Utility function to create service with environment variables
def create_service_from_env(
    default_provider: str = 'gemini', 
    env_file: Optional[str] = None
) -> ChatResponseService:
    """Create ChatResponseService using environment variables for API keys"""
    
    # Load environment variables from specified file
    load_env_config(env_file)
    
    config = {
        'default_provider': default_provider,
        'max_responses': int(os.getenv('MAX_RESPONSES', '3'))
    }
    
    # Add Gemini config if API key is available
    if os.getenv('GEMINI_API_KEY'):
        config['gemini'] = {
            'api_key': os.getenv('GEMINI_API_KEY'),
            'model': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        }
    
    return ChatResponseService(config)


# Usage Example and Testing
async def main():
    """Example usage of the ChatResponseService"""
    
    # Using .env.local specifically
    try:
        service = create_service_from_env('gemini', '.env.local')
        print("Service created successfully using .env.local!")
    except Exception as e:
        print(f"Failed to create service from .env.local: {e}")
        
        # Fallback: Try default environment loading
        try:
            service = create_service_from_env('gemini')
            print("Service created using default environment loading!")
        except Exception as e2:
            print(f"Failed to create service: {e2}")
            return
    
    # Sample data
    chat_history = {
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
    }
    
    sentiment = {
        "sentiments": {
            "positive": 1,
            "negative": 80,
            "neutral": 2,
            "excited": 10,
            "sad": 50,
            "angry": 99
        }
    }
    
    try:
        # Generate suggestions using default provider (Gemini)
        suggestions = await service.generate_suggestions(chat_history, sentiment)
        print("Gemini suggestions:")
        print(json.dumps(suggestions, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up sessions
        for provider in service.providers.values():
            if hasattr(provider, 'session') and provider.session:
                await provider.session.close()


# Synchronous wrapper for easier integration
class SyncChatResponseService:
    """Synchronous wrapper for the async ChatResponseService"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.async_service = ChatResponseService(config)
    
    def generate_suggestions(
        self, 
        chat_history: Dict[str, Any], 
        sentiment: Dict[str, Any], 
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Synchronous version of generate_suggestions"""
        return asyncio.run(
            self.async_service.generate_suggestions(chat_history, sentiment, options)
        )


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
    #asyncio.run(test())