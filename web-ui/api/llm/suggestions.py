"""
Vercel API endpoint for chat response suggestions
Endpoint: /api/llm/suggestions
"""

import json
import os
import asyncio
from typing import Dict, Any, List, Optional

# Import our chat response service
try:
    from .provider import ChatResponseService, GeminiProvider
except ImportError:
    # Fallback for local development
    try:
        from provider import ChatResponseService, GeminiProvider
    except ImportError:
        # If imports fail, create minimal implementation
        print("Warning: Could not import LLM provider")


def handler(request):
    """Main Vercel serverless function handler"""
    
    # Set CORS headers for all responses
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Content-Type': 'application/json'
    }
    
    try:
        method = request.method.upper()
        
        # Handle CORS preflight
        if method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': ''
            }
        
        # Handle GET request - API documentation
        elif method == 'GET':
            return handle_get(headers)
        
        # Handle POST request - Generate suggestions
        elif method == 'POST':
            return handle_post(request, headers)
        
        else:
            return create_error_response(405, f"Method {method} not allowed", headers)
            
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return create_error_response(500, f"Internal server error: {str(e)}", headers)


def handle_get(headers):
    """Handle GET requests - return API documentation"""
    api_docs = {
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
    }
    
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps(api_docs, indent=2)
    }


def handle_post(request, headers):
    """Handle POST requests - generate suggestions"""
    try:
        # Parse request body
        if hasattr(request, 'body'):
            body = request.body
        elif hasattr(request, 'data'):
            body = request.data
        else:
            # Try to get body from request object
            body = getattr(request, 'get_body', lambda: b'')()
        
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        
        if not body:
            return create_error_response(400, "Request body is required", headers)
        
        request_data = json.loads(body)
        
        # Validate request
        validation_error = validate_request(request_data)
        if validation_error:
            return create_error_response(400, validation_error, headers)
        
        # Extract data
        chat_history = request_data.get('chatHistory')
        sentiment = request_data.get('sentiment')
        options = request_data.get('options', {})
        
        # Generate suggestions synchronously (Vercel limitation)
        suggestions = asyncio.run(generate_suggestions_async(
            chat_history, sentiment, options
        ))
        
        # Return success response
        response_data = {
            "success": True,
            "data": suggestions,
            "message": "Suggestions generated successfully"
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response_data, ensure_ascii=False)
        }
        
    except json.JSONDecodeError:
        return create_error_response(400, "Invalid JSON in request body", headers)
    except Exception as e:
        print(f"Error in handle_post: {str(e)}")
        return create_error_response(500, f"Internal server error: {str(e)}", headers)


def create_error_response(status_code: int, message: str, headers: Dict[str, str]):
    """Create standardized error response"""
    error_data = {
        "success": False,
        "error": message,
        "data": None
    }
    
    return {
        'statusCode': status_code,
        'headers': headers,
        'body': json.dumps(error_data)
    }


def validate_request(request_data: Dict[str, Any]) -> Optional[str]:
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
    chat_history: Dict[str, Any], 
    sentiment: Dict[str, Any], 
    options: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate suggestions using the chat response service"""
    
    # Create service configuration
    config = create_service_config()
    
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


def create_service_config() -> Dict[str, Any]:
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
        'model': os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
    }
    
    return config


# Alternative: If the above doesn't work, try this simpler approach
def simple_handler(request):
    """Simplified handler for Vercel"""
    
    # CORS headers
    if request.method == 'OPTIONS':
        return ('', 200, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        })
    
    if request.method != 'POST':
        return (json.dumps({"error": "Method not allowed"}), 405, {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        })
    
    try:
        # Get request data
        request_json = request.get_json()
        
        if not request_json:
            return (json.dumps({"error": "No JSON body"}), 400, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            })
        
        # Basic validation
        if 'chatHistory' not in request_json or 'sentiment' not in request_json:
            return (json.dumps({"error": "Missing required fields"}), 400, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            })
        
        # Mock response for testing
        mock_suggestions = [
            {"id": "1", "content": "That sounds interesting! Tell me more about it."},
            {"id": "2", "content": "I understand. How are you feeling about that?"},
            {"id": "3", "content": "Thanks for sharing! What would you like to discuss next?"}
        ]
        
        response_data = {
            "success": True,
            "data": mock_suggestions,
            "message": "Suggestions generated successfully"
        }
        
        return (json.dumps(response_data), 200, {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        })
        
    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        })


# For testing locally
if __name__ == "__main__":
    # Test the handler locally
    class MockRequest:
        def __init__(self, method, body=None):
            self.method = method
            self.body = body
    
    # Test GET request
    get_request = MockRequest('GET')
    print("GET Response:", handler(get_request))
    
    # Test POST request
    test_data = {
        "chatHistory": {
            "chatId": "general",
            "messages": [
                {
                    "id": "1",
                    "content": "What's going on?",
                    "user": {"name": "test"},
                    "createdAt": "2025-06-07T11:29:07.095Z"
                }
            ],
            "timestamp": "2025-06-07T11:29:18.624Z"
        },
        "sentiment": {
            "sentiments": {
                "positive": 50,
                "negative": 7,
                "neutral": 76
            }
        }
    }
    
    post_request = MockRequest('POST', json.dumps(test_data))
    print("POST Response:", handler(post_request))