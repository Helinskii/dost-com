import json
import asyncio
import aiohttp
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ChatResponseService:
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
        self.model = config.get('model', 'gemini-pro')
    
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
        
        async with self.session.post(url, headers=headers, params=params, json=payload) as response:
            if not response.ok:
                raise Exception(f"Gemini API error: {response.status} {await response.text()}")
            
            data = await response.json()
            generated_text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            
            return self.parse_responses(generated_text, max_responses)


def create_service_from_env(default_provider: str = 'gemini') -> ChatResponseService:
    """Create ChatResponseService using environment variables for API keys"""
    config = {
        'default_provider': default_provider,
        'max_responses': int(os.getenv('MAX_RESPONSES', '3'))
    }
    
    print(os.getenv('GEMINI_API_KEY'))
    # Add Gemini config if API key is available
    if os.getenv('GEMINI_API_KEY'):
        config['gemini'] = {
            'api_key': os.getenv('GEMINI_API_KEY'),
            'model': os.getenv('GEMINI_MODEL', 'gemini-pro')
        }
    
    return ChatResponseService(config)


async def main():
    """Example usage of the ChatResponseService"""
    
    # Using environment variables (Recommended)
    try:
        service = create_service_from_env('gemini')
        print("Service created successfully using environment variables!")
    except Exception as e:
        print(f"Failed to create service from env: {e}")
        return
    
    # Sample data
    chat_history = {
        "chatId": "general",
        "messages": [
            {
                "id": "353fc390-3afe-49a4-a88a-64a32aed0c85",
                "content": "What's going on?",
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
            "positive": 50,
            "negative": 7,
            "neutral": 76,
            "excited": 76,
            "sad": 32,
            "angry": 26
        }
    }
    
    try:
        suggestions = await service.generate_suggestions(chat_history, sentiment)
        print("Gemini suggestions:")
        print(json.dumps(suggestions, indent=2))
        
        openai_suggestions = await service.generate_suggestions(
            chat_history, 
            sentiment, 
            {'provider': 'openai', 'max_responses': 2}
        )
        print("\nOpenAI suggestions:")
        print(json.dumps(openai_suggestions, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up sessions
        for provider in service.providers.values():
            if hasattr(provider, 'session') and provider.session:
                await provider.session.close()


class SyncChatResponseService:    
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


if __name__ == "__main__":
    asyncio.run(main())