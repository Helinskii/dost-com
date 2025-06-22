import os
import asyncio
from abc import ABC, abstractmethod
from typing import Any
import aiohttp

# Optional: import genai if available for Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as resp:
                result = await resp.json()
                return result["choices"][0]["message"]["content"]

    def get_name(self) -> str:
        return self.model

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-opus-20240229"):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment.")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as resp:
                result = await resp.json()
                return result["content"]

    def get_name(self) -> str:
        return self.model

class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-1.5-pro"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")
        if genai is None:
            raise ImportError("google-generativeai is not installed.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        loop = asyncio.get_event_loop()
        def _generate():
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text if hasattr(response, 'text') else str(response)
        return await loop.run_in_executor(None, _generate)

    def get_name(self) -> str:
        return self.model_name
