import os
import asyncio
from abc import ABC, abstractmethod
from typing import Any
import aiohttp
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def __init__(self, model: str = "gpt-40-mini-2025-04-14"):
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
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set in environment.")
        if genai is None:
            raise ImportError("google-generativeai is not installed.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        # Delay in seconds between requests (to avoid rate limits)
        self.request_delay = float(os.getenv("GEMINI_REQUEST_DELAY", "4.1"))  

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        loop = asyncio.get_event_loop()
        def _generate():
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            response = self.model.generate_content(prompt, generation_config=generation_config)
            # Try to extract text from response
            if hasattr(response, 'text') and response.text:
                return response.text
            # Try to extract from candidates' parts if available
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if isinstance(part, str) and part.strip():
                                return part
                            if hasattr(part, 'text') and part.text.strip():
                                return part.text
                    # Fallback: try candidate.text
                    if hasattr(candidate, 'text') and candidate.text:
                        return candidate.text
                finish_reason = getattr(response.candidates[0], 'finish_reason', None)
            else:
                finish_reason = None
            logging.error(f"No valid response from Gemini. finish_reason={finish_reason}, response={response}")
            return "No valid response from Gemini."
        return await loop.run_in_executor(None, _generate)

    def get_name(self) -> str:
        return self.model_name
