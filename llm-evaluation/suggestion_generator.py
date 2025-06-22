import time
from typing import List
from models import ChatContext, ResponseSuggestions
from prompts import PromptVariants

class SuggestionGenerator:
    """Generates response suggestions using specified LLM"""
    def __init__(self, provider):
        self.provider = provider
        self.prompt_variants = {
            'base': PromptVariants.get_base_prompt(),
            'no_positivity': PromptVariants.get_no_positivity_prompt(),
            'no_sentiment': PromptVariants.get_no_sentiment_prompt()
        }

    async def generate_suggestions(self, context: ChatContext, prompt_variant: str = 'base') -> ResponseSuggestions:
        prompt = self._build_generation_prompt(context, prompt_variant)
        start_time = time.time()
        try:
            response = await self.provider.generate(prompt, temperature=0.7, max_tokens=300)
            suggestions = self._parse_suggestions(response)
            generation_time = time.time() - start_time
            while len(suggestions) < 3:
                suggestions.append("")
            suggestions = suggestions[:3]
            return ResponseSuggestions(
                suggestion_1=suggestions[0],
                suggestion_2=suggestions[1],
                suggestion_3=suggestions[2],
                model_name=self.provider.get_name(),
                generation_time=generation_time,
                prompt_variant=prompt_variant
            )
        except Exception as e:
            import logging
            logging.error(f"Error generating suggestions: {e}")
            raise

    def _build_generation_prompt(self, context: ChatContext, prompt_variant: str) -> str:
        context_str = "\n".join([
            f"{msg.created_at} - {msg.user_name}: {msg.content}"
            for msg in context.chat_history[-10:]
        ])
        dominant_emotion = context.sentiment_probabilities.get_dominant()
        dominant_score = getattr(context.sentiment_probabilities, dominant_emotion) * 100
        prompt_template = self.prompt_variants[prompt_variant]
        if prompt_variant == 'no_sentiment':
            prompt = prompt_template.format(
                username=context.current_user,
                context=context_str
            )
        else:
            prompt = prompt_template.format(
                username=context.current_user,
                dominant=f"{dominant_emotion.upper()} ({dominant_score:.0f})",
                context=context_str
            )
        return prompt

    def _parse_suggestions(self, response: str) -> List[str]:
        suggestions = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('1.', '2.', '3.', '-', '*')):
                suggestions.append(line)
            elif line.startswith(('1.', '2.', '3.')):
                suggestions.append(line.split('.', 1)[1].strip())
            elif line.startswith(('-', '*')):
                suggestions.append(line[1:].strip())
        return suggestions[:3]
