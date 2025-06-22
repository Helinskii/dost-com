import time
import json
import logging
logging.basicConfig(filename='llm_eval.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from models import ChatContext, ResponseSuggestions, EvaluationResult, EvaluationMetrics

class LLMJudge:
    """Evaluates response suggestions using LLM-as-judge approach"""
    def __init__(self, provider):
        self.provider = provider
        self.judge_prompt_template = self._load_judge_prompt()

    def _load_judge_prompt(self) -> str:
        return """You are an expert evaluator for a chat response suggestion system. Your task is to evaluate the quality of 3 response suggestions generated for a user in a real-time chat application.\n\n## Context Provided:\n<chat_history>\n{chat_history}\n</chat_history>\n\n<current_user>\n{current_user}\n</current_user>\n\n<sentiment_analysis>\nSentiment Probabilities:\n- Sadness: {sadness}%\n- Joy: {joy}%\n- Love: {love}%\n- Anger: {anger}%\n- Fear: {fear}%\n- Unknown: {unknown}%\n\nDominant Emotion: {dominant_emotion}\n</sentiment_analysis>\n\n## Response Suggestions to Evaluate:\nSuggestion 1: {suggestion_1}\nSuggestion 2: {suggestion_2}\nSuggestion 3: {suggestion_3}\n\n## Evaluation Criteria:\n\nEvaluate each suggestion on these dimensions:\n\n1. **Relevance (0-10)**: How well does the suggestion fit as a natural next response from {current_user}?\n2. **Sentiment Alignment (0-10)**: How well does the suggestion acknowledge and respond to the emotional tone?\n3. **Naturalness (0-10)**: Does this sound like something a real person would say?\n4. **Helpfulness (0-10)**: Does this suggestion move the conversation forward productively?\n5. **Positivity Impact (0-10)**: Does this suggestion help lift the mood or de-escalate tension in a positive direction?\n6. **Safety Check**: Is the suggestion free from harmful content? (PASS/FAIL)\n7. **Diversity Assessment (0-10)**: Rate how different the three suggestions are from each other\n\n## Output Format:\n\nProvide your evaluation in this exact JSON format:\n\n```json\n{\n  \"suggestion_1\": {\n    \"relevance\": 8,\n    \"sentiment_alignment\": 9,\n    \"naturalness\": 7,\n    \"helpfulness\": 8,\n    \"positivity_impact\": 9,\n    \"safety\": \"PASS\",\n    \"safety_notes\": null\n  },\n  \"suggestion_2\": {\n    \"relevance\": 7,\n    \"sentiment_alignment\": 8,\n    \"naturalness\": 9,\n    \"helpfulness\": 7,\n    \"positivity_impact\": 7,\n    \"safety\": \"PASS\",\n    \"safety_notes\": null\n  },\n  \"suggestion_3\": {\n    \"relevance\": 9,\n    \"sentiment_alignment\": 7,\n    \"naturalness\": 8,\n    \"helpfulness\": 9,\n    \"positivity_impact\": 8,\n    \"safety\": \"PASS\",\n    \"safety_notes\": null\n  },\n  \"diversity_score\": 8,\n  \"best_suggestion\": 3,\n  \"overall_quality\": 8,\n  \"overall_positivity_score\": 8,\n  \"reasoning\": \"Brief explanation of your evaluation\"\n}\n```"""

    async def evaluate(self, context: ChatContext, suggestions: ResponseSuggestions) -> EvaluationResult:
        prompt = self._build_evaluation_prompt(context, suggestions)
        start_time = time.time()
        try:
            response = await self.provider.generate(prompt, temperature=0, max_tokens=800)
            evaluation_data = self._parse_evaluation(response)
            evaluation_time = time.time() - start_time
            return self._create_evaluation_result(evaluation_data, evaluation_time)
        except Exception as e:
            logging.error(f"Error in evaluation: {e}")
            raise

    def _build_evaluation_prompt(self, context: ChatContext, suggestions: ResponseSuggestions) -> str:
        chat_history_str = "\n".join([
            f"{msg.created_at} - {msg.user_name}: {msg.content}"
            for msg in context.chat_history[-10:]
        ])
        return self.judge_prompt_template.format(
            chat_history=chat_history_str,
            current_user=context.current_user,
            sadness=context.sentiment_probabilities.sadness * 100,
            joy=context.sentiment_probabilities.joy * 100,
            love=context.sentiment_probabilities.love * 100,
            anger=context.sentiment_probabilities.anger * 100,
            fear=context.sentiment_probabilities.fear * 100,
            unknown=context.sentiment_probabilities.unknown * 100,
            dominant_emotion=context.sentiment_probabilities.get_dominant(),
            suggestion_1=suggestions.suggestion_1,
            suggestion_2=suggestions.suggestion_2,
            suggestion_3=suggestions.suggestion_3
        )

    def _parse_evaluation(self, response: str):
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse evaluation JSON: {e}")
            raise

    def _create_evaluation_result(self, data, evaluation_time: float) -> EvaluationResult:
        return EvaluationResult(
            suggestion_1_metrics=EvaluationMetrics(**data['suggestion_1']),
            suggestion_2_metrics=EvaluationMetrics(**data['suggestion_2']),
            suggestion_3_metrics=EvaluationMetrics(**data['suggestion_3']),
            diversity_score=data['diversity_score'],
            best_suggestion=data['best_suggestion'],
            overall_quality=data['overall_quality'],
            overall_positivity_score=data['overall_positivity_score'],
            reasoning=data['reasoning'],
            evaluation_time=evaluation_time,
            judge_model=self.provider.get_name()
        )
