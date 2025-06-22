import logging
logging.basicConfig(filename='llm_eval.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import json
from llm_judge import LLMJudge
from models import ChatContext, ResponseSuggestions, EvaluationResult

class EvaluationPipeline:
    """Main pipeline for running evaluations across multiple models"""
    def __init__(self, judge_provider):
        self.judge = LLMJudge(judge_provider)
        self.results_cache = []

    async def generate_and_store_suggestions(
        self,
        test_contexts: List[ChatContext],
        model_providers: Dict[str, Any],
        prompt_variants: List[str] = ['base', 'no_positivity', 'no_sentiment'],
        output_file: str = "suggestions.jsonl"
    ):
        """Generate suggestions and store them in a JSONL file for later evaluation."""
        from suggestion_generator import SuggestionGenerator
        with open(output_file, 'w') as f:
            for context_idx, context in enumerate(test_contexts):
                logging.info(f"Generating suggestions for context {context_idx + 1}/{len(test_contexts)}")
                for model_name, provider in model_providers.items():
                    sugg_gen = SuggestionGenerator(provider)
                    for prompt_variant in prompt_variants:
                        suggestions = await sugg_gen.generate_suggestions(context, prompt_variant)
                        # Store all info needed for later evaluation
                        record = {
                            'context_id': context_idx,
                            'model_name': model_name,
                            'prompt_variant': prompt_variant,
                            'suggestions': {
                                'suggestion_1': suggestions.suggestion_1,
                                'suggestion_2': suggestions.suggestion_2,
                                'suggestion_3': suggestions.suggestion_3,
                            },
                            'generation_time': suggestions.generation_time,
                            'chat_context': context,
                            'timestamp': datetime.now().isoformat()
                        }
                        # Custom encoder for dataclasses
                        def default(obj):
                            if hasattr(obj, '__dict__'):
                                return obj.__dict__
                            return str(obj)
                        f.write(json.dumps(record, default=default) + '\n')
        logging.info(f"Suggestions written to {output_file}")

    def load_suggestions(self, suggestions_file: str) -> List[Dict[str, Any]]:
        """Load suggestions from a JSONL file."""
        records = []
        with open(suggestions_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    async def evaluate_stored_suggestions(
        self,
        suggestions_records: List[Dict[str, Any]],
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """Evaluate suggestions loaded from file and return DataFrame."""
        all_results = []
        total_evaluations = len(suggestions_records)
        completed = 0
        for record in suggestions_records:
            context = self._reconstruct_context(record['chat_context'])
            suggestions = ResponseSuggestions(
                suggestion_1=record['suggestions']['suggestion_1'],
                suggestion_2=record['suggestions']['suggestion_2'],
                suggestion_3=record['suggestions']['suggestion_3'],
                model_name=record['model_name'],
                generation_time=record['generation_time'],
                prompt_variant=record['prompt_variant']
            )
            evaluation = await self.judge.evaluate(context, suggestions)
            result_record = self._create_result_record(
                record['context_id'], context, suggestions, evaluation, record['prompt_variant']
            )
            all_results.append(result_record)
            completed += 1
            logging.info(f"  Progress: {completed}/{total_evaluations} ({completed/total_evaluations*100:.1f}%)")
            if save_intermediate and completed % 10 == 0:
                self._save_intermediate_results(all_results)
        results_df = pd.DataFrame(all_results)
        self.results_cache = all_results
        return results_df

    def _reconstruct_context(self, context_dict: dict) -> ChatContext:
        # Reconstruct ChatContext and nested dataclasses from dict
        from models import ChatMessage, SentimentProbabilities, ChatContext
        chat_history = [ChatMessage(**msg) for msg in context_dict['chat_history']]
        sentiment = SentimentProbabilities(**context_dict['sentiment_probabilities'])
        return ChatContext(
            chat_history=chat_history,
            current_user=context_dict['current_user'],
            sentiment_probabilities=sentiment
        )

    async def evaluate_models_with_prompts(
        self,
        test_contexts: List[ChatContext],
        model_providers: Dict[str, Any],
        prompt_variants: List[str] = ['base', 'no_positivity', 'no_sentiment'],
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        all_results = []
        total_evaluations = len(test_contexts) * len(model_providers) * len(prompt_variants)
        completed = 0
        for context_idx, context in enumerate(test_contexts):
            logging.info(f"Processing context {context_idx + 1}/{len(test_contexts)}")
            for model_name, provider in model_providers.items():
                for prompt_variant in prompt_variants:
                    # You must generate suggestions before evaluation
                    from suggestion_generator import SuggestionGenerator
                    sugg_gen = SuggestionGenerator(provider)
                    suggestions = await sugg_gen.generate_suggestions(context, prompt_variant)
                    evaluation = await self.judge.evaluate(context, suggestions)
                    result_record = self._create_result_record(
                        context_idx, context, suggestions, evaluation, prompt_variant
                    )
                    all_results.append(result_record)
                    completed += 1
                    logging.info(f"  Progress: {completed}/{total_evaluations} ({completed/total_evaluations*100:.1f}%)")
                    if save_intermediate and completed % 10 == 0:
                        self._save_intermediate_results(all_results)
        results_df = pd.DataFrame(all_results)
        self.results_cache = all_results
        return results_df

    def _create_result_record(
        self,
        context_idx: int,
        context: ChatContext,
        suggestions: ResponseSuggestions,
        evaluation: EvaluationResult,
        prompt_variant: str
    ) -> Dict[str, Any]:
        record = {
            'context_id': context_idx,
            'model_name': suggestions.model_name,
            'prompt_variant': prompt_variant,
            'generation_time': suggestions.generation_time,
            'judge_model': evaluation.judge_model,
            'evaluation_time': evaluation.evaluation_time,
            'suggestion_1': suggestions.suggestion_1,
            'suggestion_2': suggestions.suggestion_2,
            'suggestion_3': suggestions.suggestion_3,
            'suggestion_1_relevance': evaluation.suggestion_1_metrics.relevance,
            'suggestion_1_sentiment': evaluation.suggestion_1_metrics.sentiment_alignment,
            'suggestion_1_naturalness': evaluation.suggestion_1_metrics.naturalness,
            'suggestion_1_helpfulness': evaluation.suggestion_1_metrics.helpfulness,
            'suggestion_1_positivity': evaluation.suggestion_1_metrics.positivity_impact,
            'suggestion_1_safety': evaluation.suggestion_1_metrics.safety,
            'suggestion_2_relevance': evaluation.suggestion_2_metrics.relevance,
            'suggestion_2_sentiment': evaluation.suggestion_2_metrics.sentiment_alignment,
            'suggestion_2_naturalness': evaluation.suggestion_2_metrics.naturalness,
            'suggestion_2_helpfulness': evaluation.suggestion_2_metrics.helpfulness,
            'suggestion_2_positivity': evaluation.suggestion_2_metrics.positivity_impact,
            'suggestion_2_safety': evaluation.suggestion_2_metrics.safety,
            'suggestion_3_relevance': evaluation.suggestion_3_metrics.relevance,
            'suggestion_3_sentiment': evaluation.suggestion_3_metrics.sentiment_alignment,
            'suggestion_3_naturalness': evaluation.suggestion_3_metrics.naturalness,
            'suggestion_3_helpfulness': evaluation.suggestion_3_metrics.helpfulness,
            'suggestion_3_positivity': evaluation.suggestion_3_metrics.positivity_impact,
            'suggestion_3_safety': evaluation.suggestion_3_metrics.safety,
            'diversity_score': evaluation.diversity_score,
            'best_suggestion': evaluation.best_suggestion,
            'overall_quality': evaluation.overall_quality,
            'overall_positivity_score': evaluation.overall_positivity_score,
            'reasoning': evaluation.reasoning,
            'dominant_sentiment': context.sentiment_probabilities.get_dominant(),
            'sentiment_sadness': context.sentiment_probabilities.sadness,
            'sentiment_joy': context.sentiment_probabilities.joy,
            'sentiment_love': context.sentiment_probabilities.love,
            'sentiment_anger': context.sentiment_probabilities.anger,
            'sentiment_fear': context.sentiment_probabilities.fear,
            'sentiment_unknown': context.sentiment_probabilities.unknown,
            'chat_history_length': len(context.chat_history),
            'timestamp': datetime.now().isoformat()
        }
        return record

    def _save_intermediate_results(self, results):
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'intermediate_results_{timestamp}.csv', index=False)
        logging.info(f"Saved intermediate results: intermediate_results_{timestamp}.csv")
