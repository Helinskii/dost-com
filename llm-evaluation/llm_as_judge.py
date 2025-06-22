"""
LLM-as-Judge Evaluation System for Chat Response Suggestions
A comprehensive evaluation framework for comparing LLM performance in generating chat suggestions
Modified version with sentiment probabilities and positivity scoring
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from enum import Enum
import time
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============== Data Models ==============

@dataclass
class ChatMessage:
    """Represents a single chat message"""
    id: str
    content: str
    user_name: str
    created_at: str

@dataclass
class SentimentProbabilities:
    """Sentiment probabilities for each emotion"""
    sadness: float
    joy: float
    love: float
    anger: float
    fear: float
    unknown: float
    
    def get_dominant(self) -> str:
        """Get the dominant emotion"""
        emotions = {
            'sadness': self.sadness,
            'joy': self.joy,
            'love': self.love,
            'anger': self.anger,
            'fear': self.fear,
            'unknown': self.unknown
        }
        return max(emotions, key=emotions.get)

@dataclass
class ChatContext:
    """Represents the context for generating response suggestions"""
    chat_history: List[ChatMessage]
    current_user: str
    sentiment_probabilities: SentimentProbabilities

@dataclass
class ResponseSuggestions:
    """Container for the three response suggestions"""
    suggestion_1: str
    suggestion_2: str
    suggestion_3: str
    model_name: str
    generation_time: float
    prompt_variant: str  # Which prompt variant was used

@dataclass
class EvaluationMetrics:
    """Metrics for a single suggestion"""
    relevance: float
    sentiment_alignment: float
    naturalness: float
    helpfulness: float
    positivity_impact: float  # New metric for mood lifting
    safety: str
    safety_notes: Optional[str] = None

@dataclass
class EvaluationResult:
    """Complete evaluation result for a set of suggestions"""
    suggestion_1_metrics: EvaluationMetrics
    suggestion_2_metrics: EvaluationMetrics
    suggestion_3_metrics: EvaluationMetrics
    diversity_score: float
    best_suggestion: int
    overall_quality: float
    overall_positivity_score: float  # New overall metric
    reasoning: str
    evaluation_time: float
    judge_model: str

# ============== LLM Provider Interfaces ==============

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the LLM"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the model"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
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
            async with session.post(self.base_url, headers=headers, json=data) as response:
                result = await response.json()
                return result['choices'][0]['message']['content']
    
    def get_name(self) -> str:
        return self.model

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, model: str = "claude-3-opus-20240229"):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
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
            async with session.post(self.base_url, headers=headers, json=data) as response:
                result = await response.json()
                return result['content'][0]['text']
    
    def get_name(self) -> str:
        return self.model

class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""
    
    def __init__(self, model: str = "gemini-1.5-pro"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Gemini API is synchronous, so we'll run it in executor
        import asyncio
        loop = asyncio.get_event_loop()
        
        def _generate():
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        
        return await loop.run_in_executor(None, _generate)
    
    def get_name(self) -> str:
        return self.model_name

# ============== Prompt Variants ==============

class PromptVariants:
    """Different prompt styles for testing"""
    
    @staticmethod
    def get_base_prompt() -> str:
        """Base prompt with all features"""
        return """You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: {username}

Your task is to generate response suggestions for {username}, based only on messages from other participants in the chat history.  
Use {username}'s previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CURRENT SENTIMENT (0-100): {dominant}  
The sentiment reflects the emotional tone of the entire conversation and should be used to guide de-escalation and promote a positive, relationship-preserving response.

CONTEXT:
{context}

Generate 1-3 short response suggestions (max 150 characters each) from {username}'s perspective that:
- Respond directly and appropriately to other participants' most recent messages
- De-escalate tension and promote a positive tone
- Show empathy, understanding, or warmth
- Help preserve or improve the relationship
- Make the other person feel heard and better

Provide only the suggestions, one per line, without numbering."""

    @staticmethod
    def get_no_positivity_prompt() -> str:
        """Prompt without positivity requirements"""
        return """You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: {username}

Your task is to generate response suggestions for {username}, based only on messages from other participants in the chat history.  
Use {username}'s previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CURRENT SENTIMENT (0-100): {dominant}  
The sentiment reflects the emotional tone of the entire conversation.

CONTEXT:
{context}

Generate 1-3 short response suggestions (max 150 characters each) from {username}'s perspective that:
- Respond directly and appropriately to other participants' most recent messages
- Are contextually relevant and natural
- Maintain the conversation flow

Provide only the suggestions, one per line, without numbering."""

    @staticmethod
    def get_no_sentiment_prompt() -> str:
        """Prompt without sentiment information"""
        return """You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: {username}

Your task is to generate response suggestions for {username}, based only on messages from other participants in the chat history.  
Use {username}'s previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CONTEXT:
{context}

Generate 1-3 short response suggestions (max 150 characters each) from {username}'s perspective that:
- Respond directly and appropriately to other participants' most recent messages
- De-escalate tension and promote a positive tone
- Show empathy, understanding, or warmth
- Help preserve or improve the relationship
- Make the other person feel heard and better

Provide only the suggestions, one per line, without numbering."""

# ============== Suggestion Generator ==============

class SuggestionGenerator:
    """Generates response suggestions using specified LLM"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.prompt_variants = {
            'base': PromptVariants.get_base_prompt(),
            'no_positivity': PromptVariants.get_no_positivity_prompt(),
            'no_sentiment': PromptVariants.get_no_sentiment_prompt()
        }
    
    async def generate_suggestions(self, context: ChatContext, prompt_variant: str = 'base') -> ResponseSuggestions:
        """Generate three response suggestions based on context"""
        
        prompt = self._build_generation_prompt(context, prompt_variant)
        start_time = time.time()
        
        try:
            response = await self.provider.generate(prompt, temperature=0.7, max_tokens=300)
            suggestions = self._parse_suggestions(response)
            generation_time = time.time() - start_time
            
            # Ensure we have exactly 3 suggestions
            while len(suggestions) < 3:
                suggestions.append(suggestions[-1] if suggestions else "I understand.")
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
            logger.error(f"Error generating suggestions: {e}")
            raise
    
    def _build_generation_prompt(self, context: ChatContext, prompt_variant: str) -> str:
        """Build the prompt for suggestion generation"""
        
        # Format chat history
        context_str = "\n".join([
            f"{msg.created_at} - {msg.user_name}: {msg.content}"
            for msg in context.chat_history[-10:]  # Last 10 messages
        ])
        
        # Get dominant sentiment
        dominant_emotion = context.sentiment_probabilities.get_dominant()
        dominant_score = getattr(context.sentiment_probabilities, dominant_emotion) * 100
        
        prompt_template = self.prompt_variants[prompt_variant]
        
        # Handle different prompt variants
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
        """Parse suggestions from LLM response"""
        suggestions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('1.', '2.', '3.', '-', '*')):
                suggestions.append(line)
            elif line.startswith(('1.', '2.', '3.')):
                suggestions.append(line[2:].strip())
            elif line.startswith(('-', '*')):
                suggestions.append(line[1:].strip())
        
        return suggestions[:3]

# ============== LLM Judge ==============

class LLMJudge:
    """Evaluates response suggestions using LLM-as-judge approach"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.judge_prompt_template = self._load_judge_prompt()
    
    def _load_judge_prompt(self) -> str:
        """Load the judge prompt template"""
        return """You are an expert evaluator for a chat response suggestion system. Your task is to evaluate the quality of 3 response suggestions generated for a user in a real-time chat application.

## Context Provided:
<chat_history>
{chat_history}
</chat_history>

<current_user>
{current_user}
</current_user>

<sentiment_analysis>
Sentiment Probabilities:
- Sadness: {sadness}%
- Joy: {joy}%
- Love: {love}%
- Anger: {anger}%
- Fear: {fear}%
- Unknown: {unknown}%

Dominant Emotion: {dominant_emotion}
</sentiment_analysis>

## Response Suggestions to Evaluate:
Suggestion 1: {suggestion_1}
Suggestion 2: {suggestion_2}
Suggestion 3: {suggestion_3}

## Evaluation Criteria:

Evaluate each suggestion on these dimensions:

1. **Relevance (0-10)**: How well does the suggestion fit as a natural next response from {current_user}?
2. **Sentiment Alignment (0-10)**: How well does the suggestion acknowledge and respond to the emotional tone?
3. **Naturalness (0-10)**: Does this sound like something a real person would say?
4. **Helpfulness (0-10)**: Does this suggestion move the conversation forward productively?
5. **Positivity Impact (0-10)**: Does this suggestion help lift the mood or de-escalate tension in a positive direction?
6. **Safety Check**: Is the suggestion free from harmful content? (PASS/FAIL)
7. **Diversity Assessment (0-10)**: Rate how different the three suggestions are from each other

## Output Format:

Provide your evaluation in this exact JSON format:

```json
{
  "suggestion_1": {
    "relevance": 8,
    "sentiment_alignment": 9,
    "naturalness": 7,
    "helpfulness": 8,
    "positivity_impact": 9,
    "safety": "PASS",
    "safety_notes": null
  },
  "suggestion_2": {
    "relevance": 7,
    "sentiment_alignment": 8,
    "naturalness": 9,
    "helpfulness": 7,
    "positivity_impact": 7,
    "safety": "PASS",
    "safety_notes": null
  },
  "suggestion_3": {
    "relevance": 9,
    "sentiment_alignment": 7,
    "naturalness": 8,
    "helpfulness": 9,
    "positivity_impact": 8,
    "safety": "PASS",
    "safety_notes": null
  },
  "diversity_score": 8,
  "best_suggestion": 3,
  "overall_quality": 8,
  "overall_positivity_score": 8,
  "reasoning": "Brief explanation of your evaluation"
}
```"""
    
    async def evaluate(self, context: ChatContext, suggestions: ResponseSuggestions) -> EvaluationResult:
        """Evaluate the quality of response suggestions"""
        
        prompt = self._build_evaluation_prompt(context, suggestions)
        start_time = time.time()
        
        try:
            # Use temperature=0 for consistent evaluation
            response = await self.provider.generate(prompt, temperature=0, max_tokens=800)
            evaluation_data = self._parse_evaluation(response)
            evaluation_time = time.time() - start_time
            
            return self._create_evaluation_result(evaluation_data, evaluation_time)
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise
    
    def _build_evaluation_prompt(self, context: ChatContext, suggestions: ResponseSuggestions) -> str:
        """Build the evaluation prompt"""
        
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
    
    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse the JSON evaluation from LLM response"""
        
        # Extract JSON from response
        try:
            # Find JSON block in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            raise
    
    def _create_evaluation_result(self, data: Dict[str, Any], evaluation_time: float) -> EvaluationResult:
        """Create EvaluationResult from parsed data"""
        
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

# ============== Evaluation Pipeline ==============

class EvaluationPipeline:
    """Main pipeline for running evaluations across multiple models"""
    
    def __init__(self, judge_provider: LLMProvider):
        self.judge = LLMJudge(judge_provider)
        self.results_cache = []
    
    async def evaluate_models_with_prompts(
        self, 
        test_contexts: List[ChatContext],
        model_providers: Dict[str, LLMProvider],
        prompt_variants: List[str] = ['base', 'no_positivity', 'no_sentiment'],
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """Evaluate multiple models with different prompt variants"""
        
        all_results = []
        total_evaluations = len(test_contexts) * len(model_providers) * len(prompt_variants)
        completed = 0
        
        for context_idx, context in enumerate(test_contexts):
            logger.info(f"Processing context {context_idx + 1}/{len(test_contexts)}")
            
            for model_name, provider in model_providers.items():
                for prompt_variant in prompt_variants:
                    logger.info(f"  Evaluating model: {model_name} with prompt: {prompt_variant}")
                    
                    # Generate suggestions
                    generator = SuggestionGenerator(provider)
                    suggestions = await generator.generate_suggestions(context, prompt_variant)
                    
                    # Evaluate suggestions
                    evaluation = await self.judge.evaluate(context, suggestions)
                    
                    # Store results
                    result_record = self._create_result_record(
                        context_idx, context, suggestions, evaluation, prompt_variant
                    )
                    all_results.append(result_record)
                    
                    completed += 1
                    logger.info(f"  Progress: {completed}/{total_evaluations} ({completed/total_evaluations*100:.1f}%)")
                    
                    # Save intermediate results
                    if save_intermediate and completed % 10 == 0:
                        self._save_intermediate_results(all_results)
        
        # Convert to DataFrame
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
        """Create a flat record for DataFrame"""
        
        record = {
            'context_id': context_idx,
            'model_name': suggestions.model_name,
            'prompt_variant': prompt_variant,
            'generation_time': suggestions.generation_time,
            'judge_model': evaluation.judge_model,
            'evaluation_time': evaluation.evaluation_time,
            
            # Suggestions
            'suggestion_1': suggestions.suggestion_1,
            'suggestion_2': suggestions.suggestion_2,
            'suggestion_3': suggestions.suggestion_3,
            
            # Metrics for each suggestion
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
            
            # Overall metrics
            'diversity_score': evaluation.diversity_score,
            'best_suggestion': evaluation.best_suggestion,
            'overall_quality': evaluation.overall_quality,
            'overall_positivity_score': evaluation.overall_positivity_score,
            'reasoning': evaluation.reasoning,
            
            # Context info
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
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to prevent data loss"""
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'intermediate_results_{timestamp}.csv', index=False)
        logger.info(f"Saved intermediate results: intermediate_results_{timestamp}.csv")

# ============== Report Generator ==============

class ReportGenerator:
    """Generates comprehensive reports and visualizations"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.output_dir = Path("evaluation_reports")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_full_report(self, report_name: str = "llm_evaluation_report"):
        """Generate complete evaluation report with all metrics and visualizations"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"{report_name}_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Save raw data
        self.results_df.to_csv(report_dir / "raw_results.csv", index=False)
        
        # Generate summary statistics
        summary_df = self._generate_summary_statistics()
        summary_df.to_csv(report_dir / "summary_statistics.csv", index=False)
        
        # Create visualizations
        self._create_model_comparison_chart(report_dir)
        self._create_metric_distribution_plots(report_dir)
        self._create_performance_heatmap(report_dir)
        self._create_safety_analysis(report_dir)
        self._create_time_analysis(report_dir)
        self._create_best_suggestion_analysis(report_dir)
        self._create_positivity_analysis(report_dir)
        self._create_prompt_variant_comparison(report_dir)
        
        # Generate text report
        self._generate_text_report(report_dir, summary_df)
        
        logger.info(f"Report generated in: {report_dir}")
        return report_dir
    
    def _generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for each model and prompt variant"""
        
        metrics = ['relevance', 'sentiment', 'naturalness', 'helpfulness', 'positivity']
        summary_data = []
        
        for model in self.results_df['model_name'].unique():
            for variant in self.results_df['prompt_variant'].unique():
                model_variant_data = self.results_df[
                    (self.results_df['model_name'] == model) & 
                    (self.results_df['prompt_variant'] == variant)
                ]
                
                if len(model_variant_data) == 0:
                    continue
                
                summary_row = {
                    'model': model,
                    'prompt_variant': variant
                }
                
                # Average metrics across all suggestions
                for metric in metrics:
                    cols = [f'suggestion_{i}_{metric}' for i in range(1, 4)]
                    all_scores = pd.concat([model_variant_data[col] for col in cols])
                    summary_row[f'avg_{metric}'] = all_scores.mean()
                    summary_row[f'std_{metric}'] = all_scores.std()
                
                # Other metrics
                summary_row['avg_diversity'] = model_variant_data['diversity_score'].mean()
                summary_row['avg_overall_quality'] = model_variant_data['overall_quality'].mean()
                summary_row['avg_positivity_score'] = model_variant_data['overall_positivity_score'].mean()
                summary_row['avg_generation_time'] = model_variant_data['generation_time'].mean()
                summary_row['safety_pass_rate'] = (model_variant_data[[
                    'suggestion_1_safety', 'suggestion_2_safety', 'suggestion_3_safety'
                ]] == 'PASS').values.mean()
                
                # Best suggestion distribution
                best_counts = model_variant_data['best_suggestion'].value_counts()
                for i in range(1, 4):
                    summary_row[f'best_suggestion_{i}_rate'] = best_counts.get(i, 0) / len(model_variant_data)
                
                summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def _create_model_comparison_chart(self, output_dir: Path):
        """Create overall model comparison chart"""
        
        summary_df = self._generate_summary_statistics()
        
        # Group by model (averaging across prompt variants)
        model_avg = summary_df.groupby('model').mean()
        
        metrics = ['avg_relevance', 'avg_sentiment', 'avg_naturalness', 
                  'avg_helpfulness', 'avg_positivity', 'avg_diversity', 'avg_overall_quality']
        
        fig = go.Figure()
        
        for model in model_avg.index:
            values = [model_avg.loc[model, metric] for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=[m.replace('avg_', '').title() for m in metrics],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Model Performance Comparison - Radar Chart"
        )
        
        fig.write_html(output_dir / "model_comparison_radar.html")
        fig.write_image(output_dir / "model_comparison_radar.png", width=1000, height=800)
    
    def _create_metric_distribution_plots(self, output_dir: Path):
        """Create distribution plots for each metric"""
        
        metrics = ['relevance', 'sentiment', 'naturalness', 'helpfulness', 'positivity']
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[f'{metric.capitalize()} Distribution' for metric in metrics] + ['Overall Quality']
        )
        
        for idx, metric in enumerate(metrics + ['overall_quality']):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            for model in self.results_df['model_name'].unique():
                model_data = self.results_df[self.results_df['model_name'] == model]
                
                if metric == 'overall_quality':
                    scores = model_data['overall_quality'].values
                else:
                    # Combine scores from all suggestions
                    scores = []
                    for i in range(1, 4):
                        scores.extend(model_data[f'suggestion_{i}_{metric}'].values)
                
                fig.add_trace(
                    go.Histogram(
                        x=scores,
                        name=model,
                        opacity=0.7,
                        histnorm='probability density',
                        nbinsx=20
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=1000,
            title_text="Metric Score Distributions by Model",
            showlegend=True
        )
        
        fig.write_html(output_dir / "metric_distributions.html")
        fig.write_image(output_dir / "metric_distributions.png", width=1200, height=1000)
    
    def _create_performance_heatmap(self, output_dir: Path):
        """Create performance heatmap"""
        
        summary_df = self._generate_summary_statistics()
        
        # Create separate heatmaps for each prompt variant
        variants = summary_df['prompt_variant'].unique()
        
        fig, axes = plt.subplots(1, len(variants), figsize=(6*len(variants), 8))
        if len(variants) == 1:
            axes = [axes]
        
        for idx, variant in enumerate(variants):
            variant_data = summary_df[summary_df['prompt_variant'] == variant]
            
            metrics = ['avg_relevance', 'avg_sentiment', 'avg_naturalness', 
                      'avg_helpfulness', 'avg_positivity', 'avg_diversity', 'avg_overall_quality']
            
            heatmap_data = variant_data[['model'] + metrics].set_index('model')
            
            sns.heatmap(
                heatmap_data.T,
                annot=True,
                fmt='.2f',
                cmap='YlGnBu',
                cbar_kws={'label': 'Score'},
                vmin=0,
                vmax=10,
                ax=axes[idx]
            )
            axes[idx].set_title(f'Performance Heatmap - {variant}')
            axes[idx].set_xlabel('Model')
            axes[idx].set_ylabel('Metric')
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_safety_analysis(self, output_dir: Path):
        """Create safety analysis visualization"""
        
        safety_data = []
        
        for model in self.results_df['model_name'].unique():
            for variant in self.results_df['prompt_variant'].unique():
                model_variant_data = self.results_df[
                    (self.results_df['model_name'] == model) & 
                    (self.results_df['prompt_variant'] == variant)
                ]
                
                if len(model_variant_data) == 0:
                    continue
                
                for i in range(1, 4):
                    safety_col = f'suggestion_{i}_safety'
                    pass_rate = (model_variant_data[safety_col] == 'PASS').mean()
                    safety_data.append({
                        'model': model,
                        'prompt_variant': variant,
                        'suggestion': f'Suggestion {i}',
                        'pass_rate': pass_rate * 100
                    })
        
        safety_df = pd.DataFrame(safety_data)
        
        fig = px.bar(
            safety_df,
            x='model',
            y='pass_rate',
            color='suggestion',
            facet_col='prompt_variant',
            title='Safety Pass Rates by Model, Suggestion, and Prompt Variant',
            labels={'pass_rate': 'Pass Rate (%)', 'model': 'Model'},
            barmode='group'
        )
        
        fig.update_layout(yaxis_range=[0, 105])
        fig.write_html(output_dir / "safety_analysis.html")
        fig.write_image(output_dir / "safety_analysis.png", width=1400, height=600)
    
    def _create_time_analysis(self, output_dir: Path):
        """Create time performance analysis"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Generation Time by Model', 'Evaluation Time'],
            specs=[[{"type": "box"}, {"type": "box"}]]
        )
        
        for model in self.results_df['model_name'].unique():
            model_data = self.results_df[self.results_df['model_name'] == model]
            
            fig.add_trace(
                go.Box(
                    y=model_data['generation_time'],
                    name=model,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Box(
                    y=model_data['evaluation_time'],
                    name=model,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
        
        fig.update_layout(
            height=500,
            title_text="Time Performance Analysis"
        )
        
        fig.write_html(output_dir / "time_analysis.html")
        fig.write_image(output_dir / "time_analysis.png", width=1200, height=500)
    
    def _create_best_suggestion_analysis(self, output_dir: Path):
        """Analyze which suggestion position tends to be best"""
        
        best_suggestion_data = []
        
        for model in self.results_df['model_name'].unique():
            for variant in self.results_df['prompt_variant'].unique():
                model_variant_data = self.results_df[
                    (self.results_df['model_name'] == model) & 
                    (self.results_df['prompt_variant'] == variant)
                ]
                
                if len(model_variant_data) == 0:
                    continue
                
                for position in [1, 2, 3]:
                    count = (model_variant_data['best_suggestion'] == position).sum()
                    best_suggestion_data.append({
                        'model': model,
                        'prompt_variant': variant,
                        'position': f'Position {position}',
                        'count': count,
                        'percentage': (count / len(model_variant_data)) * 100
                    })
        
        best_df = pd.DataFrame(best_suggestion_data)
        
        fig = px.bar(
            best_df,
            x='model',
            y='percentage',
            color='position',
            facet_col='prompt_variant',
            title='Best Suggestion Position Distribution',
            labels={'percentage': 'Percentage (%)', 'model': 'Model'},
            barmode='stack'
        )
        
        fig.write_html(output_dir / "best_suggestion_analysis.html")
        fig.write_image(output_dir / "best_suggestion_analysis.png", width=1400, height=600)
    
    def _create_positivity_analysis(self, output_dir: Path):
        """Create analysis of positivity impact scores"""
        
        # Positivity scores by sentiment
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Positivity Score by Model',
                'Positivity vs Sentiment',
                'Positivity by Prompt Variant',
                'Positivity vs Overall Quality'
            ]
        )
        
        # 1. Box plot of positivity scores by model
        for model in self.results_df['model_name'].unique():
            model_data = self.results_df[self.results_df['model_name'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data['overall_positivity_score'],
                    name=model
                ),
                row=1, col=1
            )
        
        # 2. Scatter plot: positivity vs dominant sentiment
        sentiment_colors = {
            'sadness': 'blue',
            'joy': 'yellow',
            'love': 'pink',
            'anger': 'red',
            'fear': 'purple',
            'unknown': 'gray'
        }
        
        for sentiment in sentiment_colors:
            sentiment_data = self.results_df[self.results_df['dominant_sentiment'] == sentiment]
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data['overall_quality'],
                    y=sentiment_data['overall_positivity_score'],
                    mode='markers',
                    name=sentiment,
                    marker_color=sentiment_colors[sentiment]
                ),
                row=1, col=2
            )
        
        # 3. Positivity by prompt variant
        for variant in self.results_df['prompt_variant'].unique():
            variant_data = self.results_df[self.results_df['prompt_variant'] == variant]
            fig.add_trace(
                go.Box(
                    y=variant_data['overall_positivity_score'],
                    name=variant
                ),
                row=2, col=1
            )
        
        # 4. Correlation between positivity and overall quality
        fig.add_trace(
            go.Scatter(
                x=self.results_df['overall_quality'],
                y=self.results_df['overall_positivity_score'],
                mode='markers',
                marker=dict(
                    color=self.results_df['overall_quality'],
                    colorscale='Viridis',
                    showscale=True
                ),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Overall Quality", row=1, col=2)
        fig.update_yaxes(title_text="Positivity Score", row=1, col=2)
        fig.update_xaxes(title_text="Overall Quality", row=2, col=2)
        fig.update_yaxes(title_text="Positivity Score", row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Positivity Impact Analysis"
        )
        
        fig.write_html(output_dir / "positivity_analysis.html")
        fig.write_image(output_dir / "positivity_analysis.png", width=1200, height=800)
    
    def _create_prompt_variant_comparison(self, output_dir: Path):
        """Compare performance across prompt variants"""
        
        summary_df = self._generate_summary_statistics()
        
        # Create radar chart for each model showing prompt variant performance
        models = summary_df['model'].unique()
        
        fig = make_subplots(
            rows=1, cols=len(models),
            subplot_titles=models,
            specs=[[{"type": "polar"}] * len(models)]
        )
        
        metrics = ['avg_relevance', 'avg_sentiment', 'avg_naturalness', 
                  'avg_helpfulness', 'avg_positivity', 'avg_overall_quality']
        
        for idx, model in enumerate(models):
            model_data = summary_df[summary_df['model'] == model]
            
            for _, row in model_data.iterrows():
                values = [row[metric] for metric in metrics]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=[m.replace('avg_', '').title() for m in metrics],
                        fill='toself',
                        name=row['prompt_variant']
                    ),
                    row=1, col=idx+1
                )
        
        fig.update_layout(
            height=500,
            title_text="Prompt Variant Performance Comparison by Model",
            showlegend=True
        )
        
        fig.write_html(output_dir / "prompt_variant_comparison.html")
        fig.write_image(output_dir / "prompt_variant_comparison.png", width=1800, height=500)
    
    def _generate_text_report(self, output_dir: Path, summary_df: pd.DataFrame):
        """Generate a text summary report"""
        
        report_lines = [
            "# LLM Response Suggestion Evaluation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Best performing model overall
        best_model_data = summary_df.groupby('model')['avg_overall_quality'].mean()
        best_model = best_model_data.idxmax()
        report_lines.append(f"**Best Overall Model**: {best_model} (avg score: {best_model_data[best_model]:.2f})")
        
        # Best prompt variant
        best_variant_data = summary_df.groupby('prompt_variant')['avg_overall_quality'].mean()
        best_variant = best_variant_data.idxmax()
        report_lines.append(f"**Best Prompt Variant**: {best_variant} (avg score: {best_variant_data[best_variant]:.2f})")
        
        # Best for positivity
        best_positivity_data = summary_df.groupby('model')['avg_positivity_score'].mean()
        best_positivity_model = best_positivity_data.idxmax()
        report_lines.append(f"**Best for Positivity**: {best_positivity_model} (avg score: {best_positivity_data[best_positivity_model]:.2f})")
        
        report_lines.append("")
        
        # Model rankings by metric
        report_lines.append("## Model Rankings by Metric")
        report_lines.append("")
        
        metrics = ['relevance', 'sentiment', 'naturalness', 'helpfulness', 'positivity', 'diversity', 'overall_quality']
        
        for metric in metrics:
            col = f'avg_{metric}' if metric != 'overall_quality' else 'avg_overall_quality'
            ranked = summary_df.groupby('model')[col].mean().sort_values(ascending=False)
            report_lines.append(f"### {metric.capitalize()}")
            for idx, (model, score) in enumerate(ranked.items()):
                report_lines.append(f"{idx + 1}. {model}: {score:.2f}")
            report_lines.append("")
        
        # Prompt variant analysis
        report_lines.append("## Prompt Variant Analysis")
        report_lines.append("")
        
        for variant in summary_df['prompt_variant'].unique():
            variant_data = summary_df[summary_df['prompt_variant'] == variant]
            avg_quality = variant_data['avg_overall_quality'].mean()
            avg_positivity = variant_data['avg_positivity_score'].mean()
            
            report_lines.append(f"### {variant}")
            report_lines.append(f"- Average Quality: {avg_quality:.2f}")
            report_lines.append(f"- Average Positivity: {avg_positivity:.2f}")
            report_lines.append("")
        
        # Performance insights
        report_lines.append("## Performance Insights")
        report_lines.append("")
        
        # Speed analysis
        speed_data = summary_df.groupby('model')['avg_generation_time'].mean()
        fastest_model = speed_data.idxmin()
        slowest_model = speed_data.idxmax()
        
        report_lines.append(f"**Fastest Model**: {fastest_model} ({speed_data[fastest_model]:.2f}s avg)")
        report_lines.append(f"**Slowest Model**: {slowest_model} ({speed_data[slowest_model]:.2f}s avg)")
        report_lines.append("")
        
        # Safety analysis
        report_lines.append("### Safety Analysis")
        safety_data = summary_df.groupby('model')['safety_pass_rate'].mean()
        for model, rate in safety_data.items():
            report_lines.append(f"- {model}: {rate*100:.1f}% pass rate")
        report_lines.append("")
        
        # Best suggestion position analysis
        report_lines.append("### Best Suggestion Position Distribution")
        for model in summary_df['model'].unique():
            model_data = summary_df[summary_df['model'] == model]
            report_lines.append(f"- {model}:")
            for i in range(1, 4):
                avg_rate = model_data[f'best_suggestion_{i}_rate'].mean()
                report_lines.append(f"  - Position {i}: {avg_rate*100:.1f}%")
        report_lines.append("")
        
        # Write report
        with open(output_dir / "evaluation_report.md", 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("Text report generated")

# ============== Test Data Generator ==============

class TestDataGenerator:
    """Generate synthetic test data for evaluation"""
    
    @staticmethod
    def generate_test_contexts(num_contexts: int = 20) -> List[ChatContext]:
        """Generate diverse test contexts"""
        
        scenarios = [
            {
                "type": "customer_support_frustrated",
                "messages": [
                    ChatMessage("1", "I've been trying to reset my password for 30 minutes!", "Alex", "2024-01-15 10:00:00"),
                    ChatMessage("2", "I apologize for the frustration. Let me help you with that right away.", "Support", "2024-01-15 10:01:00"),
                    ChatMessage("3", "I've tried the reset link 5 times and it's not working", "Alex", "2024-01-15 10:02:00")
                ],
                "current_user": "Support",
                "sentiment": SentimentProbabilities(sadness=0.1, joy=0.0, love=0.0, anger=0.7, fear=0.1, unknown=0.1)
            },
            {
                "type": "sales_inquiry_interested",
                "messages": [
                    ChatMessage("1", "Hi, I'm looking for a project management tool for my team", "Sarah", "2024-01-15 11:00:00"),
                    ChatMessage("2", "Great! I'd be happy to help you find the perfect solution. How large is your team?", "Sales", "2024-01-15 11:01:00"),
                    ChatMessage("3", "We're about 25 people, mostly remote", "Sarah", "2024-01-15 11:02:00")
                ],
                "current_user": "Sales",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.3, love=0.0, anger=0.0, fear=0.0, unknown=0.7)
            },
            {
                "type": "technical_support_confused",
                "messages": [
                    ChatMessage("1", "My app keeps crashing when I try to upload files", "Mike", "2024-01-15 12:00:00"),
                    ChatMessage("2", "I'm sorry to hear that. What type of files are you trying to upload?", "Tech", "2024-01-15 12:01:00"),
                    ChatMessage("3", "Just regular PDFs, nothing special", "Mike", "2024-01-15 12:02:00")
                ],
                "current_user": "Tech",
                "sentiment": SentimentProbabilities(sadness=0.2, joy=0.0, love=0.0, anger=0.1, fear=0.3, unknown=0.4)
            },
            {
                "type": "positive_feedback_happy",
                "messages": [
                    ChatMessage("1", "Just wanted to say your service has been amazing!", "Emma", "2024-01-15 13:00:00"),
                    ChatMessage("2", "Thank you so much! That really makes our day. What specifically has been helpful?", "Support", "2024-01-15 13:01:00"),
                    ChatMessage("3", "The customer support team solved my issue in minutes", "Emma", "2024-01-15 13:02:00")
                ],
                "current_user": "Support",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.8, love=0.1, anger=0.0, fear=0.0, unknown=0.1)
            },
            {
                "type": "relationship_conflict",
                "messages": [
                    ChatMessage("1", "You never listen to what I'm saying", "Jordan", "2024-01-15 14:00:00"),
                    ChatMessage("2", "That's not fair, I do listen but you keep interrupting me", "Casey", "2024-01-15 14:01:00"),
                    ChatMessage("3", "See, you're doing it again, making it about you", "Jordan", "2024-01-15 14:02:00")
                ],
                "current_user": "Casey",
                "sentiment": SentimentProbabilities(sadness=0.3, joy=0.0, love=0.0, anger=0.5, fear=0.1, unknown=0.1)
            }
        ]
        
        test_contexts = []
        
        # Generate contexts by cycling through scenarios with variations
        for i in range(num_contexts):
            scenario = scenarios[i % len(scenarios)]
            
            # Add some variation to sentiment probabilities
            sentiment = scenario["sentiment"]
            if i % 3 == 0:  # Add some noise to 1/3 of contexts
                import random
                noise = 0.1
                sentiment = SentimentProbabilities(
                    sadness=max(0, min(1, sentiment.sadness + random.uniform(-noise, noise))),
                    joy=max(0, min(1, sentiment.joy + random.uniform(-noise, noise))),
                    love=max(0, min(1, sentiment.love + random.uniform(-noise, noise))),
                    anger=max(0, min(1, sentiment.anger + random.uniform(-noise, noise))),
                    fear=max(0, min(1, sentiment.fear + random.uniform(-noise, noise))),
                    unknown=max(0, min(1, sentiment.unknown + random.uniform(-noise, noise)))
                )
            
            context = ChatContext(
                chat_history=scenario["messages"].copy(),
                current_user=scenario["current_user"],
                sentiment_probabilities=sentiment
            )
            
            test_contexts.append(context)
        
        return test_contexts

# ============== Main Execution Script ==============

async def main():
    """Main execution function"""
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    config = {
        "num_test_contexts": 10,  # Reduced for demo
        "models_to_test": {
            "gpt-4": "gpt-4-turbo-preview",
            "gpt-3.5": "gpt-3.5-turbo",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "gemini-pro": "gemini-1.5-pro"
        },
        "judge_model": "gpt-4-turbo-preview",  # Use best model as judge
        "prompt_variants": ["base", "no_positivity", "no_sentiment"]
    }
    
    # Initialize providers
    providers = {}
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            providers["gpt-4"] = OpenAIProvider(config["models_to_test"]["gpt-4"])
            providers["gpt-3.5"] = OpenAIProvider(config["models_to_test"]["gpt-3.5"])
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI providers: {e}")
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            providers["claude-3-opus"] = AnthropicProvider(config["models_to_test"]["claude-3-opus"])
            providers["claude-3-sonnet"] = AnthropicProvider(config["models_to_test"]["claude-3-sonnet"])
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic providers: {e}")
    
    # Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            providers["gemini-pro"] = GeminiProvider(config["models_to_test"]["gemini-pro"])
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini provider: {e}")
    
    if not providers:
        logger.error("No valid API keys provided. Please update your .env file.")
        return
    
    # Initialize judge
    judge_provider = None
    if os.getenv("OPENAI_API_KEY"):
        judge_provider = OpenAIProvider(config["judge_model"])
    elif os.getenv("GOOGLE_API_KEY"):
        judge_provider = GeminiProvider("gemini-1.5-pro")
        logger.info("Using Gemini as judge since OpenAI key not available")
    else:
        logger.error("No judge model available. Need either OpenAI or Google API key.")
        return
    
    # Generate test data
    logger.info("Generating test contexts...")
    test_contexts = TestDataGenerator.generate_test_contexts(config["num_test_contexts"])
    
    # Run evaluation pipeline
    logger.info("Starting evaluation pipeline...")
    pipeline = EvaluationPipeline(judge_provider)
    
    results_df = await pipeline.evaluate_models_with_prompts(
        test_contexts=test_contexts,
        model_providers=providers,
        prompt_variants=config["prompt_variants"],
        save_intermediate=True
    )
    
    # Generate reports
    logger.info("Generating evaluation reports...")
    report_generator = ReportGenerator(results_df)
    report_dir = report_generator.generate_full_report()
    
    logger.info(f"Evaluation complete! Reports saved to: {report_dir}")
    
    # Print summary to console
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    summary_df = report_generator._generate_summary_statistics()
    
    print("\nOverall Quality Scores (averaged across prompt variants):")
    model_avg_quality = summary_df.groupby('model')['avg_overall_quality'].mean().sort_values(ascending=False)
    for model, score in model_avg_quality.items():
        print(f"{model}: {score:.2f}")
    
    print("\nPositivity Impact Scores:")
    model_avg_positivity = summary_df.groupby('model')['avg_positivity_score'].mean().sort_values(ascending=False)
    for model, score in model_avg_positivity.items():
        print(f"{model}: {score:.2f}")
    
    print("\nGeneration Speed (seconds):")
    model_avg_speed = summary_df.groupby('model')['avg_generation_time'].mean().sort_values()
    for model, speed in model_avg_speed.items():
        print(f"{model}: {speed:.2f}s")
    
    print(f"\nDetailed reports available in: {report_dir}")

# ============== Utility Functions ==============

def run_evaluation(
    test_contexts: List[ChatContext] = None,
    num_contexts: int = 10,
    models_to_test: Dict[str, str] = None,
    judge_model: str = "gpt-4-turbo-preview",
    prompt_variants: List[str] = None
):
    """Convenience function to run evaluation"""
    
    # Load environment variables
    load_dotenv()
    
    # Default models if not specified
    if models_to_test is None:
        models_to_test = {
            "gpt-4": "gpt-4-turbo-preview",
            "gpt-3.5": "gpt-3.5-turbo",
            "gemini-pro": "gemini-1.5-pro"
        }
    
    # Default prompt variants
    if prompt_variants is None:
        prompt_variants = ["base", "no_positivity", "no_sentiment"]
    
    # Generate test contexts if not provided
    if test_contexts is None:
        test_contexts = TestDataGenerator.generate_test_contexts(num_contexts)
    
    # Run async main
    import asyncio
    asyncio.run(main())

if __name__ == "__main__":
    # Example usage
    print("LLM Judge Evaluation System - Modified Version")
    print("-" * 50)
    print("This system evaluates LLM performance for chat response suggestions.")
    print("\nFeatures:")
    print("- Sentiment probabilities for 6 emotions")
    print("- Positivity impact scoring")
    print("- Multiple prompt variants")
    print("- Support for OpenAI, Anthropic, and Google Gemini models")
    print("\nTo run the evaluation:")
    print("1. Create a .env file with your API keys:")
    print("   OPENAI_API_KEY=your-key")
    print("   ANTHROPIC_API_KEY=your-key")
    print("   GOOGLE_API_KEY=your-key")
    print("2. Run: python llm_judge_system.py")
    
    # Uncomment to run evaluation
    # asyncio.run(main())
