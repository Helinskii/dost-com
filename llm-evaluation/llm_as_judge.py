"""
LLM-as-Judge Evaluation System for Chat Response Suggestions
A comprehensive evaluation framework for comparing LLM performance in generating chat suggestions
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============== Data Models ==============

@dataclass
class ChatContext:
    """Represents the context for generating response suggestions"""
    chat_history: List[Dict[str, str]]
    current_message: str
    rag_context: Optional[str] = None
    sentiment: Optional[str] = None
    emotional_indicators: Optional[List[str]] = None

@dataclass
class ResponseSuggestions:
    """Container for the three response suggestions"""
    suggestion_1: str
    suggestion_2: str
    suggestion_3: str
    model_name: str
    generation_time: float

@dataclass
class EvaluationMetrics:
    """Metrics for a single suggestion"""
    relevance: float
    sentiment_alignment: float
    naturalness: float
    helpfulness: float
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
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
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
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key
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

# ============== Suggestion Generator ==============

class SuggestionGenerator:
    """Generates response suggestions using specified LLM"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
    
    async def generate_suggestions(self, context: ChatContext) -> ResponseSuggestions:
        """Generate three response suggestions based on context"""
        
        prompt = self._build_generation_prompt(context)
        start_time = time.time()
        
        try:
            response = await self.provider.generate(prompt, temperature=0.7, max_tokens=300)
            suggestions = self._parse_suggestions(response)
            generation_time = time.time() - start_time
            
            return ResponseSuggestions(
                suggestion_1=suggestions[0],
                suggestion_2=suggestions[1],
                suggestion_3=suggestions[2],
                model_name=self.provider.get_name(),
                generation_time=generation_time
            )
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            raise
    
    def _build_generation_prompt(self, context: ChatContext) -> str:
        """Build the prompt for suggestion generation"""
        
        chat_history_str = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in context.chat_history[-5:]  # Last 5 messages
        ])
        
        prompt = f"""Generate 3 different response suggestions for the user in this chat.

Chat History:
{chat_history_str}

Current Message: {context.current_message}

{'RAG Context: ' + context.rag_context if context.rag_context else ''}
{'Sentiment: ' + context.sentiment if context.sentiment else ''}

Generate 3 diverse, natural responses that:
1. Are contextually relevant
2. Match the conversation tone
3. Move the conversation forward
4. Are each distinctly different approaches

Format your response as:
SUGGESTION_1: [first suggestion]
SUGGESTION_2: [second suggestion]
SUGGESTION_3: [third suggestion]"""
        
        return prompt
    
    def _parse_suggestions(self, response: str) -> List[str]:
        """Parse suggestions from LLM response"""
        suggestions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if line.startswith('SUGGESTION_'):
                suggestion = line.split(':', 1)[1].strip()
                suggestions.append(suggestion)
        
        # Fallback if parsing fails
        if len(suggestions) != 3:
            logger.warning("Failed to parse exactly 3 suggestions, using fallback")
            suggestions = response.strip().split('\n')[:3]
        
        return suggestions

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

<current_message>
{current_message}
</current_message>

<rag_context>
{rag_context}
</rag_context>

<sentiment_analysis>
Overall sentiment: {sentiment}
Emotional indicators: {emotional_indicators}
</sentiment_analysis>

## Response Suggestions to Evaluate:
Suggestion 1: {suggestion_1}
Suggestion 2: {suggestion_2}
Suggestion 3: {suggestion_3}

## Evaluation Criteria:

Evaluate each suggestion on these dimensions:

1. **Relevance (0-10)**: How well does the suggestion fit as a natural next response?
2. **Sentiment Alignment (0-10)**: How well does the suggestion match the emotional tone?
3. **Naturalness (0-10)**: Does this sound like something a real person would say?
4. **Helpfulness (0-10)**: Does this suggestion move the conversation forward productively?
5. **Safety Check**: Is the suggestion free from harmful content? (PASS/FAIL)
6. **Diversity Assessment (0-10)**: Rate how different the three suggestions are from each other

## Output Format:

Provide your evaluation in this exact JSON format:

```json
{
  "suggestion_1": {
    "relevance": 8,
    "sentiment_alignment": 9,
    "naturalness": 7,
    "helpfulness": 8,
    "safety": "PASS",
    "safety_notes": null
  },
  "suggestion_2": {
    "relevance": 7,
    "sentiment_alignment": 8,
    "naturalness": 9,
    "helpfulness": 7,
    "safety": "PASS",
    "safety_notes": null
  },
  "suggestion_3": {
    "relevance": 9,
    "sentiment_alignment": 7,
    "naturalness": 8,
    "helpfulness": 9,
    "safety": "PASS",
    "safety_notes": null
  },
  "diversity_score": 8,
  "best_suggestion": 3,
  "overall_quality": 8,
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
            f"{msg['role']}: {msg['content']}" 
            for msg in context.chat_history[-5:]
        ])
        
        emotional_indicators_str = ", ".join(context.emotional_indicators) if context.emotional_indicators else "None"
        
        return self.judge_prompt_template.format(
            chat_history=chat_history_str,
            current_message=context.current_message,
            rag_context=context.rag_context or "None provided",
            sentiment=context.sentiment or "Neutral",
            emotional_indicators=emotional_indicators_str,
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
    
    async def evaluate_models(
        self, 
        test_contexts: List[ChatContext],
        model_providers: Dict[str, LLMProvider],
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """Evaluate multiple models on test contexts"""
        
        all_results = []
        total_evaluations = len(test_contexts) * len(model_providers)
        completed = 0
        
        for context_idx, context in enumerate(test_contexts):
            logger.info(f"Processing context {context_idx + 1}/{len(test_contexts)}")
            
            for model_name, provider in model_providers.items():
                logger.info(f"  Evaluating model: {model_name}")
                
                # Generate suggestions
                generator = SuggestionGenerator(provider)
                suggestions = await generator.generate_suggestions(context)
                
                # Evaluate suggestions
                evaluation = await self.judge.evaluate(context, suggestions)
                
                # Store results
                result_record = self._create_result_record(
                    context_idx, context, suggestions, evaluation
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
        evaluation: EvaluationResult
    ) -> Dict[str, Any]:
        """Create a flat record for DataFrame"""
        
        record = {
            'context_id': context_idx,
            'model_name': suggestions.model_name,
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
            'suggestion_1_safety': evaluation.suggestion_1_metrics.safety,
            
            'suggestion_2_relevance': evaluation.suggestion_2_metrics.relevance,
            'suggestion_2_sentiment': evaluation.suggestion_2_metrics.sentiment_alignment,
            'suggestion_2_naturalness': evaluation.suggestion_2_metrics.naturalness,
            'suggestion_2_helpfulness': evaluation.suggestion_2_metrics.helpfulness,
            'suggestion_2_safety': evaluation.suggestion_2_metrics.safety,
            
            'suggestion_3_relevance': evaluation.suggestion_3_metrics.relevance,
            'suggestion_3_sentiment': evaluation.suggestion_3_metrics.sentiment_alignment,
            'suggestion_3_naturalness': evaluation.suggestion_3_metrics.naturalness,
            'suggestion_3_helpfulness': evaluation.suggestion_3_metrics.helpfulness,
            'suggestion_3_safety': evaluation.suggestion_3_metrics.safety,
            
            # Overall metrics
            'diversity_score': evaluation.diversity_score,
            'best_suggestion': evaluation.best_suggestion,
            'overall_quality': evaluation.overall_quality,
            'reasoning': evaluation.reasoning,
            
            # Context info
            'sentiment': context.sentiment,
            'has_rag_context': bool(context.rag_context),
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
        
        # Generate text report
        self._generate_text_report(report_dir, summary_df)
        
        logger.info(f"Report generated in: {report_dir}")
        return report_dir
    
    def _generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for each model"""
        
        metrics = ['relevance', 'sentiment', 'naturalness', 'helpfulness']
        summary_data = []
        
        for model in self.results_df['model_name'].unique():
            model_data = self.results_df[self.results_df['model_name'] == model]
            
            summary_row = {'model': model}
            
            # Average metrics across all suggestions
            for metric in metrics:
                cols = [f'suggestion_{i}_{metric}' for i in range(1, 4)]
                all_scores = pd.concat([model_data[col] for col in cols])
                summary_row[f'avg_{metric}'] = all_scores.mean()
                summary_row[f'std_{metric}'] = all_scores.std()
            
            # Other metrics
            summary_row['avg_diversity'] = model_data['diversity_score'].mean()
            summary_row['avg_overall_quality'] = model_data['overall_quality'].mean()
            summary_row['avg_generation_time'] = model_data['generation_time'].mean()
            summary_row['safety_pass_rate'] = (model_data[[
                'suggestion_1_safety', 'suggestion_2_safety', 'suggestion_3_safety'
            ]] == 'PASS').values.mean()
            
            # Best suggestion distribution
            best_counts = model_data['best_suggestion'].value_counts()
            for i in range(1, 4):
                summary_row[f'best_suggestion_{i}_rate'] = best_counts.get(i, 0) / len(model_data)
            
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def _create_model_comparison_chart(self, output_dir: Path):
        """Create overall model comparison chart"""
        
        summary_df = self._generate_summary_statistics()
        
        metrics = ['relevance', 'sentiment', 'naturalness', 'helpfulness', 'diversity', 'overall_quality']
        
        fig = go.Figure()
        
        for model in summary_df['model']:
            values = [summary_df[summary_df['model'] == model][f'avg_{metric}'].values[0] 
                     for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
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
        
        metrics = ['relevance', 'sentiment', 'naturalness', 'helpfulness']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{metric.capitalize()} Distribution' for metric in metrics]
        )
        
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            for model in self.results_df['model_name'].unique():
                model_data = self.results_df[self.results_df['model_name'] == model]
                
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
            height=800,
            title_text="Metric Score Distributions by Model",
            showlegend=True
        )
        
        fig.write_html(output_dir / "metric_distributions.html")
        fig.write_image(output_dir / "metric_distributions.png", width=1200, height=800)
    
    def _create_performance_heatmap(self, output_dir: Path):
        """Create performance heatmap"""
        
        summary_df = self._generate_summary_statistics()
        
        metrics = ['avg_relevance', 'avg_sentiment', 'avg_naturalness', 
                  'avg_helpfulness', 'avg_diversity', 'avg_overall_quality']
        
        heatmap_data = summary_df[['model'] + metrics].set_index('model')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            heatmap_data.T,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Score'},
            vmin=0,
            vmax=10
        )
        plt.title('Model Performance Heatmap')
        plt.xlabel('Model')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig(output_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_safety_analysis(self, output_dir: Path):
        """Create safety analysis visualization"""
        
        safety_data = []
        
        for model in self.results_df['model_name'].unique():
            model_data = self.results_df[self.results_df['model_name'] == model]
            
            for i in range(1, 4):
                safety_col = f'suggestion_{i}_safety'
                pass_rate = (model_data[safety_col] == 'PASS').mean()
                safety_data.append({
                    'model': model,
                    'suggestion': f'Suggestion {i}',
                    'pass_rate': pass_rate * 100
                })
        
        safety_df = pd.DataFrame(safety_data)
        
        fig = px.bar(
            safety_df,
            x='model',
            y='pass_rate',
            color='suggestion',
            title='Safety Pass Rates by Model and Suggestion',
            labels={'pass_rate': 'Pass Rate (%)', 'model': 'Model'},
            barmode='group'
        )
        
        fig.update_layout(yaxis_range=[0, 105])
        fig.write_html(output_dir / "safety_analysis.html")
        fig.write_image(output_dir / "safety_analysis.png", width=1000, height=600)
    
    def _create_time_analysis(self, output_dir: Path):
        """Create time performance analysis"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Generation Time', 'Evaluation Time'],
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
            model_data = self.results_df[self.results_df['model_name'] == model]
            
            for position in [1, 2, 3]:
                count = (model_data['best_suggestion'] == position).sum()
                best_suggestion_data.append({
                    'model': model,
                    'position': f'Position {position}',
                    'count': count,
                    'percentage': (count / len(model_data)) * 100
                })
        
        best_df = pd.DataFrame(best_suggestion_data)
        
        fig = px.bar(
            best_df,
            x='model',
            y='percentage',
            color='position',
            title='Best Suggestion Position Distribution',
            labels={'percentage': 'Percentage (%)', 'model': 'Model'},
            barmode='stack'
        )
        
        fig.write_html(output_dir / "best_suggestion_analysis.html")
        fig.write_image(output_dir / "best_suggestion_analysis.png", width=1000, height=600)
    
    def _generate_text_report(self, output_dir: Path, summary_df: pd.DataFrame):
        """Generate a text summary report"""
        
        report_lines = [
            "# LLM Response Suggestion Evaluation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Best performing model
        best_model = summary_df.loc[summary_df['avg_overall_quality'].idxmax(), 'model']
        report_lines.append(f"**Best Overall Model**: {best_model}")
        report_lines.append("")
        
        # Model rankings
        report_lines.append("## Model Rankings by Metric")
        report_lines.append("")
        
        metrics = ['relevance', 'sentiment', 'naturalness', 'helpfulness', 'diversity', 'overall_quality']
        
        for metric in metrics:
            col = f'avg_{metric}'
            ranked = summary_df.sort_values(col, ascending=False)
            report_lines.append(f"### {metric.capitalize()}")
            for idx, row in ranked.iterrows():
                report_lines.append(f"{idx + 1}. {row['model']}: {row[col]:.2f}")
            report_lines.append("")
        
        # Performance insights
        report_lines.append("## Performance Insights")
        report_lines.append("")
        
        # Speed analysis
        fastest_model = summary_df.loc[summary_df['avg_generation_time'].idxmin(), 'model']
        slowest_model = summary_df.loc[summary_df['avg_generation_time'].idxmax(), 'model']
        
        report_lines.append(f"**Fastest Model**: {fastest_model} "
                          f"({summary_df.loc[summary_df['model'] == fastest_model, 'avg_generation_time'].values[0]:.2f}s avg)")
        report_lines.append(f"**Slowest Model**: {slowest_model} "
                          f"({summary_df.loc[summary_df['model'] == slowest_model, 'avg_generation_time'].values[0]:.2f}s avg)")
        report_lines.append("")
        
        # Safety analysis
        report_lines.append("### Safety Analysis")
        for _, row in summary_df.iterrows():
            report_lines.append(f"- {row['model']}: {row['safety_pass_rate']*100:.1f}% pass rate")
        report_lines.append("")
        
        # Best suggestion position analysis
        report_lines.append("### Best Suggestion Position Distribution")
        for _, row in summary_df.iterrows():
            report_lines.append(f"- {row['model']}:")
            for i in range(1, 4):
                report_lines.append(f"  - Position {i}: {row[f'best_suggestion_{i}_rate']*100:.1f}%")
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
                "type": "customer_support",
                "sentiment": "frustrated",
                "chat_history": [
                    {"role": "user", "content": "I've been trying to reset my password for 30 minutes!"},
                    {"role": "assistant", "content": "I apologize for the frustration. Let me help you with that right away."},
                    {"role": "user", "content": "I've tried the reset link 5 times and it's not working"}
                ],
                "current_message": "This is ridiculous. I need to access my account NOW.",
                "emotional_indicators": ["anger", "urgency"]
            },
            {
                "type": "sales_inquiry",
                "sentiment": "interested",
                "chat_history": [
                    {"role": "user", "content": "Hi, I'm looking for a project management tool for my team"},
                    {"role": "assistant", "content": "Great! I'd be happy to help you find the perfect solution. How large is your team?"},
                    {"role": "user", "content": "We're about 25 people, mostly remote"}
                ],
                "current_message": "What features do you have for remote collaboration?",
                "emotional_indicators": ["curiosity", "business-focused"]
            },
            {
                "type": "technical_support",
                "sentiment": "confused",
                "chat_history": [
                    {"role": "user", "content": "My app keeps crashing when I try to upload files"},
                    {"role": "assistant", "content": "I'm sorry to hear that. What type of files are you trying to upload?"},
                    {"role": "user", "content": "Just regular PDFs, nothing special"}
                ],
                "current_message": "I don't understand why this is happening. It worked fine yesterday.",
                "emotional_indicators": ["confusion", "mild frustration"]
            },
            {
                "type": "positive_feedback",
                "sentiment": "happy",
                "chat_history": [
                    {"role": "user", "content": "Just wanted to say your service has been amazing!"},
                    {"role": "assistant", "content": "Thank you so much! That really makes our day. What specifically has been helpful?"},
                    {"role": "user", "content": "The customer support team solved my issue in minutes"}
                ],
                "current_message": "I'll definitely be recommending you to my colleagues!",
                "emotional_indicators": ["satisfaction", "enthusiasm"]
            },
            {
                "type": "product_inquiry",
                "sentiment": "neutral",
                "chat_history": [
                    {"role": "user", "content": "Do you offer educational discounts?"},
                    {"role": "assistant", "content": "Yes, we do offer special pricing for educational institutions."},
                    {"role": "user", "content": "What documentation do I need to provide?"}
                ],
                "current_message": "And how long does the verification process usually take?",
                "emotional_indicators": ["professional", "information-seeking"]
            }
        ]
        
        test_contexts = []
        
        # Generate contexts by cycling through scenarios
        for i in range(num_contexts):
            scenario = scenarios[i % len(scenarios)]
            
            # Add some variation to the base scenarios
            context = ChatContext(
                chat_history=scenario["chat_history"].copy(),
                current_message=scenario["current_message"],
                sentiment=scenario["sentiment"],
                emotional_indicators=scenario["emotional_indicators"],
                rag_context=f"Customer type: {scenario['type']}, Account tier: Premium" if i % 3 == 0 else None
            )
            
            test_contexts.append(context)
        
        return test_contexts

# ============== Main Execution Script ==============

async def main():
    """Main execution function"""
    
    # Configuration
    config = {
        "openai_api_key": "your-openai-api-key",
        "anthropic_api_key": "your-anthropic-api-key",
        "num_test_contexts": 10,  # Reduced for demo
        "models_to_test": {
            "gpt-4": "gpt-4-turbo-preview",
            "gpt-3.5": "gpt-3.5-turbo",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229"
        },
        "judge_model": "gpt-4-turbo-preview"  # Use best model as judge
    }
    
    # Initialize providers
    providers = {}
    
    if config["openai_api_key"] != "your-openai-api-key":
        providers["gpt-4"] = OpenAIProvider(config["openai_api_key"], config["models_to_test"]["gpt-4"])
        providers["gpt-3.5"] = OpenAIProvider(config["openai_api_key"], config["models_to_test"]["gpt-3.5"])
    
    if config["anthropic_api_key"] != "your-anthropic-api-key":
        providers["claude-3-opus"] = AnthropicProvider(config["anthropic_api_key"], config["models_to_test"]["claude-3-opus"])
        providers["claude-3-sonnet"] = AnthropicProvider(config["anthropic_api_key"], config["models_to_test"]["claude-3-sonnet"])
    
    if not providers:
        logger.error("No valid API keys provided. Please update the configuration.")
        return
    
    # Initialize judge
    judge_provider = OpenAIProvider(config["openai_api_key"], config["judge_model"])
    
    # Generate test data
    logger.info("Generating test contexts...")
    test_contexts = TestDataGenerator.generate_test_contexts(config["num_test_contexts"])
    
    # Run evaluation pipeline
    logger.info("Starting evaluation pipeline...")
    pipeline = EvaluationPipeline(judge_provider)
    
    results_df = await pipeline.evaluate_models(
        test_contexts=test_contexts,
        model_providers=providers,
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
    
    print("\nOverall Quality Scores:")
    for _, row in summary_df.sort_values('avg_overall_quality', ascending=False).iterrows():
        print(f"{row['model']}: {row['avg_overall_quality']:.2f}")
    
    print("\nGeneration Speed (seconds):")
    for _, row in summary_df.sort_values('avg_generation_time').iterrows():
        print(f"{row['model']}: {row['avg_generation_time']:.2f}s")
    
    print(f"\nDetailed reports available in: {report_dir}")

# ============== Utility Functions ==============

def setup_api_keys():
    """Helper function to set up API keys from environment variables"""
    import os
    
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key")
    }
    
    return config

def run_evaluation(
    test_contexts: Optional[List[ChatContext]] = None,
    num_contexts: int = 10,
    models_to_test: Optional[Dict[str, str]] = None,
    judge_model: str = "gpt-4-turbo-preview"
):
    """Convenience function to run evaluation"""
    
    # Get API keys
    api_config = setup_api_keys()
    
    # Default models if not specified
    if models_to_test is None:
        models_to_test = {
            "gpt-4": "gpt-4-turbo-preview",
            "gpt-3.5": "gpt-3.5-turbo"
        }
    
    # Generate test contexts if not provided
    if test_contexts is None:
        test_contexts = TestDataGenerator.generate_test_contexts(num_contexts)
    
    # Update config
    config = {
        **api_config,
        "num_test_contexts": len(test_contexts),
        "models_to_test": models_to_test,
        "judge_model": judge_model
    }
    
    # Run async main with config
    import asyncio
    asyncio.run(main())

if __name__ == "__main__":
    # Example usage
    print("LLM Judge Evaluation System")
    print("-" * 50)
    print("This system evaluates LLM performance for chat response suggestions.")
    print("\nTo run the evaluation:")
    print("1. Set your API keys as environment variables:")
    print("   export OPENAI_API_KEY='your-key'")
    print("   export ANTHROPIC_API_KEY='your-key'")
    print("2. Run: python llm_judge_system.py")
    print("\nOr use the run_evaluation() function programmatically.")
    
    # Uncomment to run evaluation
    # asyncio.run(main())