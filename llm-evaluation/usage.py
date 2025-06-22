"""
LLM Judge Evaluation System - Usage Examples and Utilities
Practical examples and helper functions for the evaluation system
"""

import asyncio
import pandas as pd
from typing import List, Dict, Optional
import json
from pathlib import Path
import os

# Assuming the main module is imported as llm_judge_system
import llm_as_judge as ljs

# ============== Quick Start Examples ==============

def example_basic_evaluation():
    """Basic example of running an evaluation"""
    
    # Set up API keys (use environment variables in production)
    # os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'
    # os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-api-key'
    
    # Run evaluation with defaults
    ljs.run_evaluation(
        num_contexts=5,  # Small number for testing
        models_to_test={
            "gpt-3.5": "gpt-3.5-turbo",
            "gpt-4": "gpt-4-turbo-preview"
        },
        judge_model="gpt-4-turbo-preview"
    )

def example_custom_test_data():
    """Example with custom test contexts"""
    
    # Create custom test contexts
    custom_contexts = [
        ljs.ChatContext(
            chat_history=[
                {"role": "user", "content": "I need help with my order"},
                {"role": "assistant", "content": "I'd be happy to help! Can you provide your order number?"},
                {"role": "user", "content": "It's #12345"}
            ],
            current_message="The delivery status shows delayed but I need it by tomorrow",
            sentiment="worried",
            emotional_indicators=["urgency", "concern"],
            rag_context="Customer is a premium member with expedited shipping"
        ),
        ljs.ChatContext(
            chat_history=[
                {"role": "user", "content": "How do I integrate your API?"},
                {"role": "assistant", "content": "Our API is REST-based. What programming language are you using?"},
                {"role": "user", "content": "Python"}
            ],
            current_message="Do you have any example code I can start with?",
            sentiment="neutral",
            emotional_indicators=["technical", "learning"],
            rag_context="Developer documentation available at docs.example.com"
        )
    ]
    
    # Run evaluation
    ljs.run_evaluation(test_contexts=custom_contexts)

# ============== Advanced Configuration ==============

class CustomEvaluationConfig:
    """Advanced configuration for evaluation runs"""
    
    def __init__(self):
        self.config = {
            # Model configurations
            "models": {
                "openai": {
                    "gpt-4": {
                        "model_name": "gpt-4-turbo-preview",
                        "temperature": 0.7,
                        "max_tokens": 300
                    },
                    "gpt-3.5": {
                        "model_name": "gpt-3.5-turbo",
                        "temperature": 0.7,
                        "max_tokens": 300
                    }
                },
                "anthropic": {
                    "claude-opus": {
                        "model_name": "claude-3-opus-20240229",
                        "temperature": 0.7,
                        "max_tokens": 300
                    },
                    "claude-sonnet": {
                        "model_name": "claude-3-sonnet-20240229",
                        "temperature": 0.7,
                        "max_tokens": 300
                    }
                }
            },
            
            # Judge configuration
            "judge": {
                "model": "gpt-4-turbo-preview",
                "temperature": 0,  # Consistent evaluation
                "max_tokens": 800
            },
            
            # Evaluation settings
            "evaluation": {
                "batch_size": 5,
                "save_intermediate": True,
                "retry_failed": True,
                "max_retries": 3
            }
        }
    
    def get_providers(self):
        """Initialize providers from configuration"""
        providers = {}
        
        # OpenAI providers
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            for name, config in self.config['models']['openai'].items():
                providers[name] = ljs.OpenAIProvider(openai_key, config['model_name'])
        
        # Anthropic providers
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            for name, config in self.config['models']['anthropic'].items():
                providers[name] = ljs.AnthropicProvider(anthropic_key, config['model_name'])
        
        return providers

# ============== Custom Test Data Loaders ==============

class TestDataLoader:
    """Load test data from various sources"""
    
    @staticmethod
    def load_from_json(file_path: str) -> List[ljs.ChatContext]:
        """Load test contexts from JSON file"""
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        contexts = []
        for item in data:
            context = ljs.ChatContext(
                chat_history=item['chat_history'],
                current_message=item['current_message'],
                sentiment=item.get('sentiment'),
                emotional_indicators=item.get('emotional_indicators', []),
                rag_context=item.get('rag_context')
            )
            contexts.append(context)
        
        return contexts
    
    @staticmethod
    def load_from_csv(file_path: str) -> List[ljs.ChatContext]:
        """Load test contexts from CSV file"""
        
        df = pd.read_csv(file_path)
        contexts = []
        
        for _, row in df.iterrows():
            # Parse chat history (assumed to be JSON string)
            chat_history = json.loads(row['chat_history'])
            
            context = ljs.ChatContext(
                chat_history=chat_history,
                current_message=row['current_message'],
                sentiment=row.get('sentiment'),
                emotional_indicators=json.loads(row.get('emotional_indicators', '[]')),
                rag_context=row.get('rag_context')
            )
            contexts.append(context)
        
        return contexts
    
    @staticmethod
    def save_contexts_to_json(contexts: List[ljs.ChatContext], file_path: str):
        """Save contexts to JSON for reuse"""
        
        data = []
        for context in contexts:
            data.append({
                'chat_history': context.chat_history,
                'current_message': context.current_message,
                'sentiment': context.sentiment,
                'emotional_indicators': context.emotional_indicators,
                'rag_context': context.rag_context
            })
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

# ============== Result Analysis Utilities ==============

class ResultAnalyzer:
    """Advanced analysis of evaluation results"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
    
    def get_model_strengths_weaknesses(self, model_name: str) -> Dict[str, List[str]]:
        """Identify strengths and weaknesses of a specific model"""
        
        model_data = self.results_df[self.results_df['model_name'] == model_name]
        
        # Calculate average scores for each metric
        metrics = ['relevance', 'sentiment', 'naturalness', 'helpfulness']
        metric_scores = {}
        
        for metric in metrics:
            scores = []
            for i in range(1, 4):
                scores.extend(model_data[f'suggestion_{i}_{metric}'].values)
            metric_scores[metric] = pd.Series(scores).mean()
        
        # Determine strengths and weaknesses
        sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        
        strengths = [m[0] for m in sorted_metrics[:2] if m[1] >= 7]
        weaknesses = [m[0] for m in sorted_metrics[-2:] if m[1] < 7]
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'scores': metric_scores
        }
    
    def find_failure_cases(self, threshold: float = 5.0) -> pd.DataFrame:
        """Find cases where models performed poorly"""
        
        failure_cases = []
        
        for idx, row in self.results_df.iterrows():
            # Check if any suggestion scored below threshold on key metrics
            low_scores = []
            
            for i in range(1, 4):
                for metric in ['relevance', 'naturalness', 'helpfulness']:
                    score = row[f'suggestion_{i}_{metric}']
                    if score < threshold:
                        low_scores.append({
                            'suggestion': i,
                            'metric': metric,
                            'score': score
                        })
            
            if low_scores:
                failure_cases.append({
                    'context_id': row['context_id'],
                    'model_name': row['model_name'],
                    'low_scores': low_scores,
                    'overall_quality': row['overall_quality']
                })
        
        return pd.DataFrame(failure_cases)
    
    def compare_models_statistical(self, model1: str, model2: str) -> Dict[str, Any]:
        """Statistical comparison between two models"""
        
        from scipy import stats
        
        model1_data = self.results_df[self.results_df['model_name'] == model1]
        model2_data = self.results_df[self.results_df['model_name'] == model2]
        
        comparisons = {}
        
        # Compare overall quality
        t_stat, p_value = stats.ttest_ind(
            model1_data['overall_quality'],
            model2_data['overall_quality']
        )
        
        comparisons['overall_quality'] = {
            'model1_mean': model1_data['overall_quality'].mean(),
            'model2_mean': model2_data['overall_quality'].mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Compare individual metrics
        for metric in ['relevance', 'sentiment', 'naturalness', 'helpfulness']:
            model1_scores = []
            model2_scores = []
            
            for i in range(1, 4):
                model1_scores.extend(model1_data[f'suggestion_{i}_{metric}'].values)
                model2_scores.extend(model2_data[f'suggestion_{i}_{metric}'].values)
            
            t_stat, p_value = stats.ttest_ind(model1_scores, model2_scores)
            
            comparisons[metric] = {
                'model1_mean': pd.Series(model1_scores).mean(),
                'model2_mean': pd.Series(model2_scores).mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return comparisons

# ============== Continuous Monitoring ==============

class ContinuousEvaluator:
    """Set up continuous evaluation for production monitoring"""
    
    def __init__(self, config: CustomEvaluationConfig):
        self.config = config
        self.results_history = []
    
    async def run_periodic_evaluation(
        self,
        test_contexts: List[ljs.ChatContext],
        interval_hours: int = 24
    ):
        """Run evaluation periodically"""
        
        while True:
            try:
                # Run evaluation
                providers = self.config.get_providers()
                judge_provider = ljs.OpenAIProvider(
                    os.getenv('OPENAI_API_KEY'),
                    self.config.config['judge']['model']
                )
                
                pipeline = ljs.EvaluationPipeline(judge_provider)
                results_df = await pipeline.evaluate_models(
                    test_contexts=test_contexts,
                    model_providers=providers
                )
                
                # Save results with timestamp
                timestamp = pd.Timestamp.now()
                results_df['evaluation_timestamp'] = timestamp
                self.results_history.append(results_df)
                
                # Generate report
                report_gen = ljs.ReportGenerator(results_df)
                report_gen.generate_full_report(f"periodic_evaluation_{timestamp.strftime('%Y%m%d_%H%M%S')}")
                
                # Check for performance degradation
                self._check_performance_degradation()
                
                # Wait for next evaluation
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                print(f"Error in periodic evaluation: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _check_performance_degradation(self):
        """Check if any model's performance has degraded"""
        
        if len(self.results_history) < 2:
            return
        
        current_results = self.results_history[-1]
        previous_results = self.results_history[-2]
        
        # Compare average overall quality
        current_avg = current_results.groupby('model_name')['overall_quality'].mean()
        previous_avg = previous_results.groupby('model_name')['overall_quality'].mean()
        
        for model in current_avg.index:
            if model in previous_avg.index:
                change = current_avg[model] - previous_avg[model]
                if change < -0.5:  # Significant degradation
                    print(f"WARNING: Performance degradation detected for {model}")
                    print(f"Previous: {previous_avg[model]:.2f}, Current: {current_avg[model]:.2f}")

# ============== Export Utilities ==============

def export_for_dashboard(results_df: pd.DataFrame, output_path: str):
    """Export results in format suitable for dashboards"""
    
    # Aggregate metrics by model
    dashboard_data = []
    
    for model in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model]
        
        # Calculate aggregated metrics
        metrics = {}
        for metric in ['relevance', 'sentiment', 'naturalness', 'helpfulness']:
            scores = []
            for i in range(1, 4):
                scores.extend(model_data[f'suggestion_{i}_{metric}'].values)
            
            metrics[f'{metric}_mean'] = pd.Series(scores).mean()
            metrics[f'{metric}_std'] = pd.Series(scores).std()
            metrics[f'{metric}_min'] = pd.Series(scores).min()
            metrics[f'{metric}_max'] = pd.Series(scores).max()
        
        dashboard_data.append({
            'model': model,
            'overall_quality': model_data['overall_quality'].mean(),
            'diversity_score': model_data['diversity_score'].mean(),
            'avg_generation_time': model_data['generation_time'].mean(),
            'total_evaluations': len(model_data),
            **metrics
        })
    
    # Save as JSON for web dashboards
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

# ============== Main Example Script ==============

if __name__ == "__main__":
    print("LLM Judge System - Examples and Utilities")
    print("-" * 50)
    
    # Example 1: Basic evaluation
    print("\nExample 1: Running basic evaluation...")
    # example_basic_evaluation()
    
    # Example 2: Custom test data
    print("\nExample 2: Using custom test contexts...")
    # example_custom_test_data()
    
    # Example 3: Load results and analyze
    print("\nExample 3: Analyzing existing results...")
    
    # Load sample results (if exists)
    results_path = Path("evaluation_reports")
    if results_path.exists():
        # Find most recent results
        csv_files = list(results_path.glob("*/raw_results.csv"))
        if csv_files:
            latest_results = pd.read_csv(csv_files[-1])
            analyzer = ResultAnalyzer(latest_results)
            
            # Get model strengths/weaknesses
            for model in latest_results['model_name'].unique():
                analysis = analyzer.get_model_strengths_weaknesses(model)
                print(f"\n{model}:")
                print(f"  Strengths: {', '.join(analysis['strengths'])}")
                print(f"  Weaknesses: {', '.join(analysis['weaknesses'])}")
    
    print("\nFor full functionality, set up API keys and run the examples.")