import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
logging.basicConfig(filename='llm_eval.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
# Visualization imports (plotly, matplotlib, seaborn, etc.)
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
from models import EvaluationResult

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
        
        logging.info(f"Report generated in: {report_dir}")
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
        # Group by model (averaging across prompt variants), only numeric columns
        numeric_cols = summary_df.select_dtypes(include='number').columns
        model_avg = summary_df.groupby('model')[numeric_cols].mean()
        
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
        
        logging.info("Text report generated")