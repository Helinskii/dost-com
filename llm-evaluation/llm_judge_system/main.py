import os
import asyncio
from dotenv import load_dotenv
from .test_data import TestDataGenerator
from .providers import OpenAIProvider, AnthropicProvider, GeminiProvider
from .pipeline import EvaluationPipeline
from .report import ReportGenerator

async def main():
    load_dotenv()
    import logging
    from datetime import datetime
    config = {
        "num_test_contexts": 10,
        "models_to_test": {
            "gpt-4": "gpt-4-turbo-preview",
            "gpt-3.5": "gpt-3.5-turbo",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "gemini-pro": "gemini-1.5-pro"
        },
        "judge_model": "gpt-4-turbo-preview",
        "prompt_variants": ["base", "no_positivity", "no_sentiment"]
    }
    providers = {}
    if os.getenv("OPENAI_API_KEY"):
        try:
            providers["gpt-4"] = OpenAIProvider(config["models_to_test"]["gpt-4"])
            providers["gpt-3.5"] = OpenAIProvider(config["models_to_test"]["gpt-3.5"])
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI providers: {e}")
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            providers["claude-3-opus"] = AnthropicProvider(config["models_to_test"]["claude-3-opus"])
            providers["claude-3-sonnet"] = AnthropicProvider(config["models_to_test"]["claude-3-sonnet"])
        except Exception as e:
            logging.warning(f"Failed to initialize Anthropic providers: {e}")
    if os.getenv("GOOGLE_API_KEY"):
        try:
            providers["gemini-pro"] = GeminiProvider(config["models_to_test"]["gemini-pro"])
        except Exception as e:
            logging.warning(f"Failed to initialize Gemini provider: {e}")
    if not providers:
        logging.error("No valid API keys provided. Please update your .env file.")
        return
    judge_provider = None
    if os.getenv("OPENAI_API_KEY"):
        judge_provider = OpenAIProvider(config["judge_model"])
    elif os.getenv("GOOGLE_API_KEY"):
        judge_provider = GeminiProvider("gemini-1.5-pro")
        logging.info("Using Gemini as judge since OpenAI key not available")
    else:
        logging.error("No judge model available. Need either OpenAI or Google API key.")
        return
    logging.info("Generating test contexts...")
    test_contexts = TestDataGenerator.generate_test_contexts(config["num_test_contexts"])
    logging.info("Starting evaluation pipeline...")
    pipeline = EvaluationPipeline(judge_provider)
    results_df = await pipeline.evaluate_models_with_prompts(
        test_contexts=test_contexts,
        model_providers=providers,
        prompt_variants=config["prompt_variants"],
        save_intermediate=True
    )
    logging.info("Generating evaluation reports...")
    report_generator = ReportGenerator(results_df)
    report_dir = report_generator.generate_full_report()
    logging.info(f"Evaluation complete! Reports saved to: {report_dir}")
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

def run_evaluation(
    test_contexts=None,
    num_contexts=10,
    models_to_test=None,
    judge_model="gpt-4-turbo-preview",
    prompt_variants=None
):
    load_dotenv()
    if models_to_test is None:
        models_to_test = {
            "gpt-4": "gpt-4-turbo-preview",
            "gpt-3.5": "gpt-3.5-turbo",
            "gemini-pro": "gemini-1.5-pro"
        }
    if prompt_variants is None:
        prompt_variants = ["base", "no_positivity", "no_sentiment"]
    if test_contexts is None:
        test_contexts = TestDataGenerator.generate_test_contexts(num_contexts)
    asyncio.run(main())

if __name__ == "__main__":
    print("LLM Judge Evaluation System - Modular Version")
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
    print("2. Run: python -m llm_judge_system.main")
    # Uncomment to run evaluation
    # asyncio.run(main())
