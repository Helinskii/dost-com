import os
import asyncio
import logging
logging.basicConfig(filename='llm_eval.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from dotenv import load_dotenv
from test_data import TestDataGenerator
from providers import OpenAIProvider, AnthropicProvider, GeminiProvider
from pipeline import EvaluationPipeline
from report import ReportGenerator

async def generate_and_store_suggestions_main():
    load_dotenv()
    from datetime import datetime
    config = {
        "num_test_contexts": 1,
        "models_to_test": {
            "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
            "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
            "gemini-flash": "gemini-2.0-flash",
            "gemini-pro": "gemini-1.5-pro"
        },
        "prompt_variants": ["base", "no_positivity", "no_sentiment"]
    }
    providers = {}
    if os.getenv("OPENAI_API_KEY"):
        for key in ["gpt-4.1-mini", "gpt-4o-mini"]:
            if key in config["models_to_test"]:
                try:
                    providers[key] = OpenAIProvider(config["models_to_test"][key])
                except Exception as e:
                    logging.warning(f"Failed to initialize OpenAI provider {key}: {e}")
    if os.getenv("GEMINI_API_KEY"):
        for key in ["gemini-flash", "gemini-pro"]:
            if key in config["models_to_test"]:
                try:
                    providers[key] = GeminiProvider(config["models_to_test"][key])
                except Exception as e:
                    logging.warning(f"Failed to initialize Gemini provider {key}: {e}")
    if not providers:
        logging.error("No valid API keys provided. Please update your .env file.")
        return
    logging.info("Generating test contexts...")
    test_contexts = TestDataGenerator.generate_test_contexts(config["num_test_contexts"])
    pipeline = EvaluationPipeline(judge_provider=None)  # No judge needed for generation
    await pipeline.generate_and_store_suggestions(
        test_contexts=test_contexts,
        model_providers=providers,
        prompt_variants=config["prompt_variants"],
        output_file="suggestions.jsonl"
    )
    print("Suggestions generated and stored in suggestions.jsonl")

async def evaluate_stored_suggestions_main():
    load_dotenv()
    from datetime import datetime
    config = {
        "judge_model": "gpt-4o-mini-2024-07-18"
    }
    # Judge provider selection
    judge_model_name = config["judge_model"]
    judge_provider = None
    if judge_model_name.startswith("gpt-") and os.getenv("OPENAI_API_KEY"):
        judge_provider = OpenAIProvider(judge_model_name)
    elif judge_model_name.startswith("gemini-") and os.getenv("GEMINI_API_KEY"):
        judge_provider = GeminiProvider(judge_model_name)
    if judge_provider is None:
        logging.error("No judge model available. Please check your config and API keys.")
        return
    pipeline = EvaluationPipeline(judge_provider)
    records = pipeline.load_suggestions("suggestions.jsonl")
    results_df = await pipeline.evaluate_stored_suggestions(records)
    report_generator = ReportGenerator(results_df)
    report_dir = report_generator.generate_full_report()
    print(f"Evaluation complete! Reports saved to: {report_dir}")
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

if __name__ == "__main__":
    print("LLM Judge Evaluation System - Modular Version")
    print("-" * 50)
    print("This system evaluates LLM performance for chat response suggestions.")
    print("\nFeatures:")
    print("- Sentiment probabilities for 6 emotions")
    print("- Positivity impact scoring")
    print("- Multiple prompt variants")
    print("- Support for OpenAI, Anthropic, and Google Gemini models")
    # By default, call the generate function
    asyncio.run(generate_and_store_suggestions_main())
