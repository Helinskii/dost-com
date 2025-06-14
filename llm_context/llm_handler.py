import openai
import os
from dotenv import load_dotenv

###########Uncomment this For Ollama LLMs############
# import requests
# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "tinyllama"


############Uncomment this for HuggingFace LLMs########
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# MODEL_NAME = "distilgpt2"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")

############Uncomment this for OpenAI LLMs############
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

## Formats the retrieved RAG chat context into a newline-separated string
def format_rag_context(chat_context):
    return "\n".join(
        f"{entry['sender']}: {entry['message']} (emotion: {entry['sentiment']})"
        for entry in chat_context
    )

## Builds the prompt for the LLM using the recent message and context history
def build_rag_prompt(recent_entry, context_history):
    formatted_context = format_rag_context(context_history)

    prompt = f"""
You are an empathetic and emotionally intelligent assistant in a group chat system.

Below is:
1. The recent message along with its detected emotion.
2. The relevant conversation history retrieved via RAG.

Your job is to understand the emotional tone and context, and suggest 1 to 3 emotionally appropriate, kind, and constructive messages that would help keep the conversation positive and supportive.

### Recent Message:
{recent_entry['sender']}: {recent_entry['message']} (emotion: {recent_entry['sentiment']})

### Context History:
{formatted_context}

### Task:
Provide 1 to 3 appropriate, supportive replies that someone in the chat could respond with next.
Do not include any explanations or labels, just the suggestions.
"""
    return prompt.strip()

############Uncomment this for OpenAI LLMs###########
def get_openai_rag_response(recent_entry, context_history):
    prompt = build_rag_prompt(recent_entry, context_history)

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a kind, context-aware assistant helping in group chats."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


##########Uncomment this for Ollama LLMs#############
# def get_tinyllama_rag_response(recent_entry, context_history):
#     prompt = build_rag_prompt(recent_entry, context_history)

#     response = requests.post(OLLAMA_URL, json={
#         "model": MODEL_NAME,
#         "prompt": prompt,
#         "stream": False
#     })

#     if response.status_code == 200:
#         return response.json()['response'].strip()
#     else:
#         return f"Error: {response.status_code} - {response.text}"


##########Uncomment this for HuggingFace LLMs##########
# def get_hf_rag_response(recent_entry, context_history, max_tokens=100):
#     prompt = build_rag_prompt(recent_entry, context_history)
#     inputs = tokenizer(prompt, return_tensors="pt")

#     output = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95
#     )

#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)
#     # Return only the new content after the prompt
#     return decoded[len(prompt):].strip()