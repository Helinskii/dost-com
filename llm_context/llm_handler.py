import openai
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests

load_dotenv()



############Uncomment this for OpenAI LLMs############



## Builds the prompt for the LLM using the recent message and context history
def build_rag_prompt(recent_entry, context_history):

    username = recent_entry['sender']
    message = recent_entry['message']
    
    # Convert probabilities to percentages and sort
    sentiment_scores = recent_entry.get("sentiment", {})
    sorted_sentiments = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
    sentiment_text = ", ".join([f"{k}: {round(v * 100)}%" for k, v in sorted_sentiments]) or "No dominant emotion detected"

    # Filter context to include the user's own messages
    if context_history:
        own_context = "\n".join(
            f"{entry['sender']}: {entry['message']} (emotion: {entry['sentiment']})"
            for entry in context_history if entry['sender'] == username
        )
    else:
        own_context = []    

    # Filter context to exclude the user's own messages
    if context_history:
        others_context = "\n".join(
            f"{entry['sender']}: {entry['message']} (emotion: {entry['sentiment']})"
            for entry in context_history if entry['sender'] != username
        )
    else:
        others_context = []

    prompt = f"""

You are an emotionally intelligent assistant embedded in a group chat support system.
The current user's name is: {username}

Your task is to Understand the emotional tone and context, and Generate kind and constructive messages that someone else in the group chat can respond with.

Take into account:
- The emotional profile of the recent message
- The user's own past messages to understand what they're going through
- Other users' past responses to maintain coherence and empathy

### Recent Message from {username}:
{message}  
Emotional Profile: {sentiment_text}

### {username}'s Previous Messages:
{own_context}

### Messages from Other Users:
{others_context}

### Task:
Suggest 1 to 3 supportive, emotionally aware replies that a group member might send in response.
Guidelines:
- Do NOT assume how many people are in the group. Avoid phrases like “just the two of us” or “three of us”.
- Do NOT include usernames or any names.
- Do NOT number the suggestions.
- Keep replies short (under 150 characters) and kind.
- Return only the raw replies (no labels or explanations).
"""
    return prompt.strip()

############Uncomment this for OpenAI LLMs###########
def get_openai_rag_response(recent_entry, context_history):

    prompt = build_rag_prompt(recent_entry, context_history)

    ############Uncomment this for OpenAI LLMs############
    openai.api_key = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4o-mini"

    ###########Uncomment this For Ollama LLMs############
    # OLLAMA_URL = "http://localhost:11434/api/generate"
    # MODEL_NAME = "tinyllama"

    ############Uncomment this for HuggingFace LLMs########
    # MODEL_NAME = "distilgpt2"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
    
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a empathetic and kind, context-aware intelligent assistant helping in group chats."},
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