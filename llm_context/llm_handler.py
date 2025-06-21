import openai
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests

load_dotenv()

## Builds the prompt for the LLM using the recent message and context history
def build_rag_prompt(recent_entry, context_history):

    username = recent_entry['Sender']
    message = recent_entry['Message']
    
    # Convert probabilities to percentages and sort
    emotion_detected = recent_entry.get("Sentiment", {})
    # sorted_sentiments = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
    # sentiment_text = ", ".join([f"{k}: {round(v * 100)}%" for k, v in sorted_sentiments]) or "No dominant emotion detected"

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
- The emotion of the recent message
- The user's own past messages to understand what they're going through
- Other users' past responses to maintain coherence and empathy

### Recent Message from {username}:
{message}  
Emotion: {emotion_detected}

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

    ############Uncomment this for OpenAI LLMs############
    openai.api_key = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4o-mini"

    prompt = build_rag_prompt(recent_entry, context_history)
    
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a empathetic and kind, context-aware intelligent assistant helping in group chats."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250,
    )
    content = response.choices[0].message.content
    raw_text = content.strip() if content is not None else ""

    suggestions = [
        line.strip().replace('\n', ' ')
        for line in raw_text.split('\n\n')
        if line.strip()
    ]

    return suggestions[:3]  # Return only the first 3 suggestions


##########Uncomment this for Ollama LLMs#############
# def get_tinyllama_rag_response(recent_entry, context_history):

#     OLLAMA_URL = "http://localhost:11434/api/generate"
#     MODEL_NAME = "tinyllama"

#     prompt = build_rag_prompt(recent_entry, context_history)

#     response = requests.post(OLLAMA_URL, json={
#         "model": MODEL_NAME,
#         "prompt": prompt,
#         "stream": False
#     })

#     if response.status_code == 200:
#         raw_text = response.json()['response'].strip()

#         Split and clean suggestions
#         suggestions = [
#           line.strip().replace('\n', ' ')
#           for line in raw_text.split('\n\n')
#           if line.strip()
#         ]
#         return suggestions[:3]  # Return only the first 3 suggestions
#     else:
#         return f"Error: {response.status_code} - {response.text}"


##########Uncomment this for HuggingFace LLMs##########
# def get_hf_rag_response(recent_entry, context_history, max_tokens=100):

#     MODEL_NAME = "distilgpt2"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
#     model.eval()
#     prompt = build_rag_prompt(recent_entry, context_history)
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
#     with torch.no_grad():
#         output = model.generate(
#         **inputs,
#         max_new_tokens=max_tokens,
#         do_sample=True,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Remove prompt from beginning of decoded output
#     generated = decoded[len(prompt):].strip()

#     # Split suggestions by line or double newline
#     suggestions = [line.strip() for line in generated.split('\n') if line.strip()]

#     return suggestions