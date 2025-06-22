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

def build_prompt_without_sentiment(recent_entry, context_history):
    username = recent_entry["sender"]
    recent_msg = recent_entry["message"]

    formatted_context = "\n".join(
        f"{entry['sender']}: {entry['message']}"
        for entry in context_history if entry["sender"] != username
    )
    own_context = "\n".join(
        f"{entry['sender']}: {entry['message']}"
        for entry in context_history if entry["sender"] == username
    )

    prompt = f"""
You are a helpful chatbot assistant in a group chat.

The current user's name is: {username}

### {username}'s Recent message: {recent_msg}

### Other participants' prior messages:
{formatted_context}

### {username}'s prior messages:
{own_context}

### Task:
Suggest 1 to 3 supportive, emotionally aware replies that any group member might send in response.
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
    prompt = build_prompt_without_sentiment(recent_entry, context_history)

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