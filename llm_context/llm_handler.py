import openai
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests

load_dotenv()

## Builds the prompt for the LLM using the recent message and context history
def build_rag_prompt(recent_entry, context_history=None):

    username = recent_entry.get('Sender', {})
    message = recent_entry.get('Message', {})
    emotion_detected = recent_entry.get("Sentiment", {})
    # sorted_sentiment = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
    # sentiment_text = ", ".join([f"{k}: {round(v * 100)}%" for k, v in sorted_sentiment]) or "No dominant emotion detected"

    # Filter context to include the user's own messages
    own_messages = [
        f"{entry['sender']}: {entry['message']}"
        for entry in context_history if entry.get('sender') == username
    ]
    own_context = f"### {username}'s Previous Messages:\n" + "\n".join(own_messages) if own_messages else ""

    # Filter context to exclude the user's own messages
    others_messages = [
        f"{entry['sender']}: {entry['message']}"
        for entry in context_history if entry.get('sender') != username
    ]
    others_context = f"### Messages from Other Users:\n" + "\n".join(others_messages) if others_messages else ""

    prompt = f"""

You are an emotionally intelligent assistant embedded in a group chat support system.
The current user's name is: {username}

Your task is to Understand the emotional tone and context, and Generate kind and constructive messages that someone else in the group chat can respond with.

Take into account:
- The emotion of the recent message
- The user's own past messages if there are any to understand what they're going through
- Other users' past responses if there are any to maintain coherence and empathy

### Recent Message from {username}:
{message}  
Emotion: {emotion_detected}

{own_context}

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

############Uncomment this for OpenAI LLM and Comment other###########
def get_llm_rag_response(recent_entry, context_history):

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
    return suggestions[:3]
    

    # #########Uncomment this for Gemini and Comment other#############
    # import google.generativeai as genai
    # genai.api_key = os.getenv("GEMINI_API_KEY")
    # MODEL_NAME = "gemini-1.5-flash"
    # prompt = build_rag_prompt(recent_entry, context_history)
    # response = genai.generate_content(
    #     model=MODEL_NAME,
    #     prompt=prompt,
    #     temperature=0.7,
    #     max_output_tokens=250,
    #     top_p=0.95,
    #     top_k=40
    # )
    # raw_text = response.candidates[0].content.strip() if response.candidates else ""
    # suggestions = [
    #     line.strip().replace('\n', ' ')
    #     for line in raw_text.split('\n\n')
    #     if line.strip()
    # ]
    # return suggestions[:3] 



    # # #########Uncomment this for MISTRAL LLM and Comment other#############
    # OLLAMA_URL = "http://localhost:11434/api/generate"
    # MODEL_NAME = "mistral"  # Use the Mistral model with Ollama
    # prompt = build_rag_prompt(recent_entry, context_history)
    # response = requests.post(OLLAMA_URL, json={
    #     "model": MODEL_NAME,
    #     "prompt": prompt,
    #     "stream": False
    #     })
    # if response.status_code == 200:
    #     raw_text = response.json()['response'].strip()
    #     # Split and clean suggestions
    #     suggestions = [
    #         line.strip().lstrip('0123456789. ').replace('"', '')
    #         for line in raw_text.split('\n')
    #         if line.strip()
    #     ]
    #     return suggestions[:3]  # Return only the first 3 suggestions
    # else:
    #     return f"Error: {response.status_code} - {response.text}"




    # #########Uncomment this for HuggingFace LLM and Comment other##########
    # MODEL_NAME = "distilgpt2"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
    # model.eval()
    # prompt = build_rag_prompt(recent_entry, context_history)
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    # with torch.no_grad():
    #     output = model.generate(
    #     **inputs,
    #     max_new_tokens=100,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_k=50,
    #     top_p=0.95,
    #     pad_token_id=tokenizer.eos_token_id
    # )
    # decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # # Remove prompt from beginning of decoded output
    # generated = decoded[len(prompt):].strip()
    # # Split suggestions by line or double newline
    # suggestions = [line.strip() for line in generated.split('\n') if line.strip()]
    # return suggestions


    # #########Uncomment this for Ollama LLM and Comment other#############
    # OLLAMA_URL = "http://localhost:11434/api/generate"
    # MODEL_NAME = "tinyllama"
    # prompt = build_rag_prompt(recent_entry, context_history)
    # response = requests.post(OLLAMA_URL, json={
    #     "model": MODEL_NAME,
    #     "prompt": prompt,
    #     "stream": False
    # })
    # if response.status_code == 200:
    #     raw_text = response.json()['response'].strip()
    #     # Split and clean suggestions
    #     suggestions = [
    #       line.strip().replace('\n', ' ')
    #       for line in raw_text.split('\n\n')
    #       if line.strip()
    #     ]
    #     return suggestions[:3]  # Return only the first 3 suggestions
    # else:
    #     return f"Error: {response.status_code} - {response.text}"





