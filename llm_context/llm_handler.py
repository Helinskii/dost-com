import openai
import os
# import json
# from guardrails import Guard
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests

# def get_system_prompt_from_rail(xml_path):
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     instructions = root.find(".//instructions")
#     return instructions.text.strip() if instructions is not None else ""

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
- For greetings or casual messages like "Hi", "Hello", "How are you?", suggest friendly, relaxed replies.

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
    load_dotenv()
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
    # load_dotenv()
    # import google.generativeai as genai
    # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # MODEL_NAME = "gemini-1.5-flash"
    # prompt = build_rag_prompt(recent_entry, context_history)
    # model = genai.GenerativeModel(MODEL_NAME)
    # response = model.generate_content(
    #     prompt,
    #     generation_config={
    #         "temperature": 0.7,
    #         "max_output_tokens": 250,
    #         "top_p": 0.95,
    #         "top_k": 40
    #     }
    # )
    # raw_text = response.text.strip() if hasattr(response, "text") else ""
    # if raw_text.startswith("```"):
    #     raw_text = raw_text.strip("`").strip()
    # if "\n\n" in raw_text:
    #     lines = raw_text.split('\n\n')
    # else:
    #     lines = raw_text.split('\n')
    # suggestions = [
    #     line.strip().replace('\n', ' ')
    #     for line in lines
    #     if line.strip() and not line.strip().startswith("```")
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



    # ############Tried GuardRail#############
    # print("=== DEBUGGING INPUT VALIDATION ===")
    # # Normalize keys in recent_entry
    # recent_entry = {
    # "sender": recent_entry.get("sender") or recent_entry.get("Sender"),
    # "message": recent_entry.get("message") or recent_entry.get("Message"),
    # "sentiment": recent_entry.get("sentiment") or recent_entry.get("Sentiment")
    # }

    # # Normalize context_history items
    # context_history = [
    # {
    #     "sender": e.get("sender"),
    #     "message": e.get("message"),
    # }
    # for e in context_history
    # ]

    # print("Input recent_entry:", json.dumps(recent_entry, indent=2))
    # print("Input context_history:", json.dumps(context_history, indent=2))

    # load_dotenv()
    # ############Uncomment this for OpenAI LLMs############
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # MODEL_NAME = "gpt-4o-mini"

    # try:
    #     # Load guard from XML file
    #     print("Loading guardrails from XML...")
    #     guard = Guard.for_rail("llm_guardrail.xml")
    #     print("Guardrails loaded successfully!")
    #     # Test input validation
    #     input_data = {
    #         "recent_entry": recent_entry,
    #         "context_history": context_history
    #     }

    #     print("Validating input data...")
    #     print("Input data structure:", json.dumps(input_data, indent=2))
        
    #     # Validate input
    #     _, validation_results, *_ = guard.validate(
    #         llm_output="{}",  # dummy output to trigger input validation
    #         prompt_params=input_data
    #     )

    #     # If any violations exist, block the request early
    #     if validation_results and validation_results.get("violations"):
    #         messages = []
    #         for v in validation_results["violations"]:
    #             field = v.get("field", "")
    #             name = v.get("name", "Unknown")
    #             description = v.get("description", "No description provided.")
    #             messages.append(f"Violation ({name}) in '{field}': {description}")
    #         return messages
    #     print("Input validation PASSED!")

    # except Exception as e:
    #     print("Input validation FAILED!")
    #     print("Error type:", type(e).__name__)
    #     print("Error message:", str(e))
        
    #     # Detailed error analysis
    #     if hasattr(e, 'errors'):
    #         print("Pydantic validation errors:")
    #         for error in e.errors():
    #             print(f"  - Field: {' -> '.join(str(x) for x in error['loc'])}")
    #             print(f"    Error: {error['msg']}")
    #             print(f"    Input: {error.get('input', 'N/A')}")
        
    #     if hasattr(e, 'error_report'):
    #         print("Guardrails error report:")
    #         for field, violations in e.error_report.items():
    #             for violation in violations:
    #                 print(f"  - Field: {field}")
    #                 print(f"    Violation: {violation.get('name', 'Unknown')}")
    #                 print(f"    Description: {violation.get('description', 'N/A')}")
        
    #     return [f"Input validation failed: {str(e)}"]

    # # If we get here, validation passed - continue with LLM call
    # try:
    #     system_prompt = get_system_prompt_from_rail("llm_guardrail.xml")
    #     prompt = build_rag_prompt(recent_entry, context_history)

    #     print("Making OpenAI API call...")
    #     response = openai.chat.completions.create(
    #         model=MODEL_NAME,
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.7,
    #         max_tokens=250,
    #     )
        
    #     content = response.choices[0].message.content
    #     raw_text = content.strip() if content is not None else ""
    #     print("LLM Response:", raw_text)

    #     suggestions = [
    #         line.strip().replace('\n', ' ')
    #         for line in raw_text.split('\n\n')
    #         if line.strip()
    #     ][:3]

    #     print("Parsed suggestions:", suggestions)

    #     # Validate output
    #     raw_output = {"suggested_replies": suggestions}
        
    #     validated_output, validation_results, *_ = guard.validate(
    #         llm_output=json.dumps(raw_output),
    #         prompt_params={
    #             "recent_entry": recent_entry,
    #             "context_history": context_history
    #         },
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": prompt}
    #         ]
    #     )

    #     if not validation_results or not validation_results.get("violations"):
    #         if isinstance(validated_output, str):
    #             validated_output = json.loads(validated_output)
    #         return validated_output.get("suggested_replies", [])
    #     else:
    #         print("Output validation failed:", validation_results)
    #         return ["Output validation failed - response violated safety policies."]

    # except Exception as e:
    #     print(f"Error during LLM processing: {e}")
    #     return [f"Error generating response: {str(e)}"]