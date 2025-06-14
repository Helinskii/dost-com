from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from llm_handler_withoutSentiment import get_openai_rag_response
from llm_handler_withSentiment import get_openai_rag_response
#from llm_handler import get_hf_rag_response
#from llama_handler import get_tinyllama_rag_response

app = FastAPI()

# Define Pydantic model
class ChatEntry(BaseModel):
    sender: str
    message: str
    sentiment: str

class RecentEntry(BaseModel):
    sender: str
    message: str
    sentiments: Dict[str, float]  # probabilities

class SuggestionRequest(BaseModel):
    recent_entry: RecentEntry
    context_history: List[ChatEntry]

@app.post("/suggest-reply/with-sentiment")
async def suggest_with_sentiment(request: SuggestionRequest):
    # Convert to standard Python dicts
    recent_dict: Dict = request.recent_entry.model_dump()
    context_list: List[Dict] = [entry.model_dump() for entry in request.context_history]

    # Call OpenAI LLM (uncomment if needed)
    suggestions = get_openai_rag_response(recent_dict, context_list)

    # Call TinyLlama LLM (uncomment if needed)
    # suggestions = get_tinyllama_rag_response(recent_dict, context_list)

    # Call HuggingFace LLM (uncomment if needed)
    # suggestions = get_hf_rag_response(recent_dict, context_list)  

    return {
        "type": "with_sentiment",
        "suggested_replies": suggestions,
        "recent_message": recent_dict
    }

@app.post("/suggest-reply/without-sentiment")
async def suggest_without_sentiment(request: SuggestionRequest):
    # Convert to standard Python dicts
    recent_dict = request.recent_entry.model_dump()
    context_list = [entry.model_dump() for entry in request.context_history]

    # Call OpenAI LLM (uncomment if needed)
    suggestions = get_openai_rag_response(recent_dict, context_list)

    # Call TinyLlama LLM (uncomment if needed)
    # suggestions = get_tinyllama_rag_response(recent_dict, context_list)

    # Call HuggingFace LLM (uncomment if needed)
    # suggestions = get_hf_rag_response(recent_dict, context_list)  

    return {
        "type": "without_sentiment",
        "suggested_replies": suggestions,
        "recent_message": recent_dict
    }