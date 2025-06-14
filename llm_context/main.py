from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from llm_handler import get_openai_rag_response
#from llm_handler import get_hf_rag_response
#from llama_handler import get_tinyllama_rag_response

app = FastAPI()

# Define Pydantic model
class ChatEntry(BaseModel):
    sender: str
    message: str
    sentiment: str

class SuggestionRequest(BaseModel):
    recent_entry: ChatEntry
    context_history: List[ChatEntry]

@app.post("/suggest-reply/")
async def suggest_reply(request: SuggestionRequest):
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
        "suggested_replies": suggestions,
        "recent_message": recent_dict,
        "context_used": context_list
    }
