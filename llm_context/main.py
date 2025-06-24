from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from llm_handler import get_llm_rag_response
#from llm_handler import get_hf_rag_response
#from llama_handler import get_tinyllama_rag_response

app = FastAPI()

# Define Pydantic model
class ChatEntry(BaseModel):
    sender: str
    message: str

class RecentEntry(BaseModel):
    sender: str
    message: str
    sentiment: str  

class SuggestionRequest(BaseModel):
    recent_entry: RecentEntry
    context_history: List[ChatEntry]

@app.post("/suggest-reply/")
async def suggest_reply(request: SuggestionRequest):
    # Convert to standard Python dicts
    recent_dict: Dict = request.recent_entry.model_dump()
    context_list: List[Dict] = [entry.model_dump() for entry in request.context_history]

    # Call LLM
    suggestions = get_llm_rag_response(recent_dict, context_list) 

    return {
        "suggested_replies": suggestions,
        "recent_message": recent_dict
    }
