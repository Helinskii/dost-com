import sys
import os

sys.path.append(os.path.abspath('../sentiment_analytics'))
sys.path.append(os.path.abspath('../RAG'))
sys.path.append(os.path.abspath('../SentimentParaphrasing'))

from sentiment_analysis import sentiment_analytics
from sentiment_infer import predict_emotion
from rag import call_rag
from message_store import message_store
import chromadb
from seq2seq_infer import paraphrase

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import logging

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# File handler
file_handler = logging.FileHandler("sentiment_api.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

# Stream (console) handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend's URL(s) instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class User(BaseModel):
    name: str

class Message(BaseModel):
    id: str
    content: str
    user: User
    createdAt: str

class ChatPayload(BaseModel):
    chatId: str
    messages: list[Message]
    timestamp: str

class DB(BaseModel):
    dbName: str

@app.post("/predict")
def predict(input: ChatPayload):
    logger.info(f"/predict called with chatId={input.chatId}, num_messages={len(input.messages)}")
    last_message = input.messages[-1].content if input.messages else ""
    logger.debug(f"Last message: {last_message}")
    emotion_last_msg, emotion_scores = predict_emotion(last_message)
    # all_msg = [msg.content for msg in input.messages]
    # emotion_score, text_emotion = predict_compl_emotion(all_msg)
    logger.info(f"Predicted emotion for last message: {emotion_last_msg}")
    return {
        "emotion_last_message": emotion_last_msg,
        "emotional_scores": emotion_scores
        # "emotion_per_text": text_emotion
        }

@app.post("/paraphrase")
def convert_text(input: ChatPayload):
    logger.info(f"/paraphrase called with chatId={input.chatId}, num_messages={len(input.messages)}")
    last_message = input.messages[-1].content if input.messages else ""
    logger.debug(f"Last message: {last_message}")
    converted_text = paraphrase(last_message)
    logger.info(f"Converted Text for last message: {converted_text}")
    return {
        "paraphrased_message": converted_text,
        }

@app.post("/analyse")
def analyse(input: ChatPayload):
    logger.info(f"/analyse called with chatId={input.chatId}, num_messages={len(input.messages)}")
    last_message = input.messages[-1].content if input.messages else ""
    logger.debug(f"Last message: {last_message}")
    emotion_last_msg, emotion_scores = predict_emotion(last_message)
    analysis_arg = input.model_dump()
    analysis_arg['messages'][0]['sentiment'] = emotion_scores
    user_emo_dist = sentiment_analytics(analysis_arg, [10])
    user_emo_dist = convert_np(user_emo_dist)
    logger.info(f"User emotion distribution calculated.")
    return user_emo_dist

@app.post("/rag")
def respond(input: ChatPayload):
    logger.info(f"/rag called with chatId={input.chatId}, num_messages={len(input.messages)}")
    collection_name = "chat_inputs"

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory="chroma_store",
        collection_metadata={"hnsw:space": "cosine"}
    )
    last_message = input.messages[-1].content if input.messages else ""
    logger.debug(f"Last message: {last_message}")
    emotion_last_msg, emotion_scores = predict_emotion(last_message)
    rag_arg = input.model_dump()
    rag_arg = rag_arg['messages'][0]
    rag_arg['sentiment'] = emotion_last_msg
    logger.debug(f"Calling RAG with arg: {rag_arg}")
    rag_response = call_rag(rag_arg, vectorstore)
    logger.info(f"RAG response: {rag_response}")
    
    message_store(rag_arg)

    return rag_response

@app.post("/clear_db")
async def clear_database(db: DB):
    try:
        logger.info(f"/clear_db called for dbName={db.dbName}")
        collection_name = db.dbName
        chroma_client = chromadb.PersistentClient(path="chroma_store")
        chroma_client.delete_collection(collection_name)
        logger.info(f"Chroma DB '{collection_name}' collection cleared.")
        os.remove("user_dist.pkl")
        logger.info(f"Removed sentiment analysis file: user_dist.pkl")
        return JSONResponse(status_code=200, content={"message": f"Chroma DB '{collection_name}' collection cleared."})
    except Exception as e:
        logger.error(f"Error clearing DB: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    

## UTILS
import numpy as np

# Convert NP arrays to python objects to be able to send
# as an API response
# Need to also handle NaNs & Infs in the analytics
def convert_np(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return [convert_np(i) for i in obj.tolist()]
    elif isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    else:
        return obj