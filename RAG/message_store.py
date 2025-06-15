import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection(name="chat_messages", metadata={"hnsw:space": "cosine"})

# Message:
'''
{
    "id": "51d179c4-1092-42e8-aed9-4cbb581a3106",
    "content": "Hi",
    "user_name": "RoboGod",
    "createdAt": "2025-06-14T12:17:35.825Z"
}
'''

def message_store(message):
    # message is a dict with keys: id, content, user_name, createdAt
    embedding = model.encode(message["content"]).tolist()
    doc_id = message["id"]
    metadata = {
        "id": message["id"],
        "user_name": message["user_name"],
        "timestamp": message["createdAt"][:19].replace("T", " ")
    }
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[message["content"]],
        metadatas=[metadata]
    )

dummy_message = {
    "id": "51d179c4-1092-42e8-aed9-4cbb581a3106",
    "content": "Hi",
    "user_name": "RoboGod",
    "createdAt": "2025-06-14T12:17:35.825Z"
}

message_store(dummy_message)