import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection(name="chat_inputs", metadata={"hnsw:space": "cosine"})

# Message Example:
'''
{
  "chatId": "general",
  "messages": [
    {
      "id": "353fc390-3afe-49a4-a88a-64a32aed0c85",
      "content": "What's going on?",
      "user": {
        "name": "test"
      },
      "sentiment": <Emotion>,
      "createdAt": "2025-06-07T11:29:07.095Z"
    }
  ],
  "timestamp": "2025-06-07T11:29:18.624Z"
}
'''

def preprocess_message(message):
    content = message["content"]
    created_at = message["createdAt"]
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))  # Handle ISO timestamp with Zulu time
    formatted_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
    username = message["user"]["name"]
    combined_text = f"{formatted_dt} {username}: {content}"
    return combined_text

def message_store(message):
    emotion = message["sentiment"]
    combined_text = preprocess_message(message)
    print(f"Text Stored: {combined_text}\tEmotion: {emotion}")
    embedding = model.encode(combined_text).tolist()
    doc_id = message["id"]
    metadata = {
        "id": message["id"],
        "user_name": message["user"]["name"],
        "timestamp": message["createdAt"][:19].replace("T", " "),
        "sentiment": emotion
    }
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[combined_text],
        metadatas=[metadata]
    )

# dummy_message = {
#     "id": "353fc391-3afe-49a4-a88a-64a32aed0c85",
#     "content": "What's going on?",
#     "user": {
#         "name": "test"
#     },
#     "sentiment": "joy",
#     "createdAt": "2025-06-07T11:29:07.095Z"
# }

# message_store(dummy_message)

# Store example conversation in JSON format in DB to check RAG
# json_path = "chat_jsonformat.json"
# with open(json_path, "r") as f:
#     data = json.load(f)

# for idx, entry in enumerate(data):
#     message = entry["chatHistory"]["messages"][0]
#     sentiment = entry["sentiment"]["overallSentiment"]["emotion_last_message"]
#     message["sentiment"] = sentiment
#     ## DEBUG
#     print(message)
#     message_store(message)
