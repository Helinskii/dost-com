import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection(name="chat_messages", metadata={"hnsw:space": "cosine"})


def preprocess_message(message, dominant_emotion):
    content = message["content"]
    created_at = message["createdAt"]
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))  # Handle ISO timestamp with Zulu time
    formatted_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
    username = message["user"]["name"]
    combined_text = f"{formatted_dt} {username}: {content}"
    return combined_text, dominant_emotion


@app.post("/upload-json/")
async def upload_json(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]

        documents = []
        ids = []
        metadatas = []

        idx = 0
        for entry in data:
            dominant_emotion = entry.get("sentiment", {}).get("dominantEmotion", "unknown")
            messages = entry.get("chatHistory", {}).get("messages", [])

            for message in messages:
                combined_text, emotion = preprocess_message(message, dominant_emotion)
                embedding = model.encode(combined_text).tolist()

                doc_id = f"{message['id']}_{idx}"
                documents.append(combined_text)
                ids.append(doc_id)
                metadatas.append({"dominantEmotion": emotion})

                collection.add(
                 ids=[doc_id],
                 embeddings=[embedding],
                 documents=[combined_text],
                 metadatas=[{
                  "dominantEmotion": emotion,
                  "sender": message["user"]["name"],
                  "timestamp": message["createdAt"][:19].replace("T", " ")  
                   }]
                )

                idx += 1

        return JSONResponse(status_code=200, content={"message": f"Stored {len(documents)} messages into Chroma DB."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    