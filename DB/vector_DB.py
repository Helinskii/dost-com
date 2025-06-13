import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize FastAPI app
app = FastAPI()

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection(name="chat_messages")

# Preprocess a single entry
def preprocess_entry(entry):
    raw_datetime = entry['datetime']
    dt = datetime.fromisoformat(raw_datetime)
    formatted_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
    combined_text = f"{formatted_dt} {entry['username']}: {entry['chat_message']}"
    return combined_text

# Process uploaded JSON
@app.post("/upload-json/")
async def upload_json(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = json.loads(content)

        documents = []
        ids = []

        for i, entry in enumerate(data):
            combined_text = preprocess_entry(entry)
            embedding = model.encode(combined_text).tolist()

            documents.append(combined_text)
            ids.append(str(i))

            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[combined_text]
            )

        return JSONResponse(status_code=200, content={"message": f"Stored {len(documents)} entries into Chroma DB."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
