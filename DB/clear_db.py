from fastapi import FastAPI
from fastapi.responses import JSONResponse
import chromadb

app = FastAPI()

chroma_client = chromadb.PersistentClient(path="chroma_store")

@app.post("/clear-db/")
async def clear_database():
    try:
        chroma_client.delete_collection("chat_messages")
        return JSONResponse(status_code=200, content={"message": "Chroma DB 'chat_messages' collection cleared."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})