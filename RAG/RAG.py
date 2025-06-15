import json
from fastapi import FastAPI, UploadFile
from fastapi.params import File
from fastapi.responses import JSONResponse
from datetime import datetime
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain import hub

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="chat_messages",
    embedding_function=embedding,
    persist_directory="chroma_store",
    collection_metadata={"hnsw:space": "cosine"}
)

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOllama(model="llama2", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": vectorstore.as_retriever(search_kwargs={"k": 5}) | format_docs,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.post("/rag")
async def rag(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = json.loads(content)

        message = data["chatHistory"]["messages"][0]
        question = message["content"].strip().lower()


        results = vectorstore.similarity_search_with_score(question, k=5)
        SIMILARITY_THRESHOLD = 0.8 

        relevant_docs = []

        print("\n--- DEBUG: Retrieved documents and scores ---")
        for doc, score in results:
            print(f"Score: {score:.4f} | Content: {doc.page_content.strip()}")
            if score is not None and score < SIMILARITY_THRESHOLD:
                relevant_docs.append(doc)

        if not relevant_docs:
            return {
                "Question": question,
                "Context": [],
                "Answer": "unknown"
            }

        context_lines = []
        for doc in relevant_docs:
            msg_content = doc.page_content.strip()
            emotion = doc.metadata.get("dominantEmotion", "unknown")
            context_line = f"{msg_content} emotions: {emotion}"
            context_lines.append(context_line)

        response = rag_chain.invoke(question)

        return {
            "Question": question,
            "Context": context_lines,
            "Answer": response
        }

    except KeyError as e:
        return JSONResponse(status_code=400, content={"error": f"Missing field: {e}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
