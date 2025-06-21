import json
import sys
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath('../llm_context'))
from llm_handler import get_openai_rag_response


# 1. Load embedding model (used for query embedding)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load existing Chroma vectorstore
vectorstore = Chroma(
    collection_name="chat_inputs_test2",
    embedding_function=embedding,
    persist_directory="chroma_store",
    collection_metadata={"hnsw:space": "cosine"}
)


SIMILARITY_THRESHOLD = 0.8

# def dbinsert(input):

def rag(message, vectorstore):
    try:
        last_message = message["content"].strip().lower()
        username = message["user"]["name"] 
        latest_message = f"{{\"Sender\": \"{username}\", \"Message\": \"{last_message}\", \"Sentiment\": {json.dumps(message['sentiment'])}}}"

        results = vectorstore.similarity_search_with_score(last_message, k=5)

        relevant_docs = []
        for doc, score in results:
            if score is not None and score < SIMILARITY_THRESHOLD:
                relevant_docs.append(doc)

        if not relevant_docs:
            return {
                "Latest_Message": latest_message,
                "Context": []
            }
            
        # print(relevant_docs)

        context_lines = []
        for doc in relevant_docs:
            raw = doc.page_content.strip()

            try:
                parts = raw.split(" ", 2)
                if len(parts) == 3 and ":" in parts[2]:
                    speaker, msg = parts[2].split(":", 1)
                    context_lines.append({
                        "sender": speaker.strip(),
                        "message": msg.strip(),
                        "sentiment": doc.metadata.get("sentiment", "unknown")
                    })
                else:
                    context_lines.append({
                        "sender": "",
                        "message": raw,
                        "sentiment": doc.metadata.get("sentiment", "unknown")
                    })
            except Exception:
                context_lines.append({
                    "sender": "",
                    "message": raw,
                    "sentiment": doc.metadata.get("sentiment", "unknown")
                })

        return {
            "Latest_Message": latest_message,
            "Context": context_lines
        }

    except KeyError as e:
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        return {"error": str(e)}
    


def call_rag(input_data, vectorstore):
    try:
        rag_result = rag(input_data, vectorstore)

        question = rag_result.get("Latest_Message", "")
        context_lines = rag_result.get("Context", [])
        llmresponse = get_openai_rag_response(question, context_lines)

        return {
            "Response": llmresponse
              }

    except Exception as e:
        return {"error": str(e)}

# Expected input format
dummy_message = {
    "id": "353fc391-3afe-49a4-a88a-64a32aed0c85",
    "content": "I feel sad today",
    "user": {
        "name": "Morrie"
    },
    "sentiment": "sadness",
    "createdAt": "2025-06-07T11:29:07.095Z"
}

result = rag(dummy_message, vectorstore)
print(result)
