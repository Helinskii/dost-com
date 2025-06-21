import json
import sys
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")

llm_context_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../llm_context'))
if llm_context_path not in sys.path:
    sys.path.insert(0, llm_context_path)
try:
    from llm_handler import get_openai_rag_response # type: ignore
except ImportError as e:
    raise ImportError(f"Could not import 'get_openai_rag_response' from 'llm_handler.py'. Make sure 'llm_handler.py' exists in '{llm_context_path}'. Original error: {e}")


# 1. Load embedding model (used for query embedding)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load existing Chroma vectorstore
vectorstore = Chroma(
    collection_name="chat_inputs_test3",
    embedding_function=embedding,
    persist_directory="chroma_store",
    collection_metadata={"hnsw:space": "cosine"}
)

# Hyperparameter
SIMILARITY_THRESHOLD = 0.8

# def dbinsert(input):

def rag(message, vectorstore):
    try:
        last_message = message["content"].strip().lower()
        username = message["user"]["name"]
        # latest_message = f"{{\"Sender\": \"{username}\", \"Message\": \"{last_message}\", \"Sentiment\": {json.dumps(message['sentiment'])}}}"
        latest_message = {
            "Sender": username,
            "Message": last_message,
            "Sentiment": message["sentiment"]
        }

        results = vectorstore.similarity_search_with_score(last_message, k=5)
        
        ## DEBUG
        # print(f"Vector Retreived: {results}")

        relevant_docs = []
        for doc, score in results:
            if score is not None and score < SIMILARITY_THRESHOLD:
                relevant_docs.append(doc)

        if not relevant_docs:
            return {
                "Latest_Message": latest_message,
                "Context": []
            }
            
        ## DEBUG
        # print(f"Relevant Docs list: {relevant_docs}")

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

        message = rag_result.get("Latest_Message", "")
        context_lines = rag_result.get("Context", [])

        # DEBUG
        # print(f"RAG Result: {rag_result}\tType: {type(rag_result)}")
        # print(f"Message: {message}\nContext: {context_lines}")
        # print(f"Message Type: {type(message)}\nContext Type: {type(context_lines)}")

        llmresponse = get_openai_rag_response(message, context_lines)

        return {
            "Response": llmresponse
              }

    except Exception as e:
        return {"error_1": str(e)}

# Expected input format
dummy_message = {
    "id": "353fc391-3afe-49a4-a88a-64a32aed0c85",
    "content": "I feel motivated after talking to you Morrie",
    "user": {
        "name": "Mitch"
    },
    "sentiment": "joy",
    "createdAt": "2025-06-07T11:29:07.095Z"
}

# DEBUG
# result = rag(dummy_message, vectorstore)
# print("Final RAG Result (delivered to LLM function for context)")
# print(f"Latest Message: {result['Latest_Message']}")

# i = 0
# for context in result['Context']:
#     print(f"Contex {i}: {context}")
#     i+=1

# rag_response = call_rag(dummy_message, vectorstore)
# print(rag_response)