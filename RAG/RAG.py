import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")

from llm_handler import get_openai_rag_response


# 1. Load embedding model (used for query embedding)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load existing Chroma vectorstore
vectorstore = Chroma(
    collection_name="chat_messages",
    embedding_function=embedding,
    persist_directory="chroma_store",
    collection_metadata={"hnsw:space": "cosine"}
)


SIMILARITY_THRESHOLD = 0.8

# def dbinsert(input):

def rag(input, vectorstore):
    try:
        # Extract message and metadata
        message = input["chatHistory"]["messages"][0]
        question = message["content"].strip().lower()
        input_message = message["content"].strip()
        username = message["user"]["name"] 
        sentiments = {
            "sentiments": input["sentiment"]["overallSentiment"]["emotional_scores"]
        }
        sentiments_str = json.dumps(sentiments)
        latest_message = f"{{\"Sender\": \"{username}\", \"Message\": \"{question}\", {sentiments_str}}}"


        # Retrieve similar documents
        results = vectorstore.similarity_search_with_score(question, k=5)

        relevant_docs = []
        for doc, score in results:
            if score is not None and score < SIMILARITY_THRESHOLD:
                relevant_docs.append(doc)

        if not relevant_docs:
            return {
                "Latest_Message": latest_message,
                "Context": []
            }

        # Format context lines
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
                        "sentiment": doc.metadata.get("dominantEmotion", "unknown")
                    })
                else:
                    context_lines.append({
                        "sender": "",
                        "message": raw,
                        "sentiment": doc.metadata.get("dominantEmotion", "unknown")
                    })
            except Exception:
                context_lines.append({
                    "sender": "",
                    "message": raw,
                    "sentiment": doc.metadata.get("dominantEmotion", "unknown")
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
        sentiments = json.dumps(input_data["sentiment"]["overallSentiment"]["emotional_scores"])
        llmresponse = get_openai_rag_response(question, sentiments, context_lines)

        return {
            "Response": llmresponse
              }

    except Exception as e:
        return {"error": str(e)}

#Expected input format
# input =  {
#     "username": "test",
#     "chatHistory": {
#         "chatId": "",
#         "messages": [
#             {
#                 "id": "2d6fc6c6-b0e1-4c52-a101-28cf5909cb14",
#                 "content": "How are you?",
#                 "user": {
#                     "name": "test"
#                 },
#                 "createdAt": "2025-06-13T15:16:04.980Z"
#             }
#         ],
#         "timestamp": "2025-06-13T15:16:06.420Z"
#     },
#     "sentiment": {
#         "overallSentiment": {
#             "emotion_last_message": "joy",
#             "emotional_scores": {
#                 "sadness": 0,
#                 "joy": 1,
#                 "love": 0,
#                 "anger": 0,
#                 "fear": 0,
#                 "unknown": 0
#             }
#         },
#         "dominantEmotion": "joy",
#         "trend": "stable",
#         "isPositive": True,
#         "messageSentiments": []
#     },
#     "timestamp": "2025-06-13T15:16:06.421Z"
# }


# result = rag(input, vectorstore)
# print(result)
