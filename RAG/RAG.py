
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.chat_models import ChatOllama
from fastapi import FastAPI, Request
from pydantic import BaseModel


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = Chroma(
    collection_name="chat_messages",
    embedding_function=embedding,
    persist_directory="chroma_store"  
)

retriever = vectorstore.as_retriever()

# Prompt
prompt = hub.pull("rlm/rag-prompt")


# llm = ChatOllama(model="llama3", temperature=0)
# llm = ChatOllama(model="llama3:8b", temperature=0)
llm = ChatOllama(model="llama2:7b", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | RunnablePassthrough()
)

# Question
# response = rag_chain.invoke("Can we review the report together?")
# print(response)

retrieved_docs = retriever.invoke("Can we review the report together?")
formatted_context = format_docs(retrieved_docs)

# Create final prompt
inputs = {"context": formatted_context, "question": "Can we review the report together?"}
final_prompt = prompt.format(**inputs)

# Print the final prompt that goes to the LLM
print("\n--- FINAL PROMPT TO LLM ---\n")
print(final_prompt)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    response = rag_chain.invoke(query.question)
    return {"response": response}
