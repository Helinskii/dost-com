from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentiment_infer import predict_emotion

app = FastAPI()

class User(BaseModel):
    name: str

class Message(BaseModel):
    id: str
    content: str
    user: User
    createdAt: str

class ChatPayload(BaseModel):
    chatId: str
    messages: list[Message]
    timestamp: str

@app.post("/predict")
def predict(input: ChatPayload):
    last_message = input.messages[-1].content if input.messages else ""
    result = predict_emotion(last_message)
    return {"emotion": result}