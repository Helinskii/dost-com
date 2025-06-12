from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentiment_infer import predict_emotion, predict_compl_emotion

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
    emotion_last_msg = predict_emotion(last_message)
    all_msg = [msg.content for msg in input.messages]
    emotion_score, text_emotion = predict_compl_emotion(all_msg)
    return {
        "emotion_last_message": emotion_last_msg,
        "emotional_scores": emotion_score,
        "emotion_per_text": text_emotion
        }