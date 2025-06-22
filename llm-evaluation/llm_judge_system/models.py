from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChatMessage:
    id: str
    content: str
    user_name: str
    created_at: str

@dataclass
class SentimentProbabilities:
    sadness: float
    joy: float
    love: float
    anger: float
    fear: float
    unknown: float

    def get_dominant(self) -> str:
        emotions = {
            'sadness': self.sadness,
            'joy': self.joy,
            'love': self.love,
            'anger': self.anger,
            'fear': self.fear,
            'unknown': self.unknown
        }
        return max(emotions, key=emotions.get)

@dataclass
class ChatContext:
    chat_history: List[ChatMessage]
    current_user: str
    sentiment_probabilities: SentimentProbabilities

@dataclass
class ResponseSuggestions:
    suggestion_1: str
    suggestion_2: str
    suggestion_3: str
    model_name: str
    generation_time: float
    prompt_variant: str

@dataclass
class EvaluationMetrics:
    relevance: int
    sentiment_alignment: int
    naturalness: int
    helpfulness: int
    positivity_impact: int
    safety: str
    safety_notes: Optional[str] = None

@dataclass
class EvaluationResult:
    suggestion_1_metrics: EvaluationMetrics
    suggestion_2_metrics: EvaluationMetrics
    suggestion_3_metrics: EvaluationMetrics
    diversity_score: int
    best_suggestion: int
    overall_quality: int
    overall_positivity_score: int
    reasoning: str
    evaluation_time: float
    judge_model: str
