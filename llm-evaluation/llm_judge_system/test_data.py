import random
from typing import List
from .models import ChatMessage, SentimentProbabilities, ChatContext

class TestDataGenerator:
    @staticmethod
    def generate_test_contexts(num_contexts: int = 20) -> List[ChatContext]:
        scenarios = [
            {
                "type": "customer_support_frustrated",
                "messages": [
                    ChatMessage("1", "I've been trying to reset my password for 30 minutes!", "Alex", "2024-01-15 10:00:00"),
                    ChatMessage("2", "I apologize for the frustration. Let me help you with that right away.", "Support", "2024-01-15 10:01:00"),
                    ChatMessage("3", "I've tried the reset link 5 times and it's not working", "Alex", "2024-01-15 10:02:00")
                ],
                "current_user": "Support",
                "sentiment": SentimentProbabilities(sadness=0.1, joy=0.0, love=0.0, anger=0.7, fear=0.1, unknown=0.1)
            },
            {
                "type": "sales_inquiry_interested",
                "messages": [
                    ChatMessage("1", "Hi, I'm looking for a project management tool for my team", "Sarah", "2024-01-15 11:00:00"),
                    ChatMessage("2", "Great! I'd be happy to help you find the perfect solution. How large is your team?", "Sales", "2024-01-15 11:01:00"),
                    ChatMessage("3", "We're about 25 people, mostly remote", "Sarah", "2024-01-15 11:02:00")
                ],
                "current_user": "Sales",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.3, love=0.0, anger=0.0, fear=0.0, unknown=0.7)
            },
            {
                "type": "technical_support_confused",
                "messages": [
                    ChatMessage("1", "My app keeps crashing when I try to upload files", "Mike", "2024-01-15 12:00:00"),
                    ChatMessage("2", "I'm sorry to hear that. What type of files are you trying to upload?", "Tech", "2024-01-15 12:01:00"),
                    ChatMessage("3", "Just regular PDFs, nothing special", "Mike", "2024-01-15 12:02:00")
                ],
                "current_user": "Tech",
                "sentiment": SentimentProbabilities(sadness=0.2, joy=0.0, love=0.0, anger=0.1, fear=0.3, unknown=0.4)
            },
            {
                "type": "positive_feedback_happy",
                "messages": [
                    ChatMessage("1", "Just wanted to say your service has been amazing!", "Emma", "2024-01-15 13:00:00"),
                    ChatMessage("2", "Thank you so much! That really makes our day. What specifically has been helpful?", "Support", "2024-01-15 13:01:00"),
                    ChatMessage("3", "The customer support team solved my issue in minutes", "Emma", "2024-01-15 13:02:00")
                ],
                "current_user": "Support",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.8, love=0.1, anger=0.0, fear=0.0, unknown=0.1)
            },
            {
                "type": "relationship_conflict",
                "messages": [
                    ChatMessage("1", "You never listen to what I'm saying", "Jordan", "2024-01-15 14:00:00"),
                    ChatMessage("2", "That's not fair, I do listen but you keep interrupting me", "Casey", "2024-01-15 14:01:00"),
                    ChatMessage("3", "See, you're doing it again, making it about you", "Jordan", "2024-01-15 14:02:00")
                ],
                "current_user": "Casey",
                "sentiment": SentimentProbabilities(sadness=0.3, joy=0.0, love=0.0, anger=0.5, fear=0.1, unknown=0.1)
            }
        ]
        test_contexts = []
        for i in range(num_contexts):
            scenario = scenarios[i % len(scenarios)]
            sentiment = scenario["sentiment"]
            if i % 3 == 0:
                noise = 0.1
                sentiment = SentimentProbabilities(
                    sadness=max(0, min(1, sentiment.sadness + random.uniform(-noise, noise))),
                    joy=max(0, min(1, sentiment.joy + random.uniform(-noise, noise))),
                    love=max(0, min(1, sentiment.love + random.uniform(-noise, noise))),
                    anger=max(0, min(1, sentiment.anger + random.uniform(-noise, noise))),
                    fear=max(0, min(1, sentiment.fear + random.uniform(-noise, noise))),
                    unknown=max(0, min(1, sentiment.unknown + random.uniform(-noise, noise)))
                )
            context = ChatContext(
                chat_history=scenario["messages"].copy(),
                current_user=scenario["current_user"],
                sentiment_probabilities=sentiment
            )
            test_contexts.append(context)
        return test_contexts
