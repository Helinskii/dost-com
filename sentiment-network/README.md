# Shubham's TODO

- [X] Integrate Sentiment Analysis as an API
- [X] Integrate single API call for last message emotion + Sentiment analysis
- [X] Integrate chat message vector store with RAG
- [X] Add API for RAG calls providing LLM responses as final output
- [X] Add logging to sentiment_api
- [X] Explore sequence modeling of predicted emotions based on past messages using BiLSTM

## Predictive Sentiment Analysis

Example conversation:

1. "It's just been one of those days."
2. "Nothing seems to be working."
3. "I give up."

"messages": [
    "I had the strangest dream last night.",
    "Are you free this weekend?",
    "That's cool. What happened next?",
    "My schedule\u2019s been hectic lately.",
    "Did you see the message he sent?",
    "They crossed the line this time!"
],


Latest emotion may be classified as Neutral without any prior context, but a hiearchical sentiment model will see that it's 'Sad'.
Such *Emotional Trajectories* can be helpful in alerting other users and biasing LLM prompts

### Some information for the paper

1. LSTM References
    a. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    b. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Training Data

Train Accuracy: 0.7264 | Precision: 0.7639 | Recall: 0.7256 | F1: 0.7109
Validation accuracy: 0.9194 (1655/1800)

### Longer Abstract
Today's chat applications are built around the same building blocks which were used when they were emerging - one-on-one chats, groups, multiple modalities (text, audio, video, emojis, etc.), and any new chat application has pretty much the same functionality.
On the other hand, more and more people are communicating over text than any other form of communication. But text has one problem - the tone (or body) of the conversation is not always apparent, and may require further clarification. It may eventually even lead to misunderstandings while communicating. This happens so often that multiple modalities are then equipped in order to clarify, elaborate or communicate the same sentiment.
This paper presents an application to enhance our texting experience, and add another dimension to communication through this medium. The chat application presented in this paper does the following:
\begin{itemize}
  \item Understanding conversation sentiment in real-time
  \item Provide live analysis about the sentiment of parts of the conversation or last (immediate) text
  \item Based on the sentiment understanding and context of the conversation, provide suggestions/prompts to the user to respond appropriately, or tailor it before sending
\end{itemize}
This application aims to solidify communication through text seamlessly by improving user understanding and judgment, while retaining the existing methods.