# dost-com
Text Sentiment Analysis Chat Application - for DA225o Deep Learning Course at IISc

## Project Overview
> Today's chat applications are built around the same building blocks used when they were emerging -- one-on-one chats, groups, multiple modalities (text, audio, video, emojis, etc.), and any new chat application has pretty much the same functionality.
>
> On the other hand, more and more people are communicating over text than any other form of communication. But text has one problem - the tone (or body) of the conversation is not always apparent, and may require further clarification or eventually lead to misunderstandings while communicating. This happens so often that multiple modalities are then equipped in order to clarify, elaborate or communicate the same *sentiment*.
>
> We have built an application to enhance our texting experience, and add another dimension to communicating through this medium. We call it **dost-com**, a chat application that does the following:
>
> 1. Understand the ongoing conversation in a text session in real-time
> 2. Provide live analysis on the *sentiment* of the conversation or recent text
> 3. Based on the sentiment understanding and context of the conversation, provide suggestions/prompts to the user to respond appropriately, or tailor it with a baseline

### Project Details
Potential Dataset - [Kaggle Data Set](https://www.kaggle.com/datasets/parulpandey/emotion-dataset?select=training.csv)

### Task Distribution
- *Web GUI*: Ishwer
- *Data Preparation*: Manish, Shubham
- *Transformer*: Deepshikhar, Rishabh
- *Sentiment Analytics*: Deepshikhar
- *NLP*: Shubham
- *LLM API*: Rishabh, Sanket
- *Database*: Yuvasree
- *Project Report*: Sanket + **ALL**

## Potential Scope
- *Add RAG based context retrieval for entire chat in order to reduce LLM API payload size (tokens)*


# Information for Reference

## Web-UI API Payload Format
```
{
  "chatId": "general",
      "messages": [
          {
              "id": "51d179c4-1092-42e8-aed9-4cbb581a3106",
              "content": "Hi",
              "user": {
                  "name": "god"
              },
              "createdAt": "2025-06-14T12:17:35.825Z"
          }
      ],
      "timestamp": "2025-06-14T12:17:36.842Z"
}
```
## RAG Vector Input Format
```
{
    "id": "353fc390-3afe-49a4-a88a-64a32aed0c85",
    "content": "What's going on?",
    "user": {
        "name": "test"
    },
    "sentiment": "joy",
    "createdAt": "2025-06-07T11:29:07.095Z"
}
```

## Sentiment Analytics Format Expectation
```
{
  "chatId": "general",
  "messages": [
    {
      "id": "353fc390-3afe-49a4-a88a-64a32aed0c85",
      "content": "What's going on?",
      "user": {
        "name": "test"
      },
      "sentiment": <Emotion>,
      "createdAt": "2025-06-07T11:29:07.095Z"
    }
  ],
  "timestamp": "2025-06-07T11:29:18.624Z"
}
```

## Vercel Web-UI

# Link to Application -> [Dost-Com](https://dostcom.vercel.app/)

##### Prototype-1 of Web GUI
![Chat application with sentiment analysis](/images/prototype-1.png)

# Open in v0

*Automatically synced with your [v0.dev](https://v0.dev) deployments*

[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com/robogods-projects/v0-open-in-v0-im)
[![Built with v0](https://img.shields.io/badge/Built%20with-v0.dev-black?style=for-the-badge)](https://v0.dev/chat/projects/eywN2GNRyVq)

## Overview

This repository will stay in sync with your deployed chats on [v0.dev](https://v0.dev).
Any changes you make to your deployed app will be automatically pushed to this repository from [v0.dev](https://v0.dev).

## Deployment

Your project is live at:

**[https://vercel.com/robogods-projects/v0-open-in-v0-im](https://vercel.com/robogods-projects/v0-open-in-v0-im)**

## Build your app

Continue building your app on:

**[https://v0.dev/chat/projects/eywN2GNRyVq](https://v0.dev/chat/projects/eywN2GNRyVq)**

## How It Works

1. Create and modify your project using [v0.dev](https://v0.dev)
2. Deploy your chats from the v0 interface
3. Changes are automatically pushed to this repository
4. Vercel deploys the latest version from this repository
