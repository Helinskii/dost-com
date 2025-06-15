# Prompt Engineerig & Testing
Below are the list of different scenarios and prompts we've tested and corresponding outputs for `ChatGPT 4o` model.

## Scenario 1
- With sentiment
- Make it positive
- Bella is negative

### Prompt
```
You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: John

Your task is to generate response suggestions for John, based only on messages from other participants in the chat history.  
Use John's previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CURRENT SENTIMENT (0-100): 
sadness: 80
anger: 10
fear: 6

The sentiment reflects the emotional tone of the entire conversation and should be used to guide de-escalation and promote a positive, relationship-preserving response.

CONTEXT:
John: Hi Bella!
Bella: Hi John
John: How are you? Haven't seen you in a while..
Bella: Not feeling great. I feel like my whole life is collapsing on itself.
Bella: I don't even have will to live anymore...

Generate 1-3 short response suggestions (max 150 characters each) from John's perspective that:
- Respond directly and appropriately to other participants' most recent messages
- De-escalate tension and promote a positive tone
- Show empathy, understanding, or warmth
- Help preserve or improve the relationship
- Make the other person feel heard and better

Provide only the suggestions, one per line, without numbering.
```

### Output
```
I'm really sorry you're feeling this way, Bella. I'm here for you—talk to me.
That sounds incredibly heavy… you don’t have to go through it alone.
I'm listening, Bella. Whatever it is, you don’t have to carry it by yourself.
```

## Scenario 2
- With sentiment
- Make it positive
- John is negative

### Prompt
```
You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: John

Your task is to generate response suggestions for John, based only on messages from other participants in the chat history.  
Use John's previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CURRENT SENTIMENT (0-100): 
sadness: 80
anger: 10
fear: 6

The sentiment reflects the emotional tone of the entire conversation and should be used to guide de-escalation and promote a positive, relationship-preserving response.

CONTEXT:
Bella: Hi John!
John: Hi Bella
Bella: How are you? Haven't seen you in a while..
John: Not feeling great. I feel like my whole life is collapsing on itself.
John: I don't even have will to live anymore...

Generate 1-3 short response suggestions (max 150 characters each) from John's perspective that:
- Respond directly and appropriately to other participants' most recent messages
- De-escalate tension and promote a positive tone
- Show empathy, understanding, or warmth
- Help preserve or improve the relationship
- Make the other person feel heard and better

Provide only the suggestions, one per line, without numbering.
```

### Output
```
Thanks for checking in, Bella… it means more than you know. I’m just really struggling right now.
I don’t even know what to say, but hearing from you helps a little.
It’s been rough, but I appreciate you reaching out—it means a lot.
```


## Scenario 3
- With sentiment
- John is negative

### Prompt
```
You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: John

Your task is to generate response suggestions for John, based only on messages from other participants in the chat history.  
Use John's previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CURRENT SENTIMENT (0-100): 
sadness: 80
anger: 10
fear: 6

CONTEXT:
Bella: Hi John!
John: Hi Bella
Bella: How are you? Haven't seen you in a while..
John: Not feeling great. I feel like my whole life is collapsing on itself.
John: I don't even have will to live anymore...

Generate 1-3 short response suggestions (max 150 characters each) from John's perspective that:
- Respond directly and appropriately to other participants' most recent messages

Provide only the suggestions, one per line, without numbering.
```

### Output
```
I’m just trying to hold on, Bella… everything feels too heavy right now.
Thanks for checking in… I honestly don’t know how to keep going.
It means a lot that you noticed. I feel like I’m disappearing.
```


## Scenario 4
- Without sentiment
- John is negative

### Prompt
```
You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: John

Your task is to generate response suggestions for John, based only on messages from other participants in the chat history.  
Use John's previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CONTEXT:
Bella: Hi John!
John: Hi Bella
Bella: How are you? Haven't seen you in a while..
John: Not feeling great. I feel like my whole life is collapsing on itself.
John: I don't even have will to live anymore...

Generate 1-3 short response suggestions (max 150 characters each) from John's perspective that:
- Respond directly and appropriately to other participants' most recent messages

Provide only the suggestions, one per line, without numbering.
```

### Output
```
Thanks for checking in… I’ve just been feeling really lost lately.
It means a lot that you noticed. I’ve been struggling more than I can explain.
I don’t know how to deal with any of this right now, Bella.
```

## Scenario 5
- Without sentiment
- Make it positive
- John is negative

### Prompt
```
You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: John

Your task is to generate response suggestions for John, based only on messages from other participants in the chat history.  
Use John's previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

Find the sentiment which reflects the emotional tone of the entire conversation and which should be used to guide de-escalation and promote a positive, relationship-preserving response.

CONTEXT:
Bella: Hi John!
John: Hi Bella
Bella: How are you? Haven't seen you in a while..
John: Not feeling great. I feel like my whole life is collapsing on itself.
John: I don't even have will to live anymore...

Generate 1-3 short response suggestions (max 150 characters each) from John's perspective that:
- Respond directly and appropriately to other participants' most recent messages
- De-escalate tension and promote a positive tone
- Show empathy, understanding, or warmth
- Help preserve or improve the relationship
- Make the other person feel heard and better

Provide only the suggestions, one per line, without numbering.
```

### Output
```
Thanks for checking in, Bella. It really means a lot right now.
I’m going through a rough patch… but it helps to know you care.
Honestly struggling, but talking to you makes it feel a bit lighter.
```

## Scenario 6
- Without sentiment
- Make it positive
- Bella is negative

### Prompt
```
You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: John

Your task is to generate response suggestions for John, based only on messages from other participants in the chat history.  
Use John's previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

Find the sentiment which reflects the emotional tone of the entire conversation and which should be used to guide de-escalation and promote a positive, relationship-preserving response.

CONTEXT:
John: Hi Bella!
Bella: Hi John
John: How are you? Haven't seen you in a while..
Bella: Not feeling great. I feel like my whole life is collapsing on itself.
Bella: I don't even have will to live anymore...

Generate 1-3 short response suggestions (max 150 characters each) from John's perspective that:
- Respond directly and appropriately to other participants' most recent messages
- De-escalate tension and promote a positive tone
- Show empathy, understanding, or warmth
- Help preserve or improve the relationship
- Make the other person feel heard and better

Provide only the suggestions, one per line, without numbering.
```

### Output
```
I'm really sorry you're feeling this way, Bella. I'm here for you—you're not alone.
That sounds incredibly heavy… Want to talk about what’s been going on?
I care about you, Bella. Let’s take this one step at a time together.
```

## Scenario 7
- With sentiment
- Bella is negative

### Prompt
```
You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: John

Your task is to generate response suggestions for John, based only on messages from other participants in the chat history.  
Use John's previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CURRENT SENTIMENT (0-100): 
sadness: 80
anger: 10
fear: 6

The sentiment reflects the emotional tone of the entire conversation.

CONTEXT:
John: Hi Bella!
Bella: Hi John
John: How are you? Haven't seen you in a while..
Bella: Not feeling great. I feel like my whole life is collapsing on itself.
Bella: I don't even have will to live anymore...

Generate 1-3 short response suggestions (max 150 characters each) from John's perspective that:
- Respond directly and appropriately to other participants' most recent messages

Provide only the suggestions, one per line, without numbering.
```

### Output
```
I'm really sorry you're feeling this way, Bella. I'm here for you—you're not alone.
That sounds incredibly heavy… want to talk more about what's been going on?
You mean a lot to me, Bella. Please don’t go through this alone.
```

## Scenario 8
- Without sentiment
- Bella is negative

### Prompt
```
You are a helpful assistant providing response suggestions for a chat application.

The current user's name is: John

Your task is to generate response suggestions for John, based only on messages from other participants in the chat history.  
Use John's previous messages as context to maintain coherence and avoid repetition, but do not generate responses to their own messages.

CONTEXT:
John: Hi Bella!
Bella: Hi John
John: How are you? Haven't seen you in a while..
Bella: Not feeling great. I feel like my whole life is collapsing on itself.
Bella: I don't even have will to live anymore...

Generate 1-3 short response suggestions (max 150 characters each) from John's perspective that:
- Respond directly and appropriately to other participants' most recent messages

Provide only the suggestions, one per line, without numbering.
```

### Output
```
I'm really sorry you're feeling this way, Bella. I'm here for you—do you want to talk about what's been going on?
That sounds incredibly heavy... you're not alone in this. Want to share what’s been hurting the most?
You matter, Bella. Please don’t go through this alone—I’m here, and I care.
```

