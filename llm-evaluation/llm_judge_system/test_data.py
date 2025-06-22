import random
from typing import List
from .models import ChatMessage, SentimentProbabilities, ChatContext

class TestDataGenerator:
    @staticmethod
    def generate_test_contexts(num_contexts: int = 30) -> List[ChatContext]:
        scenarios = [
            # 1. Argument between siblings
            {
                "type": "sibling_argument",
                "messages": [
                    ChatMessage("1", "You borrowed my car without asking AGAIN!", "Emma", "2024-01-15 10:00:00"),
                    ChatMessage("2", "I left you a note! You were sleeping and I didn't want to wake you", "Jake", "2024-01-15 10:01:00"),
                    ChatMessage("3", "A note doesn't mean you can just take it whenever you want", "Emma", "2024-01-15 10:02:00"),
                    ChatMessage("4", "I had an emergency at work, what was I supposed to do?", "Jake", "2024-01-15 10:03:00")
                ],
                "current_user": "Emma",
                "sentiment": SentimentProbabilities(sadness=0.1, joy=0.0, love=0.0, anger=0.7, fear=0.0, unknown=0.2)
            },
            # 2. Friend sharing good news
            {
                "type": "friend_good_news",
                "messages": [
                    ChatMessage("1", "OMG I GOT THE JOB!!! 🎉", "Sarah", "2024-01-15 11:00:00"),
                    ChatMessage("2", "NO WAY! That's incredible! I knew you'd get it!", "Mia", "2024-01-15 11:01:00"),
                    ChatMessage("3", "I can't believe it, I start next month! We need to celebrate!", "Sarah", "2024-01-15 11:02:00"),
                    ChatMessage("4", "Absolutely! Dinner's on me this weekend, so proud of you!", "Mia", "2024-01-15 11:03:00")
                ],
                "current_user": "Sarah",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.8, love=0.1, anger=0.0, fear=0.0, unknown=0.1)
            },
            # 3. Parent-child worry
            {
                "type": "parent_child_worry",
                "messages": [
                    ChatMessage("1", "Honey, you haven't been yourself lately. Is everything okay?", "Mom", "2024-01-15 12:00:00"),
                    ChatMessage("2", "I'm fine mom, just stressed with finals coming up", "Alex", "2024-01-15 12:01:00"),
                    ChatMessage("3", "You know you can talk to me about anything, right?", "Mom", "2024-01-15 12:02:00"),
                    ChatMessage("4", "I know... I'm just overwhelmed and don't know if I can handle it all", "Alex", "2024-01-15 12:03:00")
                ],
                "current_user": "Mom",
                "sentiment": SentimentProbabilities(sadness=0.3, joy=0.0, love=0.2, anger=0.0, fear=0.4, unknown=0.1)
            },
            # 4. Couple reconciliation
            {
                "type": "couple_reconciliation",
                "messages": [
                    ChatMessage("1", "I'm sorry about last night. I shouldn't have said those things", "Chris", "2024-01-15 13:00:00"),
                    ChatMessage("2", "I said things I didn't mean too. We both were tired and stressed", "Jordan", "2024-01-15 13:01:00"),
                    ChatMessage("3", "Can we talk about it over coffee? I miss us being okay", "Chris", "2024-01-15 13:02:00"),
                    ChatMessage("4", "Yes, let's do that. I miss us too", "Jordan", "2024-01-15 13:03:00")
                ],
                "current_user": "Chris",
                "sentiment": SentimentProbabilities(sadness=0.3, joy=0.0, love=0.4, anger=0.0, fear=0.1, unknown=0.2)
            },
            # 5. Friends planning surprise
            {
                "type": "friends_planning",
                "messages": [
                    ChatMessage("1", "Okay so for Lisa's surprise party, I've booked the venue!", "Rachel", "2024-01-15 14:00:00"),
                    ChatMessage("2", "Perfect! I'll handle the cake and decorations", "Tom", "2024-01-15 14:01:00"),
                    ChatMessage("3", "She's going to cry happy tears, I can't wait to see her face!", "Rachel", "2024-01-15 14:02:00"),
                    ChatMessage("4", "Just make sure David keeps her busy that day!", "Tom", "2024-01-15 14:03:00")
                ],
                "current_user": "Rachel",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.6, love=0.3, anger=0.0, fear=0.0, unknown=0.1)
            },
            # 6. Sibling supporting each other
            {
                "type": "sibling_support",
                "messages": [
                    ChatMessage("1", "I heard about the breakup. How are you holding up?", "Maya", "2024-01-15 15:00:00"),
                    ChatMessage("2", "Not great honestly. I thought we had something special", "Ben", "2024-01-15 15:01:00"),
                    ChatMessage("3", "I know it hurts now, but you deserve someone who sees how amazing you are", "Maya", "2024-01-15 15:02:00"),
                    ChatMessage("4", "Thanks sis. Can I come over? I don't want to be alone right now", "Ben", "2024-01-15 15:03:00")
                ],
                "current_user": "Maya",
                "sentiment": SentimentProbabilities(sadness=0.6, joy=0.0, love=0.2, anger=0.0, fear=0.1, unknown=0.1)
            },
            # 7. Friends dealing with jealousy
            {
                "type": "friend_jealousy",
                "messages": [
                    ChatMessage("1", "So I see you've been hanging out with Katie a lot lately", "Nina", "2024-01-15 16:00:00"),
                    ChatMessage("2", "Yeah, we work on the same project. Is that a problem?", "Zoe", "2024-01-15 16:01:00"),
                    ChatMessage("3", "No, I just... we used to hang out more. I feel replaced", "Nina", "2024-01-15 16:02:00"),
                    ChatMessage("4", "Nina, you could never be replaced! Let's plan something just us", "Zoe", "2024-01-15 16:03:00")
                ],
                "current_user": "Nina",
                "sentiment": SentimentProbabilities(sadness=0.4, joy=0.0, love=0.0, anger=0.2, fear=0.3, unknown=0.1)
            },
            # 8. Parent proud moment
            {
                "type": "parent_proud",
                "messages": [
                    ChatMessage("1", "Dad! I made the Dean's List!", "Sophie", "2024-01-15 17:00:00"),
                    ChatMessage("2", "That's my girl! I'm so proud of you!", "Dad", "2024-01-15 17:01:00"),
                    ChatMessage("3", "All those late study nights paid off", "Sophie", "2024-01-15 17:02:00"),
                    ChatMessage("4", "Your hard work is inspiring. Mom and I want to take you out to celebrate", "Dad", "2024-01-15 17:03:00")
                ],
                "current_user": "Dad",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.7, love=0.2, anger=0.0, fear=0.0, unknown=0.1)
            },
            # 9. Friends in conflict over money
            {
                "type": "friend_money_conflict",
                "messages": [
                    ChatMessage("1", "Hey, about that $200 I lent you last month...", "Marcus", "2024-01-15 18:00:00"),
                    ChatMessage("2", "I know, I'm sorry. I get paid Friday, I promise I'll pay you back", "Luis", "2024-01-15 18:01:00"),
                    ChatMessage("3", "You said that last week too. I have bills to pay", "Marcus", "2024-01-15 18:02:00"),
                    ChatMessage("4", "I'm really struggling right now. Can I at least give you half on Friday?", "Luis", "2024-01-15 18:03:00")
                ],
                "current_user": "Marcus",
                "sentiment": SentimentProbabilities(sadness=0.1, joy=0.0, love=0.0, anger=0.5, fear=0.2, unknown=0.2)
            },
            # 10. Cousins reconnecting
            {
                "type": "cousins_reconnecting",
                "messages": [
                    ChatMessage("1", "I can't believe it's been 5 years since we last saw each other!", "Diana", "2024-01-15 19:00:00"),
                    ChatMessage("2", "I know! You look exactly the same though!", "Carlos", "2024-01-15 19:01:00"),
                    ChatMessage("3", "We need to stop letting so much time pass. I've missed you", "Diana", "2024-01-15 19:02:00"),
                    ChatMessage("4", "Agreed! Let's make family gatherings a priority again", "Carlos", "2024-01-15 19:03:00")
                ],
                "current_user": "Diana",
                "sentiment": SentimentProbabilities(sadness=0.1, joy=0.5, love=0.3, anger=0.0, fear=0.0, unknown=0.1)
            },
            # 11. Best friends having deep talk
            {
                "type": "friends_deep_talk",
                "messages": [
                    ChatMessage("1", "Do you ever feel like you're just going through the motions?", "Kim", "2024-01-15 20:00:00"),
                    ChatMessage("2", "All the time. Like I'm on autopilot", "Jamie", "2024-01-15 20:01:00"),
                    ChatMessage("3", "What if we're wasting our lives? What if this isn't what we're meant to do?", "Kim", "2024-01-15 20:02:00"),
                    ChatMessage("4", "Maybe that's okay. Maybe figuring it out is part of the journey", "Jamie", "2024-01-15 20:03:00")
                ],
                "current_user": "Kim",
                "sentiment": SentimentProbabilities(sadness=0.3, joy=0.0, love=0.0, anger=0.0, fear=0.4, unknown=0.3)
            },
            # 12. Siblings sharing childhood memories
            {
                "type": "siblings_memories",
                "messages": [
                    ChatMessage("1", "Remember when we built that fort in the backyard?", "Lily", "2024-01-15 21:00:00"),
                    ChatMessage("2", "And dad pretended he couldn't find us for hours! 😂", "Noah", "2024-01-15 21:01:00"),
                    ChatMessage("3", "Those were the best days. Everything was so simple", "Lily", "2024-01-15 21:02:00"),
                    ChatMessage("4", "We should build another fort with our kids someday", "Noah", "2024-01-15 21:03:00")
                ],
                "current_user": "Lily",
                "sentiment": SentimentProbabilities(sadness=0.1, joy=0.5, love=0.3, anger=0.0, fear=0.0, unknown=0.1)
            },
            # 13. Friend giving tough love
            {
                "type": "friend_tough_love",
                "messages": [
                    ChatMessage("1", "You need to stop going back to him. He's not good for you", "Ashley", "2024-01-15 22:00:00"),
                    ChatMessage("2", "I know but when it's good, it's really good", "Megan", "2024-01-15 22:01:00"),
                    ChatMessage("3", "That's not enough! You deserve someone who treats you well ALL the time", "Ashley", "2024-01-15 22:02:00"),
                    ChatMessage("4", "You're right. I'm just scared to be alone", "Megan", "2024-01-15 22:03:00")
                ],
                "current_user": "Ashley",
                "sentiment": SentimentProbabilities(sadness=0.2, joy=0.0, love=0.1, anger=0.3, fear=0.3, unknown=0.1)
            },
            # 14. Parent-adult child distance
            {
                "type": "parent_distance",
                "messages": [
                    ChatMessage("1", "I feel like I never hear from you anymore", "Mom", "2024-01-15 23:00:00"),
                    ChatMessage("2", "I'm just really busy with work Mom", "Ryan", "2024-01-15 23:01:00"),
                    ChatMessage("3", "Too busy for a 5 minute call to your mother?", "Mom", "2024-01-15 23:02:00"),
                    ChatMessage("4", "You're right. I'm sorry. How about dinner this Sunday?", "Ryan", "2024-01-15 23:03:00")
                ],
                "current_user": "Mom",
                "sentiment": SentimentProbabilities(sadness=0.5, joy=0.0, love=0.1, anger=0.2, fear=0.1, unknown=0.1)
            },
            # 15. Friends excited about trip
            {
                "type": "friends_trip_planning",
                "messages": [
                    ChatMessage("1", "VEGAS BABY! I just booked our flights!", "Tyler", "2024-01-16 09:00:00"),
                    ChatMessage("2", "This is going to be EPIC! I've already started packing", "Brandon", "2024-01-16 09:01:00"),
                    ChatMessage("3", "Remember what happens in Vegas...", "Tyler", "2024-01-16 09:02:00"),
                    ChatMessage("4", "Gets posted on Instagram immediately! 😂", "Brandon", "2024-01-16 09:03:00")
                ],
                "current_user": "Tyler",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.8, love=0.0, anger=0.0, fear=0.0, unknown=0.2)
            },
            # 16. Couple discussing future
            {
                "type": "couple_future_talk",
                "messages": [
                    ChatMessage("1", "We need to talk about where this relationship is going", "Sam", "2024-01-16 10:00:00"),
                    ChatMessage("2", "I've been thinking the same thing. What do you see for us?", "Riley", "2024-01-16 10:01:00"),
                    ChatMessage("3", "I love you, but I'm not sure if we want the same things", "Sam", "2024-01-16 10:02:00"),
                    ChatMessage("4", "Let's be honest with each other. What do you want?", "Riley", "2024-01-16 10:03:00")
                ],
                "current_user": "Sam",
                "sentiment": SentimentProbabilities(sadness=0.2, joy=0.0, love=0.2, anger=0.0, fear=0.5, unknown=0.1)
            },
            # 17. Friends comforting after loss
            {
                "type": "friend_loss_comfort",
                "messages": [
                    ChatMessage("1", "I can't believe she's gone. My grandma was everything to me", "Olivia", "2024-01-16 11:00:00"),
                    ChatMessage("2", "I'm so sorry Liv. She was an amazing woman", "Hannah", "2024-01-16 11:01:00"),
                    ChatMessage("3", "I don't know how to do this without her", "Olivia", "2024-01-16 11:02:00"),
                    ChatMessage("4", "You don't have to do it alone. I'm here for whatever you need", "Hannah", "2024-01-16 11:03:00")
                ],
                "current_user": "Hannah",
                "sentiment": SentimentProbabilities(sadness=0.8, joy=0.0, love=0.1, anger=0.0, fear=0.1, unknown=0.0)
            },
            # 18. Siblings competing
            {
                "type": "sibling_competition",
                "messages": [
                    ChatMessage("1", "Of course you got into Harvard. Golden child strikes again", "Ethan", "2024-01-16 12:00:00"),
                    ChatMessage("2", "What's that supposed to mean?", "Grace", "2024-01-16 12:01:00"),
                    ChatMessage("3", "Nothing. Just must be nice to have everything come so easy", "Ethan", "2024-01-16 12:02:00"),
                    ChatMessage("4", "Easy? I worked my ass off! Why can't you just be happy for me?", "Grace", "2024-01-16 12:03:00")
                ],
                "current_user": "Ethan",
                "sentiment": SentimentProbabilities(sadness=0.2, joy=0.0, love=0.0, anger=0.5, fear=0.1, unknown=0.2)
            },
            # 19. Parent-child making amends
            {
                "type": "parent_child_amends",
                "messages": [
                    ChatMessage("1", "I know I wasn't the best father when you were growing up", "Dad", "2024-01-16 13:00:00"),
                    ChatMessage("2", "Dad, you don't have to...", "Jessica", "2024-01-16 13:01:00"),
                    ChatMessage("3", "No, I do. I was too focused on work. I missed so much", "Dad", "2024-01-16 13:02:00"),
                    ChatMessage("4", "We can't change the past, but I'm glad you're here now", "Jessica", "2024-01-16 13:03:00")
                ],
                "current_user": "Dad",
                "sentiment": SentimentProbabilities(sadness=0.4, joy=0.0, love=0.3, anger=0.0, fear=0.1, unknown=0.2)
            },
            # 20. Friends gossiping
            {
                "type": "friends_gossip",
                "messages": [
                    ChatMessage("1", "Did you hear about Mark and Lisa?!", "Chloe", "2024-01-16 14:00:00"),
                    ChatMessage("2", "No! What happened?", "Sophia", "2024-01-16 14:01:00"),
                    ChatMessage("3", "They broke up! Apparently he was texting his ex", "Chloe", "2024-01-16 14:02:00"),
                    ChatMessage("4", "I KNEW something was off at the party last week", "Sophia", "2024-01-16 14:03:00")
                ],
                "current_user": "Chloe",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.3, love=0.0, anger=0.1, fear=0.0, unknown=0.6)
            },
            # 21. Couple having baby news
            {
                "type": "couple_baby_news",
                "messages": [
                    ChatMessage("1", "I have something to tell you... I'm pregnant", "Maria", "2024-01-16 15:00:00"),
                    ChatMessage("2", "What? Are you serious?", "David", "2024-01-16 15:01:00"),
                    ChatMessage("3", "I just found out this morning. I'm scared", "Maria", "2024-01-16 15:02:00"),
                    ChatMessage("4", "Hey, we'll figure this out together. How do you feel about it?", "David", "2024-01-16 15:03:00")
                ],
                "current_user": "Maria",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.2, love=0.1, anger=0.0, fear=0.6, unknown=0.1)
            },
            # 22. Friends boundary setting
            {
                "type": "friend_boundaries",
                "messages": [
                    ChatMessage("1", "I need to be honest. Your constant negativity is draining me", "Ava", "2024-01-16 16:00:00"),
                    ChatMessage("2", "Wow, tell me how you really feel", "Bella", "2024-01-16 16:01:00"),
                    ChatMessage("3", "I care about you but I need to protect my own mental health", "Ava", "2024-01-16 16:02:00"),
                    ChatMessage("4", "I didn't realize I was affecting you like that. I'm sorry", "Bella", "2024-01-16 16:03:00")
                ],
                "current_user": "Ava",
                "sentiment": SentimentProbabilities(sadness=0.2, joy=0.0, love=0.1, anger=0.3, fear=0.2, unknown=0.2)
            },
            # 23. Grandparent-grandchild bonding
            {
                "type": "grandparent_bonding",
                "messages": [
                    ChatMessage("1", "Tell me again about how you met grandpa", "Emma", "2024-01-16 17:00:00"),
                    ChatMessage("2", "Oh sweetheart, it was at a dance in 1955", "Grandma", "2024-01-16 17:01:00"),
                    ChatMessage("3", "Was it love at first sight?", "Emma", "2024-01-16 17:02:00"),
                    ChatMessage("4", "He stepped on my feet three times before I agreed to a second dance!", "Grandma", "2024-01-16 17:03:00")
                ],
                "current_user": "Grandma",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.5, love=0.4, anger=0.0, fear=0.0, unknown=0.1)
            },
            # 24. Friends career jealousy
            {
                "type": "friend_career_jealousy",
                "messages": [
                    ChatMessage("1", "Congrats on the promotion... again", "Mike", "2024-01-16 18:00:00"),
                    ChatMessage("2", "Thanks? You don't sound very happy about it", "Dan", "2024-01-16 18:01:00"),
                    ChatMessage("3", "It's just... we started at the same time and you keep moving up", "Mike", "2024-01-16 18:02:00"),
                    ChatMessage("4", "Your time will come. You're great at what you do", "Dan", "2024-01-16 18:03:00")
                ],
                "current_user": "Mike",
                "sentiment": SentimentProbabilities(sadness=0.3, joy=0.0, love=0.0, anger=0.3, fear=0.2, unknown=0.2)
            },
            # 25. Siblings holiday planning
            {
                "type": "siblings_holiday",
                "messages": [
                    ChatMessage("1", "So Christmas at my house this year?", "Rachel", "2024-01-16 19:00:00"),
                    ChatMessage("2", "Actually, we were thinking of going to Hawaii", "Tom", "2024-01-16 19:01:00"),
                    ChatMessage("3", "What? But we always do Christmas together!", "Rachel", "2024-01-16 19:02:00"),
                    ChatMessage("4", "Maybe you guys could come with us? Make new traditions?", "Tom", "2024-01-16 19:03:00")
                ],
                "current_user": "Rachel",
                "sentiment": SentimentProbabilities(sadness=0.3, joy=0.0, love=0.1, anger=0.2, fear=0.2, unknown=0.2)
            },
            # 26. Friends workout motivation
            {
                "type": "friends_workout",
                "messages": [
                    ChatMessage("1", "5am gym tomorrow?", "Lisa", "2024-01-16 20:00:00"),
                    ChatMessage("2", "Ugh do we have to? My bed is so comfortable", "Kate", "2024-01-16 20:01:00"),
                    ChatMessage("3", "Summer bodies are made in winter! Come on!", "Lisa", "2024-01-16 20:02:00"),
                    ChatMessage("4", "Fine but you're buying the coffee after", "Kate", "2024-01-16 20:03:00")
                ],
                "current_user": "Lisa",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.4, love=0.0, anger=0.0, fear=0.0, unknown=0.6)
            },
            # 27. Parent-teen understanding
            {
                "type": "parent_teen_talk",
                "messages": [
                    ChatMessage("1", "Why won't you talk to me anymore?", "Mom", "2024-01-16 21:00:00"),
                    ChatMessage("2", "You wouldn't understand", "Zoe", "2024-01-16 21:01:00"),
                    ChatMessage("3", "Try me. I was 16 once too", "Mom", "2024-01-16 21:02:00"),
                    ChatMessage("4", "It's just... everything feels so overwhelming right now", "Zoe", "2024-01-16 21:03:00")
                ],
                "current_user": "Mom",
                "sentiment": SentimentProbabilities(sadness=0.4, joy=0.0, love=0.2, anger=0.1, fear=0.2, unknown=0.1)
            },
            # 28. Friends surprise reunion
            {
                "type": "friends_reunion",
                "messages": [
                    ChatMessage("1", "SURPRISE! Bet you didn't expect to see me!", "Jake", "2024-01-16 22:00:00"),
                    ChatMessage("2", "OH MY GOD! When did you get back from London?!", "Chris", "2024-01-16 22:01:00"),
                    ChatMessage("3", "Just landed! Had to see my best friend first", "Jake", "2024-01-16 22:02:00"),
                    ChatMessage("4", "I can't believe you're here! This is the best surprise ever!", "Chris", "2024-01-16 22:03:00")
                ],
                "current_user": "Chris",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.9, love=0.1, anger=0.0, fear=0.0, unknown=0.0)
            },
            # 29. Couple trust issues
            {
                "type": "couple_trust",
                "messages": [
                    ChatMessage("1", "Who's Jennifer? She keeps liking all your photos", "Alex", "2024-01-16 23:00:00"),
                    ChatMessage("2", "She's just a coworker. Are you going through my social media?", "Sam", "2024-01-16 23:01:00"),
                    ChatMessage("3", "It popped up on my feed. Just seems weird", "Alex", "2024-01-16 23:02:00"),
                    ChatMessage("4", "We need to talk about trust in this relationship", "Sam", "2024-01-16 23:03:00")
                ],
                "current_user": "Alex",
                "sentiment": SentimentProbabilities(sadness=0.1, joy=0.0, love=0.0, anger=0.3, fear=0.4, unknown=0.2)
            },
            # 30. Family group chat chaos
            {
                "type": "family_group_chat",
                "messages": [
                    ChatMessage("1", "URGENT: Who ate my leftover pizza?!", "Nick", "2024-01-17 09:00:00"),
                    ChatMessage("2", "Wasn't me, I'm on a diet remember?", "Mom", "2024-01-17 09:01:00"),
                    ChatMessage("3", "Sorry bro, I was hungry after practice", "Jake", "2024-01-17 09:02:00"),
                    ChatMessage("4", "You owe me a large pepperoni!", "Nick", "2024-01-17 09:03:00")
                ],
                "current_user": "Nick",
                "sentiment": SentimentProbabilities(sadness=0.0, joy=0.2, love=0.0, anger=0.4, fear=0.0, unknown=0.4)
            }
        ]
        
        test_contexts = []
        import random
        
        for i in range(num_contexts):
            scenario = scenarios[i % len(scenarios)]
            sentiment = scenario["sentiment"]
            
            # Add some variation to sentiment probabilities
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
                
                # Normalize to ensure sum is close to 1
                total = sum([sentiment.sadness, sentiment.joy, sentiment.love, 
                           sentiment.anger, sentiment.fear, sentiment.unknown])
                if total > 0:
                    sentiment = SentimentProbabilities(
                        sadness=sentiment.sadness/total,
                        joy=sentiment.joy/total,
                        love=sentiment.love/total,
                        anger=sentiment.anger/total,
                        fear=sentiment.fear/total,
                        unknown=sentiment.unknown/total
                    )
            
            context = ChatContext(
                chat_history=scenario["messages"].copy(),
                current_user=scenario["current_user"],
                sentiment_probabilities=sentiment
            )
            test_contexts.append(context)
        
        return test_contexts
