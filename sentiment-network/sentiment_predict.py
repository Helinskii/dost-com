import torch
import torch.nn as nn
from transformers import MobileBertTokenizerFast, MobileBertForSequenceClassification, AutoModelForSequenceClassification, DistilBertForSequenceClassification
from collections import Counter
from sentiment_infer import predict_emotion

model_path = "mobilebert-uncased_full_128_stage2/checkpoint-264"
tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")

# model = MobileBertForSequenceClassification.from_pretrained(model_path, num_labels=6)

# model.eval()

labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

class SentimentPredictionEngine(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=6, bert_model_name=model_path, freeze_bert=True):
        super(SentimentPredictionEngine, self).__init__()

        self.model = MobileBertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False

        self.embed_dim = self.model.config.hidden_size

        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, msg_sequences, lengths):
        batch_size = len(msg_sequences)
        max_len = max(lengths)
        device = next(self.parameters()).device
        
        all_embeddings = []
        for seq in msg_sequences:
            seq_embeddings = []
            for msg in seq:
                if msg == "":
                    seq_embeddings.append(torch.zeros(self.embed_dim, device=device))
                else:
                    inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    # Move all input tensors to the correct device
                    # Trying MPS for training/inferencing boost
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model.mobilebert(**inputs)
                        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                        cls_embedding = cls_embedding.to(device)
                    seq_embeddings.append(cls_embedding)
            seq_embeddings = torch.stack(seq_embeddings).to(device)
            all_embeddings.append(seq_embeddings)
        embeddings = torch.stack(all_embeddings).to(device)  # (batch, max_len, embed_dim)
        
        # Packing here so that all messages in a sequence are of the same length
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Get the output for the last real message in each sequence
        last_indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, output.size(2))
        last_outputs = output.gather(1, last_indices).squeeze(1)  # (batch, 2*hidden_dim)
        logits = self.classifier(last_outputs)  # (batch, num_classes)

        # Also return emotion predicted by BERT model for last message
        last_msg = msg_sequences[-1][-1]
        # DEBUG
        # print(last_msg)
        # sentiment_input = tokenizer(last_msg, return_tensors="pt", truncation=True, padding=True, max_length=128)
        # with torch.no_grad():
        #     sentiment_outputs = self.model(**sentiment_input)
        #     sentiment_logits = sentiment_outputs.logits
        #     prediction = torch.argmax(sentiment_logits, dim=1).item()
        emo_last_msg = predict_emotion(last_msg)
 
        return logits, emo_last_msg

    def debug_prints(self):
        print(self.embed_dim)

if __name__ == "__main__":
    model = SentimentPredictionEngine()
    # Load trained weights
    model.load_state_dict(torch.load("sentiment_model.pt", map_location=torch.device("cpu")))
    model.eval()
    
    # dummy_message_seq = ["Hey!", "How's it going?", "Great!"]
    dummy_message_seq = [
        "I failed my exam.",
        "I'm really upset.",
        "But it's okay, I'll try again.",
        "Whatever happens, happens."
    ]
    with torch.no_grad():
        logits, emo_last_msg = model([dummy_message_seq], torch.tensor([len(dummy_message_seq)]))
        predicted_class = torch.argmax(logits).item()
        print(f"Predicted emotion class: {predicted_class}\tLogits: {torch.sigmoid(logits)}")
        print(f"Emotion = {labels[int(predicted_class)]}")
        print(f"Last Message Emotion = {emo_last_msg}")
