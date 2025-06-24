import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentiment_predict import SentimentPredictionEngine, tokenizer
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Dummy dataset for demonstration
class ConversationDataset(Dataset):
    def __init__(self, conversations, labels):
        self.conversations = conversations  # List[List[str]]
        self.labels = labels  # List[int]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx], self.labels[idx]

# Example data
# conversations = [
#     ["Hi!", "How are you?", "I'm good, thanks!"],  # sequence of messages
#     ["Hello.", "What's up?", "Not much, you?"],
# ]
# labels = [2, 1]

with open("./datasets/synthetic_sentiment_dataset_gpt4.json", "r") as f:
    data = json.load(f)

conversations = []
labels = []
for item in data:
    conversations.append(item["messages"])
    labels.append(item["label"])

train_conversations, val_conversations, train_labels, val_labels = train_test_split(
    conversations, labels, test_size=0.3, random_state=42
)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Convert each sequence (list of str) to a list of str
    lengths = [len(seq) for seq in sequences]
    # Pad with empty string to max length
    max_len = max(lengths)
    padded_sequences = [seq + [""] * (max_len - len(seq)) for seq in sequences]
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths)

train_dataset = ConversationDataset(train_conversations, train_labels)
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

val_dataset = ConversationDataset(val_conversations, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Model, loss, optimizer
model = SentimentPredictionEngine().to(device)
# Load existing weights if available
import os
model_save_path = "sentiment_model.pt"
if os.path.exists(model_save_path):
    print(f"Loading existing model weights from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
else:
    print("No existing model weights found. Training from scratch.")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def validate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for msg_sequences, labels, lengths in dataloader:
            labels = labels.to(device)
            lengths = lengths.to(device)
            batch_logits, _ = model(msg_sequences, lengths)
            preds = torch.argmax(batch_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0
    print(f"Validation accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# Train the SentimentPredictionEngine
# Trying to train using MPS
num_epochs = 1
loss_list = []
train_acc_list = []
train_f1_list = []
train_precision_list = []
train_recall_list = []

for epoch in range(num_epochs):
    model.train()
    all_preds = []
    all_labels = []
    # correct_preds = 0
    # total = 0
    for msg_sequences, labels, lengths in tqdm(dataloader):
        labels = labels.to(device)
        lengths = lengths.to(device)
        optimizer.zero_grad()
        logits, _ = model(msg_sequences, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        # correct_preds += (preds == labels).sum().items()
        # total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    train_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    train_acc_list.append(train_acc)
    train_f1_list.append(f1)
    train_precision_list.append(precision)
    train_recall_list.append(recall)
    print(f"Train Accuracy: {train_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
          
    # Validate after each epoch
    validate(model, val_dataloader)

# Save the trained model
model_save_path = "sentiment_model.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# # Example inference
# Inferencing present in a different file 'sentiment_predict'
# model.eval()
# test_sequence = ["Hey!", "How's it going?", "Great!"]
# with torch.no_grad():
#     logits = model(test_sequence)
#     predicted_class = torch.argmax(logits).item()
#     print(f"Predicted emotion class: {predicted_class}")
