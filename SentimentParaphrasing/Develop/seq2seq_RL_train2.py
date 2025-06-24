import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load T5
t5_model = T5ForConditionalGeneration.from_pretrained("results_seq2seq_T5_def/checkpoint-1158").to(device)
t5_tokenizer = T5Tokenizer.from_pretrained("results_seq2seq_T5_def/checkpoint-1158")
# t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")

# Load frozen DistilBERT for classification/similarity
distilbert_model = DistilBertForSequenceClassification.from_pretrained('../distilbert_emotion_token128/checkpoint-3000', num_labels=6).to(device)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('../distilbert_emotion_token128/checkpoint-3000')

distilbert_model.eval()
for param in distilbert_model.parameters():
    param.requires_grad = False

# Optimizer for T5 + classifier head (T5 uses RL loss, classifier uses supervised loss optionally)
optimizer = torch.optim.Adam(list(t5_model.parameters()), lr=1e-4)

# Sample dataset
df = pd.read_csv("massive_emotion_dataset.csv")[:1000]

data = Dataset.from_pandas(df)

dataset = Dataset.from_pandas(df)
train_test_split = dataset.train_test_split(test_size=0.4, seed=42)
val_test_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

train_dataloader = DataLoader(train_test_split["train"], batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_test_split["train"], batch_size=8, shuffle=True)
test_dataloader = DataLoader(val_test_split["test"], batch_size=8, shuffle=True)

# Reward function
def compute_reward(input_texts, generated_texts):
    inputs = distilbert_tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
    gen = distilbert_tokenizer(generated_texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)

    # Get embeddings
    input_op = distilbert_model(**inputs, output_hidden_states=True, output_attentions=True)
    # inp_logits = input_op.logits
    # inp_hs = input_op.hidden_states[-1]

    gen_op = distilbert_model(**gen, output_hidden_states=True, output_attentions=True)
    gen_logits = gen_op.logits
    # gen_hs = gen_op.hidden_states[-1]

    # Classification loss on generated
    labels = torch.randint(1, 3, (gen_logits.shape[0],), device=device)  # Dummy labels (you can use real if available)
    cls_loss = F.cross_entropy(gen_logits, labels, reduction='none')
    # print(cls_loss)
    # Cosine similarity between input and generated embeddings
    cos_sim = F.cosine_similarity(torch.mean(input_op.hidden_states[-1][:, 1:, :], dim=1),
                                  torch.mean(gen_op.hidden_states[-1][:, 1:, :], dim=1), dim=1)

    # Higher reward for lower classification loss and higher cosine similarity
    reward = cos_sim - 0.00 * cls_loss  # Can weight each term
    return reward.detach()

# RL training loop (REINFORCE)
iter = 0
for epoch in range(3):
    for batch in tqdm(train_dataloader):
        iter+=1
        input_texts = batch['original']
        input_ids = t5_tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # Generate output using sampling
        outputs = t5_model(input_ids, labels=input_ids, return_dict=True)
        # outputs = t5_model.generate(input_ids, output_scores=True, return_dict_in_generate=True)
        # outputs = t5_model.generate(input_ids, do_sample=True, max_length=128, top_k=50, top_p=0.95, output_scores=True, return_dict_in_generate=True)
        generated_ids = torch.argmax(outputs.logits, dim=-1)
        generated_texts = t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        sequence_log_prob = torch.log(torch.softmax(outputs.logits, dim=-1)+0.00001)

        # Compute reward
        rewards = compute_reward(input_texts, generated_texts)

        # Compute log probs of generated sequences
        # decoder_input_ids = t5_model._shift_right(generated_ids)
        # outputs = t5_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=generated_ids)
        # sequence_log_prob = -outputs.loss  # negative log likelihood, approximate log prob

        # log_probs = F.log_softmax(outputs.logits, dim=-1)
        #
        # # Gather log-probs of the sampled tokens
        # log_probs_seq = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        #
        # # Mask padding
        # mask = (generated_ids != t5_tokenizer.pad_token_id).float()
        # sequence_log_prob = (log_probs_seq * mask).sum(dim=1)

        # Policy gradient loss
        rl_loss = - (sequence_log_prob * rewards.mean()).mean()  # Reinforce: L = -E[log π(a|s) * R]

        optimizer.zero_grad()
        rl_loss.backward()
        optimizer.step()

        print(f"RL Loss: {rl_loss.item():.4f} | Avg Reward: {rewards.mean().item():.4f}")
        if iter%20==0:
            save_path = "./t5_rl_finetuned/iter{}".format(iter)
            t5_model.save_pretrained(save_path)
            t5_tokenizer.save_pretrained(save_path)

    save_path = "./t5_rl_finetuned/iter{}".format(iter)
    t5_model.save_pretrained(save_path)
    t5_tokenizer.save_pretrained(save_path)