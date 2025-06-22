import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

s2s_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
seq2seq_model = T5ForConditionalGeneration.from_pretrained("../results_seq2seq_T5_def/checkpoint-500")

seq2seq_model = seq2seq_model.to(device)

def paraphrase(text):
    # inputs = s2s_tokenizer("paraphrase: " + text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    inputs = s2s_tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        # generated_ids = seq2seq_model.generate(**inputs, max_length=128)
        generated_ids = seq2seq_model.generate(**inputs, max_length=128, do_sample=True, top_k=20, top_p=0.99)
        generated_texts = s2s_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts

print(paraphrase("I am going to smash that window"))
