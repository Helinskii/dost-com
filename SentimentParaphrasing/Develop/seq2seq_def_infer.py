import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from sympy.stats.sampling.sample_pymc import do_sample_pymc
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
	T5Tokenizer, T5ForConditionalGeneration,
    DistilBertTokenizer, DistilBertForSequenceClassification
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# s2s_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
# s2s_tokenizer = T5Tokenizer.from_pretrained("results_seq2seq_T5_def/checkpoint-500")
# s2s_tokenizer = BartTokenizer.from_pretrained("./results_seq2seq_default/checkpoint-900")
s2s_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def preprocess(example):
    model_inputs = s2s_tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    return model_inputs

# seq2seq_model = T5ForConditionalGeneration.from_pretrained("./t5_rl_finetuned/iter20")
# seq2seq_model = T5ForConditionalGeneration.from_pretrained("results_seq2seq_T5_def_0622/checkpoint-500")
seq2seq_model = BartForConditionalGeneration.from_pretrained("results_seq2seq_BART_def_0623/checkpoint-500")
# seq2seq_model = BartForConditionalGeneration.from_pretrained("./results_seq2seq_default/checkpoint-900")

seq2seq_model = seq2seq_model.to(device)

def paraphrase(text):
    inputs = s2s_tokenizer("paraphrase: " + text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    # inputs = s2s_tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = seq2seq_model.generate(**inputs, max_length=128)
        # generated_ids = seq2seq_model.generate(**inputs, max_length=128, do_sample=True, top_k=50, top_p=0.95)
        generated_texts = s2s_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts

print(paraphrase("does it look like I care"))
print(paraphrase("Oh you can go to hell and I wouldnt blink an eyelid"))
print(paraphrase("Fuck you and your ideas "))
print(paraphrase("you and I are sworn enemies"))
print(paraphrase("The classroom was full of idiots and boring people"))
print(paraphrase("They keep complaining but do nothing"))
print(paraphrase("He told me to go die"))
print(paraphrase("The filth coming out of your mouth does not even suit the bin"))
print(paraphrase("How can a hospital be so dirty"))
print(paraphrase("A factory should have better working environment"))
print(paraphrase("They stabbed me in the back, serpents all of them"))
# for line in test_data:
#     print(line)
#     print(paraphrase(line['text']))