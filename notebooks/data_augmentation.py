from transformers import BertForMaskedLM, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load BERT for MLM
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model.eval()

# Load GPT-2 for pre-training
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


import random

def mask_random_words(text, mask_prob=0.15):
    tokens = bert_tokenizer.tokenize(text)
    masked_text = []
    masked_indices = []

    for i, token in enumerate(tokens):
        if random.random() < mask_prob:
            masked_text.append("[MASK]")
            masked_indices.append(i)
        else:
            masked_text.append(token)

    return " ".join(masked_text), masked_indices, tokens