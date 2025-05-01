# ablation_flat_classifier.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

ARXIV_LABELS = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.DS", "cs.SE", "cs.CR", "cs.NI", "cs.SI", "cs.IR"
]

def classify_bart(text):
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(ARXIV_LABELS))
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
    label_idx = torch.argmax(probs).item()
    return ARXIV_LABELS[label_idx]

def classify_bert(text):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(ARXIV_LABELS))
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
    label_idx = torch.argmax(probs).item()
    return ARXIV_LABELS[label_idx]

def classify_keyllm(text):
    # Placeholder: KeyLLM classification (replace with real model if available)
    return random.choice(ARXIV_LABELS)

def classify_majority(text):
    # Always returns the most frequent class (simulate naive baseline)
    return "cs.AI"

def get_flat_classifier(name):
    if name == "bart":
        return classify_bart
    elif name == "bert":
        return classify_bert
    elif name == "keyllm":
        return classify_keyllm
    elif name == "majority":
        return classify_majority
    else:
        raise ValueError(f"Unknown flat classifier: {name}")
