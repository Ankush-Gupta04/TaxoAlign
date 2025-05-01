import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

ARXIV_LABELS = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.DS", "cs.SE", "cs.CR", "cs.NI", "cs.SI", "cs.IR"
    # ... add all relevant categories
]

def load_papers(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    # Placeholder for KeyLLM classification (replace with actual model if available)
    # For now, returns a random label
    import random
    return random.choice(ARXIV_LABELS)

def evaluate_classifier(papers, classifier_fn):
    correct = 0
    total = 0
    for paper in papers:
        text = paper["representation_text"]
        true_label = paper["flat_label"]  # Assume ground-truth is available
        pred_label = classifier_fn(text)
        if pred_label == true_label:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    papers = load_papers("arsyta_augmented.json")  # Should have 'representation_text' and 'flat_label'
    print("Evaluating BART-large-MNLI...")
    bart_acc = evaluate_classifier(papers, classify_bart)
    print("Evaluating BERT-base-uncased...")
    bert_acc = evaluate_classifier(papers, classify_bert)
    print("Evaluating KeyLLM (placeholder)...")
    keyllm_acc = evaluate_classifier(papers, classify_keyllm)
    print(f"BART-large-MNLI accuracy: {bart_acc:.4f}")
    print(f"BERT-base-uncased accuracy: {bert_acc:.4f}")
    print(f"KeyLLM (placeholder) accuracy: {keyllm_acc:.4f}")
