from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

MODEL_NAME = "facebook/bart-large-mnli"
LABELS = [...]  # List of arXiv category names

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
    label_idx = torch.argmax(probs).item()
    return LABELS[label_idx], probs.tolist()

def classify_papers(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    for paper in papers:
        label, prob_dist = classify_text(paper["representation_text"])
        paper["flat_label"] = label
        paper["flat_label_probs"] = prob_dist
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    classify_papers("arsyta_augmented.json", "arsyta_classified.json")
