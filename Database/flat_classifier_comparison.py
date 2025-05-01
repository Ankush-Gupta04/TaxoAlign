import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define your candidate models
MODEL_CONFIGS = {
    "bart-large-mnli": {
        "model_name": "facebook/bart-large-mnli"
    },
    "bert-base-uncased": {
        "model_name": "bert-base-uncased"
    },
    # Add KeyLLM or other models as needed
    # "keyllm": {"model_name": "your-keyllm-model"}
}

# List of arXiv category names
ARXIV_LABELS = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.DS", "cs.SE", "cs.CR", "cs.NI", "cs.SI", "cs.IR"
    # ... add all relevant categories
]

def load_papers(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(ARXIV_LABELS))
    model.eval()
    return tokenizer, model

def classify_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
    label_idx = torch.argmax(probs).item()
    return ARXIV_LABELS[label_idx], probs.tolist()

def evaluate_classifier(papers, tokenizer, model):
    correct = 0
    total = 0
    for paper in papers:
        text = paper["representation_text"]
        true_label = paper["flat_label"]  # Assuming ground truth is available
        pred_label, _ = classify_text(text, tokenizer, model)
        if pred_label == true_label:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    papers = load_papers("arsyta_augmented.json")  # Should include ground-truth flat_label

    results = {}
    for model_key, config in MODEL_CONFIGS.items():
        print(f"Evaluating {model_key}...")
        tokenizer, model = get_model_and_tokenizer(config["model_name"])
        accuracy = evaluate_classifier(papers, tokenizer, model)
        results[model_key] = accuracy
        print(f"{model_key} accuracy: {accuracy:.4f}")

    # Save results
    with open("flat_classifier_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
