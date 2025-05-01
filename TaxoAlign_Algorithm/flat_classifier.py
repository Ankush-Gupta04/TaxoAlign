from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

ARXIV_LABELS = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.DS", "cs.SE", "cs.CR", "cs.NI", "cs.SI", "cs.IR"]

def flat_classify(text):
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(ARXIV_LABELS))
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
    label_idx = torch.argmax(probs).item()
    return ARXIV_LABELS[label_idx], probs.tolist()

if __name__ == "__main__":
    print(flat_classify("This paper explores advances in neural language models for text generation."))
