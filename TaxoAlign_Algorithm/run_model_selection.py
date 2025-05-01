import json
from flat_classifier_compare import evaluate_classifier, classify_bart, classify_bert, classify_keyllm
from keyword_extraction_compare import extract_keywords_keyllm, extract_keywords_scibert, extract_keywords_specter

def main():
    # Flat-layer classification comparison
    print("=== Flat-Layer Classification Model Comparison ===")
    papers = json.load(open("arsyta_augmented.json"))
    bart_acc = evaluate_classifier(papers, classify_bart)
    bert_acc = evaluate_classifier(papers, classify_bert)
    keyllm_acc = evaluate_classifier(papers, classify_keyllm)
    print(f"BART-large-MNLI accuracy: {bart_acc:.4f}")
    print(f"BERT-base-uncased accuracy: {bert_acc:.4f}")
    print(f"KeyLLM (placeholder) accuracy: {keyllm_acc:.4f}")

    # Keyword extraction comparison
    print("\n=== Keyword Extraction Model Comparison ===")
    sample = "Graph neural networks are increasingly popular for citation analysis and recommendation."
    print("KeyLLM (placeholder):", extract_keywords_keyllm(sample))
    print("SciBERT:", extract_keywords_scibert(sample))
    print("SPECTER (placeholder):", extract_keywords_specter(sample))

if __name__ == "__main__":
    main()
