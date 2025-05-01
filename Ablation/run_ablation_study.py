# run_ablation_study.py

import json
from ablation_config import ABLATION
from ablation_keyword_extraction.py import get_keyword_extractor
from ablation_flat_classifier import get_flat_classifier
from ablation_dag_matching import get_similarity_fn
from ablation_sliding_window import get_window_fn

def evaluate_ablation(test_dataset, corpus, taxonomy, mapping,
                      keyword_method, classifier_method, dag_method, window_method):
    # Load ablated modules
    keyword_extractor = get_keyword_extractor(keyword_method)
    flat_classifier = get_flat_classifier(classifier_method)
    similarity_fn = get_similarity_fn(dag_method)
    window_fn = get_window_fn(window_method)

    metrics = {"recall@10": 0, "mrr": 0, "ndcg": 0}
    total = 0

    for sample in test_dataset:
        sentences = sample["text"].split(". ")
        # Window selection
        if window_method == "sliding":
            window_idx = window_fn(sentences, 3, lambda subtext: 1)  # Use dummy scorer for now
            subtext = " ".join(sentences[window_idx[0]:window_idx[1]])
        else:
            subtext = " ".join(sentences)
        # Keyword extraction
        keywords = keyword_extractor(subtext)
        # Flat classification
        flat_label = flat_classifier(subtext)
        # DAG construction (pseudo, replace with your actual function)
        query_dag = construct_dag(subtext, flat_label, taxonomy, mapping)
        # Retrieval
        candidates = []
        for paper in corpus:
            score = similarity_fn(query_dag, paper["dag_nodes"])
            if score > 0:
                candidates.append((paper, score))
        candidates.sort(key=lambda x: -x[1])
        top10 = [c[0]["paper_id"] for c in candidates[:10]]
        # Evaluate metrics
        gold = set(sample["relevant_papers"])
        hits = len(set(top10) & gold)
        metrics["recall@10"] += hits / len(gold) if gold else 0
        # MRR
        mrr = 0
        for idx, pid in enumerate(top10):
            if pid in gold:
                mrr = 1 / (idx + 1)
                break
        metrics["mrr"] += mrr
        # NDCG (simple version)
        dcg = sum([1 / (np.log2(i + 2)) if pid in gold else 0 for i, pid in enumerate(top10)])
        idcg = sum([1 / (np.log2(i + 2)) for i in range(min(len(gold), 10))])
        metrics["ndcg"] += dcg / idcg if idcg else 0
        total += 1

    # Average metrics
    for k in metrics:
        metrics[k] /= total
    return metrics

if __name__ == "__main__":
    # Load test data and corpus
    test_dataset = json.load(open("test_dataset_400.json"))
    corpus = json.load(open("arsyta_final.json"))
    taxonomy = json.load(open("classification_tree.json"))
    mapping = json.load(open("arxiv_to_ccs_mapping.json"))

    # Run ablation for all combinations
    experiments = [
        # (keyword_method, classifier_method, dag_method, window_method)
        ("keyllm", "bart", "hierarchical", "sliding"),      # Full model
        ("scibert", "bart", "hierarchical", "sliding"),     # Replace keyword extractor
        ("keyllm", "bert", "hierarchical", "sliding"),      # Replace classifier
        ("keyllm", "bart", "leaf", "sliding"),              # Flat/leaf-only matching
        ("keyllm", "bart", "hierarchical", "whole"),        # No sliding window
        ("naive", "majority", "leaf", "whole"),             # All baselines
    ]
    results = []
    for exp in experiments:
        print(f"Running ablation: {exp}")
        metrics = evaluate_ablation(test_dataset, corpus, taxonomy, mapping, *exp)
        results.append({"config": exp, "metrics": metrics})
        print(f"Metrics: {metrics}")

    # Save results
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
