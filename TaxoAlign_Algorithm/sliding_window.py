from keyword_extraction import extract_keywords
from flat_classifier import flat_classify
from taxonomy_dag import construct_dag

def find_best_window(sentences, window_size, corpus, taxonomy, mapping):
    best_score = 0
    best_window = None
    for i in range(len(sentences) - window_size + 1):
        subtext = " ".join(sentences[i:i+window_size])
        keywords = extract_keywords(subtext)
        flat_label, _ = flat_classify(subtext)
        query_dag = construct_dag(subtext, flat_label, taxonomy, mapping)
        candidates = retrieve_papers(query_dag, corpus, top_k=1)
        if candidates and candidates[0][1] > best_score:
            best_score = candidates[0][1]
            best_window = (i, i+window_size)
    return best_window

if __name__ == "__main__":
    # Example usage
    pass
