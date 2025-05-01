import json
from sliding_window import find_best_window
from keyword_extraction import extract_keywords
from flat_classifier import flat_classify
from taxonomy_dag import construct_dag
from similarity_retrieval import retrieve_papers

def taxoalign_recommend(input_text, corpus, taxonomy, mapping, window_size=3, top_k=10):
    sentences = input_text.split(". ")
    best_window = find_best_window(sentences, window_size, corpus, taxonomy, mapping)
    if best_window is None:
        return []
    subtext = " ".join(sentences[best_window[0]:best_window[1]])
    keywords = extract_keywords(subtext)
    flat_label, _ = flat_classify(subtext)
    query_dag = construct_dag(subtext, flat_label, taxonomy, mapping)
    recommendations = retrieve_papers(query_dag, corpus, top_k=top_k)
    return [{"title": rec[0]["title"], "insertion_point": best_window[0], "score": rec[1]} for rec in recommendations]

if __name__ == "__main__":
    taxonomy = json.load(open("classification_tree.json"))
    mapping = json.load(open("arxiv_to_ccs_mapping.json"))
    corpus = json.load(open("arsyta_final.json"))
    input_text = "Your research passage here."
    results = taxoalign_recommend(input_text, corpus, taxonomy, mapping)
    for r in results:
        print(r)
