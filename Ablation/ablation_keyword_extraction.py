# ablation_keyword_extraction.py

import spacy
from cosine_scibert import get_scibert_embeddings

nlp = spacy.load("en_core_web_sm")

def extract_keywords_keyllm(text, top_k=8):
    # Placeholder: KeyLLM extraction (replace with real model if available)
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks][:top_k]

def extract_keywords_scibert(text, top_k=8):
    doc = nlp(text)
    candidates = [chunk.text for chunk in doc.noun_chunks]
    if not candidates:
        return []
    text_emb = get_scibert_embeddings([text])[0]
    cand_embs = get_scibert_embeddings(candidates)
    scores = (text_emb @ cand_embs.T).tolist()
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [candidates[i] for i in top_indices]

def extract_keywords_specter(text, top_k=8):
    # Placeholder: SPECTER extraction (replace with real model if available)
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks][-top_k:]

def extract_keywords_naive(text, top_k=8):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks][:top_k]

def get_keyword_extractor(name):
    if name == "keyllm":
        return extract_keywords_keyllm
    elif name == "scibert":
        return extract_keywords_scibert
    elif name == "specter":
        return extract_keywords_specter
    elif name == "naive":
        return extract_keywords_naive
    else:
        raise ValueError(f"Unknown keyword extractor: {name}")
