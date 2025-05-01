from cosine_scibert import get_scibert_embeddings
import spacy

nlp = spacy.load("en_core_web_sm")

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

def extract_keywords_keyllm(text, top_k=8):
    # Placeholder for KeyLLM (replace with actual model if available)
    doc = nlp(text)
    candidates = [chunk.text for chunk in doc.noun_chunks]
    return candidates[:top_k] if candidates else []

def extract_keywords_specter(text, top_k=8):
    # Placeholder for SPECTER (replace with actual model if available)
    doc = nlp(text)
    candidates = [chunk.text for chunk in doc.noun_chunks]
    return candidates[-top_k:] if candidates else []

def compare_extraction(text):
    print("Text:", text)
    print("KeyLLM (placeholder):", extract_keywords_keyllm(text))
    print("SciBERT:", extract_keywords_scibert(text))
    print("SPECTER (placeholder):", extract_keywords_specter(text))

if __name__ == "__main__":
    sample = "Graph neural networks are increasingly popular for citation analysis and recommendation."
    compare_extraction(sample)
