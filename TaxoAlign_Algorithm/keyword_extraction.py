from cosine_scibert import get_scibert_embeddings
import spacy

# Load spaCy for noun phrase extraction
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text, top_k=8):
    doc = nlp(text)
    candidates = [chunk.text for chunk in doc.noun_chunks]
    if not candidates:
        return []
    text_emb = get_scibert_embeddings([text])[0]
    cand_embs = get_scibert_embeddings(candidates)
    scores = (text_emb @ cand_embs.T).tolist()
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [candidates[i] for i in top_indices]

if __name__ == "__main__":
    sample = "Graph neural networks are increasingly popular for citation analysis and recommendation."
    print(extract_keywords(sample))
