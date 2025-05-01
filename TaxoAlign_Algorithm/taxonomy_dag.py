import json
from cosine_scibert import get_scibert_embeddings

def load_taxonomy(json_path="classification_tree.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_node_description(taxonomy, node_path):
    node = taxonomy
    for key in node_path.split("/"):
        node = node.get(key, {})
    return node.get("text", "")

def construct_dag(text, flat_label, taxonomy, mapping, top_k=3, threshold=0.4):
    candidate_paths = mapping.get(flat_label, [])
    if not candidate_paths:
        return []
    text_emb = get_scibert_embeddings([text])[0]
    node_descs = [get_node_description(taxonomy, path) for path in candidate_paths]
    node_embs = get_scibert_embeddings(node_descs)
    sims = (text_emb @ node_embs.T).tolist()
    scored = [(path, sim) for path, sim in zip(candidate_paths, sims) if sim > threshold]
    scored = sorted(scored, key=lambda x: -x[1])[:top_k]
    return [{"path": path, "sim": sim, "embedding": node_embs[i].tolist()} for i, (path, sim) in enumerate(scored)]

if __name__ == "__main__":
    taxonomy = load_taxonomy()
    # Example mapping: {"cs.AI": ["B/1/1", "I/2/6/1"]}
    mapping = {"cs.AI": ["B/1/1", "I/2/6/1"]}
    dag = construct_dag("Sample text about AI.", "cs.AI", taxonomy, mapping)
    print(dag)
