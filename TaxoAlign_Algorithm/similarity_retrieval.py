import torch

def compute_hierarchical_similarity(query_dag, paper_dag):
    # Product of cosine similarities at each level (here, just leaf)
    sim = 1.0
    for q_node, p_node in zip(query_dag, paper_dag):
        q_emb = torch.tensor(q_node["embedding"])
        p_emb = torch.tensor(p_node["embedding"])
        level_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), p_emb.unsqueeze(0)).item()
        sim *= level_sim
    return sim if sim > 0.3 else 0

def retrieve_papers(query_dag, corpus, top_k=10):
    scored = []
    for paper in corpus:
        paper_dag = paper["dag_nodes"]  # List of nodes with "embedding"
        sim = compute_hierarchical_similarity(query_dag, paper_dag)
        if sim > 0:
            scored.append((paper, sim))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]

# Example usage:
if __name__ == "__main__":
    # query_dag and corpus should be prepared as per previous steps
    pass
