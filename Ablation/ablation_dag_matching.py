# ablation_dag_matching.py

import torch

def hierarchical_similarity(query_dag, paper_dag):
    sim = 1.0
    for q_node, p_node in zip(query_dag, paper_dag):
        q_emb = torch.tensor(q_node["embedding"])
        p_emb = torch.tensor(p_node["embedding"])
        level_sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), p_emb.unsqueeze(0)).item()
        sim *= level_sim
    return sim if sim > 0.3 else 0

def leaf_similarity(query_dag, paper_dag):
    # Only compare last (leaf) node
    q_emb = torch.tensor(query_dag[-1]["embedding"])
    p_emb = torch.tensor(paper_dag[-1]["embedding"])
    sim = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), p_emb.unsqueeze(0)).item()
    return sim if sim > 0.3 else 0

def get_similarity_fn(name):
    if name == "hierarchical":
        return hierarchical_similarity
    elif name == "leaf":
        return leaf_similarity
    else:
        raise ValueError(f"Unknown similarity function: {name}")
