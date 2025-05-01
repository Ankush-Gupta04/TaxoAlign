import json
from cosine_scibert import get_scibert_embeddings  # Provided in cosine_scibert.py[2]

def load_taxonomy_nodes(taxonomy_json_path):
    with open(taxonomy_json_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    # Flatten the taxonomy tree to a list of nodes with their paths and descriptions
    nodes = []
    def traverse(node, path):
        if "text" in node:
            nodes.append({"path": path, "description": node["text"]})
        for k, v in node.items():
            if isinstance(v, dict):
                traverse(v, path + [k])
    for k, v in taxonomy.items():
        traverse(v, [k])
    return nodes

def compute_embeddings_for_nodes(nodes):
    descriptions = [node["description"] for node in nodes]
    embeddings = get_scibert_embeddings(descriptions)
    for i, node in enumerate(nodes):
        node["embedding"] = embeddings[i].tolist()
    return nodes

def augment_papers_with_dag(input_path, taxonomy_nodes, output_path, threshold=0.4):
    with open(input_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    paper_texts = [paper["representation_text"] for paper in papers]
    paper_embeddings = get_scibert_embeddings(paper_texts)
    for i, paper in enumerate(papers):
        # Find candidate nodes based on flat_label mapping (implement this mapping as needed)
        candidates = [node for node in taxonomy_nodes if paper["flat_label"] in node["description"]]
        similarities = []
        for node in candidates:
            node_emb = torch.tensor(node["embedding"])
            paper_emb = paper_embeddings[i]
            sim = torch.nn.functional.cosine_similarity(paper_emb.unsqueeze(0), node_emb.unsqueeze(0)).item()
            similarities.append((node, sim))
        # Select top 3 DAGs above threshold
        selected = sorted([x for x in similarities if x[1] > threshold], key=lambda x: -x[1])[:3]
        paper["dag_nodes"] = [x[0]["path"] for x in selected]
        paper["dag_embeddings"] = [x[0]["embedding"] for x in selected]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    nodes = load_taxonomy_nodes("classification_tree.json")
    nodes = compute_embeddings_for_nodes(nodes)
    augment_papers_with_dag("arsyta_classified.json", nodes, "arsyta_final.json")
