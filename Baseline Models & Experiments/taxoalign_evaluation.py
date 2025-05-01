import json
import numpy as np
import torch
from sklearn.metrics import ndcg_score
from cosine_scibert import get_scibert_embeddings

class TaxoAlignEvaluation:
    def __init__(self, database_path, test_data_path):
        # Load the augmented database and test data
        with open(database_path, 'r', encoding='utf-8') as f:
            self.database = json.load(f)
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        # Create a mapping for faster paper lookup
        self.paper_map = {paper.get("paper_id", ""): paper for paper in self.database}
        
    def process_query(self, query_text):
        # Process the query using the TaxoAlign pipeline
        from prepare_database import generate_representation_text
        from flat_classifier import classify_text
        from dag_augmentation import compute_embeddings_for_nodes
        
        # 1. Extract keywords and classify
        representation = generate_representation_text({"title": "", "abstract": query_text, "cited_texts": []})
        flat_label, _ = classify_text(representation)
        
        # 2. Construct DAG representation
        # (This would call the DAG construction code from dag_augmentation.py)
        # For simplicity, we'll use a placeholder implementation here
        query_dag = self._construct_dag_for_query(query_text, flat_label)
        
        return query_text, flat_label, query_dag
    
    def _construct_dag_for_query(self, query_text, flat_label):
        # Simplified DAG construction for evaluation
        # In a real implementation, this would use the same logic as in dag_augmentation.py
        query_embedding = get_scibert_embeddings([query_text])[0]
        
        # Find taxonomy nodes related to the flat label
        # (This is a simplified version of the actual implementation)
        related_nodes = []
        for paper in self.database:
            if paper.get("flat_label") == flat_label:
                if "dag_nodes" in paper:
                    for node in paper["dag_nodes"]:
                        if node not in related_nodes:
                            related_nodes.append(node)
        
        # Return a simplified DAG representation
        return {
            "flat_label": flat_label,
            "embedding": query_embedding.tolist(),
            "nodes": related_nodes[:3]  # Take top 3 nodes
        }
    
    def retrieve_papers(self, query_text, flat_label, query_dag, top_k=10):
        # Implement TaxoAlign retrieval logic
        scores = []
        
        for paper_id, paper in self.paper_map.items():
            # Skip papers with different flat labels
            if paper.get("flat_label") != flat_label:
                continue
            
            # Calculate hierarchical similarity
            similarity = self._calculate_hierarchical_similarity(query_dag, paper)
            if similarity > 0.3:  # Apply threshold as in the paper
                scores.append((paper_id, similarity))
        
        # Sort by similarity and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _calculate_hierarchical_similarity(self, query_dag, paper):
        # Calculate hierarchical similarity between query DAG and paper
        # This is a simplified version of the actual implementation
        
        # Check if paper has DAG nodes
        if "dag_nodes" not in paper or not paper["dag_nodes"]:
            return 0
        
        # Calculate node overlap
        query_nodes = set(query_dag["nodes"])
        paper_nodes = set(paper["dag_nodes"])
        node_overlap = len(query_nodes.intersection(paper_nodes))
        
        # If no node overlap, return 0
        if node_overlap == 0:
            return 0
        
        # Calculate embedding similarity
        query_embedding = torch.tensor(query_dag["embedding"])
        
        # Use the first DAG embedding from the paper
        if "dag_embeddings" in paper and paper["dag_embeddings"]:
            paper_embedding = torch.tensor(paper["dag_embeddings"][0])
            embedding_similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), paper_embedding.unsqueeze(0)
            ).item()
        else:
            embedding_similarity = 0
        
        # Combine node overlap and embedding similarity
        # The actual implementation would use a product of similarities across levels
        similarity = (node_overlap / max(len(query_nodes), len(paper_nodes))) * embedding_similarity
        
        return similarity
    
    def evaluate(self, top_k_values=[5, 10, 15]):
        # Evaluate TaxoAlign on test data
        results = []
        
        for k in top_k_values:
            recall_sum = 0
            mrr_sum = 0
            ndcg_sum = 0
            count = 0
            
            for test_item in self.test_data:
                query = test_item.get("query_text", "")
                relevant_papers = set(test_item.get("relevant_papers", []))
                
                if not query or not relevant_papers:
                    continue
                
                # Process query
                query_text, flat_label, query_dag = self.process_query(query)
                
                # Retrieve papers
                retrieved_results = self.retrieve_papers(query_text, flat_label, query_dag, top_k=k)
                retrieved_papers = [paper_id for paper_id, _ in retrieved_results]
                
                # Calculate metrics
                hits = len(set(retrieved_papers) & relevant_papers)
                recall = hits / len(relevant_papers) if relevant_papers else 0
                recall_sum += recall
                
                # MRR
                mrr = 0
                for i, paper_id in enumerate(retrieved_papers):
                    if paper_id in relevant_papers:
                        mrr = 1 / (i + 1)
                        break
                mrr_sum += mrr
                
                # NDCG
                y_true = np.zeros(len(retrieved_papers))
                for i, paper_id in enumerate(retrieved_papers):
                    if paper_id in relevant_papers:
                        y_true[i] = 1
                y_score = np.array([score for _, score in retrieved_results])
                
                if np.sum(y_true) > 0:  # Only calculate NDCG if there are relevant items
                    ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
                    ndcg_sum += ndcg
                
                count += 1
            
            # Calculate average metrics
            avg_recall = recall_sum / count if count > 0 else 0
            avg_mrr = mrr_sum / count if count > 0 else 0
            avg_ndcg = ndcg_sum / count if count > 0 else 0
            
            result = {
                "model": "TaxoAlign",
                f"recall@{k}": avg_recall,
                "mrr": avg_mrr,
                "ndcg": avg_ndcg,
                "samples": count
            }
            
            results.append(result)
            print(f"TaxoAlign, Top-{k} - Recall: {avg_recall:.4f}, MRR: {avg_mrr:.4f}, NDCG: {avg_ndcg:.4f}")
        
        return results

if __name__ == "__main__":
    # Example usage
    evaluation = TaxoAlignEvaluation(
        database_path="arsyta_final.json",
        test_data_path="test_dataset_400.json"
    )
    
    # Run evaluation and save results
    results = evaluation.evaluate()
    with open("taxoalign_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
