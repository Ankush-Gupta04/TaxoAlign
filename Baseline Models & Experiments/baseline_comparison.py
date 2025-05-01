import json
import numpy as np
import torch
from sklearn.metrics import ndcg_score
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

class BaselineModelComparison:
    def __init__(self, database_path, test_data_path):
        # Load the augmented database and test data
        with open(database_path, 'r', encoding='utf-8') as f:
            self.database = json.load(f)
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        # Initialize models
        self.initialize_models()
        
        # Prepare corpus for BM25
        self.prepare_bm25_corpus()
        
    def initialize_models(self):
        # Initialize SciBERT for embeddings
        self.scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        
        # Initialize SciNCL (if available)
        try:
            self.scincl_tokenizer = AutoTokenizer.from_pretrained("malteos/scincl")
            self.scincl_model = AutoModel.from_pretrained("malteos/scincl")
        except:
            print("SciNCL model not available, skipping initialization")
            self.scincl_model = None
    
    def prepare_bm25_corpus(self):
        # Create tokenized corpus for BM25
        corpus = [paper.get("representation_text", "") for paper in self.database]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_ids = [paper.get("paper_id", "") for paper in self.database]
    
    def get_scibert_embedding(self, text):
        # Generate SciBERT embedding for a text
        inputs = self.scibert_tokenizer(text, return_tensors="pt", 
                                       max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.scibert_model(**inputs)
        # Use mean pooling for sentence embedding
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding
    
    def get_scincl_embedding(self, text):
        # Generate SciNCL embedding if available
        if self.scincl_model is None:
            return None
        
        inputs = self.scincl_tokenizer(text, return_tensors="pt", 
                                      max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.scincl_model(**inputs)
        # Use mean pooling for sentence embedding
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding
    
    def bm25_search(self, query, top_k=10):
        # Search using BM25
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.corpus_ids[idx], scores[idx]) for idx in top_indices]
        return results
    
    def scibert_search(self, query, top_k=10):
        # Search using SciBERT embeddings
        query_embedding = self.get_scibert_embedding(query)
        
        results = []
        for paper in self.database:
            paper_id = paper.get("paper_id", "")
            if "scibert_embedding" in paper:
                paper_embedding = torch.tensor(paper["scibert_embedding"])
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding, paper_embedding.unsqueeze(0)
                ).item()
                results.append((paper_id, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def scincl_search(self, query, top_k=10):
        # Search using SciNCL embeddings if available
        if self.scincl_model is None:
            return []
            
        query_embedding = self.get_scincl_embedding(query)
        if query_embedding is None:
            return []
            
        results = []
        for paper in self.database:
            paper_id = paper.get("paper_id", "")
            if "scincl_embedding" in paper:
                paper_embedding = torch.tensor(paper["scincl_embedding"])
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding, paper_embedding.unsqueeze(0)
                ).item()
                results.append((paper_id, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def hatten_search(self, query, top_k=10):
        # Simulate HAtten search using a combination of BM25 and SciBERT
        # This is a simplified version as the actual HAtten implementation would be more complex
        bm25_results = self.bm25_search(query, top_k=top_k*2)
        scibert_results = self.scibert_search(query, top_k=top_k*2)
        
        # Combine results with a weighted score
        combined_results = {}
        for paper_id, score in bm25_results:
            combined_results[paper_id] = score * 0.4
            
        for paper_id, score in scibert_results:
            if paper_id in combined_results:
                combined_results[paper_id] += score * 0.6
            else:
                combined_results[paper_id] = score * 0.6
        
        # Sort and return top_k
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def evaluate_model(self, model_name, top_k=10):
        # Evaluate a specific model on test data
        recall_sum = 0
        mrr_sum = 0
        ndcg_sum = 0
        count = 0
        
        for test_item in self.test_data:
            query = test_item.get("query_text", "")
            relevant_papers = set(test_item.get("relevant_papers", []))
            
            if not query or not relevant_papers:
                continue
                
            # Get results based on model name
            if model_name == "bm25":
                results = self.bm25_search(query, top_k)
            elif model_name == "scibert":
                results = self.scibert_search(query, top_k)
            elif model_name == "scincl":
                results = self.scincl_search(query, top_k)
            elif model_name == "hatten":
                results = self.hatten_search(query, top_k)
            else:
                continue
                
            # Calculate metrics
            retrieved_papers = [paper_id for paper_id, _ in results]
            
            # Recall@k
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
            y_score = np.array([score for _, score in results])
            
            if np.sum(y_true) > 0:  # Only calculate NDCG if there are relevant items
                ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
                ndcg_sum += ndcg
            
            count += 1
        
        # Calculate average metrics
        avg_recall = recall_sum / count if count > 0 else 0
        avg_mrr = mrr_sum / count if count > 0 else 0
        avg_ndcg = ndcg_sum / count if count > 0 else 0
        
        return {
            "model": model_name,
            f"recall@{top_k}": avg_recall,
            "mrr": avg_mrr,
            "ndcg": avg_ndcg,
            "samples": count
        }
    
    def run_comparison(self, top_k_values=[5, 10, 15]):
        # Run comparison for all models and different k values
        models = ["bm25", "scibert"]
        if self.scincl_model is not None:
            models.append("scincl")
        models.append("hatten")
        
        results = []
        for model in models:
            for k in top_k_values:
                result = self.evaluate_model(model, k)
                results.append(result)
                print(f"Model: {model}, Top-{k} - Recall: {result[f'recall@{k}']:.4f}, MRR: {result['mrr']:.4f}, NDCG: {result['ndcg']:.4f}")
        
        return results

if __name__ == "__main__":
    # Example usage
    comparison = BaselineModelComparison(
        database_path="arsyta_final.json",
        test_data_path="test_dataset_400.json"
    )
    
    # Run comparison and save results
    results = comparison.run_comparison()
    with open("baseline_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
