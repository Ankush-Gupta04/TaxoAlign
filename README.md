# TaxoAlign: Multi-Taxonomy Contextually Aligned Citation Recommendations

- **Database:** The core database used in this project is [ArSyTa](https://huggingface.co/datasets/goyalkaraniit/ArSyTa), a large-scale, up-to-date scholarly citation dataset hosted on HuggingFace Datasets.  
---

## Brief Description

TaxoAlign addresses the challenge of finding relevant and contextually appropriate citations in the ever-expanding landscape of academic literature. It works by:

- **Extracting domain-specific keywords** from input text using SciBERT and LLMs.
- **Classifying the input** into flat arXiv categories using advanced classifiers (BART-large-MNLI, BERT, KeyLLM).
- **Constructing a hierarchical DAG representation** by mapping flat categories to ACM CCS taxonomy nodes.
- **Matching and retrieving relevant papers** via hierarchical DAG and semantic similarity.
- **Identifying the optimal citation insertion point** using a sliding window over the text.

TaxoAlign outperforms strong baselines (BM25, HAtten, SciNCL) in both accuracy and contextual alignment, as demonstrated on a test set of 400 recent research papers.

---


## Setup Instructions

### 1. Clone the repository


### 2. Install dependencies

We recommend using a virtual environment.



### 3. Download the ArSyTa database

The main database used in this paper is hosted at HuggingFace Datasets:

- [ArSyTa on HuggingFace](https://huggingface.co/datasets/goyalkaraniit/ArSyTa)




