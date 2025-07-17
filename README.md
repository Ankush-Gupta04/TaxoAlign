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

# TaxoAlign: Multi-Taxonomy Contextually Aligned Recommendations for Scholarly Papers

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

...

## License

This repository and its code are released under the [Creative Commons BY-NC-ND 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).




