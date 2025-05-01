import json

def load_arsyta_papers(input_path):
    # Load ArSyTa database JSON or CSV (assume JSON for this example)
    with open(input_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    return papers

def generate_representation_text(paper):
    # Concatenate title, abstract, and all citation contexts
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    cited_texts = " ".join(paper.get("cited_texts", []))
    return f"{title} {abstract} {cited_texts}"

def augment_papers(papers):
    for paper in papers:
        paper["representation_text"] = generate_representation_text(paper)
    return papers

def save_augmented_papers(papers, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    papers = load_arsyta_papers("arsyta_raw.json")
    papers = augment_papers(papers)
    save_augmented_papers(papers, "arsyta_augmented.json")
