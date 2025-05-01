from bs4 import BeautifulSoup
import json

def parse_classification_tree(soup):
    tree = {}
    def insert_node(path, text, reference=None):
        node = tree
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = {"text": text}
        if reference:
            node[path[-1]]["reference"] = reference
    for li in soup.find_all("li"):
        name_tag = li.find("a", attrs={"name": True})
        href_tag = li.find("a", href=True)
        if name_tag:
            path = name_tag["name"].split(".")
            text = li.get_text(strip=True).split("\n", 1)[0]
            reference = href_tag.get_text(strip=True) if href_tag else None
            insert_node(path, text, reference)
    return tree

if __name__ == "__main__":
    file_path = 'The ACM Computing Classification System (1998).html'
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    classification_tree = parse_classification_tree(soup)
    output_file = 'classification_tree.json'
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(classification_tree, file, indent=4, ensure_ascii=False)
