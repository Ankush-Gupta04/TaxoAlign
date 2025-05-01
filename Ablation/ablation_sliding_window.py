# ablation_sliding_window.py

def find_best_window(sentences, window_size, scorer_fn):
    best_score = 0
    best_window = (0, window_size)
    for i in range(len(sentences) - window_size + 1):
        subtext = " ".join(sentences[i:i+window_size])
        score = scorer_fn(subtext)
        if score > best_score:
            best_score = score
            best_window = (i, i+window_size)
    return best_window

def use_whole_text(sentences, scorer_fn):
    subtext = " ".join(sentences)
    score = scorer_fn(subtext)
    return (0, len(sentences)), score

def get_window_fn(name):
    if name == "sliding":
        return find_best_window
    elif name == "whole":
        return use_whole_text
    else:
        raise ValueError(f"Unknown window function: {name}")
