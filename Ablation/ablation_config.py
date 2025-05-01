# ablation_config.py

# Set these flags to True to ENABLE that component, False to ablate (disable/replace)
ABLATION = {
    "keyword_extraction": True,      # If False, use random or noun-phrase only keywords
    "flat_classifier": True,         # If False, use random or majority class
    "dag_matching": True,            # If False, use flat matching only
    "hierarchical_matching": True,   # If False, use only leaf node similarity
    "sliding_window": True           # If False, use whole text as a single window
}
