from transformers import pipeline

# Define all categories with detailed information
detailed_categories = [
    {'category_code': 'cs.AI', 'category_name': 'Artificial Intelligence', 'acm_subject_names': [
        'Artificial Intelligence', 'Applications and Expert Systems', 'Deduction and Theorem Proving',
        'Knowledge Representation Formalisms and Methods', 'Problem Solving, Control Methods, and Search',
        'Distributed Artificial Intelligence'
    ]},
    {'category_code': 'cs.AR', 'category_name': 'Hardware Architecture', 'acm_subject_names': [
        'Computer Systems Organization', 'Processor Architectures', 'Computer System Implementation'
    ]},
    {'category_code': 'cs.CC', 'category_name': 'Computational Complexity', 'acm_subject_names': [
        'Computation by Abstract Devices', 'Tradeoffs between Complexity Measures', 'Formal Languages',
        'Numerical Algorithms and Problems', 'Nonnumerical Algorithms and Problems'
    ]},
    {'category_code': 'cs.CE', 'category_name': 'Computational Engineering, Finance, and Science', 'acm_subject_names': [
        'Physical Sciences and Engineering', 'Life and Medical Sciences', 'Social and Behavioral Sciences'
    ]},
    {'category_code': 'cs.CG', 'category_name': 'Computational Geometry', 'acm_subject_names': [
        'Computational Geometry and Object Modeling', 'Nonnumerical Algorithms and Problems'
    ]},
    {'category_code': 'cs.CL', 'category_name': 'Computation and Language', 'acm_subject_names': [
        'Natural Language Processing'
    ]},
    {'category_code': 'cs.CR', 'category_name': 'Cryptography and Security', 'acm_subject_names': [
        'Security and Protection', 'Data Encryption'
    ]},
    {'category_code': 'cs.CV', 'category_name': 'Computer Vision and Pattern Recognition', 'acm_subject_names': [
        'Vision and Scene Understanding', 'Image Processing and Computer Vision', 'Pattern Recognition'
    ]},
    {'category_code': 'cs.CY', 'category_name': 'Computers and Society', 'acm_subject_names': [
        'Computing Milieux', 'History of Computing', 'Computers and Education', 'Computers and Society',
        'Legal Aspects of Computing', 'Computing Profession'
    ]},
    {'category_code': 'cs.DB', 'category_name': 'Databases', 'acm_subject_names': [
        'Data Storage Representations', 'Files', 'Information Systems', 'Database Management',
        'Administrative Data Processing'
    ]},
    {'category_code': 'cs.DC', 'category_name': 'Distributed, Parallel, and Cluster Computing', 'acm_subject_names': [
        'Multiple Data Stream Architectures', 'Parallel Architectures', 'Distributed Systems',
        'Concurrent Programming', 'Reliability', 'Organization and Design', 'Data Structures'
    ]},
    {'category_code': 'cs.DL', 'category_name': 'Digital Libraries', 'acm_subject_names': [
        'Online Information Services', 'Library Automation', 'Digital Libraries', 'Document and Text Processing'
    ]},
    {'category_code': 'cs.DM', 'category_name': 'Discrete Mathematics', 'acm_subject_names': [
        'Discrete Mathematics', 'Probability and Statistics'
    ]},
    {'category_code': 'cs.DS', 'category_name': 'Data Structures and Algorithms', 'acm_subject_names': [
        'Data Structures', 'Data Storage Representations', 'Numerical Algorithms and Problems',
        'Nonnumerical Algorithms and Problems'
    ]},
    {'category_code': 'cs.ET', 'category_name': 'Emerging Technologies', 'acm_subject_names': [
        'CMOS-based technologies', 'Nanoscale Electronics', 'Photonics', 'Spintronics', 'Superconductors',
        'Mechanical and Biochemical Technologies', 'Quantum Technologies'
    ]},
    {'category_code': 'cs.FL', 'category_name': 'Formal Languages and Automata Theory', 'acm_subject_names': [
        'Models of Computation', 'Formal Languages'
    ]},
    {'category_code': 'cs.GL', 'category_name': 'General Literature', 'acm_subject_names': [
        'Introductory and Survey', 'References, Dictionaries, Encyclopedias, Glossaries'
    ]},
    {'category_code': 'cs.GR', 'category_name': 'Graphics', 'acm_subject_names': [
        'Graphics Systems', 'Picture or Image Generation', 'Graphics Utilities', 'Three-Dimensional Graphics and Realism'
    ]},
    {'category_code': 'cs.GT', 'category_name': 'Computer Science and Game Theory', 'acm_subject_names': [
        'Mechanism Design', 'Learning in Games', 'Foundations of Agent Modeling in Games',
        'Coordination in Non-Cooperative Environments'
    ]},
    {'category_code': 'cs.HC', 'category_name': 'Human-Computer Interaction', 'acm_subject_names': [
        'User Interfaces', 'Group and Organization Interfaces', 'Hypertext or Hypermedia', 'Sound and Music Computing'
    ]},
    {'category_code': 'cs.IR', 'category_name': 'Information Retrieval', 'acm_subject_names': [
        'Content Analysis and Indexing', 'Information Storage', 'Information Search and Retrieval', 'Systems and Software'
    ]},
    {'category_code': 'cs.IT', 'category_name': 'Information Theory', 'acm_subject_names': [
        'Systems and Information Theory', 'Coding and Information Theory'
    ]},
    {'category_code': 'cs.LG', 'category_name': 'Machine Learning', 'acm_subject_names': [
        'Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'Bandit Problems', 
        'Robustness, Explanation, Fairness'
    ]},
    {'category_code': 'cs.LO', 'category_name': 'Logic in Computer Science', 'acm_subject_names': [
        'Software or Program Verification', 'Specifying and Verifying and Reasoning About Programs',
        'Mathematical Logic and Formal Languages', 'Grammars and Other Rewriting Systems', 'Formal Languages'
    ]},
    {'category_code': 'cs.MA', 'category_name': 'Multiagent Systems', 'acm_subject_names': [
        'Distributed Artificial Intelligence'
    ]},
    {'category_code': 'cs.MM', 'category_name': 'Multimedia', 'acm_subject_names': [
        'Multimedia Information Systems'
    ]},
    {'category_code': 'cs.MS', 'category_name': 'Mathematical Software', 'acm_subject_names': [
        'Mathematical Software'
    ]},
    {'category_code': 'cs.NA', 'category_name': 'Numerical Analysis', 'acm_subject_names': [
        'Numerical Analysis'
    ]},
    {'category_code': 'cs.NE', 'category_name': 'Neural and Evolutionary Computing', 'acm_subject_names': [
        'Other Architecture Styles', 'Learning', 'Pattern Recognition'
    ]},
    {'category_code': 'cs.NI', 'category_name': 'Networking and Internet Architecture', 'acm_subject_names': [
        'Network Architecture and Design', 'Network Protocols', 'Network Operations', 'Distributed Systems',
        'Local and Wide-Area Networks', 'Internetworking'
    ]},
    {'category_code': 'cs.OH', 'category_name': 'Other Computer Science', 'acm_subject_names': [
        'Miscellaneous'
    ]},
    # Add the remaining categories similarly
]


# Extract detailed category descriptions
category_descriptions = [
    f"{cat['category_name']} ({cat['category_code']}): {', '.join(cat['acm_subject_names'])}"
    for cat in detailed_categories
]

# Initialize the zero-shot classification pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0  # Use GPU if available
)

# Function to classify input text
def classify_text(input_text):
    # Construct the classification prompt with categories and their subcategories
    classification_prompt = (
        f"The following are the categories and their subcategories:\n"
        + "\n".join(category_descriptions)
        + f"\n\nGiven the text: \"{input_text}\", which category does it belong to?"
    )

    # Perform zero-shot classification
    result = classifier(
        classification_prompt,
        [cat["category_name"] for cat in detailed_categories]
    )
    
    # Get the top prediction
    top_prediction = result["labels"][0]
    confidence_score = result["scores"][0]

    # Map the top prediction to the detailed category
    for category in detailed_categories:
        if category["category_name"] == top_prediction:
            return {
                "predicted_class": category["category_name"],
                "category_code": category["category_code"],
                "acm_subject_names": category["acm_subject_names"],
                "confidence_score": confidence_score,
            }
    return None

# Example usage
input_text = """
"Deep learning, a subset of machine learning, has revolutionized the field of artificial intelligence by enabling computers to process and analyze large datasets. Convolutional neural networks (CNNs) have become the backbone of computer vision tasks, excelling in applications such as image recognition, object detection, and medical image analysis. Despite their success, these models often require extensive computational resources and large amounts of labeled data. Recent advancements in transfer learning and pre-trained models have mitigated these challenges by allowing models to generalize knowledge across tasks, reducing the need for task-specific training data."
"""

classification_result = classify_text(input_text)

# Print detailed results
print("Classification Result:")
print(f"Predicted Class: {classification_result['predicted_class']}")
print(f"Category Code: {classification_result['category_code']}")
print(f"ACM Subject Names: {', '.join(classification_result['acm_subject_names'])}")
print(f"Confidence Score: {classification_result['confidence_score']:.2f}")
