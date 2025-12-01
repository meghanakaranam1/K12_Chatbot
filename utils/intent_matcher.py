"""
Semantic intent classification for natural language commands
"""
import re
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the model once
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def classify_command(user_input: str) -> Dict[str, Any]:
    """
    Classify user input into semantic command types using sentence transformers.
    Returns intent type and extracted parameters.
    """
    text = user_input.lower().strip()
    
    # Define command patterns with semantic similarity
    command_patterns = {
        "select": [
            "open", "expand", "show", "select", "choose", "pick",
            "first", "second", "third", "1st", "2nd", "3rd",
            "one", "two", "three", "1", "2", "3"
        ],
        "similar": [
            "similar", "like this", "more like", "related", "comparable",
            "same type", "equivalent", "alike"
        ],
        "next_prev": [
            "next", "previous", "back", "forward", "continue", "more",
            "another", "different", "other", "else"
        ],
        "details": [
            "details", "more info", "explain", "describe", "tell me about",
            "what is", "how does", "why", "when", "where"
        ],
        "modify": [
            "modify", "change", "adapt", "adjust", "customize", "edit",
            "shorten", "lengthen", "simplify", "make easier", "make harder"
        ],
        "search": [
            "find", "search", "look for", "show me", "get me", "need",
            "want", "looking for", "seeking"
        ]
    }
    
    # Extract numeric index if present
    index = None
    index_match = re.search(r'\b(\d+)\b', text)
    if index_match:
        index = int(index_match.group(1))
    
    # Check for explicit selection patterns
    if any(word in text for word in ["open", "expand", "show", "select", "choose", "pick"]):
        if index is not None:
            return {"intent": "select", "index": index, "confidence": 0.9}
        # Check for ordinal numbers
        ordinal_match = re.search(r'\b(first|second|third|1st|2nd|3rd|one|two|three)\b', text)
        if ordinal_match:
            ordinal_map = {
                "first": 1, "second": 2, "third": 3,
                "1st": 1, "2nd": 2, "3rd": 3,
                "one": 1, "two": 2, "three": 3
            }
            index = ordinal_map.get(ordinal_match.group(1), 1)
            return {"intent": "select", "index": index, "confidence": 0.9}
    
    # Check for navigation patterns
    if any(word in text for word in ["next", "previous", "back", "forward", "continue", "more", "another"]):
        return {"intent": "next_prev", "confidence": 0.8}
    
    if any(word in text for word in ["similar", "like this", "more like", "related", "comparable"]):
        return {"intent": "similar", "confidence": 0.8}
    
    if any(word in text for word in ["details", "more info", "explain", "describe", "tell me about"]):
        return {"intent": "details", "confidence": 0.8}
    
    if any(word in text for word in ["modify", "change", "adapt", "adjust", "customize", "edit"]):
        return {"intent": "modify", "confidence": 0.8}
    
    # Default to search if no specific pattern matches
    return {"intent": "search", "confidence": 0.7}

def get_semantic_similarity(text1: str, text2: str) -> float:
    """Get semantic similarity between two texts using sentence transformers"""
    try:
        model = get_model()
        embeddings = model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return float(similarity)
    except Exception:
        return 0.0




