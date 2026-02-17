import re

# Mock of the existing extractor from rag.py to test improvements
def _resolve_coreferences(query, history):
    # Current logic (simplified copy from rag.py for testing)
    # We want to improve this to handle "both"
    if not history: return query
    
    # ... logic to find last user msg ...
    last_user_msg = "compare hp spectre with macbook pro" # Mock
    
    q_lower = query.lower()
    
    # NEW LOGIC TO TEST
    if "both" in q_lower:
        # Simplistic extraction from "compare A with B"
        # In reality we should use the same extraction logic
        # For this test, assume we can get [A, B] from history
        
        # Mock extraction from history
        prev_entities = ["hp spectre", "macbook pro"] 
        
        # Replace "both" with "hp spectre and macbook pro"
        replacement = " and ".join(prev_entities)
        query = re.sub(r'\bboth\b', replacement, query, flags=re.IGNORECASE)
        print(f"DEBUG: Resolved 'both' to '{replacement}'")
        
    return query

def _extract_entities_from_query(query):
    # New logic to return LIST of entities
    q = query.lower()
    
    # Handle typo "comapre"
    q = q.replace("comapre", "compare")
    
    # Remove "compare ", "comparison of ", "tell me about "
    for prefix in ["compare ", "comparison of ", "tell me about ", "contrast "]:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
            
    # Split by delimiters: vs, with, and, comma
    # Regex split
    # "hp spectre and macbook pro with dell n5110"
    # "hp spectre, macbook pro, and dell n5110"
    
    tokens = re.split(r'\s+(?:vs\.?|v\.?|with|and|,)\s+', q)
    entities = [t.strip() for t in tokens if t.strip() and t.strip() not in ["between"]]
    
    return entities

# Test
history = [{"role": "user", "content": "compare hp spectre with macbook pro"}]
query = "comapre both with dell n5110"

print(f"Original: {query}")
resolved = _resolve_coreferences(query, history)
print(f"Resolved: {resolved}")

entities = _extract_entities_from_query(resolved)
print(f"Extracted: {entities}")

# Expected: ['hp spectre', 'macbook pro', 'dell n5110']
