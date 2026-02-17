import re

def extract_subject_from_previous_query(prev_query):
    """
    Simulate the extraction logic to find the main entity in the previous query.
    e.g. "tell me about macbook pro" -> "macbook pro"
    """
    if not prev_query:
        return None
    
    q = prev_query.lower()
    
    # "tell me about X"
    if q.startswith("tell me about "):
        return prev_query[14:].strip()
    
    # "what is X"
    if q.startswith("what is "):
        return prev_query[8:].strip()
    
    # "review X"
    if q.startswith("review "):
        return prev_query[7:].strip()
        
    return prev_query.strip()

def resolve_coreferences(query, history):
    """
    Replace 'it', 'this', 'that' with the subject of the last user query.
    """
    if not history:
        return query
        
    # Find last user message
    last_user_msg = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content")
            break
            
    if not last_user_msg:
        return query
        
    # Check for coreference markers
    markers = [" it ", " this ", " that ", " it's "]
    # We also check end of string
    q_lower = query.lower()
    
    has_marker = False
    for m in markers:
        if m in " " + q_lower + " ":
            has_marker = True
            break
            
    if has_marker:
        subject = extract_subject_from_previous_query(last_user_msg)
        if subject:
            print(f"DEBUG: Resolved 'it' to '{subject}'")
            # Simple replacement isn't perfect but works for "compare it with X"
            # Regex replace to handle boundaries?
            # Or just append? "compare [subject] with ..."
            
            # Strategy: If "compare it with", reconstruct command.
            if "compare it with" in q_lower:
                return query.lower().replace("compare it with", f"compare {subject} with")
            if "compare it vs" in q_lower:
                 return query.lower().replace("compare it vs", f"compare {subject} vs")
            if "compare it to" in q_lower:
                 return query.lower().replace("compare it to", f"compare {subject} to")
                 
            # Fallback: simple text replacement
            resolved = re.sub(r'\bit\b', subject, query, flags=re.IGNORECASE)
            return resolved
            
    return query

# Test Cases
history = [
    {"role": "user", "content": "tell me about macbook pro"},
    {"role": "assistant", "content": "The MacBook Pro is..."}
]

print("Test 1:", resolve_coreferences("compare it with hp spectre", history))
print("Test 2:", resolve_coreferences("does it have 16gb ram?", history))

history2 = [
    {"role": "user", "content": "review Dell XPS 13"},
    {"role": "assistant", "content": "..."}
]
print("Test 3:", resolve_coreferences("compare it with Asus Zenbook", history2))
