import re

def _extract_entities_from_query(query: str):
    print(f"DEBUG: Extracting from '{query}'")
    if not query:
        return (None, None)
    q = query.strip()
    patterns = [
        # Add explicit compare ... vs ... pattern
        r"compare\s+(.+?)\s+vs\.?\s+(.+)$",
        r"compare\s+(.+?)\s+with\s+(.+)$",
        r"(.+?)\s+vs\.?\s+(.+)$",
        r"(.+?)\s+v[s]?\.?\s+(.+)$",
        r"compare\s+(.+?)\s+and\s+(.+)$",
        r"(.+?)\s+versus\s+(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            a = m.group(1).strip(" \t\n\r'\"")
            b = m.group(2).strip(" \t\n\r'\"")
            # Cleanup 'compare' if caught in the first group by generic pattern
            if a.lower().startswith("compare "):
                a = a[8:].strip()
            return (a, b)
    
    # Fallback split logic
    for sep in [" vs ", " v ", " vs. ", " - "]:
        if sep in q.lower():
            parts = q.split(sep)
            if len(parts) >= 2:
                a = parts[0].strip()
                if a.lower().startswith("compare "):
                    a = a[8:].strip()
                return (a, parts[1].strip())
    
    return (None, None)

def check_context(ent_a, ent_b, context):
    missing = []
    if ent_a and ent_a.lower() not in context.lower():
        missing.append(ent_a)
    if ent_b and ent_b.lower() not in context.lower():
        missing.append(ent_b)
    return missing

query = "compare dell inspiron n5110 vs Dell Inspirion 15 5502"
a, b = _extract_entities_from_query(query)
print(f"Result: '{a}' | '{b}'")

# Simulate context where one 5502 is present but n5110 is not
context = "Specs for Dell Inspiron 15 5502: CPU i7..."
missing = check_context(a, b, context)
print(f"Missing entities: {missing}")
