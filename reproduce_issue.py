import rag
import os

# Force fallback to match production
os.environ["FORCE_LOCAL_FALLBACK"] = "1"

queries = [
    "compare macbook air and mac book pro",
    "compare Dell XPS 13 vs MacBook Air M2"
]

for query in queries:
    print(f"\n--- Testing Query: '{query}' ---")
    
    # 1. Local Retrieval
    docs = rag.local_retrieval(query)
    context = "\n".join(d.page_content for d in docs)
    print(f"Local Docs Found: {len(docs)}")
    print(f"Context Snippet (first 1000 chars):\n{context[:1000]}\n")
    
    # 2. Entity Extraction
    ent_a, ent_b = rag._extract_entities_from_query(query)
    print(f"Entities: '{ent_a}', '{ent_b}'")
    
    # 3. Context Check
    missing = []
    if ent_a:
        found_a = ent_a.lower() in context.lower()
        print(f"Entity '{ent_a}' found in local context? {found_a}")
        if not found_a: missing.append(ent_a)
        
    if ent_b:
        found_b = ent_b.lower() in context.lower()
        print(f"Entity '{ent_b}' found in local context? {found_b}")
        if not found_b: missing.append(ent_b)
        
    print(f"Missing (would trigger web search): {missing}")
    
    # 4. Full Pipeline Check
    answer, source = rag.answer_question(query)
    print(f"Final Reported Source: {source}")

