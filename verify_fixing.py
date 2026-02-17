import rag
import os

# Force fallback to match production environment
os.environ["FORCE_LOCAL_FALLBACK"] = "1"

query = "compare dell inspiron n5110 vs Dell Inspirion 15 5502"
print(f"TESTING QUERY: {query}\n")

print("--- STEP 1: RETRIEVAL & CONTEXT ---")
try:
    # We call the internal logic steps manually to inspect the context before LLM
    docs = rag.local_retrieval(query)
    context = "\n".join(d.page_content for d in docs)
    
    # Hybrid logic simulation
    ent_a, ent_b = rag._extract_entities_from_query(query)
    print(f"Entities extracted: {ent_a}, {ent_b}")
    
    missing = []
    if ent_a and ent_a.lower() not in context.lower(): missing.append(ent_a)
    if ent_b and ent_b.lower() not in context.lower(): missing.append(ent_b)
    
    print(f"Missing entities: {missing}")
    
    if missing:
        for m in missing:
            print(f"Searching web for: {m}")
            web_res = rag.web_search(m)
            print(f"Web result length: {len(web_res)}")
            print(f"Web result snippet: {web_res[:200]}...")
            if web_res:
                context += f"\n\n[Web search for '{m}']\n{web_res}"

    print(f"\nFINAL CONTEXT SENT TO LLM (first 500 chars):\n{context[:500]}\n...")
except Exception as e:
    print(f"Error during context building: {e}")

print("\n--- STEP 2: LLM GENERATION ---")
try:
    # Now call the actual full function
    answer, source = rag.answer_question(query)
    print(f"Source: {source}")
    print(f"FINAL ANSWER:\n{answer}")
except Exception as e:
    print(f"Error during generation: {e}")
