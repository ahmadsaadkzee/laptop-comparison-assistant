import rag
import os

# Set env to ensure fallback is on (matching prod)
os.environ["FORCE_LOCAL_FALLBACK"] = "1"

query = "compare dell inspiron n5110 vs Dell Inspirion 15 5502"
print(f"DEBUG: Query: {query}")

# 1. Test Extraction
ent_a, ent_b = rag._extract_entities_from_query(query)
print(f"DEBUG: Extracted: A='{ent_a}', B='{ent_b}'")

# 2. Test Local Retrieval
print("DEBUG: Running local retrieval...")
docs = rag.local_retrieval(query)
context = "\n".join(d.page_content for d in docs)
print(f"DEBUG: Context length: {len(context)}")
print(f"DEBUG: Context snippet: {context[:100]}...")

# 3. Test Missing Logic
missing = []
if ent_a and ent_a.lower() not in context.lower():
    missing.append(ent_a)
    print(f"DEBUG: '{ent_a}' marked as missing.")
else:
    print(f"DEBUG: '{ent_a}' found in context (or None).")

if ent_b and ent_b.lower() not in context.lower():
    missing.append(ent_b)
    print(f"DEBUG: '{ent_b}' marked as missing.")
else:
    print(f"DEBUG: '{ent_b}' found in context (or None).")

# 4. Test Web Search specifically
if missing:
    print(f"DEBUG: Attempting web search for: {missing}")
    for m in missing:
        print(f"DEBUG: Searching web for '{m}'...")
        res = rag.web_search(m)
        print(f"DEBUG: Web search result length: {len(res)}")
        if not res:
            print("DEBUG: Web search returned EMPTY string!")
            # Try to debug WHY
            try:
                from duckduckgo_search import DDGS
                print("DEBUG: Testing raw DDGS...")
                with DDGS() as ddgs:
                    results = list(ddgs.text(m, max_results=3))
                    print(f"DEBUG: DDGS raw results: {len(results)}")
            except Exception as e:
                print(f"DEBUG: DDGS raw test failed: {e}")
        else:
             print(f"DEBUG: Web search success. Snippet: {res[:50]}...")
else:
    print("DEBUG: No missing entities detected, skipping web search.")
