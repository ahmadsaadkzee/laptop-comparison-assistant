
import sys
import os

# Ensure we can import rag.py
sys.path.append(os.getcwd())

from rag import web_search, answer_question

# Test 1: Web Search Query Construction
print("Test 1: Web Search Query Construction")
region = "Pakistan"
query = "Dell N5110"
res = web_search(query, region=region)
print(f"Searching for '{query}' in '{region}'...")
# We can't easily see the internal query, but we can check if the results contain PKR or local sites
if "PKR" in res or "pakistan" in res.lower() or "olx" in res.lower() or "daraz" in res.lower():
    print("SUCCESS: Web search results contain local context/currency.")
else:
    print("WARNING: Web search results might not be localized. Check debug output.")
print(res[:500] + "..." if len(res) > 500 else res)
print("-" * 20)

# Test 2: Full RAG Answer
print("\nTest 2: Full RAG Answer")
ans, source, ctx = answer_question("Price of Dell N5110", region="Pakistan")
print(f"Answer:\n{ans}")
print("-" * 20)

if "PKR" in ans or "Rs" in ans or "Pakistan" in ans:
    print("SUCCESS: Answer mentions local currency/region.")
else:
    print("FAILURE: Answer does not mention local currency/region.")
