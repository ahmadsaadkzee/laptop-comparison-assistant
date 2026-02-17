import sys
import os
import time

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

print("Starting debug script...")

try:
    print("Importing rag module...")
    import rag
    print("rag module imported.")
except Exception as e:
    print(f"Failed to import rag: {e}")
    sys.exit(1)

query = "Which laptop is best for coding?"
print(f"Testing with query: '{query}'")

# Test 1: Local Retrieval
print("\n--- Test 1: Local Retrieval ---")
start_time = time.time()
try:
    docs = rag.local_retrieval(query)
    print(f"Retrieved {len(docs)} documents.")
    for i, doc in enumerate(docs):
        print(f"Doc {i+1}: {(doc.page_content if hasattr(doc, 'page_content') else str(doc))[:50]}...")
except Exception as e:
    print(f"Local retrieval failed: {e}")
print(f"Time taken: {time.time() - start_time:.2f}s")


# Test 2: LLM Connection
print("\n--- Test 2: LLM Connection ---")
start_time = time.time()
try:
    llm, err = rag.get_llm()
    if err:
        print(f"LLM setup error: {err}")
    else:
        print("LLM callable obtained.")
        # Test a simple prompt
        print("Testing simple prompt...")
        response = llm("Hello, are you working?")
        print(f"LLM Response: {response}")
except Exception as e:
    print(f"LLM test failed: {e}")
print(f"Time taken: {time.time() - start_time:.2f}s")

# Test 3: Full Pipeline
print("\n--- Test 3: Full Pipeline ---")
start_time = time.time()
try:
    answer, source = rag.answer_question(query)
    print(f"Answer: {answer[:100]}...")
    print(f"Source: {source}")
except Exception as e:
    print(f"Full pipeline failed: {e}")
print(f"Time taken: {time.time() - start_time:.2f}s")
