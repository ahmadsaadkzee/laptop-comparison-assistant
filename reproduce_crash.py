import rag
import os

# Ensure fallback is on
os.environ["FORCE_LOCAL_FALLBACK"] = "1"

query = "tell me about macbook pro"
print(f"Testing query: '{query}'")

try:
    ans, source = rag.answer_question(query, history=[])
    print("Success!")
    print(f"Source: {source}")
except Exception as e:
    print(f"Caught Exception: {e}")
# Test 2: Query that yields None entities but hits logic
query = "tell me about macbook pro"
print(f"\nTesting robustness with None entities...")

# Manually test the is_in_context function with None
try:
    print("Testing is_in_context(None, ...)")
    rag.is_in_context(None, "some context", "somecontext")
    print("is_in_context(None) passed")
except Exception as e:
    print(f"is_in_context crashed: {e}")

# Manually test other functions that might crash
try:
    print("Testing answer_question with weird input")
    # This specific query might return (None, None) from extract
    rag.answer_question("laptops 2024", history=None)
except Exception as e:
    print(f"answer_question crashed: {e}")
    import traceback
    traceback.print_exc()
