
import rag
import time

def run_test(name, query, history=[]):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"QUERY: {query}")
    print(f"HISTORY: {history}")
    print(f"{'-'*60}")
    
    start = time.time()
    response, source, context = rag.answer_question(query, history)
    duration = time.time() - start
    
    print(f"SOURCE: {source}")
    print(f"TIME: {duration:.2f}s")
    print(f"CONTEXT LENGTH: {len(context)}")
    print(f"RESPONSE SNIPPET: {response[:200]}...")
    print(f"{'='*60}\n")

# 1. Edge Case: Old specific model (Tests Entity Extraction + Junk Filter)
run_test("1. Old Specific Model", "Tell me about Dell N5110")

# 2. Edge Case: Generic Apple Query (Tests Auto-Append 'M3' logic)
run_test("2. Generic Apple Query", "Tell me about MacBook Pro")

# 3. Edge Case: Typo + Multi-Entity (Tests 'comapre' fix + extraction)
run_test("3. Typo + Multi-Entity", "Comapre Macbok Air M2 and Dell XPS 13 Plus")

# 4. Edge Case: Fake Model (Tests Hallucination/Not Found)
run_test("4. Non-Existent Model", "Tell me about the Foobar Laptop 3000")

# 5. Edge Case: Domain Restriction (Tests 'Laptops Only' prompt)
run_test("5. Domain Restriction", "Compare iPhone 15 and Samsung S24")

# 6. Edge Case: Coreference 'It' (Tests Context Retention)
history = [{"role": "user", "content": "Tell me about the HP Spectre x360"}]
run_test("6. Coreference (It)", "How is its battery life?", history)

# 7. Edge Case: Coreference 'Both' (Tests Multi-Entity Coref)
history_both = [{"role": "user", "content": "Compare XPS 13 and Spectre x360"}]
run_test("7. Coreference (Both)", "What are both of their screen sizes?", history_both)

# 8. Edge Case: Ambiguous / Broad (Tests Search Quality)
run_test("8. Broad Query", "What is the best gaming laptop 2024?")

# 9. Edge Case: Missing Entity in Local DB (Tests Hybrid Fallback)
# Assuming 'Lenovo Legion 9i' is not in local Markdown but valid on web
run_test("9. Hybrid Fallback", "Compare Dell XPS 13 and Lenovo Legion 9i")

# 10. Edge Case: Junk Term Filtering (search for 'Acne' or 'Google')
# We want to ensure we don't get junk. Using a query that historically triggered junk.
run_test("10. Generic/Nav Query", "Tell me about supportassist") 
