
import rag
import time

def run_conversation_test(scenario_name, conversation_flow):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'-'*60}")
    
    history = []
    
    for i, user_input in enumerate(conversation_flow):
        print(f"\nTurn {i+1}: User asks '{user_input}'")
        
        # We need to simulate the resolved query logic that happens inside answer_question
        # But answer_question does it internally, so we just call it and see the result.
        
        start = time.time()
        response, source, context = rag.answer_question(user_input, history)
        duration = time.time() - start
        
        print(f"   -> Source: {source}")
        print(f"   -> Time: {duration:.2f}s")
        # print(f"   -> Response Snippet: {response[:100]}...")
        
        # Check for key phrases to verify context
        if i > 0:
            resolved = rag._resolve_coreferences(user_input, history)
            print(f"   -> Resolved Query (Internal): {resolved}")
            
        # Update history for next turn
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
    print(f"{'='*60}\n")

# Scenario 1: Simple Follow-up
# Expectation: "it" in Turn 2 refers to "Dell XPS 13"
run_conversation_test("1. Simple Follow-up", [
    "Tell me about Dell XPS 13",
    "How is its battery life?"
])

# Scenario 2: Comparison Follow-up
# Expectation: "both" in Turn 2 refers to "HP Spectre" and "Dell XPS"
run_conversation_test("2. Comparison Follow-up", [
    "Compare HP Spectre x360 and Dell XPS 13",
    "What are both of their display sizes?"
])

# Scenario 3: Pivot / Topic Switch
# Expectation: Turn 3 "it" refers to "MacBook Air", NOT "Dell XPS"
run_conversation_test("3. Topic Switch", [
    "Tell me about Dell XPS 13",
    "Actually, tell me about MacBook Air M2 instead",
    "Does it have a fan?"
])

# Scenario 4: Implicit Comparison
# Expectation: Turn 2 "it" refers to "N5110", comparing against "Spectre"
run_conversation_test("4. Implicit Comparison", [
    "Tell me about Dell N5110",
    "Compare it with HP Spectre x360"
])
