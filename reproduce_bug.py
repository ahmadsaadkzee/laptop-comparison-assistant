
def check_validation(entity, result_title):
    required_term = entity.lower()
    item_lower = result_title.lower()
    
    print(f"Checking entity '{entity}' against title '{result_title}'")
    
    if required_term not in item_lower:
        print("❌ FAIL: Exact substring match failed.")
    else:
        print("✅ PASS: Exact substring match succeeded.")

    # Proposed Fix: Token-based matching
    tokens = required_term.split()
    if all(t in item_lower for t in tokens):
        print("✅ PROPOSED FIX: Token-based match succeeded.")
    else:
        print("❌ PROPOSED FIX: Token-based match failed.")

print("--- Scenario 1: Exact Match ---")
check_validation("Dell N5110", "Review of the Dell N5110 Laptop")

print("\n--- Scenario 2: Interpolated Word (Real World) ---")
check_validation("Dell N5110", "Dell Inspiron 15R N5110 Specs")

print("\n--- Scenario 3: Mixed Order ---")
check_validation("Samsung S24", "S24 Samsung Galaxy Review")
