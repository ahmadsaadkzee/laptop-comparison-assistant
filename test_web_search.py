from duckduckgo_search import DDGS
import time

def web_search(query):
    print(f"DEBUG: Searching for '{query}'...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            print(f"DEBUG: Found {len(results)} results.")
            for r in results:
                print(f" - {r.get('title', 'No Title')}: {r.get('href', 'No URL')}")
            return results
    except Exception as e:
        print(f"DEBUG: Search failed: {e}")
        return []

if __name__ == "__main__":
    q = "dell inspiron n5110 specs"
    web_search(q)
    
    q2 = "Dell Inspirion 15 5502 specs"
    web_search(q2)
