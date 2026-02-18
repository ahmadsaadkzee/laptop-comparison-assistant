import os
import re
import traceback
from typing import List, Tuple, Optional, Any

# Third-party imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Langchain / Embeddings
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    Chroma = None
    HuggingFaceEmbeddings = None

# Web Search (DDGS) - Try new package, fallback to old
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# Constants
DB_PATH = "chroma_db"
# Fallback GROQ key (hardcoded per user request as backup)
os.environ.setdefault('GROQ_API_KEY', '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')

# Global Vector DB Reference
_db = None

def get_local_docs_with_fallback(query: str, k: int = 4) -> List[Any]:
    """
    Retrieve documents using Chroma DB if available.
    Falls back to simple keyword search over markdown files in 'data/laptops' if Chroma fails or is empty.
    """
    global _db
    
    # 1. Try Chroma Retrieval
    try:
        # Check env var to force fallback (for testing/debugging)
        if os.getenv('FORCE_LOCAL_FALLBACK') or os.getenv('CHROMA_FORCE_FALLBACK'):
            raise RuntimeError('Forced local fallback')
            
        # Only attempt if DB directory exists and has content
        if os.path.isdir(DB_PATH) and any(os.scandir(DB_PATH)):
            if _db is None:
                if Chroma is None or HuggingFaceEmbeddings is None:
                    raise ImportError("LangChain/Chroma dependencies missing")
                    
                # Lazy load embeddings to avoid startup overhead if not needed
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                _db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            
            return _db.similarity_search(query, k=k)
    except Exception:
        # Silently fail back to file search
        pass

    # 2. Fallback: Simple File Search
    # This runs if Chroma fails, is empty, or is forced off.
    scores = []
    qterms = [t.lower() for t in query.split() if t.strip()]
    
    # Robust path calculation for Cloud environment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data/laptops')
    
    if not os.path.exists(data_dir):
        return []

    # Walk through markdown files
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith('.md'):
                continue
            
            p = os.path.join(root, fname)
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    text = fh.read()
            except Exception:
                continue
                
            # access content
            txt_lower = text.lower()
            score = sum(txt_lower.count(t) for t in qterms)
            
            if score > 0:
                # Create a lightweight object mimicking LangChain Document
                class LightweightDoc:
                    def __init__(self, page_content, metadata):
                        self.page_content = page_content
                        self.metadata = metadata
                
                scores.append((score, LightweightDoc(text, {"source": p})))
    
    # Sort by score descending and return top k
    scores.sort(key=lambda x: x[0], reverse=True)
    return [d for _s, d in scores[:k]]


def get_llm():
    """
    Lazily import and return an LLM callable. 
    Returns: (llm_callable, error_message or None)
    Prioritizes GROQ, then OpenAI.
    """
    
    def _extract_content(resp):
        """Helper to extract text content from various LLM response formats."""
        try:
            # 1. Object-like with 'choices' (OpenAI-style object)
            choices = getattr(resp, 'choices', None)
            if choices:
                first = choices[0]
                if hasattr(first, 'message'):
                    msg = first.message
                    return getattr(msg, 'content', getattr(msg, 'text', None) or str(msg))
                if hasattr(first, 'text'):
                    return first.text

            # 2. Dict-like or Pydantic model dump
            try:
                d = resp if isinstance(resp, dict) else getattr(resp, '__dict__', {})
            except Exception:
                d = {}
                
            if isinstance(d, dict):
                ch = d.get('choices') or d.get('outputs')
                if ch:
                    c0 = ch[0]
                    if isinstance(c0, dict):
                        m = c0.get('message') or c0.get('text') or c0.get('output')
                        if isinstance(m, dict):
                            return m.get('content') or m.get('text') or str(m)
                        return m
            return str(resp)
        except Exception:
            return repr(resp)

    # 1. Try GROQ (Preferred)
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        try:
            import groq
            client = groq.Groq(api_key=groq_key)

            def _groq_call(prompt, system=None, model='llama-3.3-70b-versatile'):
                msgs = []
                if system:
                    msgs.append({'role': 'system', 'content': system})
                msgs.append({'role': 'user', 'content': prompt})
                
                resp = client.chat.completions.create(model=model, messages=msgs)
                return _extract_content(resp)

            return _groq_call, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, f"GROQ setup failed: {e}\n{tb}"

    # 2. Try OpenAI
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key

            def _openai_call(prompt, system=None, model='gpt-3.5-turbo'):
                msgs = []
                if system:
                    msgs.append({'role': 'system', 'content': system})
                msgs.append({'role': 'user', 'content': prompt})
                
                resp = openai.ChatCompletion.create(model=model, messages=msgs)
                try:
                    return resp['choices'][0]['message']['content']
                except Exception:
                    return _extract_content(resp)

            return _openai_call, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, f"OpenAI setup failed: {e}\n{tb}"

    return None, "No GROQ_API_KEY or OPENAI_API_KEY found."


def web_search(query: str) -> str:
    """
    Search the web for `query` using multiple strategies:
    1. `ddgs` Python package (Primary)
    2. `duckduckgo_search` legacy package (Secondary)
    3. HTML scraping via requests (Final Fallback)
    
    Returns a formatted string of results or empty string if failed.
    """
    import html
    
    def _format_item(d):
        if isinstance(d, dict):
            parts = []
            # Title
            t = d.get('title') or d.get('text') or d.get('snippet')
            if t: parts.append(f"Title: {t}")
            # Body/Snippet
            b = d.get('body') or d.get('snippet') or d.get('text')
            if b: parts.append(f"Snippet: {b}")
            # URL
            href = d.get('href') or d.get('url') or d.get('source')
            if href: parts.append(f"URL: {href}")
            
            return " | ".join(parts) if parts else str(d)
        return str(d)

    tried = set()
    combined_results = []

    def _try_single_query(q, max_results=5):
        if not q or q in tried:
            return []
        tried.add(q)
        got = []
        
        # Strategy 1: DDGS (New)
        if DDGS:
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(q, max_results=max_results):
                        got.append(_format_item(r))
            except Exception as e:
                print(f"WEB_SEARCH ERROR (DDGS Primary): {e}")

        # Strategy 2: Google Search (Fallback 1)
        if not got:
            try:
                from googlesearch import search
                count = 0
                for url in search(q, num_results=max_results, advanced=True):
                    title = "Google Result"
                    snippet = "See URL for details"
                    href = url
                    
                    if hasattr(url, 'title'): title = url.title
                    if hasattr(url, 'description'): snippet = url.description
                    if hasattr(url, 'url'): href = url.url
                    
                    got.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}")
                    count += 1
                    if count >= max_results: break
            except Exception as e:
                print(f"WEB_SEARCH ERROR (Google Fallback): {e}")

        # Strategy 3: Old DDGS (Fallback 2)
        if not got:
            try:
                from duckduckgo_search import DDGS as OldDDGS
                with OldDDGS() as ddgs:
                    for r in ddgs.text(q, max_results=max_results):
                        got.append(_format_item(r))
            except Exception:
                pass

        # Strategy 4: HTML Scraping (Last Resort)
        if not got:
            try:
                import requests
                from urllib.parse import unquote
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                payload = {'q': q}
                resp = requests.post("https://html.duckduckgo.com/html/", data=payload, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    text = resp.text
                    links = re.findall(r'<a class="result__a" href="([^"]+)">([^<]+)</a>', text)
                    count = 0
                    for href, title in links:
                        if count >= max_results: break
                        if "uddg=" in href:
                             try:
                                 href = unquote(href.split("uddg=")[1].split("&")[0])
                             except Exception: pass
                        snippet = "Fallback search result"
                        start_idx = text.find(href)
                        if start_idx != -1:
                             snippet_match = re.search(r'<a class="result__snippet"[^>]*>(.*?)</a>', text[start_idx:])
                             if snippet_match:
                                 snippet = html.unescape(re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip())
                        got.append(f"Title: {html.unescape(title)}\nURL: {href}\nSnippet: {snippet}")
                        count += 1
            except Exception as e:
                 print(f"WEB_SEARCH ERROR (HTML Fallback): {e}")

        return got

    # Execute search with prioritized variants
    # Ensure 'laptop' context is present to disambiguate model numbers (e.g. "N5110" -> TV channel vs Laptop)
    base_query = query
    query_lower = query.lower()
    
    if "laptop" not in query_lower:
        base_query = f"{query} laptop"

    # Special handling for generic Apple queries to avoid "rumor" pages about M5/M6
    if "macbook" in query_lower and not any(c in query_lower for c in ['m1', 'm2', 'm3', 'm4', 'air', 'pro']):
        # If user just says "macbook", default to latest widely available logic
        base_query += " m3 specifications"
    elif "macbook pro" in query_lower and not any(c in query_lower for c in ['m1', 'm2', 'm3', 'm4']):
        base_query += " m3 pro specifications"
    elif "macbook air" in query_lower and not any(c in query_lower for c in ['m1', 'm2', 'm3']):
        base_query += " m3 specifications"

    variants = [
        f"{base_query} specs", 
        f"{base_query} specifications", 
        f"{base_query} review"
    ]
    
    for v in variants:
        # Debug check
        print(f"DEBUG: Searching Google for '{v}'")
        combined_results.append(f"DEBUG: Searching for '{v}'")
        items = _try_single_query(v, max_results=5)
        
        # Filter out junk results
        valid_items = []
        
        # Robust Keyword Validation:
        # Instead of using the raw query (which might be "tell me about macbook pro laptop m3 specs"),
        # we extract the core entity (e.g. "macbook pro") and check for that.
        extracted_entities = _extract_all_entities(query)
        required_term = extracted_entities[0].lower() if extracted_entities else None
        
        # If extraction failed (e.g. generic query), fall back to a simpler check
        if not required_term:
             # Remove common "command" words to find something significant
             cleaned = query.lower().replace("tell me about", "").replace("compare", "").strip()
             required_term = cleaned.split()[0] if cleaned else "laptop"

        for item in items:
            # Junk titles
            item_lower = item.lower()
            if "title: google" in item_lower or "title: sign in" in item_lower or "title: acne" in item_lower:
                continue
            
            # Strict Validation
            # The result MUST contain the specific entity name matches
            # We check if ALL tokens of the required term are present in the item
            # e.g. "Dell N5110" -> "dell", "n5110". Result "Dell Inspiron N5110" contains both.
            required_tokens = required_term.split()
            if not all(t in item_lower for t in required_tokens):
                # Double check: sometimes model numbers are joined or split differently
                # But for high precision, we require the tokens to be there.
                continue
                
            valid_items.append(item)
            
        if valid_items:
            combined_results.extend(valid_items)
            
        if len(combined_results) >= 8:
            break
            
    # Fallback: If Google failed to give VALID results, try strict DuckDuckGo
    if len(combined_results) <= 1: # Only header debug line
        print("DEBUG: Google failed validation, falling back to DDGS")
        combined_results.append("DEBUG: Falling back to DDGS")
        try:
             # Try DDGS with the specific variants
             if DDGS:
                with DDGS() as ddgs:
                    for v in variants:
                        for r in ddgs.text(v, max_results=3):
                             combined_results.append(_format_item(r))
        except Exception as e:
            print(f"Fallback DDGS Error: {e}")

    if not combined_results:
        return ""

    # Deduplicate
    seen = set()
    final_output = []
    for item in combined_results:
        if item not in seen:
            seen.add(item)
            final_output.append(item)

    return "[Web search results]\n" + "\n---\n".join(final_output)


def _extract_all_entities(query: str) -> List[str]:
    """
    Extract a list of entities (e.g. laptop names) from a query.
    Handles 'compare A, B, and C', 'A vs B', etc.
    """
    if not query:
        return []
    
    q = query.lower()
    q = q.replace("comapre", "compare") # Typo fix
    
    # Remove common command prefixes
    prefixes = ["compare ", "comparison of ", "tell me about ", "what is ", "review ", 
                "specs for ", "specifications for ", "difference between "]
    for p in prefixes:
        if q.startswith(p):
            q = q[len(p):]
            break
            
    # Split by delimiters (vs, and, with, comma)
    tokens = re.split(r'\s+(?:vs\.?|v\.?|with|and|,|versus)\s+', q)
    
    entities = []
    seen = set()
    for t in tokens:
        clean = t.strip(" \t\n\r'\".,")
        if clean and clean not in seen and clean not in ["between"]:
            entities.append(clean)
            seen.add(clean)
            
    return entities


def _resolve_coreferences(query: str, history: List[dict]) -> str:
    """
    Enhanced coreference resolution:
    - Replaces 'it', 'this', 'that' with the subject of the previous query.
    - Replaces 'both', 'these', 'all of them' with ALL entities from the previous query.
    """
    if not history:
        return query
        
    # valid history is list of dicts with 'role' and 'content'
    last_user_msg = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content")
            break
            
    if not last_user_msg:
        return query
        
    q_lower = query.lower()
    prev_entities = _extract_all_entities(last_user_msg)
    
    if not prev_entities:
        return query

    # 1. Handle Plural/Group references ("both", "these")
    group_markers = ["both", "these", "all of them", "all"]
    if any(m in q_lower for m in group_markers):
        replacement = " and ".join(prev_entities)
        for marker in group_markers:
            pattern = r'\b' + re.escape(marker) + r'\b'
            if re.search(pattern, q_lower):
                query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

    # 2. Handle Singular references ("it", "this")
    # If multiple entities exist, map "it" to the FIRST one (heuristic)
    singular_markers = ["it", "this", "that"]
    if any(re.search(r'\b' + re.escape(m) + r'\b', q_lower) for m in singular_markers):
        primary_subject = prev_entities[0]
        for marker in singular_markers:
            query = re.sub(r'\b' + re.escape(marker) + r'\b', primary_subject, query, flags=re.IGNORECASE)

    return query


def answer_question(query: str, history: List[dict] = []) -> Tuple[str, str, str]:
    """
    Main RAG pipeline entry point.
    
    Steps:
    1. Resolve coreferences (handle "it", "both").
    2. Local Retrieval (Chroma/Markdown).
    3. Check sufficiency (do we have 2+ docs?).
    4. Web Search fallback/augmentation if needed.
    5. Entity Completeness Check (did we miss one laptop?).
    6. LLM Generation.
    
    Returns: (final_answer, retrieval_source, resolved_query)
    """
    # 1. Coreference Resolution
    resolved_query = _resolve_coreferences(query, history)

    # 2. Local Retrieval
    docs = get_local_docs_with_fallback(resolved_query)

    # 3. Determine Source
    if len(docs) >= 2:
        context = "\n".join(d.page_content for d in docs)
        source = "local documents"
    else:
        context = web_search(resolved_query)
        source = "web search"

    # 4. Consistency Check: Did local search miss explicit entities?
    if source == "local documents":
        entities = _extract_all_entities(resolved_query)
        missing = []
        
        ctx_lower = context.lower()
        # Simplified context for fuzzy matching (remove spaces/dashes)
        ctx_simplified = ctx_lower.replace(" ", "").replace("-", "").replace(".", "")

        for ent in entities:
            # Check normal match
            if ent.lower() in ctx_lower: continue
            
            # Check simplified match
            e_simp = ent.lower().replace(" ", "").replace("-", "").replace(".", "")
            if len(e_simp) > 2 and e_simp in ctx_simplified: continue
            
            missing.append(ent)

        if missing:
            # Perform specific web searches for missing entities
            web_contexts = []
            for m in missing:
                w = web_search(m)
                if w:
                    web_contexts.append(f"[Web search for '{m}']\n{w}")

            if web_contexts:
                context += "\n\n" + "\n".join(web_contexts)
                # Update source label
                local_found = [e for e in entities if e not in missing]
                src_parts = []
                if local_found:
                    src_parts.append(f"Local ({', '.join(local_found)})")
                else:
                    src_parts.append("Local (Partial)")
                src_parts.append(f"Web ({', '.join(missing)})")
                source = ", ".join(src_parts)
            
            # Final safety net: if we still have missing info and no specific web results, try full query
            if len(web_contexts) == 0 and len(missing) > 0:
                w = web_search(resolved_query)
                if w:
                    context += "\n\n[Web search for full query]\n" + w
                    source += " + Web (Full Query)"

    # 5. Construct Prompt
    system_prompt = (
        "You are a professional laptop reviewer. When asked to compare laptops, provide a detailed, "
        "side-by-side comparison of key specifications (Processor, RAM, Storage, Display, Battery, OS, etc.). "
        "Use all the context provided (including web search results) to fill in the details. "
        "Do NOT apologize for missing information. If a specific spec is absolutely not found in the context, "
        "briefly mention 'Not specified' for that row, but do not start your response with a disclaimer. "
        "OUTPUT FORMATTING:\n"
        "1. If the user explicitly asks to COMPARE laptops (e.g., 'compare X and Y', 'vs', 'difference between'), "
        "   format the output as a MARKDOWN TABLE. The first column should be the Specification, "
        "   and subsequent columns should be the laptops being compared.\n"
        "2. If the user asks for general information, a single laptop review, or a follow-up question that isn't a direct comparison, "
        "   provide a natural language TEXT response. Do NOT use a table unless requested.\n"
        "Pay close attention to the Conversation History. If the user asks a follow-up question (e.g., 'which is better?') "
        "or asks about what was discussed previously (e.g., 'what laptops did I ask about?'), "
        "refer to the Conversation History to answer accurately.\n"
        "STRICTLY limit your response to the laptops explicitly requested by the user. "
        "Do NOT provide comparisons for other laptops found in the context unless asked. "
        "Do NOT offer 'extra suggestions' or 'other laptops you might like'.\n"
        "CRITICAL: If the user asks for a specific model (e.g. 'Dell N510') and you cannot find info for it, "
        "DO NOT SUBSTITUTE it with a similar model (e.g. 'Dell XPS'). "
        "Just state that you have no data for the requested model or leave its column as 'Not Found'. "
        "It is better to say 'Not Found' than to provide the wrong laptop.\n"
        "DOMAIN RESTRICTION: You are a LAPTOP expert. If the user asks to compare mobile phones, tablets, or other non-laptop devices, "
        "politely decline and state that you only compare laptops. Do NOT generate comparisons for phones."
    )

    # Format History
    formatted_history = ""
    if history:
        for msg in history[-5:]: # Keep last 5 turns
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_history += f"{role.capitalize()}: {content}\n"

    full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nConversation History:\n{formatted_history}\n\nQuestion:\n{resolved_query}"

    # 6. Call LLM
    llm_func, err_msg = get_llm()
    if err_msg or not llm_func:
        return f"System Error: {err_msg or 'LLM Unavailable'}", source, context

    try:
        response = llm_func(full_prompt)
        return response, source, context
    except Exception as e:
        return f"Error generating response: {str(e)}", source, context
