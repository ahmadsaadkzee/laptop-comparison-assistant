import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import traceback
import re

# Load .env if present and ensure GROQ_API_KEY is available as a fallback
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
# Fallback GROQ key (hardcoded per user request) if not provided in env/.env
os.environ.setdefault('GROQ_API_KEY', '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')

# Prefer the `ddgs` package (new name); fall back to `duckduckgo_search` if necessary.
try:
    from ddgs import DDGS
except Exception:
    try:
        from duckduckgo_search import DDGS
    except Exception:
        DDGS = None


DB_PATH = "chroma_db"

_db = None


def get_local_docs_with_fallback(query, k=4):
    """Try to use Chroma for retrieval; if unavailable, fall back to a simple keyword search over
    markdown files in `DATA_PATH`."""
    global _db
    # Try Chroma first
    try:
        # Only attempt to construct Chroma if the DB directory exists and appears populated.
        import os
        # allow forcing the lightweight file-search fallback to avoid heavy imports
        if os.getenv('FORCE_LOCAL_FALLBACK') or os.getenv('CHROMA_FORCE_FALLBACK'):
             raise RuntimeError('Forced local fallback')
        if os.path.isdir(DB_PATH) and any(os.scandir(DB_PATH)):
            if _db is None:
                # lazy import and construction to avoid heavy deps at module import
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                _db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            return _db.similarity_search(query, k=k)
    except Exception as e:
        print(f"DEBUG: Chroma retrieval failed or skipped: {e}")
        pass

    # Fallback: simple file search (Always run if Chroma didn't return)
    import os
    scores = []
    qterms = [t.lower() for t in query.split() if t.strip()]
    # Use absolute path calculation for robustness in Cloud environment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data/laptops')
    
    if not os.path.exists(data_dir):
        print(f"DEBUG: Data directory not found at {data_dir}")
        return []

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
            txt = text.lower()
            score = sum(txt.count(t) for t in qterms)
            if score > 0:
                # create a lightweight doc-like object compatible with code below
                class D:
                    def __init__(self, page_content, metadata):
                        self.page_content = page_content
                        self.metadata = metadata
                scores.append((score, D(text, {"source": p})))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [d for _s, d in scores[:k]]


def get_llm():
    """Lazily import and return an LLM callable. Returns (llm_callable, None) on success,
    or (None, error_message) on failure."""
    import os

    def _extract_content(resp):
        try:
            # object-like response
            choices = getattr(resp, 'choices', None)
            if choices:
                first = choices[0]
                # common shapes
                if hasattr(first, 'message'):
                    msg = first.message
                    return getattr(msg, 'content', getattr(msg, 'text', None) or str(msg))
                if hasattr(first, 'text'):
                    return first.text

            # dict-like response
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

    # 1) Try GROQ if API key present
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
                # Call the requested GROQ model directly (no model-list fallback)
                resp = client.chat.completions.create(model=model, messages=msgs)
                return _extract_content(resp)

            return _groq_call, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, f'GROQ client setup failed: {e}\n{tb}'

    # 2) Try OpenAI if API key present
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key

            def _openai_call(prompt, system=None, model='gpt-3.5-turbo'):
                messages = []
                if system:
                    messages.append({'role': 'system', 'content': system})
                messages.append({'role': 'user', 'content': prompt})
                resp = openai.ChatCompletion.create(model=model, messages=messages)
                # standard OpenAI response
                try:
                    return resp['choices'][0]['message']['content']
                except Exception:
                    return _extract_content(resp)

            return _openai_call, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, f'OpenAI client setup failed: {e}\n{tb}'

    return None, 'No GROQ_API_KEY or OPENAI_API_KEY found in environment.'

def local_retrieval(query):
    return get_local_docs_with_fallback(query, k=4)

def is_sufficient(docs):
    return len(docs) >= 2

def web_search(query):
    """Search the web for `query` and return a concise combined string of titles/snippets/urls.

    Tries several strategies in order:
    1. `ddgs` package (preferred)
    2. `duckduckgo_search` package as a fallback
    3. Try alternate query forms ("specs", "specifications", "review")
    Returns an empty string if no usable results are found.
    """
    def _format_item(d):
        # Accept dict-like or string items and return a readable block
        if isinstance(d, dict):
            parts = []
            t = d.get('title') or d.get('text') or d.get('snippet')
            if t:
                parts.append(f"Title: {t}")
            b = d.get('body') or d.get('snippet') or d.get('text')
            if b:
                parts.append(f"Snippet: {b}")
            href = d.get('href') or d.get('url') or d.get('source')
            if href:
                parts.append(f"URL: {href}")
            return " | ".join(parts) if parts else str(d)
        return str(d)

    tried = set()
    combined = []

    # Helper: try a single query using ddgs or duckduckgo_search
    def _try_query(q, max_results=5):
        if not q or q in tried:
            return []
        tried.add(q)
        got = []
        
        # 1. Try DDGS (new package)
        if DDGS is not None:
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(q, max_results=max_results):
                        got.append(_format_item(r))
            except Exception as e:
                print(f"DDGS primary failed: {e}")

        # 2. Try duckduckgo_search (old package) if no results
        if not got:
            try:
                from duckduckgo_search import DDGS as OldDDGS
                with OldDDGS() as ddgs:
                    for r in ddgs.text(q, max_results=max_results):
                        got.append(_format_item(r))
            except Exception:
                pass

        # 3. HTML Scraping Fallback (last resort)
        if not got:
                # 3. HTML Scraping Fallback (last resort)
                print(f"DEBUG: Attempting HTML fallback for '{q}'")
                import requests
                from html import unescape
                import re
                
                # Use a real user-agent to avoid being blocked
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                # Use params payload correctly
                payload = {'q': q}
                resp = requests.post("https://html.duckduckgo.com/html/", data=payload, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    text = resp.text
                    # Regex to find links in the HTML result table
                    # Matches: <a class="result__a" href="...">...</a>
                    # Result link format often: <a class="result__a" href="//duckduckgo.com/l/?uddg=...">Title</a>
                    links = re.findall(r'<a class="result__a" href="([^"]+)">([^<]+)</a>', text)
                    
                    count = 0
                    for href, title in links:
                        if count >= max_results: break
                        
                        # Cleanup URL (sometimes it's a redirect wrapper)
                        # If it starts with //duckduckgo.com/l/?uddg=, we need to decode
                        if href.startswith("//duckduckgo.com/l/?uddg=") or "uddg=" in href:
                             try:
                                 from urllib.parse import unquote
                                 # simple extract
                                 href = unquote(href.split("uddg=")[1].split("&")[0])
                             except:
                                 pass
                                 
                        snippet = "Fallback search result" 
                        # Try to find snippet div? usually <a class="result__snippet" ...>
                        
                        got.append(f"Title: {unescape(title)}\nURL: {href}\nSnippet: {snippet}")
                        count += 1
                else:
                    print(f"DEBUG: HTML fallback failed with status {resp.status_code}")
            except Exception as e:
                print(f"DEBUG: HTML fallback failed: {e}")

        return got

    # Try primary query and a few variants aimed at specifications/reviews
    # Prioritize 'specs' to get technical details first
    variants = [f"{query} specs", f"{query} specifications", query, f"{query} review"]
    for v in variants:
        items = _try_query(v, max_results=3)
        if items:
            combined.extend(items)
        if len(combined) >= 8: # Increased limit to gather more diverse info
            break

    if not combined:
        return ""

    # Deduplicate while preserving order
    seen = set()
    out = []
    for i in combined:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)

    return "[Web search results]\n" + "\n---\n".join(out)


def _extract_all_entities(query: str):
    """
    Extract a list of entities from a query.
    Handles 'compare A, B, and C', 'A vs B vs C', etc.
    Returns a list of strings.
    """
    if not query:
        return []
    
    q = query.lower()
    
    # Handle typos
    q = q.replace("comapre", "compare")
    
    # Remove command prefixes
    for prefix in ["compare ", "comparison of ", "tell me about ", "what is ", "review ", "specs for ", "specifications for ", "difference between "]:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
            
    # Split by common delimiters
    # We want to split by: " vs ", " v ", " vs. ", " with ", " and ", ",", " versus "
    # Note: "between" is often part of "difference between", handled by prefix removal above
    
    # Use regex to split
    tokens = re.split(r'\s+(?:vs\.?|v\.?|with|and|,|versus)\s+', q)
    
    # Clean up tokens
    entities = []
    seen = set()
    for t in tokens:
        clean_t = t.strip(" \t\n\r'\".,")
        if clean_t and clean_t not in seen:
            # Filter out common stop words that might remain if split failed
            if clean_t in ["between"]:
                continue
            entities.append(clean_t)
            seen.add(clean_t)
            
    return entities

def _resolve_coreferences(query, history):
    """
    Replace 'it', 'this', 'that', 'both' with entities from the last user query.
    """
    if not history:
        return query
        
    last_user_msg = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content")
            break
            
    if not last_user_msg:
        return query
        
    q_lower = query.lower()
    
    # helper to find subject(s)
    prev_entities = _extract_all_entities(last_user_msg)
    
    # Handle "both" or "these" -> join all previous entities
    if "both" in q_lower or "these" in q_lower or "all of them" in q_lower:
        if prev_entities:
            replacement = " and ".join(prev_entities)
            # Regex replace
            for keyword in ["both", "these", "all of them"]:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, q_lower):
                    print(f"DEBUG: Resolved '{keyword}' to '{replacement}'")
                    query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                    # We continue incase there are other markers
    
    # Handle singular "it", "this", "that" -> use first entity if multiple, or the only one
    markers = ["it", "this", "that"]
    has_marker = False
    for m in markers:
        if re.search(r'\b' + re.escape(m) + r'\b', q_lower):
            has_marker = True
            break
            
    if has_marker and prev_entities:
        subject = prev_entities[0] # Best guess
        print(f"DEBUG: Resolved coreference in '{query}' -> replacing with '{subject}'")
        for m in markers:
             query = re.sub(r'\b' + re.escape(m) + r'\b', subject, query, flags=re.IGNORECASE)

    return query


def answer_question(query, history=[]):
    print(f"DEBUG: answer_question called with query='{query}' type(history)={type(history)}")
    
    # Resolve "it", "this", "both"
    original_query = query
    query = _resolve_coreferences(query, history)
    if query != original_query:
        print(f"DEBUG: Query resolved to: '{query}'")

    # try: removed to fix syntax error
    docs = local_retrieval(query)
    print(f"DEBUG: local_retrieval returned docs of type {type(docs)} len={len(docs) if docs is not None else 'None'}")


    if is_sufficient(docs):
        context = "\n".join(d.page_content for d in docs)
        source = "local documents"
    else:
        context = web_search(query)
        source = "web search"

    # If we used local documents but they don't mention some entities, check specifically
    if source == "local documents":
        entities = _extract_all_entities(query)
        missing_entities = []
        
        # Prepare context for robust matching
        ctx_lower = context.lower()
        ctx_simplified = ctx_lower.replace(" ", "").replace("-", "").replace(".", "")

        def is_in_context(entity, context_lower, context_simplified):
            if not entity: return True
            e = entity.lower()
            if e in context_lower: return True
            # Try simplified match (ignoring spaces/dashes)
            e_simp = e.replace(" ", "").replace("-", "").replace(".", "")
            if len(e_simp) > 2 and e_simp in context_simplified:
                return True
            return False

        # Check ALL extracted entities
        for ent in entities:
            if not is_in_context(ent, ctx_lower, ctx_simplified):
                missing_entities.append(ent)

        if missing_entities:
            web_sources = []
            for missing in missing_entities:
                web = web_search(missing)
                if web:
                    context = context + f"\n\n[Web search for '{missing}']\n" + web
                    web_sources.append(missing)
            
            if web_sources:
                local_entities = [e for e in entities if e not in missing_entities]
                
                source_parts = []
                if local_entities:
                     source_parts.append(f"Local ({', '.join(local_entities)})")
                else:
                     source_parts.append("Local (Partial)")
                
                source_parts.append(f"Web ({', '.join(web_sources)})")
                source = ", ".join(source_parts)

            # If specific searches didn't return anything, try full query fallback
            if source == "local documents" and missing_entities: 
                 web = web_search(query)
                 if web:
                    context = context + "\n\n[Web search appended]\n" + web
                    source = "Local + Web (Full Query)"

    system = (
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
        "DOMAIN RESTRICTION: You are a LAPTOP expert. If the user asks to compare mobile phones, tablets, or other non-laptop devices, "
        "politely decline and state that you only compare laptops. Do NOT generate comparisons for phones."
    )

    # Limit history to last 5 turns to prevent context overflow
    formatted_history = ""
    if history:
        print(f"DEBUG: Received history of length {len(history)}")
        for msg in history[-5:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_history += f"{role.capitalize()}: {content}\n"
        print(f"DEBUG: Formatted history:\n{formatted_history}")
    else:
        print("DEBUG: No history received.")

    prompt = f"{system}\n\nContext:\n{context}\n\nConversation History:\n{formatted_history}\n\nQuestion:\n{query}"

    llm, err = get_llm()
    if err or llm is None:
        msg = (
            "LLM not available. Install `langchain_groq` and its dependencies, or run this "
            "script with the project virtualenv.\nError: " + (err or "unknown")
        )
        return msg, source

    try:
        # Attempt a simple text call; ChatGroq implementations may accept a string input.
        out = llm(prompt)
        # If the model returns an object with `content`, try to extract it.
        if hasattr(out, 'content'):
            return out.content, source
        # If the model returns a dict-like object
        try:
            return str(out), source
        except Exception:
            return repr(out), source
    except Exception as e:
        tb = traceback.format_exc()
        return f"LLM call failed: {e}\n{tb}", source
