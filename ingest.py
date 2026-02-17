from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

DATA_PATH = "data/laptops"
DB_PATH = "chroma_db"


class SimpleDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def load_markdown_files(path):
    docs = []
    for root, _dirs, files in os.walk(path):
        for fname in files:
            if not fname.lower().endswith('.md'):
                continue
            p = os.path.join(root, fname)
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    text = fh.read()
            except Exception:
                continue
            docs.append(SimpleDoc(text, {"source": p}))
    return docs


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=600, chunk_overlap=100):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split_documents(self, docs):
        out = []
        for doc in docs:
            text = doc.page_content or ""
            n = len(text)
            start = 0
            while start < n:
                end = min(start + self.chunk_size, n)
                chunk = text[start:end]
                md = dict(doc.metadata) if getattr(doc, 'metadata', None) else {}
                md.update({"chunk_start": start, "chunk_end": end})
                out.append(SimpleDoc(chunk, md))
                if end == n:
                    break
                start = end - self.chunk_overlap
        return out


def ingest():
    docs = load_markdown_files(DATA_PATH)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )

    print("Ingestion complete")


if __name__ == "__main__":
    ingest()
