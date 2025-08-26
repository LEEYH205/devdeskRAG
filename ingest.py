# ingest.py
import os, glob, requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

def load_docs():
    docs = []
    # PDFs
    for pdf in glob.glob(f"{DATA_DIR}/*.pdf"):
        try:
            docs += PyPDFLoader(pdf).load()
        except Exception as e:
            print(f"[WARN] PDF load failed: {pdf} -> {e}")

    # Markdown/Text (treat as plain text)
    for path in glob.glob(f"{DATA_DIR}/*.md") + glob.glob(f"{DATA_DIR}/*.txt"):
        try:
            docs += TextLoader(path, encoding="utf-8").load()
        except Exception as e:
            print(f"[WARN] Text load failed: {path} -> {e}")

    # Optional URL crawl
    if os.path.exists("urls.txt"):
        for url in open("urls.txt", "r", encoding="utf-8"):
            url = url.strip()
            if not url:
                continue
            try:
                html = requests.get(url, timeout=20).text
                text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
                docs.append(Document(page_content=text, metadata={"source": url}))
            except Exception as e:
                print(f"[WARN] URL fetch failed: {url} -> {e}")
    return docs

if __name__ == "__main__":
    raw_docs = load_docs()
    if not raw_docs:
        print("[INFO] No documents found. Put PDFs/MD/TXT under data/ or add urls.txt.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(raw_docs)
    print(f"[INFO] Split into {len(chunks)} chunks. Building embeddings with {EMBED_MODEL}...")
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma.from_documents(chunks, embed, persist_directory=DB_DIR)
    vs.persist()
    print(f"[OK] Indexed {len(chunks)} chunks -> {DB_DIR}")
