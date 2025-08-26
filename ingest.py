# ingest.py
import os, glob, requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import time

load_dotenv()

DATA_DIR = "data"
DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# 성능 최적화 설정
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

def load_docs():
    docs = []
    start_time = time.time()
    
    # PDFs
    for pdf in glob.glob(f"{DATA_DIR}/*.pdf"):
        try:
            print(f"[INFO] Loading PDF: {pdf}")
            docs += PyPDFLoader(pdf).load()
        except Exception as e:
            print(f"[WARN] PDF load failed: {pdf} -> {e}")

    # Markdown/Text (treat as plain text)
    for path in glob.glob(f"{DATA_DIR}/*.md") + glob.glob(f"{DATA_DIR}/*.txt"):
        try:
            print(f"[INFO] Loading text: {path}")
            docs += TextLoader(path, encoding="utf-8").load()
        except Exception as e:
            print(f"[WARN] Text load failed: {path} -> {e}")

    # Optional URL crawl
    if os.path.exists("urls.txt"):
        print("[INFO] Loading web pages from urls.txt...")
        for url in open("urls.txt", "r", encoding="utf-8"):
            url = url.strip()
            if not url:
                continue
            try:
                print(f"[INFO] Loading URL: {url}")
                html = requests.get(url, timeout=20).text
                text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
                docs.append(Document(page_content=text, metadata={"source": url}))
            except Exception as e:
                print(f"[WARN] URL fetch failed: {url} -> {e}")
    
    load_time = time.time() - start_time
    print(f"[INFO] Document loading completed in {load_time:.2f}s")
    return docs

if __name__ == "__main__":
    print("=== DevDesk-RAG Document Ingestion v2.0 ===")
    
    # 문서 로드
    print("\n1. Loading documents...")
    raw_docs = load_docs()
    
    if not raw_docs:
        print("[INFO] No documents found. Put PDFs/MD/TXT under data/ or add urls.txt.")
        exit(1)
    
    print(f"[INFO] Loaded {len(raw_docs)} documents")
    
    # 텍스트 분할 (성능 측정)
    print(f"\n2. Splitting documents into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})...")
    split_start = time.time()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(raw_docs)
    split_time = time.time() - split_start
    
    print(f"[INFO] Created {len(chunks)} chunks in {split_time:.2f}s")
    
    # 임베딩 모델 로드 및 벡터DB 생성 (성능 측정)
    print(f"\n3. Building embeddings with {EMBED_MODEL}...")
    embed_start = time.time()
    
    try:
        embed = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vs = Chroma.from_documents(chunks, embed, persist_directory=DB_DIR)
        vs.persist()
        
        embed_time = time.time() - embed_start
        print(f"[OK] Indexed {len(chunks)} chunks -> {DB_DIR} in {embed_time:.2f}s")
        
        # 검색 테스트
        print("\n4. Testing search functionality...")
        test_start = time.time()
        test_query = "test"
        results = vs.similarity_search(test_query, k=1)
        test_time = time.time() - test_start
        
        if results:
            print(f"✓ Search test successful in {test_time:.3f}s")
        else:
            print("⚠ Search test returned no results")
            
    except Exception as e:
        print(f"[ERROR] Failed to create vector database: {e}")
        exit(1)
    
    total_time = time.time() - split_start
    print(f"\n=== Ingestion Complete ===")
    print(f"Documents: {len(raw_docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Database: {DB_DIR}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per chunk: {total_time/len(chunks):.3f}s")
    print(f"\nYou can now run: python app.py")
