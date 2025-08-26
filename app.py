import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")
MODE = os.getenv("MODE", "ollama")  # 'ollama' | 'together'
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# 성능 최적화 설정
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
SEARCH_K = int(os.getenv("SEARCH_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# 전역 변수
embed = None
vs = None
retriever = None
llm = None

def initialize_components():
    """컴포넌트들을 초기화합니다"""
    global embed, vs, retriever, llm
    
    try:
        # 임베딩 모델 초기화
        logger.info("Initializing embedding model...")
        embed = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 벡터 스토어 초기화
        logger.info("Initializing vector store...")
        if not os.path.exists(DB_DIR):
            raise FileNotFoundError(f"Vector database not found at {DB_DIR}. Please run ingest.py first.")
        
        vs = Chroma(persist_directory=DB_DIR, embedding_function=embed)
        retriever = vs.as_retriever(search_kwargs={"k": SEARCH_K})
        
        # LLM 초기화
        logger.info(f"Initializing LLM in {MODE} mode...")
        if MODE == "together":
            together_api_key = os.getenv("TOGETHER_API_KEY")
            if not together_api_key:
                raise ValueError("TOGETHER_API_KEY environment variable is required for Together mode")
            
            llm = ChatTogether(
                model=os.getenv("TOGETHER_MODEL", "lgai/exaone-deep-32b"),
                temperature=TEMPERATURE,
                together_api_key=together_api_key
            )
            logger.info(f"Using Together API with model: {os.getenv('TOGETHER_MODEL')}")
        else:
            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b"),
                temperature=TEMPERATURE
            )
            logger.info(f"Using Ollama with model: {os.getenv('OLLAMA_MODEL')}")
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 한국어/영어 문서에 정통한 어시스턴트야. "
     "주어진 '컨텍스트'에 근거해 한국어로 간결하게 답변하고, "
     "출처(파일명/URL)도 함께 요약해줘. 근거가 모자라면 '모르겠다'고 말해."),
    ("human", "질문: {question}\n\n컨텍스트:\n{context}")
])

def format_docs(docs):
    """검색된 문서들을 포맷팅합니다"""
    lines = []
    for d in docs:
        src = d.metadata.get("source", "local")
        snippet = d.page_content[:500].replace("\n", " ")
        lines.append(f"- ({src}) {snippet}")
    return "\n".join(lines)

def process_question(question: str):
    """질문을 처리하는 최적화된 RAG 함수"""
    try:
        # 1. 관련 문서 검색 (성능 측정)
        search_start = time.time()
        docs = retriever.invoke(question)  # 최신 메서드 사용
        search_time = time.time() - search_start
        
        # 2. 문서 포맷팅
        format_start = time.time()
        context = format_docs(docs)
        format_time = time.time() - format_start
        
        # 3. 프롬프트 생성
        prompt_start = time.time()
        prompt = PROMPT.format(question=question, context=context)
        prompt_time = time.time() - prompt_start
        
        # 4. LLM으로 답변 생성 (성능 측정)
        llm_start = time.time()
        response = llm.invoke(prompt)
        llm_time = time.time() - llm_start
        
        # 성능 로깅
        logger.info(f"Performance - Search: {search_time:.3f}s, Format: {format_time:.3f}s, Prompt: {prompt_time:.3f}s, LLM: {llm_time:.3f}s")
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error in process_question: {e}")
        raise e

class Q(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    performance: dict
    sources: list

app = FastAPI(
    title="DevDesk-RAG",
    description="나만의 ChatGPT - RAG 기반 문서 Q&A 시스템",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행되는 이벤트"""
    logger.info("Starting DevDesk-RAG API v2.0...")
    if not initialize_components():
        logger.error("Failed to initialize components. Check your configuration.")
        raise RuntimeError("Component initialization failed")

@app.post("/chat", response_model=ChatResponse)
def chat(q: Q):
    """질문에 답변합니다 (성능 모니터링 포함)"""
    try:
        if not llm or not retriever:
            raise HTTPException(status_code=500, detail="System not properly initialized")
        
        # 전체 처리 시간 측정
        total_start = time.time()
        
        # RAG 처리
        logger.info(f"Processing question: {q.question}")
        result = process_question(q.question)
        
        total_time = time.time() - total_start
        
        # 출처 추출
        sources = []
        try:
            docs = retriever.invoke(q.question)
            sources = [doc.metadata.get("source", "unknown") for doc in docs]
        except:
            pass
        
        return ChatResponse(
            answer=result,
            performance={
                "total_time": round(total_time, 3),
                "search_k": SEARCH_K,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP
            },
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
def health():
    """시스템 상태를 확인합니다"""
    try:
        db_status = os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0
        return {
            "status": "healthy" if db_status else "warning",
            "mode": MODE,
            "model": os.getenv("OLLAMA_MODEL" if MODE == "ollama" else "TOGETHER_MODEL"),
            "vector_db": db_status,
            "performance_config": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "search_k": SEARCH_K,
                "temperature": TEMPERATURE
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui")
async def get_ui():
    """웹 UI 페이지를 제공합니다"""
    return FileResponse("static/index.html")

@app.get("/")
def root():
    return {
        "message": "DevDesk-RAG API v2.0", 
        "endpoints": ["/chat", "/health", "/ui", "/config"],
        "features": ["Performance Monitoring", "Optimized RAG", "CORS Support", "Web UI"],
        "web_ui": "/ui"
    }

@app.get("/config")
def get_config():
    """현재 시스템 설정을 확인합니다"""
    return {
        "mode": MODE,
        "embed_model": EMBED_MODEL,
        "performance": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "search_k": SEARCH_K,
            "temperature": TEMPERATURE
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
