import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")
MODE = os.getenv("MODE", "ollama")  # 'ollama' | 'together'
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

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
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        
        # LLM 초기화
        logger.info(f"Initializing LLM in {MODE} mode...")
        if MODE == "together":
            together_api_key = os.getenv("TOGETHER_API_KEY")
            if not together_api_key:
                raise ValueError("TOGETHER_API_KEY environment variable is required for Together mode")
            
            llm = ChatTogether(
                model=os.getenv("TOGETHER_MODEL", "lgai/exaone-deep-32b"),
                temperature=0.1,
                together_api_key=together_api_key
            )
            logger.info(f"Using Together API with model: {os.getenv('TOGETHER_MODEL')}")
        else:
            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b"),
                temperature=0.1
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
    """질문을 처리하는 간단한 RAG 함수"""
    try:
        # 1. 관련 문서 검색
        docs = retriever.get_relevant_documents(question)
        
        # 2. 문서 포맷팅
        context = format_docs(docs)
        
        # 3. 프롬프트 생성
        prompt = PROMPT.format(question=question, context=context)
        
        # 4. LLM으로 답변 생성
        response = llm.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error in process_question: {e}")
        raise e

class Q(BaseModel):
    question: str

app = FastAPI(title="DevDesk-RAG")

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행되는 이벤트"""
    logger.info("Starting DevDesk-RAG API...")
    if not initialize_components():
        logger.error("Failed to initialize components. Check your configuration.")
        raise RuntimeError("Component initialization failed")

@app.post("/chat")
def chat(q: Q):
    """질문에 답변합니다"""
    try:
        if not llm or not retriever:
            raise HTTPException(status_code=500, detail="System not properly initialized")
        
        # RAG 처리
        logger.info(f"Processing question: {q.question}")
        result = process_question(q.question)
        
        return {"answer": result}
        
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
            "vector_db": db_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "DevDesk-RAG API", "endpoints": ["/chat", "/health"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
