import os
import time
import asyncio
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import uuid
import json
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from langchain_together import TogetherEmbeddings  # exaone 임베딩 추가

# 채팅 히스토리 모듈 import
from chat_history import ChatHistoryManager

# 문서 처리 모듈 import
from document_processor import DocumentProcessor

# 성능 모니터링 모듈 import
from performance.performance_monitor import record_chat_metrics, get_performance_dashboard_data

# 고급 검색 알고리즘 모듈 import
from advanced_search.advanced_search import advanced_search_engine, get_search_insights

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODE = os.getenv("MODE", "ollama")  # 'ollama' | 'together'
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_MODE = os.getenv("EMBED_MODE", "huggingface")  # 'huggingface' | 'together'
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# 성능 최적화 설정
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
SEARCH_K = int(os.getenv("SEARCH_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# 고급 검색 알고리즘 설정
USE_ADVANCED_SEARCH = os.getenv("USE_ADVANCED_SEARCH", "true").lower() == "true"
ADVANCED_SEARCH_ALGORITHM = os.getenv("ADVANCED_SEARCH_ALGORITHM", "weighted_hybrid")

# 전역 변수
embed = None
vs = None
retriever = None
llm = None
chat_manager = None
doc_processor = None

# 지원하는 파일 확장자
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF 문서',
    '.md': 'Markdown 문서',
    '.txt': '텍스트 문서',
    '.doc': 'Word 문서',
    '.docx': 'Word 문서',
    '.html': 'HTML 문서',
    '.htm': 'HTML 문서'
}

def initialize_components():
    """컴포넌트들을 초기화합니다"""
    global embed, vs, retriever, llm, chat_manager, doc_processor
    
    try:
        # 임베딩 모델 초기화
        logger.info("Initializing embedding model...")
        if EMBED_MODE == "together":
            together_api_key = os.getenv("TOGETHER_API_KEY")
            if not together_api_key:
                raise ValueError("TOGETHER_API_KEY environment variable is required for Together embedding mode")
            
            # exaone 임베딩 모델 사용
            embed = TogetherEmbeddings(
                model="lgai/exaone-deep-32b",
                together_api_key=together_api_key
            )
            logger.info("Using Together API with exaone embedding model")
        else:
            # 기존 HuggingFace 임베딩 모델 사용
            embed = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Using HuggingFace embedding model: {EMBED_MODEL}")
        
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
        
        # 채팅 히스토리 매니저 초기화
        logger.info("Initializing chat history manager...")
        chat_manager = ChatHistoryManager(REDIS_URL)
        
        # 문서 처리기 초기화
        logger.info("Initializing document processor...")
        doc_processor = DocumentProcessor(data_dir=DATA_DIR, db_dir=DB_DIR, embed_model=EMBED_MODEL)
        doc_processor.initialize()
        
        # 고급 검색 엔진 초기화
        logger.info("Initializing advanced search engine...")
        try:
            # BM25 리트리버 생성 (간단한 구현)
            from langchain_community.retrievers import BM25Retriever
            from langchain_core.documents import Document
            
            # 기존 문서들을 BM25용으로 변환
            all_docs = []
            if vs:
                collection = vs._collection
                results = collection.get()
                if results and results['documents']:
                    for i, doc_content in enumerate(results['documents']):
                        metadata = results['metadatas'][i] if results['metadatas'] else {}
                        all_docs.append(Document(
                            page_content=doc_content,
                            metadata=metadata
                        ))
            
            bm25_retriever = BM25Retriever.from_documents(all_docs) if all_docs else None
            
            # 고급 검색 엔진 초기화
            advanced_search_engine.initialize_search_system(
                vector_store=vs,
                retriever=retriever,
                embedding_model=embed,
                bm25_retriever=bm25_retriever
            )
            logger.info("Advanced search engine initialized successfully")
        except Exception as e:
            logger.warning(f"Advanced search engine initialization failed: {e}")
            logger.info("Continuing with basic search functionality")
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False

async def get_chat_manager():
    """채팅 히스토리 매니저 의존성"""
    if not chat_manager:
        raise HTTPException(status_code=500, detail="Chat history manager not initialized")
    return chat_manager

def validate_file_extension(filename: str) -> bool:
    """파일 확장자 검증"""
    file_ext = Path(filename).suffix.lower()
    return file_ext in SUPPORTED_EXTENSIONS

def get_file_info(filename: str) -> dict:
    """파일 정보 반환"""
    file_ext = Path(filename).suffix.lower()
    return {
        "filename": filename,
        "extension": file_ext,
        "type": SUPPORTED_EXTENSIONS.get(file_ext, "알 수 없는 파일"),
        "supported": file_ext in SUPPORTED_EXTENSIONS
    }

async def save_uploaded_file(file: UploadFile, session_id: str = None) -> dict:
    """업로드된 파일 저장"""
    try:
        # 파일 확장자 검증
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다: {file.filename}"
            )
        
        # data 디렉토리 생성
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 고유한 파일명 생성 (중복 방지)
        file_ext = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(DATA_DIR, unique_filename)
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_info = get_file_info(file.filename)
        file_info.update({
            "saved_path": file_path,
            "saved_filename": unique_filename,
            "session_id": session_id,
            "upload_time": time.time()
        })
        
        logger.info(f"파일 업로드 성공: {file.filename} -> {file_path}")
        return file_info
        
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")

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

def process_question(question: str, user_id: str = None, use_advanced_search: bool = True):
    """질문을 처리하는 최적화된 RAG 함수 (고급 검색 알고리즘 통합)"""
    try:
        # 1. 고급 검색 알고리즘을 사용한 문서 검색
        search_start = time.time()
        
        if use_advanced_search and advanced_search_engine:
            # 고급 검색 사용
            search_results = advanced_search_engine.search(
                query=question,
                user_id=user_id or "anonymous"
            )
            
            # SearchResult를 Document로 변환
            docs = []
            for result in search_results:
                from langchain_core.documents import Document
                doc = Document(
                    page_content=result.content,
                    metadata=result.metadata
                )
                docs.append(doc)
            
            # 검색 품질 메트릭 추출
            search_quality = result.relevance_score if hasattr(result, 'relevance_score') else 0.8
            algorithm_used = "advanced_search"
            
        else:
            # 기존 검색 사용
            docs = retriever.invoke(question)
            search_quality = 0.7  # 기본값
            algorithm_used = "traditional_rag"
        
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
        
        # 성능 로깅 (고급 검색 정보 포함)
        logger.info(f"Performance - Search: {search_time:.3f}s, Format: {format_time:.3f}s, Prompt: {prompt_time:.3f}s, LLM: {llm_time:.3f}s, Algorithm: {algorithm_used}, Quality: {search_quality:.3f}")
        
        # 성능 메트릭 기록 (고급 검색 메트릭 포함)
        try:
            record_chat_metrics(
                session_id="",  # 나중에 세션 ID 추가
                user_id=user_id or "",
                question=question,
                search_time=search_time,
                format_time=format_time,
                prompt_time=prompt_time,
                llm_time=llm_time,
                search_results_count=len(docs),
                chunks_retrieved=len(docs),
                relevance_score=search_quality
            )
        except Exception as e:
            logger.error(f"성능 메트릭 기록 실패: {e}")
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error in process_question: {e}")
        raise e

async def process_question_stream(question: str):
    """질문을 처리하는 스트리밍 RAG 함수"""
    try:
        # 1. 관련 문서 검색
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        # 2. 프롬프트 생성
        prompt = PROMPT.format(question=question, context=context)
        
        # 3. 스트리밍 응답 생성
        response = llm.stream(prompt)
        
        for chunk in response:
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                yield content
        
    except Exception as e:
        logger.error(f"Error in process_question_stream: {e}")
        yield f"오류가 발생했습니다: {str(e)}"

class Q(BaseModel):
    question: str
    session_id: str = None

class ChatResponse(BaseModel):
    answer: str
    performance: dict
    sources: list
    session_id: str

class SessionCreate(BaseModel):
    user_id: str = None

class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    created_at: str
    last_activity: str
    message_count: int

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    file_info: dict = None

app = FastAPI(
    title="DevDesk-RAG",
    description="나만의 ChatGPT - RAG 기반 문서 Q&A 시스템",
    version="2.1.0"
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

# 성능 대시보드 정적 파일 서빙
@app.get("/performance_dashboard.html")
async def get_performance_dashboard():
    """성능 모니터링 대시보드 페이지를 제공합니다"""
    return FileResponse("performance/performance_dashboard.html")

@app.get("/performance_dashboard")
async def get_performance_dashboard_alt():
    """성능 모니터링 대시보드 페이지를 제공합니다 (대체 경로)"""
    return FileResponse("performance/performance_dashboard.html")

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행되는 이벤트"""
    logger.info("Starting DevDesk-RAG API v2.1...")
    if not initialize_components():
        logger.error("Failed to initialize components. Check your configuration.")
        raise RuntimeError("Component initialization failed")
    
    # Redis 연결
    try:
        await chat_manager.connect()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 실행되는 이벤트"""
    if chat_manager:
        await chat_manager.disconnect()
        logger.info("Redis connection closed")

@app.post("/chat", response_model=ChatResponse)
async def chat(q: Q, chat_mgr: ChatHistoryManager = Depends(get_chat_manager)):
    """질문에 답변합니다 (채팅 히스토리 저장 포함)"""
    try:
        if not llm or not retriever:
            raise HTTPException(status_code=500, detail="System not properly initialized")
        
        # 세션 ID가 없으면 새로 생성
        if not q.session_id:
            q.session_id = await chat_mgr.create_session()
        
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
        
        # 사용자 질문 저장
        await chat_mgr.add_message(q.session_id, {
            "sender": "user",
            "content": q.question,
            "metadata": {"sources": sources, "performance": {"total_time": total_time}}
        })
        
        # 봇 답변 저장
        await chat_mgr.add_message(q.session_id, {
            "sender": "bot",
            "content": result,
            "metadata": {"sources": sources, "performance": {"total_time": total_time}}
        })
        
        return ChatResponse(
            answer=result,
            performance={
                "total_time": round(total_time, 3),
                "search_k": SEARCH_K,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP
            },
            sources=sources,
            session_id=q.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(q: Q, chat_mgr: ChatHistoryManager = Depends(get_chat_manager)):
    """질문에 답변합니다 (실시간 스트리밍 응답)"""
    try:
        if not llm or not retriever:
            raise HTTPException(status_code=500, detail="System not properly initialized")
        
        # 세션 ID가 없으면 새로 생성
        if not q.session_id:
            q.session_id = await chat_mgr.create_session()
        
        # 출처 추출
        sources = []
        try:
            docs = retriever.invoke(q.question)
            sources = [doc.metadata.get("source", "unknown") for doc in docs]
        except:
            pass
        
        # 사용자 질문 저장
        await chat_mgr.add_message(q.session_id, {
            "sender": "user",
            "content": q.question,
            "metadata": {"sources": sources}
        })
        
        async def generate_stream():
            """스트리밍 응답 생성"""
            try:
                # 시작 신호
                yield "data: {\"type\": \"start\", \"session_id\": \"" + q.session_id + "\"}\n\n"
                
                # 스트리밍 응답 생성
                full_response = ""
                async for chunk in process_question_stream(q.question):
                    full_response += chunk
                    # SSE 형식으로 데이터 전송
                    yield f"data: {{\"type\": \"chunk\", \"content\": \"{chunk}\"}}\n\n"
                    await asyncio.sleep(0.01)  # 약간의 지연으로 자연스러운 스트리밍
                
                # 완료 신호
                yield f"data: {{\"type\": \"complete\", \"session_id\": \"{q.session_id}\", \"sources\": {json.dumps(sources)}}}\n\n"
                
                # 봇 답변 저장
                await chat_mgr.add_message(q.session_id, {
                    "sender": "bot",
                    "content": full_response,
                    "metadata": {"sources": sources}
                })
                
            except Exception as e:
                error_msg = f"스트리밍 오류: {str(e)}"
                yield f"data: {{\"type\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
                logger.error(f"Streaming error: {e}")
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat stream: {e}")
        raise HTTPException(status_code=500, detail=f"Error in chat stream: {str(e)}")

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(None)
):
    """파일 업로드 (드래그 앤 드롭 지원)"""
    try:
        # 파일 크기 제한 (100MB)
        if file.size and file.size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기는 100MB를 초과할 수 없습니다")
        
        # 파일 저장
        file_info = await save_uploaded_file(file, session_id)
        
        # 문서 처리 및 RAG 시스템에 추가
        if doc_processor:
            try:
                success = doc_processor.process_single_file(file_info["saved_path"])
                if success:
                    logger.info(f"Document added to RAG system: {file_info['filename']}")
                    file_info["rag_status"] = "added"
                else:
                    logger.warning(f"Failed to add document to RAG system: {file_info['filename']}")
                    file_info["rag_status"] = "failed"
            except Exception as e:
                logger.warning(f"Document processing failed: {e}")
                file_info["rag_status"] = "error"
                file_info["rag_error"] = str(e)
        
        return FileUploadResponse(
            success=True,
            message="파일 업로드 성공",
            file_info=file_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 업로드 실패: {e}")
        return FileUploadResponse(
            success=False,
            message=f"파일 업로드 실패: {str(e)}"
        )

@app.post("/upload/multiple")
async def upload_multiple_files(
    files: list[UploadFile] = File(...),
    session_id: str = Form(None)
):
    """다중 파일 업로드"""
    results = []
    
    for file in files:
        try:
            file_info = await save_uploaded_file(file, session_id)
            
            # 문서 처리 및 RAG 시스템에 추가
            if doc_processor and file_info.get("success"):
                try:
                    success = doc_processor.process_single_file(file_info["file_info"]["saved_path"])
                    if success:
                        file_info["file_info"]["rag_status"] = "added"
                        logger.info(f"Document added to RAG system: {file.filename}")
                    else:
                        file_info["file_info"]["rag_status"] = "failed"
                        logger.warning(f"Failed to add document to RAG system: {file.filename}")
                except Exception as e:
                    logger.warning(f"Document processing failed: {e}")
                    file_info["file_info"]["rag_status"] = "error"
                    file_info["file_info"]["rag_error"] = str(e)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "file_info": file_info
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "message": f"{len(files)}개 파일 처리 완료",
        "results": results,
        "success_count": sum(1 for r in results if r["success"]),
        "error_count": sum(1 for r in results if not r["success"])
    }

@app.get("/files")
async def list_uploaded_files():
    """업로드된 파일 목록 조회"""
    try:
        if not os.path.exists(DATA_DIR):
            return {"files": [], "count": 0}
        
        files = []
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.isfile(file_path):
                file_info = get_file_info(filename)
                file_info.update({
                    "size": os.path.getsize(file_path),
                    "modified": os.path.getmtime(file_path)
                })
                files.append(file_info)
        
        # 수정 시간순 정렬 (최신 파일이 먼저)
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "files": files,
            "count": len(files),
            "supported_extensions": list(SUPPORTED_EXTENSIONS.keys())
        }
        
    except Exception as e:
        logger.error(f"파일 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 목록 조회 실패: {str(e)}")

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """업로드된 파일 삭제"""
    try:
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
        
        # RAG 시스템에서도 문서 제거
        if doc_processor:
            try:
                doc_processor.remove_document(file_path)
                logger.info(f"Document removed from RAG system: {filename}")
            except Exception as e:
                logger.warning(f"Failed to remove document from RAG system: {e}")
        
        os.remove(file_path)
        logger.info(f"파일 삭제 완료: {filename}")
        
        return {"message": f"파일 {filename} 삭제 완료"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {str(e)}")

@app.post("/documents/process")
async def process_all_documents():
    """모든 문서를 RAG 시스템에 처리"""
    try:
        if not doc_processor:
            raise HTTPException(status_code=500, detail="Document processor not initialized")
        
        result = doc_processor.process_all_files()
        return result
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.get("/documents/status")
async def get_document_status():
    """RAG 시스템의 문서 상태 조회"""
    try:
        if not doc_processor:
            raise HTTPException(status_code=500, detail="Document processor not initialized")
        
        info = doc_processor.get_vector_store_info()
        return info
        
    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")

@app.post("/documents/refresh")
async def refresh_rag_system():
    """RAG 시스템 새로고침 (모든 문서 재처리)"""
    try:
        if not doc_processor:
            raise HTTPException(status_code=500, detail="Document processor not initialized")
        
        # 기존 벡터 스토어 초기화
        if doc_processor:
            doc_processor.remove_document("")  # 빈 문자열로 전체 초기화
            logger.info("Vector store cleared")
        
        # 모든 문서 재처리
        result = doc_processor.process_all_files()
        return {
            "message": "RAG system refreshed successfully",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"RAG system refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG system refresh failed: {str(e)}")

@app.post("/sessions", response_model=SessionInfo)
async def create_session(session_data: SessionCreate, chat_mgr: ChatHistoryManager = Depends(get_chat_manager)):
    """새로운 채팅 세션을 생성합니다"""
    try:
        session_id = await chat_mgr.create_session(session_data.user_id)
        session_info = await chat_mgr.redis.hgetall(f"session:{session_id}")
        
        return SessionInfo(
            session_id=session_id,
            user_id=session_info.get("user_id", ""),
            created_at=session_info.get("created_at", ""),
            last_activity=session_info.get("last_activity", ""),
            message_count=int(session_info.get("message_count", 0))
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 50, chat_mgr: ChatHistoryManager = Depends(get_chat_manager)):
    """세션의 메시지 목록을 조회합니다"""
    try:
        messages = await chat_mgr.get_session_messages(session_id, limit)
        return {"session_id": session_id, "messages": messages, "count": len(messages)}
    except Exception as e:
        logger.error(f"Error getting session messages: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting session messages: {str(e)}")

@app.get("/sessions/{session_id}/search")
async def search_session_messages(session_id: str, query: str, chat_mgr: ChatHistoryManager = Depends(get_chat_manager)):
    """세션 내 메시지를 검색합니다"""
    try:
        results = await chat_mgr.search_messages(session_id, query)
        return {"session_id": session_id, "query": query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error searching session messages: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching session messages: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, chat_mgr: ChatHistoryManager = Depends(get_chat_manager)):
    """세션을 삭제합니다"""
    try:
        success = await chat_mgr.delete_session(session_id)
        if success:
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.get("/health")
def health():
    """시스템 상태를 확인합니다"""
    try:
        db_status = os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0
        redis_status = chat_manager is not None and chat_manager.redis is not None
        
        return {
            "status": "healthy" if db_status and redis_status else "warning",
            "mode": MODE,
            "model": os.getenv("OLLAMA_MODEL" if MODE == "ollama" else "TOGETHER_MODEL"),
            "vector_db": db_status,
            "redis": redis_status,
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

@app.get("/advanced_search_dashboard")
async def get_advanced_search_dashboard():
    """고급 검색 알고리즘 대시보드 페이지를 제공합니다"""
    return FileResponse("advanced_search/advanced_search_dashboard.html")

@app.get("/")
def root():
    return {
        "message": "DevDesk-RAG API v2.1", 
        "endpoints": ["/chat", "/chat/stream", "/upload", "/files", "/sessions", "/health", "/ui", "/config"],
        "features": ["Performance Monitoring", "Optimized RAG", "CORS Support", "Web UI", "Chat History", "Streaming Response", "File Upload"],
        "web_ui": "/ui"
    }

@app.get("/config")
def get_config():
    """현재 시스템 설정을 확인합니다"""
    return {
        "mode": MODE,
        "embed_model": EMBED_MODEL,
        "redis_url": REDIS_URL,
        "data_dir": DATA_DIR,
        "supported_extensions": list(SUPPORTED_EXTENSIONS.keys()),
        "performance": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "search_k": SEARCH_K,
            "temperature": TEMPERATURE
        },
        "advanced_search": {
            "enabled": USE_ADVANCED_SEARCH,
            "algorithm": ADVANCED_SEARCH_ALGORITHM,
            "available_algorithms": ["vector_only", "bm25_only", "hybrid", "weighted_hybrid", "adaptive"]
        }
    }

@app.get("/performance/dashboard")
def get_performance_dashboard():
    """성능 모니터링 대시보드 데이터를 제공합니다"""
    try:
        dashboard_data = get_performance_dashboard_data()
        return {
            "status": "success",
            "data": dashboard_data,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"성능 대시보드 데이터 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"성능 대시보드 데이터 조회 실패: {str(e)}")

@app.get("/performance/session/{session_id}")
def get_session_performance(session_id: str):
    """특정 세션의 성능 메트릭을 제공합니다"""
    try:
        from performance.performance_monitor import performance_monitor
        session_metrics = performance_monitor.get_session_metrics(session_id)
        return {
            "status": "success",
            "session_id": session_id,
            "metrics": session_metrics,
            "count": len(session_metrics)
        }
    except Exception as e:
        logger.error(f"세션 성능 메트릭 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"세션 성능 메트릭 조회 실패: {str(e)}")

# ===== 고급 검색 알고리즘 API 엔드포인트 =====

@app.get("/search/advanced")
def advanced_search(query: str, user_id: str = None, algorithm: str = None):
    """고급 검색 알고리즘을 사용한 검색을 실행합니다"""
    try:
        results = advanced_search_engine.search(
            query=query,
            user_id=user_id or "anonymous",
            algorithm=algorithm
        )
        
        return {
            "status": "success",
            "query": query,
            "algorithm": algorithm or "weighted_hybrid",
            "results": [r.to_dict() for r in results],
            "count": len(results),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"고급 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"고급 검색 실패: {str(e)}")

@app.get("/search/insights")
def get_search_insights():
    """고급 검색 시스템의 성능 인사이트를 제공합니다"""
    try:
        insights = advanced_search_engine.get_performance_insights()
        return {
            "status": "success",
            "insights": insights,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"검색 인사이트 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"검색 인사이트 조회 실패: {str(e)}")

@app.get("/search/experiments")
def get_search_experiments():
    """현재 진행 중인 A/B 테스트 실험 정보를 제공합니다"""
    try:
        experiments = advanced_search_engine.ab_test_framework.experiments
        return {
            "status": "success",
            "experiments": experiments,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"A/B 테스트 실험 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"고급 검색 실험 정보 조회 실패: {str(e)}")

@app.post("/search/experiments")
def create_search_experiment(experiment_data: dict):
    """새로운 A/B 테스트 실험을 생성합니다"""
    try:
        experiment_id = experiment_data.get("experiment_id")
        variants = experiment_data.get("variants", [])
        traffic_split = experiment_data.get("traffic_split")
        
        success = advanced_search_engine.ab_test_framework.create_experiment(
            experiment_id=experiment_id,
            variants=variants,
            traffic_split=traffic_split
        )
        
        if success:
            return {
                "status": "success",
                "message": f"실험 {experiment_id} 생성 완료",
                "experiment_id": experiment_id
            }
        else:
            raise HTTPException(status_code=400, detail="실험 생성 실패")
            
    except Exception as e:
        logger.error(f"A/B 테스트 실험 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"A/B 테스트 실험 생성 실패: {str(e)}")

# ===== Phase 2.1: 하이브리드 검색 강화 API 엔드포인트 =====

@app.get("/search/hybrid")
def hybrid_search(query: str, user_id: str = None, domain: str = None):
    """도메인별 가중치를 적용한 하이브리드 검색을 실행합니다"""
    try:
        # 도메인 자동 분류 (사용자가 지정하지 않은 경우)
        if not domain:
            domain = advanced_search_engine.classify_query_domain(query)
        
        # 도메인별 가중치 조회
        domain_weights = advanced_search_engine.get_domain_weights(domain)
        
        # 하이브리드 검색 실행
        results = advanced_search_engine.search(
            query=query,
            user_id=user_id or "anonymous",
            algorithm="weighted_hybrid"
        )
        
        return {
            "status": "success",
            "query": query,
            "domain": domain,
            "domain_weights": domain_weights,
            "algorithm": "weighted_hybrid",
            "results": [r.to_dict() for r in results],
            "count": len(results),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"하이브리드 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"하이브리드 검색 실패: {str(e)}")

@app.get("/search/domain-analysis")
def analyze_query_domain(query: str):
    """쿼리의 도메인을 분석하고 적절한 검색 전략을 제안합니다"""
    try:
        domain = advanced_search_engine.classify_query_domain(query)
        domain_weights = advanced_search_engine.get_domain_weights(domain)
        
        # 도메인별 검색 전략 제안
        strategy_suggestions = {
            'technical': {
                'description': '기술적 질문 - 벡터 검색에 높은 가중치',
                'recommended_algorithm': 'weighted_hybrid',
                'optimization_tips': ['정확한 용어 사용', '코드 예시 포함', 'API 문서 참조']
            },
            'general': {
                'description': '일반적 질문 - 균형잡힌 하이브리드 검색',
                'recommended_algorithm': 'hybrid',
                'optimization_tips': ['명확한 질문 작성', '구체적인 예시 요청']
            },
            'code': {
                'description': '코드 관련 질문 - 벡터 검색과 키워드 검색 병합',
                'recommended_algorithm': 'weighted_hybrid',
                'optimization_tips': ['에러 메시지 포함', '프로그래밍 언어 명시', '코드 블록 사용']
            }
        }
        
        return {
            "status": "success",
            "query": query,
            "detected_domain": domain,
            "domain_weights": domain_weights,
            "strategy": strategy_suggestions.get(domain, strategy_suggestions['general']),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"도메인 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"도메인 분석 실패: {str(e)}")

@app.get("/search/quality-metrics")
def get_search_quality_metrics(query: str = None, algorithm: str = None, limit: int = 100):
    """검색 품질 메트릭을 제공합니다"""
    try:
        if not advanced_search_engine.search_metrics_history:
            return {
                "status": "success",
                "message": "아직 검색 메트릭이 수집되지 않았습니다",
                "metrics": {},
                "timestamp": time.time()
            }
        
        # 필터링된 메트릭 수집
        filtered_metrics = []
        for metric in advanced_search_engine.search_metrics_history:
            if query and query.lower() not in metric.query.lower():
                continue
            if algorithm and metric.algorithm != algorithm:
                continue
            filtered_metrics.append(metric)
        
        # 최근 N개로 제한
        recent_metrics = filtered_metrics[-limit:] if filtered_metrics else []
        
        if not recent_metrics:
            return {
                "status": "success",
                "message": "조건에 맞는 메트릭이 없습니다",
                "metrics": {},
                "timestamp": time.time()
            }
        
        # 품질 메트릭 계산
        avg_search_time = np.mean([m.search_time for m in recent_metrics])
        avg_results_count = np.mean([m.results_count for m in recent_metrics])
        avg_score = np.mean([m.avg_score for m in recent_metrics])
        
        # 알고리즘별 성능 비교
        algorithm_performance = {}
        for metric in recent_metrics:
            if metric.algorithm not in algorithm_performance:
                algorithm_performance[metric.algorithm] = {
                    'count': 0,
                    'avg_search_time': 0,
                    'avg_results_count': 0,
                    'avg_score': 0
                }
            
            algo_perf = algorithm_performance[metric.algorithm]
            algo_perf['count'] += 1
            algo_perf['avg_search_time'] += metric.search_time
            algo_perf['avg_results_count'] += metric.results_count
            algo_perf['avg_score'] += metric.avg_score
        
        # 평균 계산
        for algo_perf in algorithm_performance.values():
            if algo_perf['count'] > 0:
                algo_perf['avg_search_time'] /= algo_perf['count']
                algo_perf['avg_results_count'] /= algo_perf['count']
                algo_perf['avg_score'] /= algo_perf['count']
        
        return {
            "status": "success",
            "query_filter": query,
            "algorithm_filter": algorithm,
            "total_metrics": len(recent_metrics),
            "overall_performance": {
                "avg_search_time": round(avg_search_time, 3),
                "avg_results_count": round(avg_results_count, 3),
                "avg_score": round(avg_score, 3)
            },
            "algorithm_performance": algorithm_performance,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"검색 품질 메트릭 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"검색 품질 메트릭 조회 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
