"""
DevDesk-RAG 실시간 문서 처리 모듈
- 업로드된 파일을 자동으로 RAG 시스템에 추가
- 다양한 문서 형식 지원 (PDF, MD, TXT, DOC, HTML)
- 벡터 DB 실시간 업데이트
"""

import os
import glob
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Any
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import TogetherEmbeddings  # exaone 임베딩 추가
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """실시간 문서 처리 및 RAG 시스템 업데이트"""
    
    def __init__(self, data_dir: str = "data", db_dir: str = "chroma_db", embed_model: str = "BAAI/bge-m3"):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.embed_model = embed_model
        self.embed_mode = os.getenv("EMBED_MODE", "huggingface")  # 'huggingface' | 'together'
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        
        # 임베딩 모델 및 벡터 스토어 초기화
        self.embed = None
        self.vs = None
        self.initialized = False
        
    def initialize(self):
        """컴포넌트 초기화"""
        try:
            if not self.initialized:
                logger.info("Initializing document processor...")
                
                # 임베딩 모델 초기화
                if self.embed_mode == "together":
                    together_api_key = os.getenv("TOGETHER_API_KEY")
                    if not together_api_key:
                        logger.warning("TOGETHER_API_KEY not found, falling back to HuggingFace")
                        self.embed_mode = "huggingface"
                    
                    if self.embed_mode == "together":
                        self.embed = TogetherEmbeddings(
                            model="lgai/exaone-deep-32b",
                            together_api_key=together_api_key
                        )
                        logger.info("Using Together API with exaone embedding model")
                
                if self.embed_mode == "huggingface":
                    self.embed = HuggingFaceEmbeddings(
                        model_name=self.embed_model,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    logger.info(f"Using HuggingFace embedding model: {self.embed_model}")
                
                # 벡터 스토어 초기화
                if os.path.exists(self.db_dir):
                    self.vs = Chroma(persist_directory=self.db_dir, embedding_function=self.embed)
                    logger.info(f"Vector store loaded from {self.db_dir}")
                else:
                    logger.warning(f"Vector store not found at {self.db_dir}")
                    return False
                
                self.initialized = True
                logger.info("Document processor initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing document processor: {e}")
            return False
    
    def process_single_file(self, file_path: str) -> bool:
        """단일 파일 처리 및 RAG 시스템에 추가"""
        try:
            if not self.initialized:
                if not self.initialize():
                    return False
            
            logger.info(f"Processing file: {file_path}")
            
            # 파일 로드
            docs = self._load_document(file_path)
            if not docs:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            # 문서 분할
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.split_documents(docs)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            
            # 벡터 스토어에 추가
            if self.vs:
                # 기존 문서와 중복 체크 (파일명 기준)
                existing_docs = self.vs.get(where={"source": file_path})
                if existing_docs and len(existing_docs['documents']) > 0:
                    logger.info(f"Document {file_path} already exists in vector store, updating...")
                    # 기존 문서 제거 후 새로 추가
                    self.vs.delete(where={"source": file_path})
                
                # 새 문서 추가
                self.vs.add_documents(chunks)
                
                logger.info(f"Successfully added {len(chunks)} chunks from {file_path} to vector store")
                return True
            else:
                logger.error("Vector store not initialized")
                return False
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    def process_all_files(self) -> Dict[str, Any]:
        """모든 파일 처리 및 RAG 시스템 업데이트"""
        try:
            if not self.initialized:
                if not self.initialize():
                    return {"success": False, "error": "Failed to initialize processor"}
            
            logger.info("Processing all files in data directory...")
            
            # 지원하는 파일 확장자
            supported_extensions = {
                '*.pdf': 'PDF 문서',
                '*.md': 'Markdown 문서', 
                '*.txt': '텍스트 문서',
                '*.doc': 'Word 문서',
                '*.docx': 'Word 문서',
                '*.html': 'HTML 문서',
                '*.htm': 'HTML 문서'
            }
            
            all_files = []
            processed_files = []
            failed_files = []
            
            # 각 확장자별로 파일 검색
            for pattern, description in supported_extensions.items():
                files = glob.glob(os.path.join(self.data_dir, pattern))
                all_files.extend(files)
            
            logger.info(f"Found {len(all_files)} files to process")
            
            # 각 파일 처리
            for file_path in all_files:
                try:
                    if self.process_single_file(file_path):
                        processed_files.append(file_path)
                    else:
                        failed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    failed_files.append(file_path)
            
            # 결과 요약
            result = {
                "success": True,
                "total_files": len(all_files),
                "processed_files": len(processed_files),
                "failed_files": len(failed_files),
                "processed_list": processed_files,
                "failed_list": failed_files
            }
            
            logger.info(f"Processing complete: {len(processed_files)}/{len(all_files)} files processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in process_all_files: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_document(self, file_path: str) -> List[Document]:
        """문서 로드 (파일 형식별 처리)"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # 메타데이터에 source 추가
                for doc in docs:
                    doc.metadata["source"] = file_path
                return docs
                
            elif file_ext in ['.md', '.txt']:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                # 메타데이터에 source 추가
                for doc in docs:
                    doc.metadata["source"] = file_path
                return docs
                
            elif file_ext in ['.html', '.htm']:
                # HTML 파일 처리
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text()
                
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path}
                )
                return [doc]
                
            elif file_ext in ['.doc', '.docx']:
                # Word 문서는 현재 지원하지 않음 (추후 확장 가능)
                logger.warning(f"Word document format not yet supported: {file_path}")
                return []
                
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """벡터 스토어 정보 조회"""
        try:
            if not self.vs:
                return {"error": "Vector store not initialized"}
            
            # 벡터 스토어 통계
            collection = self.vs._collection
            count = collection.count()
            
            # 문서 출처별 통계
            results = collection.get()
            sources = {}
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'source' in metadata:
                        source = metadata['source']
                        sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_documents": count,
                "sources": sources,
                "db_path": self.db_dir
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}")
            return {"error": str(e)}
    
    def remove_document(self, file_path: str) -> bool:
        """문서를 벡터 스토어에서 제거"""
        try:
            if not self.initialized or not self.vs:
                return False
            
            # 해당 파일의 모든 청크 제거
            if file_path:
                self.vs.delete(where={"source": file_path})
                logger.info(f"Successfully removed document {file_path} from vector store")
            else:
                # 전체 벡터 스토어 초기화
                self.vs._collection.delete(where={})
                logger.info("Vector store cleared completely")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}")
            return False

# 사용 예시
def main():
    """테스트 함수"""
    processor = DocumentProcessor()
    
    # 모든 파일 처리
    result = processor.process_all_files()
    print(f"Processing result: {result}")
    
    # 벡터 스토어 정보 조회
    info = processor.get_vector_store_info()
    print(f"Vector store info: {info}")

if __name__ == "__main__":
    main()
