"""
DevDesk-RAG 고급 검색 모듈
- 하이브리드 검색 (벡터 + BM25)
- 재랭킹 시스템
- 검색 품질 향상
"""

import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import ChatTogether, TogetherEmbeddings  # exaone 임베딩 추가
import logging

logger = logging.getLogger(__name__)

class AdvancedRetriever:
    """고급 검색 및 재랭킹 시스템"""
    
    def __init__(self, 
                 vector_store: Chroma,
                 embedding_model: HuggingFaceEmbeddings = None,
                 together_api_key: str = None,
                 search_k: int = 8,
                 rerank_k: int = 4):
        
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.together_api_key = together_api_key
        self.search_k = search_k
        self.rerank_k = rerank_k
        
        # BM25 검색기 초기화
        self.bm25_retriever = self._init_bm25_retriever()
        
        # 재랭킹 모델 초기화
        self.rerank_model = self._init_rerank_model()
    
    def _init_bm25_retriever(self) -> BM25Retriever:
        """BM25 검색기 초기화"""
        try:
            # 벡터 스토어에서 모든 문서 가져오기
            all_docs = self.vector_store.get()
            documents = []
            
            for i, content in enumerate(all_docs['documents']):
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            return BM25Retriever.from_documents(documents)
            
        except Exception as e:
            logger.warning(f"BM25 검색기 초기화 실패: {e}")
            return None
    
    def _init_rerank_model(self):
        """재랭킹 모델 초기화"""
        if not self.together_api_key:
            logger.info("Together API 키가 없어 재랭킹을 건너뜁니다")
            return None
        
        try:
            return ChatTogether(
                model="lgai/exaone-deep-32b",
                temperature=0.1,
                together_api_key=self.together_api_key
            )
        except Exception as e:
            logger.warning(f"재랭킹 모델 초기화 실패: {e}")
            return None
    
    def hybrid_search(self, query: str) -> List[Document]:
        """하이브리드 검색 (벡터 + BM25)"""
        try:
            # 1. 벡터 검색
            vector_docs = self.vector_store.similarity_search(query, k=self.search_k)
            vector_scores = [1.0 - (i * 0.1) for i in range(len(vector_docs))]  # 순위 기반 점수
            
            # 2. BM25 검색
            bm25_docs = []
            bm25_scores = []
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                bm25_scores = [1.0 - (i * 0.1) for i in range(len(bm25_docs))]
            
            # 3. 결과 통합 및 점수 계산
            combined_results = {}
            
            # 벡터 검색 결과 추가
            for i, doc in enumerate(vector_docs):
                doc_id = self._get_doc_id(doc)
                combined_results[doc_id] = {
                    'doc': doc,
                    'vector_score': vector_scores[i],
                    'bm25_score': 0.0,
                    'final_score': vector_scores[i]
                }
            
            # BM25 검색 결과 추가/업데이트
            for i, doc in enumerate(bm25_docs):
                doc_id = self._get_doc_id(doc)
                if doc_id in combined_results:
                    combined_results[doc_id]['bm25_score'] = bm25_scores[i]
                    combined_results[doc_id]['final_score'] = (
                        combined_results[doc_id]['vector_score'] * 0.7 + 
                        bm25_scores[i] * 0.3
                    )
                else:
                    combined_results[doc_id] = {
                        'doc': doc,
                        'vector_score': 0.0,
                        'bm25_score': bm25_scores[i],
                        'final_score': bm25_scores[i] * 0.3
                    }
            
            # 4. 최종 점수로 정렬
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x['final_score'],
                reverse=True
            )
            
            # 상위 결과 반환
            final_docs = [result['doc'] for result in sorted_results[:self.search_k]]
            
            logger.info(f"하이브리드 검색 완료: {len(final_docs)}개 문서")
            return final_docs
            
        except Exception as e:
            logger.error(f"하이브리드 검색 실패: {e}")
            # 폴백: 벡터 검색만 사용
            return self.vector_store.similarity_search(query, k=self.search_k)
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """문서 재랭킹"""
        if not self.rerank_model or len(documents) <= 1:
            return documents
        
        try:
            # 재랭킹을 위한 프롬프트 생성
            rerank_prompt = self._create_rerank_prompt(query, documents)
            
            # 재랭킹 실행
            response = self.rerank_model.invoke(rerank_prompt)
            
            # 응답 파싱 및 재정렬
            reranked_docs = self._parse_rerank_response(response.content, documents)
            
            logger.info(f"재랭킹 완료: {len(reranked_docs)}개 문서")
            return reranked_docs[:self.rerank_k]
            
        except Exception as e:
            logger.warning(f"재랭킹 실패: {e}")
            return documents[:self.rerank_k]
    
    def _create_rerank_prompt(self, query: str, documents: List[Document]) -> str:
        """재랭킹을 위한 프롬프트 생성"""
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"{i+1}. {doc.page_content[:300]}...\n\n"
        
        prompt = f"""
        다음 질문에 대해 주어진 문서들을 관련성 순으로 재정렬해주세요.
        
        질문: {query}
        
        문서들:
        {docs_text}
        
        가장 관련성이 높은 문서부터 순서대로 번호만 나열해주세요.
        예시: 3,1,5,2,4
        
        답변:
        """
        
        return prompt
    
    def _parse_rerank_response(self, response: str, documents: List[Document]) -> List[Document]:
        """재랭킹 응답 파싱"""
        try:
            # 응답에서 숫자 추출
            numbers = [int(n.strip()) for n in response.split(',') if n.strip().isdigit()]
            
            # 유효한 인덱스만 필터링
            valid_indices = [n-1 for n in numbers if 1 <= n <= len(documents)]
            
            # 재정렬된 문서 반환
            reranked = [documents[i] for i in valid_indices if i < len(documents)]
            
            # 누락된 문서 추가
            for i, doc in enumerate(documents):
                if i not in valid_indices:
                    reranked.append(doc)
            
            return reranked
            
        except Exception as e:
            logger.warning(f"재랭킹 응답 파싱 실패: {e}")
            return documents
    
    def _get_doc_id(self, doc: Document) -> str:
        """문서 ID 생성"""
        content_hash = hash(doc.page_content[:100])
        source = doc.metadata.get('source', 'unknown')
        return f"{source}_{content_hash}"
    
    def search(self, query: str) -> List[Document]:
        """통합 검색 (하이브리드 + 재랭킹)"""
        # 1. 하이브리드 검색
        hybrid_results = self.hybrid_search(query)
        
        # 2. 재랭킹
        final_results = self.rerank_documents(query, hybrid_results)
        
        return final_results

# 사용 예시
if __name__ == "__main__":
    # 환경 변수 로드
    from dotenv import load_dotenv
    load_dotenv()
    
    # 컴포넌트 초기화
    embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vs = Chroma(persist_directory="chroma_db", embedding_function=embed)
    
    # 고급 검색기 초기화
    advanced_retriever = AdvancedRetriever(
        vector_store=vs,
        embedding_model=embed,
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        search_k=8,
        rerank_k=4
    )
    
    # 테스트
    query = "DevDesk-RAG 시스템의 특징은?"
    results = advanced_retriever.search(query)
    
    print(f"검색 결과: {len(results)}개 문서")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:100]}...")
        print(f"   출처: {doc.metadata.get('source', 'unknown')}")
        print()
