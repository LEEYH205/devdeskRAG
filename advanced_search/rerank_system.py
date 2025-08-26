"""
DevDesk-RAG 고급 재랭킹 시스템
Together API Rerank를 활용한 컨텍스트 기반 문서 재랭킹
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import httpx
import numpy as np
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RerankStrategy(Enum):
    """재랭킹 전략"""
    CONTEXT_AWARE = "context_aware"      # 컨텍스트 기반
    FEEDBACK_LEARNING = "feedback_learning"  # 피드백 학습
    HYBRID = "hybrid"                    # 하이브리드
    ADAPTIVE = "adaptive"                # 적응형

@dataclass
class RerankRequest:
    """재랭킹 요청 데이터"""
    query: str
    documents: List[str]
    metadata: List[Dict[str, Any]]
    strategy: RerankStrategy = RerankStrategy.CONTEXT_AWARE
    user_id: Optional[str] = None
    context: Optional[str] = None
    
    def to_together_format(self) -> Dict[str, Any]:
        """Together API 형식으로 변환"""
        return {
            "query": self.query,
            "documents": self.documents,
            "top_n": len(self.documents),
            "return_metadata": True
        }

@dataclass
class RerankResult:
    """재랭킹 결과 데이터"""
    document: str
    metadata: Dict[str, Any]
    original_rank: int
    new_rank: int
    rerank_score: float
    improvement: float
    strategy_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document': self.document,
            'metadata': self.metadata,
            'original_rank': self.original_rank,
            'new_rank': self.new_rank,
            'rerank_score': self.rerank_score,
            'improvement': self.improvement,
            'strategy_used': self.strategy_used
        }

class TogetherRerankClient:
    """Together API Rerank 클라이언트"""
    
    def __init__(self):
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1/rerank"
        self.model = "togethercomputer/m2-bert-80M-8k-base"  # 기본 재랭킹 모델
        
        if not self.api_key:
            logger.warning("TOGETHER_API_KEY가 설정되지 않음. 재랭킹 기능이 제한됩니다.")
        
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def rerank_documents(self, request: RerankRequest) -> List[RerankResult]:
        """문서 재랭킹 실행"""
        if not self.api_key:
            logger.warning("Together API 키가 없어 기본 재랭킹만 수행")
            return self._fallback_rerank(request)
        
        try:
            # Together API 호출
            response = await self._call_together_api(request)
            
            # 결과 파싱 및 변환
            results = self._parse_rerank_response(request, response)
            
            logger.info(f"재랭킹 완료: {len(results)}개 문서, 전략: {request.strategy.value}")
            return results
            
        except Exception as e:
            logger.error(f"Together API 재랭킹 실패: {e}")
            logger.info("기본 재랭킹으로 폴백")
            return self._fallback_rerank(request)
    
    async def _call_together_api(self, request: RerankRequest) -> Dict[str, Any]:
        """Together API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = request.to_together_format()
        
        async with self.client as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    def _parse_rerank_response(self, request: RerankRequest, response: Dict[str, Any]) -> List[RerankResult]:
        """API 응답 파싱"""
        results = []
        
        if 'results' not in response:
            logger.error(f"예상치 못한 API 응답 형식: {response}")
            return self._fallback_rerank(request)
        
        for i, result in enumerate(response['results']):
            original_rank = i
            new_rank = result.get('index', i)
            rerank_score = result.get('relevance_score', 0.0)
            
            # 개선도 계산
            improvement = self._calculate_improvement(original_rank, new_rank, rerank_score)
            
            rerank_result = RerankResult(
                document=request.documents[i],
                metadata=request.metadata[i] if request.metadata else {},
                original_rank=original_rank,
                new_rank=new_rank,
                rerank_score=rerank_score,
                improvement=improvement,
                strategy_used=request.strategy.value
            )
            results.append(rerank_result)
        
        # 재랭킹 점수로 정렬
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # 새로운 순위 업데이트
        for i, result in enumerate(results):
            result.new_rank = i
        
        return results
    
    def _fallback_rerank(self, request: RerankRequest) -> List[RerankResult]:
        """기본 재랭킹 (API 실패 시)"""
        logger.info("기본 재랭킹 수행")
        
        results = []
        for i, (doc, meta) in enumerate(zip(request.documents, request.metadata or [{}] * len(request.documents))):
            # 간단한 키워드 매칭 기반 점수 계산
            score = self._calculate_keyword_score(request.query, doc)
            
            rerank_result = RerankResult(
                document=doc,
                metadata=meta,
                original_rank=i,
                new_rank=i,
                rerank_score=score,
                improvement=0.0,
                strategy_used="fallback_keyword"
            )
            results.append(rerank_result)
        
        # 점수로 정렬
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i
        
        return results
    
    def _calculate_keyword_score(self, query: str, document: str) -> float:
        """키워드 매칭 기반 점수 계산"""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # 키워드 겹침 계산
        overlap = len(query_words.intersection(doc_words))
        total_query_words = len(query_words)
        
        if total_query_words == 0:
            return 0.0
        
        # 기본 점수 + 키워드 매칭 보너스
        base_score = 0.3
        keyword_bonus = (overlap / total_query_words) * 0.7
        
        return min(1.0, base_score + keyword_bonus)
    
    def _calculate_improvement(self, original_rank: int, new_rank: int, score: float) -> float:
        """순위 개선도 계산"""
        if original_rank == new_rank:
            return 0.0
        
        # 순위 개선 (낮은 숫자가 더 좋은 순위)
        rank_improvement = original_rank - new_rank
        
        # 점수 개선 (높은 점수가 더 좋음)
        score_improvement = score - 0.5  # 0.5를 기준점으로
        
        # 종합 개선도 (순위 70%, 점수 30%)
        total_improvement = (rank_improvement * 0.7) + (score_improvement * 0.3)
        
        return total_improvement

class ContextAwareReranker:
    """컨텍스트 기반 재랭킹 시스템"""
    
    def __init__(self):
        self.together_client = TogetherRerankClient()
        self.context_cache = {}
        self.strategy_weights = {
            RerankStrategy.CONTEXT_AWARE: 0.4,
            RerankStrategy.FEEDBACK_LEARNING: 0.3,
            RerankStrategy.HYBRID: 0.2,
            RerankStrategy.ADAPTIVE: 0.1
        }
    
    async def rerank_with_context(self, 
                                query: str, 
                                documents: List[str], 
                                metadata: List[Dict[str, Any]],
                                context: Optional[str] = None,
                                user_id: Optional[str] = None) -> List[RerankResult]:
        """컨텍스트를 고려한 재랭킹"""
        
        # 컨텍스트 분석
        enhanced_context = self._analyze_context(query, context, user_id)
        
        # 컨텍스트 기반 문서 강화
        enhanced_docs = self._enhance_documents_with_context(documents, enhanced_context)
        
        # 재랭킹 요청 생성
        request = RerankRequest(
            query=query,
            documents=enhanced_docs,
            metadata=metadata,
            strategy=RerankStrategy.CONTEXT_AWARE,
            user_id=user_id,
            context=enhanced_context
        )
        
        # Together API로 재랭킹
        results = await self.together_client.rerank_documents(request)
        
        # 컨텍스트 점수 보정
        results = self._apply_context_correction(results, enhanced_context)
        
        return results
    
    def _analyze_context(self, query: str, context: Optional[str], user_id: Optional[str]) -> str:
        """컨텍스트 분석 및 강화"""
        enhanced_context = query
        
        if context:
            enhanced_context += f" [Context: {context}]"
        
        if user_id and user_id in self.context_cache:
            user_context = self.context_cache[user_id]
            enhanced_context += f" [User History: {user_context}]"
        
        return enhanced_context
    
    def _enhance_documents_with_context(self, documents: List[str], context: str) -> List[str]:
        """컨텍스트를 고려한 문서 강화"""
        enhanced_docs = []
        
        for doc in documents:
            # 컨텍스트 키워드가 문서에 포함되어 있는지 확인
            context_keywords = set(context.lower().split())
            doc_keywords = set(doc.lower().split())
            
            # 컨텍스트 관련성 점수
            relevance = len(context_keywords.intersection(doc_keywords)) / max(len(context_keywords), 1)
            
            if relevance > 0.1:  # 10% 이상 관련성
                enhanced_doc = f"[Context-Relevant: {relevance:.2f}] {doc}"
            else:
                enhanced_doc = doc
            
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs
    
    def _apply_context_correction(self, results: List[RerankResult], context: str) -> List[RerankResult]:
        """컨텍스트 점수 보정"""
        for result in results:
            # 컨텍스트 관련성에 따른 점수 보정
            context_relevance = self._calculate_context_relevance(result.document, context)
            
            # 원래 점수와 컨텍스트 점수를 결합
            corrected_score = (result.rerank_score * 0.7) + (context_relevance * 0.3)
            result.rerank_score = min(1.0, corrected_score)
        
        # 보정된 점수로 재정렬
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i
        
        return results
    
    def _calculate_context_relevance(self, document: str, context: str) -> float:
        """컨텍스트와 문서의 관련성 계산"""
        context_words = set(context.lower().split())
        doc_words = set(document.lower().split())
        
        if not context_words:
            return 0.5
        
        overlap = len(context_words.intersection(doc_words))
        relevance = overlap / len(context_words)
        
        return min(1.0, relevance)

class FeedbackLearningReranker:
    """피드백 기반 학습 재랭킹 시스템"""
    
    def __init__(self):
        self.feedback_history = {}
        self.learning_rate = 0.01
        self.feedback_weights = {}
    
    def update_feedback(self, user_id: str, query: str, document_id: str, feedback_score: float):
        """사용자 피드백 업데이트"""
        if user_id not in self.feedback_history:
            self.feedback_history[user_id] = {}
        
        if query not in self.feedback_history[user_id]:
            self.feedback_history[user_id][query] = {}
        
        self.feedback_history[user_id][query][document_id] = feedback_score
        
        # 피드백 가중치 업데이트
        self._update_feedback_weights(user_id, query, document_id, feedback_score)
    
    def _update_feedback_weights(self, user_id: str, query: str, document_id: str, feedback_score: float):
        """피드백 가중치 업데이트"""
        key = f"{user_id}:{query}"
        
        if key not in self.feedback_weights:
            self.feedback_weights[key] = {}
        
        current_weight = self.feedback_weights[key].get(document_id, 0.5)
        
        # 피드백 점수에 따른 가중치 조정
        if feedback_score > 0.7:  # 높은 만족도
            new_weight = current_weight + self.learning_rate
        elif feedback_score < 0.3:  # 낮은 만족도
            new_weight = current_weight - self.learning_rate
        else:  # 중간 만족도
            new_weight = current_weight
        
        # 가중치 범위 제한 (0.1 ~ 1.0)
        self.feedback_weights[key][document_id] = max(0.1, min(1.0, new_weight))
    
    def apply_feedback_correction(self, results: List[RerankResult], user_id: str, query: str) -> List[RerankResult]:
        """피드백 기반 점수 보정"""
        key = f"{user_id}:{query}"
        
        if key not in self.feedback_weights:
            return results
        
        feedback_weights = self.feedback_weights[key]
        
        for result in results:
            # 문서 ID 추출 (메타데이터에서)
            doc_id = result.metadata.get('id', str(hash(result.document)))
            
            if doc_id in feedback_weights:
                # 피드백 가중치 적용
                feedback_weight = feedback_weights[doc_id]
                corrected_score = (result.rerank_score * 0.8) + (feedback_weight * 0.2)
                result.rerank_score = min(1.0, corrected_score)
        
        # 보정된 점수로 재정렬
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i
        
        return results

class AdvancedRerankSystem:
    """고급 재랭킹 시스템 (통합)"""
    
    def __init__(self):
        self.context_reranker = ContextAwareReranker()
        self.feedback_reranker = FeedbackLearningReranker()
        self.together_client = TogetherRerankClient()
        
        # 재랭킹 설정
        self.rerank_config = {
            'enabled': True,
            'min_documents': 3,  # 최소 3개 문서일 때만 재랭킹
            'max_documents': 20,  # 최대 20개 문서까지 재랭킹
            'context_weight': 0.4,
            'feedback_weight': 0.3,
            'rerank_weight': 0.3
        }
    
    async def rerank_documents(self, 
                              query: str, 
                              documents: List[str], 
                              metadata: List[Dict[str, Any]],
                              user_id: Optional[str] = None,
                              context: Optional[str] = None,
                              strategy: RerankStrategy = RerankStrategy.HYBRID) -> List[RerankResult]:
        """통합 재랭킹 실행"""
        
        if not self.rerank_config['enabled']:
            logger.info("재랭킹이 비활성화됨")
            return self._create_basic_results(documents, metadata)
        
        if len(documents) < self.rerank_config['min_documents']:
            logger.info(f"문서 수가 부족하여 재랭킹 건너뜀: {len(documents)}개")
            return self._create_basic_results(documents, metadata)
        
        if len(documents) > self.rerank_config['max_documents']:
            logger.info(f"문서 수가 많아 상위 {self.rerank_config['max_documents']}개만 재랭킹")
            documents = documents[:self.rerank_config['max_documents']]
            metadata = metadata[:self.rerank_config['max_documents']]
        
        try:
            # 1단계: 컨텍스트 기반 재랭킹
            context_results = await self.context_reranker.rerank_with_context(
                query, documents, metadata, context, user_id
            )
            
            # 2단계: 피드백 기반 보정
            if user_id:
                context_results = self.feedback_reranker.apply_feedback_correction(
                    context_results, user_id, query
                )
            
            # 3단계: 최종 점수 계산 및 정렬
            final_results = self._calculate_final_scores(context_results, strategy)
            
            logger.info(f"재랭킹 완료: {len(final_results)}개 문서, 전략: {strategy.value}")
            return final_results
            
        except Exception as e:
            logger.error(f"재랭킹 실패: {e}")
            return self._create_basic_results(documents, metadata)
    
    def _create_basic_results(self, documents: List[str], metadata: List[Dict[str, Any]]) -> List[RerankResult]:
        """기본 결과 생성 (재랭킹 실패 시)"""
        results = []
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            result = RerankResult(
                document=doc,
                metadata=meta,
                original_rank=i,
                new_rank=i,
                rerank_score=0.5,
                improvement=0.0,
                strategy_used="basic_no_rerank"
            )
            results.append(result)
        return results
    
    def _calculate_final_scores(self, results: List[RerankResult], strategy: RerankStrategy) -> List[RerankResult]:
        """최종 점수 계산"""
        for result in results:
            # 전략별 가중치 적용
            if strategy == RerankStrategy.CONTEXT_AWARE:
                result.rerank_score *= 1.0  # 컨텍스트 점수 그대로
            elif strategy == RerankStrategy.FEEDBACK_LEARNING:
                result.rerank_score *= 1.1  # 피드백 점수 10% 가중치
            elif strategy == RerankStrategy.HYBRID:
                result.rerank_score *= 1.05  # 하이브리드 5% 가중치
            elif strategy == RerankStrategy.ADAPTIVE:
                # 적응형 가중치 (점수에 따라 동적 조정)
                if result.rerank_score > 0.8:
                    result.rerank_score *= 1.1  # 높은 점수는 더 높게
                elif result.rerank_score < 0.4:
                    result.rerank_score *= 0.9  # 낮은 점수는 더 낮게
            
            # 점수 범위 제한
            result.rerank_score = max(0.0, min(1.0, result.rerank_score))
        
        # 최종 점수로 정렬
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i
        
        return results
    
    def update_user_feedback(self, user_id: str, query: str, document_id: str, feedback_score: float):
        """사용자 피드백 업데이트"""
        self.feedback_reranker.update_feedback(user_id, query, document_id, feedback_score)
    
    def get_rerank_stats(self) -> Dict[str, Any]:
        """재랭킹 통계 정보"""
        return {
            'enabled': self.rerank_config['enabled'],
            'min_documents': self.rerank_config['min_documents'],
            'max_documents': self.rerank_config['max_documents'],
            'feedback_users': len(self.feedback_reranker.feedback_history),
            'context_cache_size': len(self.context_reranker.context_cache),
            'together_api_available': bool(self.together_client.api_key)
        }

# 전역 재랭킹 시스템 인스턴스
advanced_rerank_system = AdvancedRerankSystem()

if __name__ == "__main__":
    # 테스트 코드
    async def test_rerank_system():
        print("🚀 DevDesk-RAG 고급 재랭킹 시스템 테스트")
        
        # 테스트 데이터
        query = "DevDesk-RAG 시스템의 성능은 어떨까요?"
        documents = [
            "DevDesk-RAG는 고급 검색 알고리즘을 사용합니다.",
            "성능 모니터링 시스템이 실시간으로 작동합니다.",
            "사용자 피드백을 기반으로 지속 개선됩니다."
        ]
        metadata = [
            {'id': 'doc1', 'source': 'performance.md'},
            {'id': 'doc2', 'source': 'monitoring.md'},
            {'id': 'doc3', 'source': 'feedback.md'}
        ]
        
        # 재랭킹 실행
        results = await advanced_rerank_system.rerank_documents(
            query=query,
            documents=documents,
            metadata=metadata,
            user_id="test_user",
            strategy=RerankStrategy.HYBRID
        )
        
        print(f"✅ 재랭킹 완료: {len(results)}개 결과")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.document[:50]}... (점수: {result.rerank_score:.3f})")
        
        # 통계 정보
        stats = advanced_rerank_system.get_rerank_stats()
        print(f"📊 재랭킹 통계: {stats}")
        
        print("🎉 고급 재랭킹 시스템 테스트 완료!")
    
    # 비동기 테스트 실행
    asyncio.run(test_rerank_system())
