"""
DevDesk-RAG 고급 검색 알고리즘
동적 가중치 조정, A/B 테스트, 성능 병목 분석, 자동 최적화
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import logging
import random
import numpy as np
from enum import Enum

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchAlgorithm(Enum):
    """검색 알고리즘 유형"""
    VECTOR_ONLY = "vector_only"           # 벡터 검색만
    BM25_ONLY = "bm25_only"              # BM25만
    HYBRID = "hybrid"                     # 하이브리드 (벡터 + BM25)
    WEIGHTED_HYBRID = "weighted_hybrid"  # 가중치 기반 하이브리드
    ADAPTIVE = "adaptive"                 # 적응형 가중치

class SearchResult:
    """검색 결과 데이터 클래스"""
    
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, source: str):
        self.content = content
        self.metadata = metadata
        self.score = score
        self.source = source
        self.rank = 0
        self.relevance_score = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'score': self.score,
            'source': self.source,
            'rank': self.rank,
            'relevance_score': self.relevance_score
        }

@dataclass
class SearchMetrics:
    """검색 성능 메트릭"""
    timestamp: float
    query: str
    algorithm: str
    search_time: float
    results_count: int
    avg_score: float
    top_score: float
    user_feedback: Optional[float] = None
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class WeightOptimizer:
    """동적 가중치 최적화 시스템"""
    
    def __init__(self):
        self.weights_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.current_weights = {
            'vector_weight': 0.7,
            'bm25_weight': 0.3,
            'rerank_weight': 0.5
        }
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
    
    def update_weights(self, performance_metrics: List[SearchMetrics]) -> Dict[str, float]:
        """성능 메트릭을 기반으로 가중치 업데이트"""
        if not performance_metrics:
            return self.current_weights
        
        # 최근 성능 계산
        recent_metrics = performance_metrics[-100:]  # 최근 100개
        avg_relevance = np.mean([m.relevance_score for m in recent_metrics if m.relevance_score > 0])
        avg_search_time = np.mean([m.search_time for m in recent_metrics])
        
        # 가중치 조정 방향 결정
        if avg_relevance > 0.8:  # 높은 관련성
            # 현재 가중치 유지하면서 미세 조정
            self.current_weights['vector_weight'] += self.learning_rate * 0.1
            self.current_weights['bm25_weight'] -= self.learning_rate * 0.1
        elif avg_relevance < 0.6:  # 낮은 관련성
            # 더 균형잡힌 가중치로 조정
            self.current_weights['vector_weight'] = 0.6
            self.current_weights['bm25_weight'] = 0.4
        
        # 탐색적 조정 (10% 확률)
        if random.random() < self.exploration_rate:
            self._explore_new_weights()
        
        # 가중치 정규화
        total_weight = sum(self.current_weights.values())
        self.current_weights = {k: v/total_weight for k, v in self.current_weights.items()}
        
        # 히스토리 저장
        self.weights_history.append({
            'timestamp': time.time(),
            'weights': self.current_weights.copy(),
            'performance': {'relevance': avg_relevance, 'search_time': avg_search_time}
        })
        
        return self.current_weights
    
    def _explore_new_weights(self):
        """새로운 가중치 조합 탐색"""
        # 가우시안 노이즈 추가
        noise = np.random.normal(0, 0.05)
        self.current_weights['vector_weight'] += noise
        self.current_weights['bm25_weight'] -= noise
        
        # 경계 제한
        self.current_weights['vector_weight'] = max(0.3, min(0.9, self.current_weights['vector_weight']))
        self.current_weights['bm25_weight'] = max(0.1, min(0.7, self.current_weights['bm25_weight']))

class ABTestFramework:
    """A/B 테스트 프레임워크"""
    
    def __init__(self):
        self.experiments = {}
        self.user_assignments = {}
        self.results = defaultdict(list)
    
    def create_experiment(self, experiment_id: str, variants: List[Dict[str, Any]], 
                         traffic_split: List[float] = None) -> bool:
        """새로운 A/B 테스트 실험 생성"""
        try:
            if traffic_split is None:
                traffic_split = [1.0 / len(variants)] * len(variants)
            
            if abs(sum(traffic_split) - 1.0) > 0.01:
                raise ValueError("Traffic split must sum to 1.0")
            
            self.experiments[experiment_id] = {
                'variants': variants,
                'traffic_split': traffic_split,
                'start_time': time.time(),
                'status': 'active'
            }
            
            logger.info(f"A/B 테스트 실험 생성: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"A/B 테스트 실험 생성 실패: {e}")
            return False
    
    def get_variant(self, experiment_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """사용자에게 테스트 변형 할당"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        if experiment['status'] != 'active':
            return None
        
        # 사용자 할당 확인
        if user_id not in self.user_assignments:
            # 새로운 사용자 할당
            variant_index = self._assign_variant(experiment['traffic_split'])
            self.user_assignments[user_id] = variant_index
        
        variant_index = self.user_assignments[user_id]
        return experiment['variants'][variant_index]
    
    def _assign_variant(self, traffic_split: List[float]) -> int:
        """트래픽 분할에 따른 변형 할당"""
        rand = random.random()
        cumulative = 0
        
        for i, split in enumerate(traffic_split):
            cumulative += split
            if rand <= cumulative:
                return i
        
        return len(traffic_split) - 1
    
    def record_result(self, experiment_id: str, user_id: str, variant_index: int, 
                     metrics: Dict[str, Any]) -> bool:
        """A/B 테스트 결과 기록"""
        try:
            result = {
                'timestamp': time.time(),
                'user_id': user_id,
                'variant_index': variant_index,
                'metrics': metrics
            }
            
            self.results[experiment_id].append(result)
            logger.debug(f"A/B 테스트 결과 기록: {experiment_id} - {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"A/B 테스트 결과 기록 실패: {e}")
            return False
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """A/B 테스트 결과 분석"""
        if experiment_id not in self.results:
            return {}
        
        results = self.results[experiment_id]
        if not results:
            return {}
        
        # 변형별 성능 분석
        variant_performance = defaultdict(list)
        for result in results:
            variant_index = result['variant_index']
            variant_performance[variant_index].append(result['metrics'])
        
        analysis = {}
        for variant_index, metrics_list in variant_performance.items():
            if metrics_list:
                # 평균 성능 계산
                avg_relevance = np.mean([m.get('relevance_score', 0) for m in metrics_list])
                avg_search_time = np.mean([m.get('search_time', 0) for m in metrics_list])
                avg_user_feedback = np.mean([m.get('user_feedback', 0) for m in metrics_list if m.get('user_feedback')])
                
                analysis[f'variant_{variant_index}'] = {
                    'count': len(metrics_list),
                    'avg_relevance': round(avg_relevance, 3),
                    'avg_search_time': round(avg_search_time, 3),
                    'avg_user_feedback': round(avg_user_feedback, 3) if avg_user_feedback > 0 else None
                }
        
        return analysis

class PerformanceAnalyzer:
    """성능 병목 지점 분석 도구"""
    
    def __init__(self):
        self.bottleneck_history = deque(maxlen=1000)
        self.thresholds = {
            'search_time': 1.0,      # 1초 이상이면 병목
            'relevance_score': 0.6,   # 0.6 이하면 낮은 품질
            'results_count': 2        # 2개 이하면 부족한 결과
        }
    
    def analyze_bottleneck(self, metrics: SearchMetrics) -> Dict[str, Any]:
        """성능 병목 지점 분석"""
        bottlenecks = []
        
        # 검색 시간 병목
        if metrics.search_time > self.thresholds['search_time']:
            bottlenecks.append({
                'type': 'search_time',
                'severity': 'high' if metrics.search_time > 2.0 else 'medium',
                'value': metrics.search_time,
                'threshold': self.thresholds['search_time'],
                'suggestion': '검색 알고리즘 최적화 또는 인덱스 개선 필요'
            })
        
        # 관련성 점수 병목
        if metrics.relevance_score < self.thresholds['relevance_score']:
            bottlenecks.append({
                'type': 'relevance_score',
                'severity': 'high' if metrics.relevance_score < 0.4 else 'medium',
                'value': metrics.relevance_score,
                'threshold': self.thresholds['relevance_score'],
                'suggestion': '임베딩 모델 개선 또는 검색 파라미터 조정 필요'
            })
        
        # 결과 수 병목
        if metrics.results_count < self.thresholds['results_count']:
            bottlenecks.append({
                'type': 'results_count',
                'severity': 'medium',
                'value': metrics.results_count,
                'threshold': self.thresholds['results_count'],
                'suggestion': '검색 범위 확대 또는 청킹 전략 개선 필요'
            })
        
        # 병목 히스토리 저장
        if bottlenecks:
            self.bottleneck_history.append({
                'timestamp': metrics.timestamp,
                'query': metrics.query,
                'algorithm': metrics.algorithm,
                'bottlenecks': bottlenecks
            })
        
        return {
            'has_bottleneck': len(bottlenecks) > 0,
            'bottlenecks': bottlenecks,
            'overall_severity': self._calculate_overall_severity(bottlenecks)
        }
    
    def _calculate_overall_severity(self, bottlenecks: List[Dict[str, Any]]) -> str:
        """전체 병목 심각도 계산"""
        if not bottlenecks:
            return 'none'
        
        severity_scores = {'low': 1, 'medium': 2, 'high': 3}
        max_severity = max(severity_scores.get(b['severity'], 0) for b in bottlenecks)
        
        if max_severity >= 3:
            return 'critical'
        elif max_severity >= 2:
            return 'high'
        else:
            return 'low'
    
    def get_bottleneck_trends(self, hours: int = 24) -> Dict[str, Any]:
        """병목 트렌드 분석"""
        cutoff_time = time.time() - (hours * 3600)
        recent_bottlenecks = [b for b in self.bottleneck_history if b['timestamp'] > cutoff_time]
        
        if not recent_bottlenecks:
            return {}
        
        # 병목 유형별 발생 빈도
        bottleneck_types = defaultdict(int)
        algorithm_bottlenecks = defaultdict(int)
        
        for bottleneck in recent_bottlenecks:
            for b in bottleneck['bottlenecks']:
                bottleneck_types[b['type']] += 1
            algorithm_bottlenecks[bottleneck['algorithm']] += 1
        
        return {
            'total_bottlenecks': len(recent_bottlenecks),
            'bottleneck_types': dict(bottleneck_types),
            'algorithm_bottlenecks': dict(algorithm_bottlenecks),
            'trend': self._analyze_trend(recent_bottlenecks)
        }
    
    def _analyze_trend(self, bottlenecks: List[Dict[str, Any]]) -> str:
        """병목 트렌드 분석"""
        if len(bottlenecks) < 2:
            return 'insufficient_data'
        
        # 시간순으로 정렬
        sorted_bottlenecks = sorted(bottlenecks, key=lambda x: x['timestamp'])
        
        # 최근 1/3과 이전 1/3 비교
        third = len(sorted_bottlenecks) // 3
        recent = sorted_bottlenecks[-third:]
        previous = sorted_bottlenecks[:third]
        
        recent_count = len(recent)
        previous_count = len(previous)
        
        if recent_count < previous_count * 0.7:
            return 'improving'
        elif recent_count > previous_count * 1.3:
            return 'worsening'
        else:
            return 'stable'

class AdvancedSearchEngine:
    """고급 검색 엔진 - 동적 가중치 조정, A/B 테스트, 성능 병목 분석"""
    
    def __init__(self):
        self.weight_optimizer = WeightOptimizer()
        self.ab_test_framework = ABTestFramework()
        self.performance_analyzer = PerformanceAnalyzer()
        self.search_metrics_history = deque(maxlen=1000)
        
        # 실제 검색 시스템과의 연동을 위한 변수들
        self.vector_store = None
        self.retriever = None
        self.embedding_model = None
        self.bm25_retriever = None
        
        # 검색 설정
        self.search_config = {
            'max_results': 10,
            'min_score_threshold': 0.3,
            'rerank_enabled': True,
            'domain_weights': {
                'technical': {'vector': 0.8, 'bm25': 0.2},
                'general': {'vector': 0.6, 'bm25': 0.4},
                'code': {'vector': 0.7, 'bm25': 0.3}
            }
        }
        
        # 도메인 분류기 (간단한 키워드 기반)
        self.domain_classifier = {
            'technical': ['api', 'system', 'performance', 'algorithm', 'optimization', 'architecture'],
            'general': ['what', 'how', 'explain', 'describe', 'introduction', 'overview'],
            'code': ['code', 'implementation', 'function', 'class', 'method', 'syntax', 'error']
        }
    
    def initialize_search_system(self, vector_store, retriever, embedding_model, bm25_retriever=None):
        """실제 검색 시스템과 연동 초기화"""
        self.vector_store = vector_store
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.bm25_retriever = bm25_retriever
        
        logger.info("고급 검색 엔진 초기화 완료")
        logger.info(f"벡터 스토어: {type(vector_store).__name__}")
        logger.info(f"리트리버: {type(retriever).__name__}")
        logger.info(f"임베딩 모델: {type(embedding_model).__name__}")
        if bm25_retriever:
            logger.info(f"BM25 리트리버: {type(bm25_retriever).__name__}")
    
    def classify_query_domain(self, query: str) -> str:
        """쿼리의 도메인 분류"""
        query_lower = query.lower()
        
        for domain, keywords in self.domain_classifier.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return 'general'  # 기본값
    
    def get_domain_weights(self, domain: str) -> Dict[str, float]:
        """도메인별 가중치 반환"""
        return self.search_config['domain_weights'].get(domain, self.search_config['domain_weights']['general'])
    
    def search(self, query: str, user_id: str = None, 
               algorithm: str = None, use_ab_test: bool = True) -> List[SearchResult]:
        """고급 검색 실행"""
        start_time = time.time()
        
        # A/B 테스트 변형 할당
        if use_ab_test and user_id:
            variant = self.ab_test_framework.get_variant("search_algorithm_comparison", user_id)
            if variant:
                algorithm = variant['algorithm']
        
        # 알고리즘별 검색 실행
        if algorithm == 'vector_only':
            results = self._vector_search(query)
        elif algorithm == 'bm25_only':
            results = self._bm25_search(query)
        elif algorithm == 'hybrid':
            results = self._hybrid_search(query)
        elif algorithm == 'weighted_hybrid':
            results = self._weighted_hybrid_search(query)
        else:
            # 기본값: 가중치 기반 하이브리드
            results = self._weighted_hybrid_search(query)
        
        search_time = time.time() - start_time
        
        # 성능 메트릭 생성
        metrics = SearchMetrics(
            timestamp=start_time,
            query=query,
            algorithm=algorithm or 'weighted_hybrid',
            search_time=search_time,
            results_count=len(results),
            avg_score=np.mean([r.score for r in results]) if results else 0,
            top_score=results[0].score if results else 0
        )
        
        # 성능 분석
        bottleneck_analysis = self.performance_analyzer.analyze_bottleneck(metrics)
        
        # 메트릭 히스토리 저장
        self.search_metrics_history.append(metrics)
        
        # A/B 테스트 결과 기록
        if user_id and use_ab_test:
            variant_index = self._get_variant_index(algorithm)
            self.ab_test_framework.record_result(
                "search_algorithm_comparison",
                user_id,
                variant_index,
                {
                    'search_time': search_time,
                    'results_count': len(results),
                    'avg_score': metrics.avg_score,
                    'top_score': metrics.top_score,
                    'bottleneck_severity': bottleneck_analysis['overall_severity']
                }
            )
        
        # 가중치 최적화
        self.weight_optimizer.update_weights(list(self.search_metrics_history))
        
        return results
    
    def _vector_search(self, query: str) -> List[SearchResult]:
        """실제 벡터 검색 실행"""
        if not self.retriever:
            logger.warning("벡터 리트리버가 초기화되지 않음")
            return []
        
        try:
            # 기존 리트리버를 사용하여 검색
            docs = self.retriever.invoke(query)
            
            # SearchResult로 변환
            results = []
            for i, doc in enumerate(docs):
                # 간단한 점수 계산 (순위 기반)
                score = max(0.1, 1.0 - (i * 0.1))
                
                result = SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    source='vector_search'
                )
                result.rank = i + 1
                results.append(result)
            
            logger.debug(f"벡터 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    def _bm25_search(self, query: str) -> List[SearchResult]:
        """실제 BM25 검색 실행"""
        if not self.bm25_retriever:
            logger.warning("BM25 리트리버가 초기화되지 않음")
            return []
        
        try:
            # BM25 검색 실행
            docs = self.bm25_retriever.invoke(query)
            
            # SearchResult로 변환
            results = []
            for i, doc in enumerate(docs):
                # BM25 점수를 정규화 (0.1 ~ 1.0)
                score = max(0.1, min(1.0, doc.metadata.get('score', 0.5)))
                
                result = SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    source='bm25_search'
                )
                result.rank = i + 1
                results.append(result)
            
            logger.debug(f"BM25 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"BM25 검색 실패: {e}")
            return []
    
    def _hybrid_search(self, query: str) -> List[SearchResult]:
        """하이브리드 검색"""
        vector_results = self._vector_search(query)
        bm25_results = self._bm25_search(query)
        
        # 결과 병합 및 재랭킹
        all_results = vector_results + bm25_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:5]  # 상위 5개 결과
    
    def _weighted_hybrid_search(self, query: str) -> List[SearchResult]:
        """도메인별 가중치 기반 하이브리드 검색"""
        # 도메인 분류 및 가중치 결정
        domain = self.classify_query_domain(query)
        domain_weights = self.get_domain_weights(domain)
        
        logger.debug(f"쿼리 도메인: {domain}, 가중치: {domain_weights}")
        
        # 각 검색 알고리즘 실행
        vector_results = self._vector_search(query)
        bm25_results = self._bm25_search(query) if self.bm25_retriever else []
        
        # 도메인별 가중치 적용
        for result in vector_results:
            result.score *= domain_weights['vector']
            result.relevance_score = result.score  # 관련성 점수 설정
        
        for result in bm25_results:
            result.score *= domain_weights['bm25']
            result.relevance_score = result.score  # 관련성 점수 설정
        
        # 결과 병합 및 중복 제거
        all_results = self._merge_and_deduplicate_results(vector_results, bm25_results)
        
        # 최종 점수 계산 및 재랭킹
        all_results = self._calculate_final_scores(all_results, query)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # 최대 결과 수 제한
        max_results = self.search_config['max_results']
        return all_results[:max_results]
    
    def _merge_and_deduplicate_results(self, vector_results: List[SearchResult], 
                                     bm25_results: List[SearchResult]) -> List[SearchResult]:
        """검색 결과 병합 및 중복 제거"""
        merged_results = {}
        
        # 벡터 검색 결과 추가
        for result in vector_results:
            content_key = result.content[:100]  # 내용의 첫 100자로 키 생성
            if content_key not in merged_results:
                merged_results[content_key] = result
            else:
                # 더 높은 점수로 업데이트
                if result.score > merged_results[content_key].score:
                    merged_results[content_key] = result
        
        # BM25 검색 결과 추가/병합
        for result in bm25_results:
            content_key = result.content[:100]
            if content_key not in merged_results:
                merged_results[content_key] = result
            else:
                # 기존 결과와 점수 병합
                existing = merged_results[content_key]
                existing.score = (existing.score + result.score) / 2
                existing.relevance_score = max(existing.relevance_score, result.relevance_score)
        
        return list(merged_results.values())
    
    def _calculate_final_scores(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """최종 점수 계산 (컨텍스트 및 품질 고려)"""
        for result in results:
            # 기본 점수
            base_score = result.score
            
            # 컨텍스트 품질 점수 (메타데이터 기반)
            context_score = self._calculate_context_score(result, query)
            
            # 최종 점수 계산 (가중 평균)
            final_score = (base_score * 0.7) + (context_score * 0.3)
            result.score = min(1.0, max(0.1, final_score))
            
            # 관련성 점수 업데이트
            result.relevance_score = result.score
        
        return results
    
    def _calculate_context_score(self, result: SearchResult, query: str) -> float:
        """컨텍스트 품질 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 메타데이터 품질 평가
        metadata = result.metadata
        
        # 소스 파일 유형
        if 'source' in metadata:
            source = metadata['source'].lower()
            if source.endswith('.md') or source.endswith('.txt'):
                score += 0.1  # 마크다운/텍스트 파일 우선
            elif source.endswith('.pdf'):
                score += 0.05  # PDF 파일
        
        # 청크 길이 (적절한 길이 우선)
        content_length = len(result.content)
        if 100 <= content_length <= 1000:
            score += 0.1  # 적절한 길이
        elif content_length > 1000:
            score += 0.05  # 긴 내용
        
        # 쿼리와의 키워드 매칭
        query_words = set(query.lower().split())
        content_words = set(result.content.lower().split())
        keyword_overlap = len(query_words.intersection(content_words))
        if keyword_overlap > 0:
            score += min(0.2, keyword_overlap * 0.05)
        
        return min(1.0, score)
    
    def _get_variant_index(self, algorithm: str) -> int:
        """알고리즘에 따른 변형 인덱스 반환"""
        algorithm_mapping = {
            'vector_only': 0,
            'hybrid': 1,
            'weighted_hybrid': 2
        }
        return algorithm_mapping.get(algorithm, 2)
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """성능 인사이트 제공"""
        return {
            'current_weights': self.weight_optimizer.current_weights,
            'bottleneck_trends': self.performance_analyzer.get_bottleneck_trends(),
            'ab_test_results': self.ab_test_framework.get_experiment_results("search_algorithm_comparison"),
            'search_metrics_summary': self._get_metrics_summary()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """검색 메트릭 요약"""
        if not self.search_metrics_history:
            return {}
        
        recent_metrics = list(self.search_metrics_history)[-100:]  # 최근 100개
        
        return {
            'total_searches': len(self.search_metrics_history),
            'recent_searches': len(recent_metrics),
            'avg_search_time': np.mean([m.search_time for m in recent_metrics]),
            'avg_results_count': np.mean([m.results_count for m in recent_metrics]),
            'algorithm_distribution': self._get_algorithm_distribution(recent_metrics)
        }
    
    def _get_algorithm_distribution(self, metrics: List[SearchMetrics]) -> Dict[str, int]:
        """알고리즘별 사용 분포"""
        distribution = defaultdict(int)
        for metric in metrics:
            distribution[metric.algorithm] += 1
        return dict(distribution)

# 전역 고급 검색 엔진 인스턴스
advanced_search_engine = AdvancedSearchEngine()

def get_search_insights() -> Dict[str, Any]:
    """검색 성능 인사이트 조회"""
    return advanced_search_engine.get_performance_insights()

if __name__ == "__main__":
    # 테스트 코드
    print("🔍 DevDesk-RAG 고급 검색 알고리즘 테스트")
    
    # 검색 테스트
    results = advanced_search_engine.search(
        query="DevDesk-RAG 시스템의 성능은 어떨까요?",
        user_id="test_user_123",
        algorithm="weighted_hybrid"
    )
    
    print(f"✅ 검색 완료: {len(results)}개 결과")
    
    # 성능 인사이트 조회
    insights = get_search_insights()
    print(f"📊 성능 인사이트: {insights}")
    
    print("🎉 고급 검색 알고리즘 테스트 완료!")
