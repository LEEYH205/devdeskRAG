"""
DevDesk-RAG 고급 검색 시스템
통합된 고급 검색, 재랭킹, 개인화 검색 기능

Features:
- 고급 검색 알고리즘 (하이브리드 검색, 동적 가중치 최적화)
- 재랭킹 시스템 (Together API, 컨텍스트 기반, 피드백 학습)
- 개인화 검색 (사용자별 프로필, 실시간 학습)
- A/B 테스트 및 성능 모니터링
- 통합 대시보드 및 분석 도구
"""

# 고급 검색 엔진
from .advanced_search import (
    advanced_search_engine,
    get_search_insights,
    AdvancedSearchEngine,
    WeightOptimizer,
    ABTestFramework,
    PerformanceAnalyzer,
    SearchResult,
    SearchMetrics,
    SearchAlgorithm
)

# 재랭킹 시스템
from .rerank_system import (
    advanced_rerank_system,
    RerankStrategy,
    RerankResult,
    RerankRequest,
    AdvancedRerankSystem,
    ContextAwareReranker,
    FeedbackLearningReranker,
    TogetherRerankClient
)

# 개인화 검색 시스템
from .personalized_search import (
    behavior_tracker,
    personalized_search_engine,
    real_time_learning_system,
    UserBehavior,
    UserProfile,
    SearchSession,
    PersonalizedResult,
    SearchContext
)

__version__ = "2.2.0"
__author__ = "DevDesk-RAG Team"
__all__ = [
    # 고급 검색
    "advanced_search_engine",
    "get_search_insights",
    "AdvancedSearchEngine",
    "WeightOptimizer", 
    "ABTestFramework",
    "PerformanceAnalyzer",
    "SearchResult",
    "SearchMetrics",
    "SearchAlgorithm",
    
    # 재랭킹
    "advanced_rerank_system",
    "RerankStrategy",
    "RerankResult",
    "RerankRequest",
    "AdvancedRerankSystem",
    "ContextAwareReranker",
    "FeedbackLearningReranker",
    "TogetherRerankClient",
    
    # 개인화 검색
    "behavior_tracker",
    "personalized_search_engine",
    "real_time_learning_system",
    "UserBehavior",
    "UserProfile",
    "SearchSession",
    "PersonalizedResult",
    "SearchContext"
]
