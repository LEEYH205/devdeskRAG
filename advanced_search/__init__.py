"""
DevDesk-RAG 고급 검색 알고리즘 패키지
동적 가중치 조정, A/B 테스트, 성능 병목 분석, 자동 최적화
"""

from .advanced_search import (
    AdvancedSearchEngine,
    WeightOptimizer,
    ABTestFramework,
    PerformanceAnalyzer,
    SearchResult,
    SearchMetrics,
    SearchAlgorithm,
    advanced_search_engine,
    get_search_insights
)

__version__ = "1.0.0"
__author__ = "DevDesk-RAG Team"

__all__ = [
    "AdvancedSearchEngine",
    "WeightOptimizer", 
    "ABTestFramework",
    "PerformanceAnalyzer",
    "SearchResult",
    "SearchMetrics",
    "SearchAlgorithm",
    "advanced_search_engine",
    "get_search_insights"
]
