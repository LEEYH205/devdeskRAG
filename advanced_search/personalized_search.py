"""
DevDesk-RAG 개인화 검색 시스템
사용자별 검색 히스토리 기반 개인화 및 실시간 학습

Features:
- 사용자별 검색 패턴 분석
- 선호도 기반 가중치 조정
- 실시간 학습 및 개선
- 개인화된 검색 결과 제공
"""

import os
import time
import logging
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime, timedelta
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserBehavior(Enum):
    """사용자 행동 유형"""
    SEARCH = "search"           # 검색
    CLICK = "click"            # 결과 클릭
    SCROLL = "scroll"          # 스크롤
    STAY = "stay"              # 체류
    FEEDBACK = "feedback"      # 피드백
    UPLOAD = "upload"          # 파일 업로드

class SearchContext(Enum):
    """검색 컨텍스트 유형"""
    TECHNICAL = "technical"     # 기술 문서
    GENERAL = "general"         # 일반 문서
    CODE = "code"              # 코드 관련
    TUTORIAL = "tutorial"      # 튜토리얼
    REFERENCE = "reference"    # 참조 문서

@dataclass
class UserProfile:
    """사용자 프로필"""
    user_id: str
    created_at: datetime
    last_active: datetime
    search_count: int = 0
    total_clicks: int = 0
    avg_session_time: float = 0.0
    preferred_topics: List[str] = None
    search_patterns: Dict[str, int] = None
    
    def __post_init__(self):
        if self.preferred_topics is None:
            self.preferred_topics = []
        if self.search_patterns is None:
            self.search_patterns = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'search_count': self.search_count,
            'total_clicks': self.total_clicks,
            'avg_session_time': self.avg_session_time,
            'preferred_topics': self.preferred_topics,
            'search_patterns': self.search_patterns
        }

@dataclass
class SearchSession:
    """검색 세션"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    queries: List[str] = None
    clicked_results: List[str] = None
    session_duration: float = 0.0
    
    def __post_init__(self):
        if self.queries is None:
            self.queries = []
        if self.clicked_results is None:
            self.clicked_results = []
    
    def end_session(self):
        """세션 종료"""
        self.end_time = datetime.now()
        if self.start_time:
            self.session_duration = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'queries': self.queries,
            'clicked_results': self.clicked_results,
            'session_duration': self.session_duration
        }

@dataclass
class PersonalizedResult:
    """개인화된 검색 결과"""
    document_id: str
    original_score: float
    personalized_score: float
    personalization_factors: Dict[str, float]
    user_relevance: float
    context_match: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'original_score': self.original_score,
            'personalized_score': self.personalized_score,
            'personalization_factors': self.personalization_factors,
            'user_relevance': self.user_relevance,
            'context_match': self.context_match
        }

class UserBehaviorTracker:
    """사용자 행동 추적기"""
    
    def __init__(self, db_path: str = "personalization.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    last_active TEXT,
                    search_count INTEGER,
                    total_clicks INTEGER,
                    avg_session_time REAL,
                    preferred_topics TEXT,
                    search_patterns TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    queries TEXT,
                    clicked_results TEXT,
                    session_duration REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_behaviors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_id TEXT,
                    behavior_type TEXT,
                    query TEXT,
                    document_id TEXT,
                    timestamp TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_contexts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    context_type TEXT,
                    confidence REAL,
                    timestamp TEXT
                )
            """)
    
    def track_behavior(self, user_id: str, session_id: str, behavior: UserBehavior, 
                      query: str = "", document_id: str = "", metadata: Dict[str, Any] = None):
        """사용자 행동 추적"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_behaviors 
                    (user_id, session_id, behavior_type, query, document_id, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, session_id, behavior.value, query, document_id,
                    datetime.now().isoformat(),
                    json.dumps(metadata) if metadata else "{}"
                ))
                
                # 사용자 프로필 업데이트
                self._update_user_profile(user_id, behavior)
                
        except Exception as e:
            logger.error(f"행동 추적 실패: {e}")
    
    def _update_user_profile(self, user_id: str, behavior: UserBehavior):
        """사용자 프로필 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 기존 프로필 확인
                cursor = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
                profile = cursor.fetchone()
                
                if profile:
                    # 기존 프로필 업데이트
                    if behavior == UserBehavior.SEARCH:
                        conn.execute("""
                            UPDATE user_profiles 
                            SET search_count = search_count + 1, last_active = ?
                            WHERE user_id = ?
                        """, (datetime.now().isoformat(), user_id))
                    elif behavior == UserBehavior.CLICK:
                        conn.execute("""
                            UPDATE user_profiles 
                            SET total_clicks = total_clicks + 1, last_active = ?
                            WHERE user_id = ?
                        """, (datetime.now().isoformat(), user_id))
                else:
                    # 새 프로필 생성
                    conn.execute("""
                        INSERT INTO user_profiles 
                        (user_id, created_at, last_active, search_count, total_clicks, 
                         avg_session_time, preferred_topics, search_patterns)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, datetime.now().isoformat(), datetime.now().isoformat(),
                        1 if behavior == UserBehavior.SEARCH else 0,
                        1 if behavior == UserBehavior.CLICK else 0,
                        0.0, "[]", "{}"
                    ))
                    
        except Exception as e:
            logger.error(f"프로필 업데이트 실패: {e}")

class PersonalizedSearchEngine:
    """개인화 검색 엔진"""
    
    def __init__(self, behavior_tracker: UserBehaviorTracker):
        self.behavior_tracker = behavior_tracker
        self.user_profiles = {}
        self.search_contexts = {}
        self.learning_rate = 0.01
        self.decay_factor = 0.95
        
        # 개인화 가중치
        self.personalization_weights = {
            'topic_preference': 0.3,
            'search_history': 0.25,
            'click_behavior': 0.25,
            'context_similarity': 0.2
        }
    
    def personalize_search_results(self, user_id: str, query: str, 
                                original_results: List[Dict[str, Any]]) -> List[PersonalizedResult]:
        """검색 결과 개인화"""
        try:
            # 사용자 프로필 로드
            user_profile = self._load_user_profile(user_id)
            
            # 검색 컨텍스트 분석
            context_type = self._analyze_search_context(query)
            
            # 개인화된 결과 생성
            personalized_results = []
            
            for result in original_results:
                # 개인화 점수 계산
                personalization_score = self._calculate_personalization_score(
                    user_profile, result, query, context_type
                )
                
                # 원본 점수와 개인화 점수 결합
                final_score = self._combine_scores(
                    result.get('score', 0.0), personalization_score
                )
                
                personalized_result = PersonalizedResult(
                    document_id=result.get('id', ''),
                    original_score=result.get('score', 0.0),
                    personalized_score=final_score,
                    personalization_factors=personalization_score,
                    user_relevance=self._calculate_user_relevance(user_profile, result),
                    context_match=self._calculate_context_match(context_type, result)
                )
                
                personalized_results.append(personalized_result)
            
            # 개인화 점수로 재정렬
            personalized_results.sort(key=lambda x: x.personalized_score, reverse=True)
            
            # 행동 추적
            self.behavior_tracker.track_behavior(
                user_id, self._get_session_id(user_id), UserBehavior.SEARCH, query
            )
            
            return personalized_results
            
        except Exception as e:
            logger.error(f"개인화 검색 실패: {e}")
            return self._create_fallback_results(original_results)
    
    def _load_user_profile(self, user_id: str) -> UserProfile:
        """사용자 프로필 로드"""
        try:
            with sqlite3.connect(self.behavior_tracker.db_path) as conn:
                cursor = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
                profile_data = cursor.fetchone()
                
                if profile_data:
                    return UserProfile(
                        user_id=profile_data[0],
                        created_at=datetime.fromisoformat(profile_data[1]),
                        last_active=datetime.fromisoformat(profile_data[2]),
                        search_count=profile_data[3],
                        total_clicks=profile_data[4],
                        avg_session_time=profile_data[5],
                        preferred_topics=json.loads(profile_data[6]),
                        search_patterns=json.loads(profile_data[7])
                    )
                else:
                    # 기본 프로필 생성
                    return UserProfile(
                        user_id=user_id,
                        created_at=datetime.now(),
                        last_active=datetime.now()
                    )
                    
        except Exception as e:
            logger.error(f"프로필 로드 실패: {e}")
            return UserProfile(
                user_id=user_id,
                created_at=datetime.now(),
                last_active=datetime.now()
            )
    
    def _analyze_search_context(self, query: str) -> SearchContext:
        """검색 컨텍스트 분석"""
        query_lower = query.lower()
        
        # 기술적 키워드
        technical_keywords = ['api', 'function', 'class', 'method', 'algorithm', 'architecture', 'system']
        if any(keyword in query_lower for keyword in technical_keywords):
            return SearchContext.TECHNICAL
        
        # 코드 관련 키워드
        code_keywords = ['code', 'programming', 'syntax', 'bug', 'error', 'debug']
        if any(keyword in query_lower for keyword in code_keywords):
            return SearchContext.CODE
        
        # 튜토리얼 키워드
        tutorial_keywords = ['how to', 'tutorial', 'guide', 'example', 'step by step']
        if any(keyword in query_lower for keyword in tutorial_keywords):
            return SearchContext.TUTORIAL
        
        # 참조 키워드
        reference_keywords = ['reference', 'documentation', 'manual', 'specification']
        if any(keyword in query_lower for keyword in reference_keywords):
            return SearchContext.REFERENCE
        
        return SearchContext.GENERAL
    
    def _calculate_personalization_score(self, user_profile: UserProfile, 
                                       result: Dict[str, Any], query: str, 
                                       context_type: SearchContext) -> Dict[str, float]:
        """개인화 점수 계산"""
        scores = {}
        
        # 1. 토픽 선호도 점수
        topic_score = self._calculate_topic_preference_score(user_profile, result)
        scores['topic_preference'] = topic_score
        
        # 2. 검색 히스토리 점수
        history_score = self._calculate_search_history_score(user_profile, query)
        scores['search_history'] = history_score
        
        # 3. 클릭 행동 점수
        click_score = self._calculate_click_behavior_score(user_profile, result)
        scores['click_behavior'] = click_score
        
        # 4. 컨텍스트 유사도 점수
        context_score = self._calculate_context_similarity_score(context_type, result)
        scores['context_similarity'] = context_score
        
        return scores
    
    def _calculate_topic_preference_score(self, user_profile: UserProfile, 
                                        result: Dict[str, Any]) -> float:
        """토픽 선호도 점수 계산"""
        if not user_profile.preferred_topics:
            return 0.5
        
        # 문서 메타데이터에서 토픽 추출
        document_topics = result.get('metadata', {}).get('topics', [])
        if not document_topics:
            return 0.5
        
        # 선호도와 매칭되는 토픽 수
        matching_topics = set(user_profile.preferred_topics) & set(document_topics)
        
        if not user_profile.preferred_topics:
            return 0.5
        
        return len(matching_topics) / len(user_profile.preferred_topics)
    
    def _calculate_search_history_score(self, user_profile: UserProfile, query: str) -> float:
        """검색 히스토리 점수 계산"""
        if not user_profile.search_patterns:
            return 0.5
        
        # 쿼리와 유사한 이전 검색 패턴 찾기
        query_words = set(query.lower().split())
        max_similarity = 0.0
        
        for pattern, count in user_profile.search_patterns.items():
            pattern_words = set(pattern.lower().split())
            if pattern_words:
                similarity = len(query_words & pattern_words) / len(query_words | pattern_words)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_click_behavior_score(self, user_profile: UserProfile, 
                                       result: Dict[str, Any]) -> float:
        """클릭 행동 점수 계산"""
        if user_profile.total_clicks == 0:
            return 0.5
        
        # 클릭률 기반 점수 (간단한 구현)
        click_rate = min(user_profile.total_clicks / max(user_profile.search_count, 1), 1.0)
        return click_rate
    
    def _calculate_context_similarity_score(self, context_type: SearchContext, 
                                          result: Dict[str, Any]) -> float:
        """컨텍스트 유사도 점수 계산"""
        # 문서의 컨텍스트 타입과 검색 컨텍스트 타입 매칭
        document_context = result.get('metadata', {}).get('context_type', SearchContext.GENERAL.value)
        
        if document_context == context_type.value:
            return 1.0
        elif document_context == SearchContext.GENERAL.value:
            return 0.7
        else:
            return 0.3
    
    def _combine_scores(self, original_score: float, personalization_scores: Dict[str, float]) -> float:
        """원본 점수와 개인화 점수 결합"""
        weighted_personalization = sum(
            score * self.personalization_weights[factor]
            for factor, score in personalization_scores.items()
            if factor in self.personalization_weights
        )
        
        # 원본 점수 70%, 개인화 점수 30%
        final_score = (original_score * 0.7) + (weighted_personalization * 0.3)
        return max(0.0, min(1.0, final_score))
    
    def _calculate_user_relevance(self, user_profile: UserProfile, 
                                 result: Dict[str, Any]) -> float:
        """사용자 관련성 점수"""
        # 사용자 활동 수준 기반 점수
        activity_score = min(user_profile.search_count / 100, 1.0)
        return 0.5 + (activity_score * 0.5)
    
    def _calculate_context_match(self, context_type: SearchContext, 
                                result: Dict[str, Any]) -> float:
        """컨텍스트 매칭 점수"""
        document_context = result.get('metadata', {}).get('context_type', SearchContext.GENERAL.value)
        
        if document_context == context_type.value:
            return 1.0
        elif document_context == SearchContext.GENERAL.value:
            return 0.6
        else:
            return 0.3
    
    def _get_session_id(self, user_id: str) -> str:
        """세션 ID 생성"""
        return hashlib.md5(f"{user_id}_{int(time.time() / 3600)}".encode()).hexdigest()
    
    def _create_fallback_results(self, original_results: List[Dict[str, Any]]) -> List[PersonalizedResult]:
        """폴백 결과 생성"""
        return [
            PersonalizedResult(
                document_id=result.get('id', ''),
                original_score=result.get('score', 0.0),
                personalized_score=result.get('score', 0.0),
                personalization_factors={},
                user_relevance=0.5,
                context_match=0.5
            )
            for result in original_results
        ]

class RealTimeLearningSystem:
    """실시간 학습 시스템"""
    
    def __init__(self, behavior_tracker: UserBehaviorTracker):
        self.behavior_tracker = behavior_tracker
        self.learning_rate = 0.01
        self.min_samples = 10
        
    def learn_from_feedback(self, user_id: str, query: str, document_id: str, 
                           feedback_score: float, context: Dict[str, Any] = None):
        """사용자 피드백으로부터 학습"""
        try:
            # 피드백 데이터 저장
            self._store_feedback(user_id, query, document_id, feedback_score, context)
            
            # 사용자 프로필 업데이트
            self._update_user_preferences(user_id, query, document_id, feedback_score)
            
            # 검색 컨텍스트 학습
            if context:
                self._learn_search_context(user_id, query, context)
            
            logger.info(f"피드백 학습 완료: 사용자 {user_id}, 문서 {document_id}, 점수 {feedback_score}")
            
        except Exception as e:
            logger.error(f"피드백 학습 실패: {e}")
    
    def _store_feedback(self, user_id: str, query: str, document_id: str, 
                       feedback_score: float, context: Dict[str, Any] = None):
        """피드백 데이터 저장"""
        try:
            with sqlite3.connect(self.behavior_tracker.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_behaviors 
                    (user_id, session_id, behavior_type, query, document_id, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, self._get_session_id(user_id), UserBehavior.FEEDBACK.value,
                    query, document_id, datetime.now().isoformat(),
                    json.dumps({'feedback_score': feedback_score, 'context': context or {}})
                ))
                
        except Exception as e:
            logger.error(f"피드백 저장 실패: {e}")
    
    def _update_user_preferences(self, user_id: str, query: str, document_id: str, 
                                feedback_score: float):
        """사용자 선호도 업데이트"""
        try:
            with sqlite3.connect(self.behavior_tracker.db_path) as conn:
                # 기존 프로필 로드
                cursor = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
                profile_data = cursor.fetchone()
                
                if profile_data:
                    # 검색 패턴 업데이트
                    search_patterns = json.loads(profile_data[7])
                    search_patterns[query] = search_patterns.get(query, 0) + 1
                    
                    # 선호도 업데이트 (간단한 구현)
                    if feedback_score > 0.7:  # 높은 만족도
                        # 문서 메타데이터에서 토픽 추출하여 선호도에 추가
                        # 실제 구현에서는 문서 메타데이터를 활용해야 함
                        pass
                    
                    # 업데이트된 프로필 저장
                    conn.execute("""
                        UPDATE user_profiles 
                        SET search_patterns = ?
                        WHERE user_id = ?
                    """, (json.dumps(search_patterns), user_id))
                    
        except Exception as e:
            logger.error(f"선호도 업데이트 실패: {e}")
    
    def _learn_search_context(self, user_id: str, query: str, context: Dict[str, Any]):
        """검색 컨텍스트 학습"""
        try:
            with sqlite3.connect(self.behavior_tracker.db_path) as conn:
                # 컨텍스트 타입 추출
                context_type = context.get('type', 'general')
                confidence = context.get('confidence', 0.5)
                
                conn.execute("""
                    INSERT INTO search_contexts 
                    (user_id, query, context_type, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, query, context_type, confidence, datetime.now().isoformat()))
                
        except Exception as e:
            logger.error(f"컨텍스트 학습 실패: {e}")
    
    def _get_session_id(self, user_id: str) -> str:
        """세션 ID 생성"""
        return hashlib.md5(f"{user_id}_{int(time.time() / 3600)}".encode()).hexdigest()

# 전역 인스턴스
behavior_tracker = UserBehaviorTracker()
personalized_search_engine = PersonalizedSearchEngine(behavior_tracker)
real_time_learning_system = RealTimeLearningSystem(behavior_tracker)

if __name__ == "__main__":
    # 테스트 코드
    print("🚀 DevDesk-RAG 개인화 검색 시스템 테스트")
    
    # 사용자 행동 추적 테스트
    user_id = "test_user_001"
    session_id = "test_session_001"
    
    # 검색 행동 추적
    behavior_tracker.track_behavior(
        user_id, session_id, UserBehavior.SEARCH, "DevDesk-RAG 성능 최적화"
    )
    
    # 클릭 행동 추적
    behavior_tracker.track_behavior(
        user_id, session_id, UserBehavior.CLICK, "", "doc_001"
    )
    
    # 개인화 검색 테스트
    test_results = [
        {'id': 'doc1', 'score': 0.8, 'metadata': {'topics': ['performance', 'optimization']}},
        {'id': 'doc2', 'score': 0.7, 'metadata': {'topics': ['search', 'algorithm']}},
        {'id': 'doc3', 'score': 0.9, 'metadata': {'topics': ['rag', 'system']}}
    ]
    
    personalized_results = personalized_search_engine.personalize_search_results(
        user_id, "DevDesk-RAG 성능 최적화", test_results
    )
    
    print(f"✅ 개인화 검색 완료: {len(personalized_results)}개 결과")
    for i, result in enumerate(personalized_results[:3]):
        print(f"{i+1}. 문서 {result.document_id}: 원본 {result.original_score:.3f} → 개인화 {result.personalized_score:.3f}")
    
    # 실시간 학습 테스트
    real_time_learning_system.learn_from_feedback(
        user_id, "DevDesk-RAG 성능 최적화", "doc_001", 0.9
    )
    
    print("🎉 개인화 검색 시스템 테스트 완료!")
