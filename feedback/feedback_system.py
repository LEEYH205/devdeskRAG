"""
DevDesk-RAG 사용자 피드백 시스템
사용자 만족도 평가, 개선 제안 수집, 피드백 분석 기능
"""

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import threading
import logging
from enum import Enum

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """피드백 유형"""
    RATING = "rating"           # 별점 평가
    IMPROVEMENT = "improvement" # 개선 제안
    BUG_REPORT = "bug_report"   # 버그 신고
    FEATURE_REQUEST = "feature" # 기능 요청
    GENERAL = "general"         # 일반 피드백

class FeedbackStatus(Enum):
    """피드백 처리 상태"""
    PENDING = "pending"         # 대기 중
    IN_PROGRESS = "in_progress" # 처리 중
    RESOLVED = "resolved"       # 해결됨
    CLOSED = "closed"           # 종료됨

@dataclass
class UserFeedback:
    """사용자 피드백 데이터 클래스"""
    id: Optional[int] = None
    timestamp: float = None
    session_id: str = ""
    user_id: str = ""
    question: str = ""
    answer: str = ""
    
    # 피드백 내용
    feedback_type: str = FeedbackType.RATING.value
    rating: Optional[int] = None  # 1-5 별점
    feedback_text: str = ""
    
    # 메타데이터
    response_time: float = 0.0
    search_quality: float = 0.0
    answer_relevance: float = 0.0
    
    # 처리 상태
    status: str = FeedbackStatus.PENDING.value
    admin_notes: str = ""
    resolved_at: Optional[float] = None
    
    # 시스템 정보
    user_agent: str = ""
    ip_address: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class FeedbackAnalyzer:
    """피드백 분석 및 인사이트 생성"""
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['좋다', '훌륭하다', '정확하다', '빠르다', '유용하다', '만족', '추천'],
            'negative': ['나쁘다', '느리다', '부정확하다', '불만족', '문제', '오류', '개선필요'],
            'neutral': ['보통', '일반적', '평범하다', '괜찮다']
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """텍스트 감정 분석"""
        text_lower = text.lower()
        scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[sentiment] += 1
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """주요 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용 가능)
        words = text.split()
        word_count = Counter(words)
        
        # 한 글자 단어 제외
        filtered_words = {word: count for word, count in word_count.items() 
                         if len(word) > 1 and count > 1}
        
        return sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def generate_insights(self, feedbacks: List[UserFeedback]) -> Dict[str, Any]:
        """피드백 데이터로부터 인사이트 생성"""
        if not feedbacks:
            return {}
        
        # 기본 통계
        total_feedbacks = len(feedbacks)
        avg_rating = sum(f.rating or 0 for f in feedbacks if f.rating) / len([f for f in feedbacks if f.rating])
        
        # 피드백 유형별 분포
        type_distribution = Counter(f.feedback_type for f in feedbacks)
        
        # 별점 분포
        rating_distribution = Counter(f.rating for f in feedbacks if f.rating)
        
        # 시간대별 분석
        hourly_feedback = Counter()
        for f in feedbacks:
            hour = datetime.fromtimestamp(f.timestamp).hour
            hourly_feedback[hour] += 1
        
        # 감정 분석
        all_text = " ".join([f.feedback_text for f in feedbacks if f.feedback_text])
        sentiment_scores = self.analyze_sentiment(all_text)
        
        # 키워드 분석
        keywords = self.extract_keywords(all_text)
        
        return {
            'total_feedbacks': total_feedbacks,
            'avg_rating': round(avg_rating, 2),
            'type_distribution': dict(type_distribution),
            'rating_distribution': dict(rating_distribution),
            'hourly_distribution': dict(hourly_feedback),
            'sentiment_analysis': sentiment_scores,
            'top_keywords': keywords,
            'satisfaction_rate': self._calculate_satisfaction_rate(feedbacks)
        }
    
    def _calculate_satisfaction_rate(self, feedbacks: List[UserFeedback]) -> float:
        """만족도 비율 계산 (4-5점 기준)"""
        rated_feedbacks = [f for f in feedbacks if f.rating]
        if not rated_feedbacks:
            return 0.0
        
        satisfied = len([f for f in rated_feedbacks if f.rating >= 4])
        return round(satisfied / len(rated_feedbacks) * 100, 1)

class FeedbackManager:
    """사용자 피드백 관리 시스템"""
    
    def __init__(self, db_path: str = "feedback/feedback.db"):
        self.db_path = db_path
        self.analyzer = FeedbackAnalyzer()
        self.lock = threading.Lock()
        
        # 데이터베이스 초기화
        self._init_database()
    
    def _init_database(self):
        """피드백 데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 피드백 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        session_id TEXT,
                        user_id TEXT,
                        question TEXT,
                        answer TEXT,
                        feedback_type TEXT,
                        rating INTEGER,
                        feedback_text TEXT,
                        response_time REAL,
                        search_quality REAL,
                        answer_relevance REAL,
                        status TEXT,
                        admin_notes TEXT,
                        resolved_at REAL,
                        user_agent TEXT,
                        ip_address TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON user_feedback(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON user_feedback(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON user_feedback(feedback_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating ON user_feedback(rating)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON user_feedback(status)")
                
                conn.commit()
                logger.info("피드백 데이터베이스 초기화 완료")
                
        except Exception as e:
            logger.error(f"피드백 데이터베이스 초기화 실패: {e}")
    
    def submit_feedback(self, feedback: UserFeedback) -> bool:
        """피드백 제출"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO user_feedback (
                        timestamp, session_id, user_id, question, answer,
                        feedback_type, rating, feedback_text, response_time,
                        search_quality, answer_relevance, status, user_agent, ip_address
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.timestamp, feedback.session_id, feedback.user_id,
                    feedback.question, feedback.answer, feedback.feedback_type,
                    feedback.rating, feedback.feedback_text, feedback.response_time,
                    feedback.search_quality, feedback.answer_relevance, feedback.status,
                    feedback.user_agent, feedback.ip_address
                ))
                
                conn.commit()
                logger.info(f"피드백 제출 완료: {feedback.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"피드백 제출 실패: {e}")
            return False
    
    def get_feedback_by_session(self, session_id: str) -> List[UserFeedback]:
        """세션별 피드백 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM user_feedback WHERE session_id = ? ORDER BY timestamp DESC
                """, (session_id,))
                
                rows = cursor.fetchall()
                return [self._row_to_feedback(row) for row in rows]
                
        except Exception as e:
            logger.error(f"세션 피드백 조회 실패: {e}")
            return []
    
    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """피드백 요약 통계"""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM user_feedback WHERE timestamp > ?
                """, (cutoff_time,))
                
                rows = cursor.fetchall()
                feedbacks = [self._row_to_feedback(row) for row in rows]
                
                return self.analyzer.generate_insights(feedbacks)
                
        except Exception as e:
            logger.error(f"피드백 요약 조회 실패: {e}")
            return {}
    
    def get_recent_feedbacks(self, limit: int = 50) -> List[UserFeedback]:
        """최근 피드백 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM user_feedback ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                return [self._row_to_feedback(row) for row in rows]
                
        except Exception as e:
            logger.error(f"최근 피드백 조회 실패: {e}")
            return []
    
    def update_feedback_status(self, feedback_id: int, status: str, admin_notes: str = "") -> bool:
        """피드백 상태 업데이트 (관리자용)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                resolved_at = time.time() if status == FeedbackStatus.RESOLVED.value else None
                
                cursor.execute("""
                    UPDATE user_feedback 
                    SET status = ?, admin_notes = ?, resolved_at = ?
                    WHERE id = ?
                """, (status, admin_notes, resolved_at, feedback_id))
                
                conn.commit()
                logger.info(f"피드백 상태 업데이트 완료: {feedback_id} -> {status}")
                return True
                
        except Exception as e:
            logger.error(f"피드백 상태 업데이트 실패: {e}")
            return False
    
    def _row_to_feedback(self, row: tuple) -> UserFeedback:
        """데이터베이스 행을 UserFeedback 객체로 변환"""
        return UserFeedback(
            id=row[0],
            timestamp=row[1],
            session_id=row[2],
            user_id=row[3],
            question=row[4],
            answer=row[5],
            feedback_type=row[6],
            rating=row[7],
            feedback_text=row[8],
            response_time=row[9],
            search_quality=row[10],
            answer_relevance=row[11],
            status=row[12],
            admin_notes=row[13],
            resolved_at=row[14],
            user_agent=row[15],
            ip_address=row[16]
        )
    
    def cleanup_old_feedbacks(self, days: int = 365):
        """오래된 피드백 데이터 정리"""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM user_feedback WHERE timestamp < ?", (cutoff_time,))
                deleted_count = cursor.rowcount
                
                conn.commit()
                logger.info(f"{deleted_count}개의 오래된 피드백 데이터 정리 완료")
                
        except Exception as e:
            logger.error(f"오래된 피드백 정리 실패: {e}")

# 전역 피드백 매니저 인스턴스
feedback_manager = FeedbackManager()

def submit_chat_feedback(
    session_id: str,
    user_id: str,
    question: str,
    answer: str,
    rating: int,
    feedback_text: str = "",
    feedback_type: str = FeedbackType.RATING.value,
    response_time: float = 0.0,
    search_quality: float = 0.0,
    answer_relevance: float = 0.0,
    user_agent: str = "",
    ip_address: str = ""
) -> bool:
    """채팅 피드백 제출 헬퍼 함수"""
    try:
        feedback = UserFeedback(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=answer,
            feedback_type=feedback_type,
            rating=rating,
            feedback_text=feedback_text,
            response_time=response_time,
            search_quality=search_quality,
            answer_relevance=answer_relevance,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        return feedback_manager.submit_feedback(feedback)
        
    except Exception as e:
        logger.error(f"채팅 피드백 제출 실패: {e}")
        return False

def get_feedback_analytics(days: int = 30) -> Dict[str, Any]:
    """피드백 분석 데이터 조회"""
    return feedback_manager.get_feedback_summary(days)

if __name__ == "__main__":
    # 테스트 코드
    print("💬 DevDesk-RAG 사용자 피드백 시스템 테스트")
    
    # 샘플 피드백 제출
    success = submit_chat_feedback(
        session_id="test_session_123",
        user_id="test_user_456",
        question="테스트 질문입니다.",
        answer="테스트 답변입니다.",
        rating=5,
        feedback_text="매우 만족스럽습니다!",
        response_time=2.5,
        search_quality=0.9,
        answer_relevance=0.95
    )
    
    if success:
        print("✅ 피드백 제출 성공!")
        
        # 피드백 분석
        analytics = get_feedback_analytics()
        print(f"📊 피드백 분석: {analytics}")
    else:
        print("❌ 피드백 제출 실패!")
    
    print("🎉 사용자 피드백 시스템 테스트 완료!")
