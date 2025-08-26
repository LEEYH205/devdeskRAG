"""
DevDesk-RAG 성능 모니터링 시스템
성능 메트릭 수집, 저장, 분석 및 대시보드 제공
"""

import time
import psutil
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    timestamp: float
    session_id: str
    user_id: str
    question: str
    
    # 응답 시간 메트릭
    search_time: float
    format_time: float
    prompt_time: float
    llm_time: float
    total_time: float
    
    # 검색 품질 메트릭
    relevance_score: float
    search_results_count: int
    chunks_retrieved: int
    
    # 시스템 리소스 메트릭
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    
    # API 호출 메트릭
    api_status: str = "success"  # "success", "error"
    error_message: Optional[str] = None
    
    # 사용자 피드백 (나중에 추가)
    user_rating: Optional[int] = None
    user_feedback: Optional[str] = None

class PerformanceMonitor:
    """DevDesk-RAG 성능 모니터링 시스템"""
    
    def __init__(self, db_path: str = "performance/performance_metrics.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=1000)  # 메모리 버퍼
        self.lock = threading.Lock()
        
        # 실시간 통계
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'avg_search_time': 0.0,
            'avg_llm_time': 0.0,
            'avg_relevance_score': 0.0,
            'memory_usage_history': deque(maxlen=100),
            'response_time_history': deque(maxlen=100)
        }
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 백그라운드 저장 스레드 시작
        self._start_background_saver()
    
    def _init_database(self):
        """성능 메트릭 데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 성능 메트릭 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        session_id TEXT,
                        user_id TEXT,
                        question TEXT,
                        search_time REAL,
                        format_time REAL,
                        prompt_time REAL,
                        llm_time REAL,
                        total_time REAL,
                        relevance_score REAL,
                        search_results_count INTEGER,
                        chunks_retrieved INTEGER,
                        memory_usage_mb REAL,
                        cpu_usage_percent REAL,
                        gpu_memory_mb REAL,
                        api_status TEXT,
                        error_message TEXT,
                        user_rating INTEGER,
                        user_feedback TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON performance_metrics(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_status ON performance_metrics(api_status)")
                
                conn.commit()
                logger.info("성능 모니터링 데이터베이스 초기화 완료")
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def _start_background_saver(self):
        """백그라운드에서 메트릭을 데이터베이스에 저장하는 스레드"""
        def save_metrics():
            while True:
                try:
                    time.sleep(5)  # 5초마다 저장
                    self._save_buffered_metrics()
                except Exception as e:
                    logger.error(f"백그라운드 저장 중 오류: {e}")
        
        thread = threading.Thread(target=save_metrics, daemon=True)
        thread.start()
        logger.info("백그라운드 메트릭 저장 스레드 시작")
    
    def _save_buffered_metrics(self):
        """버퍼된 메트릭을 데이터베이스에 저장"""
        if not self.metrics_buffer:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                with self.lock:
                    metrics_to_save = list(self.metrics_buffer)
                    self.metrics_buffer.clear()
                
                for metrics in metrics_to_save:
                    cursor.execute("""
                        INSERT INTO performance_metrics (
                            timestamp, session_id, user_id, question,
                            search_time, format_time, prompt_time, llm_time, total_time,
                            relevance_score, search_results_count, chunks_retrieved,
                            memory_usage_mb, cpu_usage_percent, gpu_memory_mb,
                            api_status, error_message, user_rating, user_feedback
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics.timestamp, metrics.session_id, metrics.user_id, metrics.question,
                        metrics.search_time, metrics.format_time, metrics.prompt_time, 
                        metrics.llm_time, metrics.total_time, metrics.relevance_score,
                        metrics.search_results_count, metrics.chunks_retrieved,
                        metrics.memory_usage_mb, metrics.cpu_usage_percent, metrics.gpu_memory_mb,
                        metrics.api_status, metrics.error_message, metrics.user_rating, metrics.user_feedback
                    ))
                
                conn.commit()
                logger.debug(f"{len(metrics_to_save)}개 메트릭 저장 완료")
                
        except Exception as e:
            logger.error(f"메트릭 저장 실패: {e}")
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """성능 메트릭 기록"""
        try:
            with self.lock:
                # 버퍼에 추가
                self.metrics_buffer.append(metrics)
                
                # 실시간 통계 업데이트
                self._update_real_time_stats(metrics)
                
                # 메모리 사용량 히스토리 추가
                self.stats['memory_usage_history'].append({
                    'timestamp': metrics.timestamp,
                    'memory_mb': metrics.memory_usage_mb
                })
                
                # 응답 시간 히스토리 추가
                self.stats['response_time_history'].append({
                    'timestamp': metrics.timestamp,
                    'total_time': metrics.total_time
                })
                
            logger.debug(f"메트릭 기록 완료: {metrics.session_id}")
            
        except Exception as e:
            logger.error(f"메트릭 기록 실패: {e}")
    
    def _update_real_time_stats(self, metrics: PerformanceMetrics):
        """실시간 통계 업데이트"""
        self.stats['total_requests'] += 1
        
        if metrics.api_status == 'success':
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        # 평균 응답 시간 업데이트 (이동 평균)
        current_avg = self.stats['avg_response_time']
        self.stats['avg_response_time'] = (current_avg * 0.9) + (metrics.total_time * 0.1)
        
        # 평균 검색 시간 업데이트
        current_search_avg = self.stats['avg_search_time']
        self.stats['avg_search_time'] = (current_search_avg * 0.9) + (metrics.search_time * 0.1)
        
        # 평균 LLM 시간 업데이트
        current_llm_avg = self.stats['avg_llm_time']
        self.stats['avg_llm_time'] = (current_llm_avg * 0.9) + (metrics.llm_time * 0.1)
        
        # 평균 관련성 점수 업데이트
        if metrics.relevance_score > 0:
            current_relevance_avg = self.stats['avg_relevance_score']
            self.stats['avg_relevance_score'] = (current_relevance_avg * 0.9) + (metrics.relevance_score * 0.1)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """현재 시스템 리소스 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 메모리 사용량
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # 디스크 사용량
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 네트워크 I/O
            network = psutil.net_io_counters()
            
            # GPU 메모리 (MPS/CUDA 지원 시)
            gpu_memory_mb = None
            try:
                import torch
                if torch.backends.mps.is_available():
                    # MPS GPU 메모리 정보 (제한적)
                    gpu_memory_mb = 0.0  # MPS에서는 직접적인 메모리 정보 접근 어려움
                elif torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except ImportError:
                pass
            
            return {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_percent': memory.percent,
                'disk_percent': disk_percent,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'gpu_memory_mb': gpu_memory_mb
            }
            
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 실패: {e}")
            return {}
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """지정된 시간 동안의 성능 요약"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 기본 통계
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_requests,
                        COUNT(CASE WHEN api_status = 'success' THEN 1 END) as successful_requests,
                        COUNT(CASE WHEN api_status = 'error' THEN 1 END) as failed_requests,
                        AVG(total_time) as avg_total_time,
                        AVG(search_time) as avg_search_time,
                        AVG(llm_time) as avg_llm_time,
                        AVG(relevance_score) as avg_relevance_score,
                        AVG(memory_usage_mb) as avg_memory_usage,
                        AVG(cpu_usage_percent) as avg_cpu_usage
                    FROM performance_metrics 
                    WHERE timestamp > ?
                """, (cutoff_time,))
                
                row = cursor.fetchone()
                if row:
                    basic_stats = {
                        'total_requests': row[0],
                        'successful_requests': row[1],
                        'failed_requests': row[2],
                        'avg_total_time': row[3] or 0.0,
                        'avg_search_time': row[4] or 0.0,
                        'avg_llm_time': row[5] or 0.0,
                        'avg_relevance_score': row[6] or 0.0,
                        'avg_memory_usage': row[7] or 0.0,
                        'avg_cpu_usage': row[8] or 0.0
                    }
                else:
                    basic_stats = {}
                
                # 응답 시간 분포
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN total_time < 1 THEN '0-1s'
                            WHEN total_time < 5 THEN '1-5s'
                            WHEN total_time < 10 THEN '5-10s'
                            WHEN total_time < 30 THEN '10-30s'
                            ELSE '30s+'
                        END as time_range,
                        COUNT(*) as count
                    FROM performance_metrics 
                    WHERE timestamp > ? AND total_time IS NOT NULL
                    GROUP BY time_range
                    ORDER BY 
                        CASE time_range
                            WHEN '0-1s' THEN 1
                            WHEN '1-5s' THEN 2
                            WHEN '5-10s' THEN 3
                            WHEN '10-30s' THEN 4
                            ELSE 5
                        END
                """, (cutoff_time,))
                
                response_time_distribution = dict(cursor.fetchall())
                
                # 시간대별 요청 수
                cursor.execute("""
                    SELECT 
                        strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                        COUNT(*) as count
                    FROM performance_metrics 
                    WHERE timestamp > ?
                    GROUP BY hour
                    ORDER BY hour
                """, (cutoff_time,))
                
                hourly_requests = dict(cursor.fetchall())
                
                return {
                    'basic_stats': basic_stats,
                    'response_time_distribution': response_time_distribution,
                    'hourly_requests': hourly_requests,
                    'current_stats': self.stats,
                    'system_metrics': self.get_system_metrics()
                }
                
        except Exception as e:
            logger.error(f"성능 요약 조회 실패: {e}")
            return {}
    
    def get_session_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        """특정 세션의 메트릭 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        timestamp, question, total_time, search_time, llm_time,
                        relevance_score, api_status, user_rating, user_feedback
                    FROM performance_metrics 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                """, (session_id,))
                
                rows = cursor.fetchall()
                return [
                    {
                        'timestamp': row[0],
                        'question': row[1],
                        'total_time': row[2],
                        'search_time': row[3],
                        'llm_time': row[4],
                        'relevance_score': row[5],
                        'api_status': row[6],
                        'user_rating': row[7],
                        'user_feedback': row[8]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"세션 메트릭 조회 실패: {e}")
            return []
    
    def cleanup_old_metrics(self, days: int = 30):
        """오래된 메트릭 데이터 정리"""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_time,))
                deleted_count = cursor.rowcount
                
                conn.commit()
                logger.info(f"{deleted_count}개의 오래된 메트릭 데이터 정리 완료")
                
        except Exception as e:
            logger.error(f"오래된 메트릭 정리 실패: {e}")

# 전역 성능 모니터 인스턴스
performance_monitor = PerformanceMonitor()

def record_chat_metrics(
    session_id: str,
    user_id: str,
    question: str,
    search_time: float,
    format_time: float,
    prompt_time: float,
    llm_time: float,
    relevance_score: float = 0.0,
    search_results_count: int = 0,
    chunks_retrieved: int = 0,
    api_status: str = "success",
    error_message: str = None
) -> None:
    """채팅 성능 메트릭 기록 헬퍼 함수"""
    try:
        # 시스템 메트릭 수집
        system_metrics = performance_monitor.get_system_metrics()
        
        # 총 응답 시간 계산
        total_time = search_time + format_time + prompt_time + llm_time
        
        # 메트릭 생성
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            session_id=session_id,
            user_id=user_id,
            question=question,
            search_time=search_time,
            format_time=format_time,
            prompt_time=prompt_time,
            llm_time=llm_time,
            total_time=total_time,
            relevance_score=relevance_score,
            search_results_count=search_results_count,
            chunks_retrieved=chunks_retrieved,
            memory_usage_mb=system_metrics.get('memory_mb', 0.0),
            cpu_usage_percent=system_metrics.get('cpu_percent', 0.0),
            gpu_memory_mb=system_metrics.get('gpu_memory_mb'),
            api_status=api_status,
            error_message=error_message
        )
        
        # 메트릭 기록
        performance_monitor.record_metrics(metrics)
        
    except Exception as e:
        logger.error(f"채팅 메트릭 기록 실패: {e}")

def get_performance_dashboard_data() -> Dict[str, Any]:
    """대시보드용 성능 데이터 조회"""
    return performance_monitor.get_performance_summary()

if __name__ == "__main__":
    # 테스트 코드
    print("🚀 DevDesk-RAG 성능 모니터링 시스템 테스트")
    
    # 샘플 메트릭 기록
    record_chat_metrics(
        session_id="test_session",
        user_id="test_user",
        question="테스트 질문입니다.",
        search_time=0.5,
        format_time=0.1,
        prompt_time=0.2,
        llm_time=2.0,
        relevance_score=0.85,
        search_results_count=5,
        chunks_retrieved=3
    )
    
    # 성능 요약 조회
    summary = get_performance_dashboard_data()
    print(f"📊 성능 요약: {summary['basic_stats']}")
    
    print("✅ 성능 모니터링 시스템 테스트 완료!")
