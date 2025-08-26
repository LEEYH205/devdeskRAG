"""
DevDesk-RAG 채팅 히스토리 관리 모듈
- Redis 기반 대화 기록 저장
- 세션 관리 및 사용자별 대화 분리
- 대화 검색 및 필터링
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from redis.asyncio import Redis
import logging

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """Redis 기반 채팅 히스토리 관리자"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.session_expiry = 24 * 60 * 60  # 24시간
        
    async def connect(self):
        """Redis 연결"""
        try:
            self.redis = Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # 연결 테스트
            await self.redis.ping()
            logger.info("Redis 연결 성공")
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise
    
    async def disconnect(self):
        """Redis 연결 해제"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis 연결 해제")
    
    async def create_session(self, user_id: str = None) -> str:
        """새로운 채팅 세션 생성"""
        if not user_id:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        session_id = f"session_{uuid.uuid4().hex}"
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "message_count": 0
        }
        
        # 세션 정보 저장
        await self.redis.hset(
            f"session:{session_id}",
            mapping=session_data
        )
        
        # 세션 만료 시간 설정
        await self.redis.expire(f"session:{session_id}", self.session_expiry)
        
        # 사용자별 세션 목록에 추가
        await self.redis.sadd(f"user_sessions:{user_id}", session_id)
        await self.redis.expire(f"user_sessions:{user_id}", self.session_expiry)
        
        logger.info(f"새 세션 생성: {session_id} (사용자: {user_id})")
        return session_id
    
    async def add_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """메시지 추가"""
        try:
            # 메시지 ID 생성
            message_id = f"msg_{uuid.uuid4().hex}"
            
            # 메시지 데이터 준비
            message_data = {
                "id": message_id,
                "timestamp": datetime.now().isoformat(),
                "sender": message.get("sender", "unknown"),
                "content": message.get("content", ""),
                "metadata": json.dumps(message.get("metadata", {}))
            }
            
            # 메시지 저장
            await self.redis.hset(
                f"message:{session_id}:{message_id}",
                mapping=message_data
            )
            
            # 메시지 만료 시간 설정
            await self.redis.expire(f"message:{session_id}:{message_id}", self.session_expiry)
            
            # 세션의 메시지 목록에 추가
            await self.redis.lpush(f"session_messages:{session_id}", message_id)
            await self.redis.expire(f"session_messages:{session_id}", self.session_expiry)
            
            # 세션 정보 업데이트
            await self.redis.hincrby(f"session:{session_id}", "message_count", 1)
            await self.redis.hset(f"session:{session_id}", "last_activity", datetime.now().isoformat())
            
            logger.info(f"메시지 추가: {message_id} (세션: {session_id})")
            return True
            
        except Exception as e:
            logger.error(f"메시지 추가 실패: {e}")
            return False
    
    async def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """세션의 메시지 목록 조회"""
        try:
            # 메시지 ID 목록 조회
            message_ids = await self.redis.lrange(f"session_messages:{session_id}", 0, limit - 1)
            
            messages = []
            for msg_id in message_ids:
                message_data = await self.redis.hgetall(f"message:{session_id}:{msg_id}")
                if message_data:
                    # metadata 파싱
                    try:
                        metadata = json.loads(message_data.get("metadata", "{}"))
                        message_data["metadata"] = metadata
                    except:
                        message_data["metadata"] = {}
                    
                    messages.append(message_data)
            
            # 시간순 정렬 (최신 메시지가 마지막)
            messages.reverse()
            
            logger.info(f"세션 메시지 조회: {len(messages)}개 (세션: {session_id})")
            return messages
            
        except Exception as e:
            logger.error(f"세션 메시지 조회 실패: {e}")
            return []
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """사용자의 세션 목록 조회"""
        try:
            session_ids = await self.redis.smembers(f"user_sessions:{user_id}")
            
            sessions = []
            for session_id in session_ids:
                session_data = await self.redis.hgetall(f"session:{session_id}")
                if session_data:
                    # 최근 메시지 미리보기
                    recent_messages = await self.get_session_messages(session_id, 3)
                    session_data["recent_messages"] = recent_messages
                    sessions.append(session_data)
            
            # 최근 활동순 정렬
            sessions.sort(key=lambda x: x.get("last_activity", ""), reverse=True)
            
            logger.info(f"사용자 세션 조회: {len(sessions)}개 (사용자: {user_id})")
            return sessions
            
        except Exception as e:
            logger.error(f"사용자 세션 조회 실패: {e}")
            return []
    
    async def search_messages(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """세션 내 메시지 검색"""
        try:
            # 모든 메시지 조회
            all_messages = await self.get_session_messages(session_id, 1000)
            
            # 간단한 텍스트 검색
            matching_messages = []
            query_lower = query.lower()
            
            for message in all_messages:
                content = message.get("content", "").lower()
                if query_lower in content:
                    matching_messages.append(message)
            
            logger.info(f"메시지 검색: {len(matching_messages)}개 결과 (쿼리: {query})")
            return matching_messages
            
        except Exception as e:
            logger.error(f"메시지 검색 실패: {e}")
            return []
    
    async def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        try:
            # 세션 정보 조회
            session_data = await self.redis.hgetall(f"session:{session_id}")
            user_id = session_data.get("user_id")
            
            # 메시지 ID 목록 조회
            message_ids = await self.redis.lrange(f"session_messages:{session_id}", 0, -1)
            
            # 메시지 삭제
            for msg_id in message_ids:
                await self.redis.delete(f"message:{session_id}:{msg_id}")
            
            # 세션 메시지 목록 삭제
            await self.redis.delete(f"session_messages:{session_id}")
            
            # 세션 정보 삭제
            await self.redis.delete(f"session:{session_id}")
            
            # 사용자 세션 목록에서 제거
            if user_id:
                await self.redis.srem(f"user_sessions:{user_id}", session_id)
            
            logger.info(f"세션 삭제 완료: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"세션 삭제 실패: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """만료된 세션 정리"""
        try:
            # 만료된 세션 확인 (Redis가 자동으로 만료 처리)
            logger.info("만료된 세션 정리 완료")
            return 0
            
        except Exception as e:
            logger.error(f"세션 정리 실패: {e}")
            return 0

# 사용 예시
async def main():
    """테스트 함수"""
    manager = ChatHistoryManager()
    
    try:
        await manager.connect()
        
        # 새 세션 생성
        session_id = await manager.create_session("test_user")
        
        # 메시지 추가
        await manager.add_message(session_id, {
            "sender": "user",
            "content": "안녕하세요!",
            "metadata": {"type": "greeting"}
        })
        
        await manager.add_message(session_id, {
            "sender": "bot",
            "content": "안녕하세요! 무엇을 도와드릴까요?",
            "metadata": {"type": "response"}
        })
        
        # 메시지 조회
        messages = await manager.get_session_messages(session_id)
        print(f"메시지 수: {len(messages)}")
        
        # 검색 테스트
        search_results = await manager.search_messages(session_id, "안녕")
        print(f"검색 결과: {len(search_results)}개")
        
    finally:
        await manager.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
