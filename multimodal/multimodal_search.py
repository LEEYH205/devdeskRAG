"""
멀티모달 검색 시스템
텍스트와 이미지를 통합하여 검색하는 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .ocr_system import OCRSystem, OCRResult
from .image_processor import ImageProcessor, ImageFeature

logger = logging.getLogger(__name__)

@dataclass
class MultimodalQuery:
    """멀티모달 쿼리 데이터 클래스"""
    text: Optional[str] = None
    image_path: Optional[str] = None
    image_features: Optional[ImageFeature] = None
    query_type: str = "text_only"  # text_only, image_only, multimodal

@dataclass
class MultimodalResult:
    """멀티모달 검색 결과 데이터 클래스"""
    content_id: str
    content_type: str  # text, image, document
    similarity_score: float
    text_content: Optional[str] = None
    image_path: Optional[str] = None
    ocr_results: Optional[List[OCRResult]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """numpy 타입을 Python 기본 타입으로 변환 - 임시 비활성화"""
        # 임시로 비활성화하여 오류 방지
        pass
        
        # # similarity_score 변환
        # if hasattr(self, 'similarity_score') and self.similarity_score is not None:
        #     if isinstance(self.similarity_score, (np.integer, np.floating)):
        #         self.similarity_score = float(self.similarity_score)
        #     elif isinstance(self.similarity_score, np.ndarray):
        #         self.similarity_score = float(self.similarity_score.item())
        
        # # metadata의 numpy 타입도 변환 (안전하게)
        # if self.metadata and isinstance(self.metadata, dict):
        #     try:
        #         self._convert_numpy_types(self.metadata)
        #     except Exception as e:
        #         logger.warning(f"metadata 변환 실패: {e}")
        #         self.metadata = {}
    
    def _convert_numpy_types(self, obj):
        """재귀적으로 numpy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if isinstance(value, (np.integer, np.floating)):
                    obj[key] = float(value)
                elif isinstance(value, np.ndarray):
                    obj[key] = value.tolist()
                elif isinstance(value, dict):
                    self._convert_numpy_types(value)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            self._convert_numpy_types(item)
                        elif isinstance(item, (np.integer, np.floating)):
                            value[i] = float(item)
                        elif isinstance(item, np.ndarray):
                            value[i] = item.tolist()
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    self._convert_numpy_types(item)
                elif isinstance(item, (np.integer, np.floating)):
                    obj[i] = float(item)
                elif isinstance(item, np.ndarray):
                    obj[i] = item.tolist()

class MultimodalSearch:
    """멀티모달 검색 시스템"""
    
    def __init__(self, ocr_system: OCRSystem = None, image_processor: ImageProcessor = None):
        self.ocr_system = ocr_system or OCRSystem()
        self.image_processor = image_processor or ImageProcessor()
        self.text_index = {}  # 텍스트 인덱스
        self.image_index = {}  # 이미지 인덱스
        self.multimodal_index = {}  # 통합 인덱스
        
    def add_text_content(self, content_id: str, text: str, metadata: Dict[str, Any] = None):
        """텍스트 콘텐츠를 인덱스에 추가"""
        try:
            self.text_index[content_id] = {
                "text": text,
                "metadata": metadata or {},
                "type": "text"
            }
            logger.info(f"텍스트 콘텐츠 추가: {content_id}")
            
        except Exception as e:
            logger.error(f"텍스트 콘텐츠 추가 실패: {e}")
    
    def add_image_content(self, content_id: str, image_path: str, metadata: Dict[str, Any] = None):
        """이미지 콘텐츠를 인덱스에 추가"""
        try:
            # 이미지 로드
            image = self.image_processor.load_image(image_path)
            if image is None:
                return
            
            # 이미지 정보 추출
            image_info = self.image_processor.get_image_info(image)
            
            # CLIP 특징 추출
            image_features = self.image_processor.extract_clip_features(image)
            
            # OCR 텍스트 추출
            ocr_results = self.ocr_system.extract_text_from_pil(image)
            
            # 인덱스에 저장
            self.image_index[content_id] = {
                "image_path": image_path,
                "image_info": image_info,
                "image_features": image_features,
                "ocr_results": ocr_results,
                "metadata": metadata or {},
                "type": "image"
            }
            
            # OCR 텍스트도 텍스트 인덱스에 추가
            if ocr_results:
                ocr_text = " ".join([result.text for result in ocr_results])
                metadata_combined = {"source": content_id, "source_type": "image_ocr"}
                if metadata:
                    metadata_combined.update(metadata)
                self.add_text_content(f"{content_id}_ocr", ocr_text, metadata_combined)
            
            logger.info(f"이미지 콘텐츠 추가: {content_id}")
            
        except Exception as e:
            logger.error(f"이미지 콘텐츠 추가 실패: {e}")
    
    def add_document_content(self, content_id: str, text: str, image_paths: List[str] = None, metadata: Dict[str, Any] = None):
        """문서 콘텐츠를 인덱스에 추가 (텍스트 + 이미지)"""
        try:
            # 텍스트 콘텐츠 추가
            self.add_text_content(content_id, text, metadata)
            
            # 이미지가 있다면 추가
            if image_paths:
                for i, image_path in enumerate(image_paths):
                    image_content_id = f"{content_id}_image_{i}"
                    metadata_combined = {
                        "source": content_id,
                        "source_type": "document_image",
                        "image_index": i
                    }
                    if metadata:
                        metadata_combined.update(metadata)
                    self.add_image_content(image_content_id, image_path, metadata_combined)
            
            # 통합 인덱스에 문서 정보 저장
            self.multimodal_index[content_id] = {
                "text_id": content_id,
                "image_ids": [f"{content_id}_image_{i}" for i in range(len(image_paths or []))],
                "metadata": metadata or {},
                "type": "document"
            }
            
            logger.info(f"문서 콘텐츠 추가: {content_id}")
            
        except Exception as e:
            logger.error(f"문서 콘텐츠 추가 실패: {e}")
    
    def search(self, query: MultimodalQuery, top_k: int = 10) -> List[MultimodalResult]:
        """멀티모달 검색 실행 - 실제 검색 로직 복원"""
        try:
            logger.info(f"검색 요청: {query.query_type}, 쿼리: {query.text}")
            
            results = []
            
            # 쿼리 타입에 따른 검색 실행
            if query.query_type == "text_only":
                if query.text:
                    results = self._text_search(query.text, top_k)
                else:
                    logger.warning("텍스트 쿼리가 비어있습니다")
                    
            elif query.query_type == "image_only":
                if query.image_features:
                    results = self._image_search(query.image_features, top_k)
                else:
                    logger.warning("이미지 특징이 제공되지 않았습니다")
                    
            elif query.query_type == "multimodal":
                # 멀티모달 통합 검색
                results = self._multimodal_search(query, top_k)
                
            else:
                logger.warning(f"지원하지 않는 검색 타입: {query.query_type}")
                return []
            
            # 유사도 점수로 정렬
            if results:
                results.sort(key=lambda x: x.similarity_score, reverse=True)
                results = results[:top_k]
            
            logger.info(f"멀티모달 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"멀티모달 검색 실패: {e}")
            logger.error(f"오류 타입: {type(e)}")
            logger.error(f"오류 상세: {str(e)}")
            return []
    
    def _text_search(self, query_text: str, top_k: int) -> List[MultimodalResult]:
        """텍스트 기반 검색 - 단순화된 버전"""
        if not query_text:
            return []
        
        logger.info(f"텍스트 검색 시작: 쿼리='{query_text}', top_k={top_k}")
        logger.info(f"텍스트 인덱스 크기: {len(self.text_index)}")
        logger.info(f"텍스트 인덱스 키: {list(self.text_index.keys())}")
        
        results = []
        query_lower = query_text.lower()
        
        try:
            for content_id, content_data in self.text_index.items():
                logger.info(f"검색 중인 콘텐츠: {content_id}")
                logger.info(f"콘텐츠 데이터: {content_data}")
                
                if "text" not in content_data:
                    logger.warning(f"콘텐츠에 'text' 키가 없음: {content_data}")
                    continue
                
                text_lower = content_data["text"].lower()
                logger.info(f"텍스트 내용 (소문자): {text_lower}")
                
                # 간단한 키워드 매칭
                if query_lower in text_lower:
                    logger.info(f"키워드 매칭 성공: '{query_lower}' in '{text_lower}'")
                    
                    # 간단한 유사도 점수 계산 (0.0 ~ 1.0)
                    similarity = 0.8  # 임시로 고정값 사용
                    
                    try:
                        result = MultimodalResult(
                            content_id=content_id,
                            content_type=content_data.get("type", "text"),
                            similarity_score=similarity,
                            text_content=content_data["text"],
                            metadata=content_data.get("metadata", {})
                        )
                        results.append(result)
                        logger.info(f"검색 결과 추가 성공: {content_id}")
                    except Exception as e:
                        logger.error(f"MultimodalResult 생성 실패: {e}")
                        continue
                else:
                    logger.info(f"키워드 매칭 실패: '{query_lower}' not in '{text_lower}'")
                    
        except Exception as e:
            logger.error(f"텍스트 검색 중 오류 발생: {e}")
            logger.error(f"오류 타입: {type(e)}")
            logger.error(f"오류 상세: {str(e)}")
        
        logger.info(f"텍스트 검색 완료: {len(results)}개 결과")
        return results
    
    def _image_search(self, query_features: ImageFeature, top_k: int) -> List[MultimodalResult]:
        """이미지 기반 검색"""
        if not query_features:
            return []
        
        results = []
        
        for content_id, content_data in self.image_index.items():
            if content_data["image_features"]:
                # CLIP 특징 벡터 간 유사도 계산
                similarity = self._calculate_feature_similarity(
                    query_features.feature_vector,
                    content_data["image_features"].feature_vector
                )
                
                result = MultimodalResult(
                    content_id=content_id,
                    content_type=content_data["type"],
                    similarity_score=similarity,
                    image_path=content_data["image_path"],
                    ocr_results=content_data["ocr_results"],
                    metadata=content_data["metadata"]
                )
                results.append(result)
        
        return results
    
    def _multimodal_search(self, query: MultimodalQuery, top_k: int) -> List[MultimodalResult]:
        """멀티모달 통합 검색"""
        results = []
        
        # 텍스트 검색 결과
        text_results = self._text_search(query.text, top_k * 2) if query.text else []
        
        # 이미지 검색 결과
        image_results = self._image_search(query.image_features, top_k * 2) if query.image_features else []
        
        # 결과 통합 및 가중치 적용
        all_results = {}
        
        # 텍스트 결과 처리
        for result in text_results:
            content_id = result.content_id
            if content_id not in all_results:
                all_results[content_id] = result
            else:
                # 기존 결과와 통합
                all_results[content_id].similarity_score = max(
                    all_results[content_id].similarity_score,
                    result.similarity_score
                )
        
        # 이미지 결과 처리
        for result in image_results:
            content_id = result.content_id
            if content_id not in all_results:
                all_results[content_id] = result
            else:
                # 기존 결과와 통합
                all_results[content_id].similarity_score = max(
                    all_results[content_id].similarity_score,
                    result.similarity_score
                )
        
        # 최종 결과 생성
        results = list(all_results.values())
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:top_k]
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """텍스트 유사도 계산 (간단한 키워드 기반)"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """특징 벡터 유사도 계산 (코사인 유사도)"""
        try:
            # 코사인 유사도 계산
            similarity = cosine_similarity(
                features1.reshape(1, -1),
                features2.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"특징 벡터 유사도 계산 실패: {e}")
            return 0.0
    
    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 반환"""
        return {
            "text_content_count": len(self.text_index),
            "image_content_count": len(self.image_index),
            "multimodal_content_count": len(self.multimodal_index),
            "total_content_count": len(self.text_index) + len(self.image_index)
        }
    
    def clear_index(self):
        """인덱스 초기화"""
        self.text_index.clear()
        self.image_index.clear()
        self.multimodal_index.clear()
        logger.info("멀티모달 검색 인덱스 초기화 완료")
