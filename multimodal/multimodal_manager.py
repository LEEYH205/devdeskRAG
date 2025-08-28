"""
멀티모달 시스템 통합 관리자
OCR, 이미지 처리, 멀티모달 검색을 통합하여 관리하는 시스템
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
from .ocr_system import OCRSystem, OCRConfig, OCRResult
from .image_processor import ImageProcessor, ImageInfo, ImageFeature
from .multimodal_search import MultimodalSearch, MultimodalQuery, MultimodalResult

logger = logging.getLogger(__name__)

@dataclass
class MultimodalConfig:
    """멀티모달 시스템 설정 클래스"""
    enable_ocr: bool = False  # 기본적으로 비활성화
    enable_clip: bool = False  # 기본적으로 비활성화
    enable_image_enhancement: bool = True
    max_image_size: Tuple[int, int] = (1024, 1024)
    supported_formats: List[str] = None
    cache_dir: str = "multimodal_cache"
    model_dir: str = "models"

@dataclass
class ProcessingResult:
    """처리 결과 데이터 클래스"""
    success: bool
    content_id: str
    content_type: str
    processing_time: float
    results: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 가능한 딕셔너리로 변환"""
        return {
            "success": self.success,
            "content_id": self.content_id,
            "content_type": self.content_type,
            "processing_time": self.processing_time,
            "results": self._convert_numpy_types(self.results),
            "error_message": self.error_message
        }
    
    def _convert_numpy_types(self, obj):
        """numpy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, dict):
            converted = {}
            for key, value in obj.items():
                converted[key] = self._convert_numpy_types(value)
            return converted
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # 객체를 딕셔너리로 변환
            return self._convert_numpy_types(obj.__dict__)
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj

class MultimodalManager:
    """멀티모달 시스템 통합 관리자"""
    
    def __init__(self, config: MultimodalConfig = None):
        self.config = config or MultimodalConfig()
        self.ocr_system = None
        self.image_processor = None
        self.multimodal_search = None
        self.processing_history = []
        self.content_registry = {}
        
        # 기본 지원 형식 설정
        if self.config.supported_formats is None:
            self.config.supported_formats = [
                'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif',
                'pdf', 'docx', 'txt', 'md'
            ]
        
        # 시스템 초기화 (안전 모드)
        try:
            self._initialize_systems()
        except Exception as e:
            print(f"멀티모달 시스템 초기화 중 오류 발생: {e}")
            # 기본 기능만 활성화
            self.ocr_system = None
            self.image_processor = None
            self.multimodal_search = None
    
    def _initialize_systems(self):
        """멀티모달 시스템 초기화"""
        try:
            logger.info("멀티모달 시스템 초기화 중...")
            
            # OCR 시스템 초기화 (선택적)
            if self.config.enable_ocr:
                try:
                    ocr_config = OCRConfig(
                        languages=['ko', 'en'],
                        gpu=False,
                        model_storage_directory=self.config.model_dir
                    )
                    self.ocr_system = OCRSystem(ocr_config)
                    logger.info("OCR 시스템 초기화 완료")
                except Exception as e:
                    logger.warning(f"OCR 시스템 초기화 실패: {e}")
                    self.ocr_system = None
            
            # 이미지 처리 시스템 초기화 (선택적)
            try:
                self.image_processor = ImageProcessor(use_clip=self.config.enable_clip)
                logger.info("이미지 처리 시스템 초기화 완료")
            except Exception as e:
                logger.warning(f"이미지 처리 시스템 초기화 실패: {e}")
                self.image_processor = None
            
            # 멀티모달 검색 시스템 초기화 (선택적)
            try:
                self.multimodal_search = MultimodalSearch(
                    ocr_system=self.ocr_system,
                    image_processor=self.image_processor
                )
                logger.info("멀티모달 검색 시스템 초기화 완료")
            except Exception as e:
                logger.warning(f"멀티모달 검색 시스템 초기화 실패: {e}")
                self.multimodal_search = None
            
            # 캐시 디렉토리 생성
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
            logger.info("멀티모달 시스템 초기화 완료!")
            
        except Exception as e:
            logger.error(f"멀티모달 시스템 초기화 실패: {e}")
    
    def process_file(self, file_path: str, content_id: str = None) -> Dict[str, Any]:
        """파일 처리 (텍스트, 이미지, 문서) - JSON 직렬화 문제 해결된 버전"""
        start_time = time.time()
        
        try:
            logger.info(f"파일 처리 시작: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
            
            # 파일 확장자 확인
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            logger.info(f"파일 확장자: {file_ext}")
            
            if file_ext not in self.config.supported_formats:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_ext}")
            
            # content_id 생성
            if content_id is None:
                content_id = f"{os.path.splitext(os.path.basename(file_path))[0]}_{int(start_time)}"
            
            # 파일 타입별 처리
            if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif']:
                result = self._process_image_file(file_path, content_id)
            elif file_ext in ['pdf', 'docx']:
                result = self._process_document_file(file_path, content_id)
            elif file_ext in ['txt', 'md']:
                result = self._process_text_file(file_path, content_id)
            else:
                raise ValueError(f"알 수 없는 파일 형식: {file_ext}")
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 생성 (JSON 직렬화 가능한 형태)
            processing_result = {
                "success": True,
                "content_id": content_id,
                "content_type": result.get("type", "unknown"),
                "processing_time": round(processing_time, 3),
                "results": self._convert_numpy_types(result),
                "error_message": None
            }
            
            # 처리 히스토리에 추가
            self.processing_history.append(processing_result)
            
            # 콘텐츠 레지스트리에 저장
            self.content_registry[content_id] = {
                "file_path": file_path,
                "content_type": result.get("type", "unknown"),
                "processed_at": datetime.now().isoformat(),
                "processing_time": round(processing_time, 3),
                "results": self._convert_numpy_types(result)
            }
            
            logger.info(f"파일 처리 완료: {content_id} ({processing_time:.3f}초)")
            return processing_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"파일 처리 실패: {error_msg}")
            
            return {
                "success": False,
                "content_id": content_id or "unknown",
                "content_type": "unknown",
                "processing_time": round(processing_time, 3),
                "results": {},
                "error_message": error_msg
            }
    
    def _convert_numpy_types(self, obj):
        """numpy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, dict):
            converted = {}
            for key, value in obj.items():
                converted[key] = self._convert_numpy_types(value)
            return converted
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    def _process_image_file(self, file_path: str, content_id: str) -> Dict[str, Any]:
        """이미지 파일 처리"""
        results = {
            "type": "image",
            "file_path": file_path,
            "content_id": content_id
        }
        
        # 이미지 로드
        image = self.image_processor.load_image(file_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다")
        
        # 이미지 정보 추출
        image_info = self.image_processor.get_image_info(image)
        results["image_info"] = image_info.__dict__ if image_info else {}
        
        # 이미지 품질 향상
        if self.config.enable_image_enhancement:
            enhanced_image = self.image_processor.enhance_image_quality(image)
            results["enhancement_applied"] = True
        else:
            enhanced_image = image
            results["enhancement_applied"] = False
        
        # CLIP 특징 추출
        if self.config.enable_clip:
            image_features = self.image_processor.extract_clip_features(enhanced_image)
            results["clip_features"] = {
                "available": image_features is not None,
                "dimension": len(image_features.feature_vector) if image_features else 0
            } if image_features else {"available": False}
        
        # OCR 텍스트 추출
        if self.config.enable_ocr and self.ocr_system:
            ocr_results = self.ocr_system.extract_text_from_pil(enhanced_image)
            results["ocr_results"] = [result.__dict__ for result in ocr_results]
            
            # OCR 요약 정보
            if ocr_results:
                ocr_summary = self.ocr_system.get_text_summary(ocr_results)
                results["ocr_summary"] = ocr_summary
        
        # 텍스트 영역 감지
        text_regions = self.image_processor.detect_text_regions(enhanced_image)
        results["text_regions"] = text_regions
        
        # 멀티모달 검색 인덱스에 추가
        self.multimodal_search.add_image_content(content_id, file_path, results)
        
        return results
    
    def _process_document_file(self, file_path: str, content_id: str) -> Dict[str, Any]:
        """문서 파일 처리 (PDF, DOCX)"""
        results = {
            "type": "document",
            "file_path": file_path,
            "content_id": content_id
        }
        
        # 현재는 기본적인 텍스트 추출만 구현
        # 향후 PDF/DOCX 파서 추가 예정
        results["note"] = "문서 파싱 기능은 향후 구현 예정입니다."
        
        # 멀티모달 검색 인덱스에 추가
        self.multimodal_search.add_document_content(content_id, "", [], results)
        
        return results
    
    def _process_text_file(self, file_path: str, content_id: str) -> Dict[str, Any]:
        """텍스트 파일 처리"""
        results = {
            "type": "text",
            "file_path": file_path,
            "content_id": content_id
        }
        
        try:
            # 텍스트 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            results["text_content"] = text_content
            results["text_length"] = len(text_content)
            results["text_lines"] = len(text_content.splitlines())
            
            # 멀티모달 검색 인덱스에 추가
            self.multimodal_search.add_text_content(content_id, text_content, results)
            
        except Exception as e:
            logger.error(f"텍스트 파일 처리 실패: {e}")
            results["error"] = str(e)
        
        return results
    
    def search(self, query: str, query_type: str = "text_only", top_k: int = 10) -> List[Dict[str, Any]]:
        """멀티모달 검색 실행 - 매우 간단한 테스트 버전"""
        try:
            logger.info(f"멀티모달 검색 요청: {query}, 타입: {query_type}")
            
            # 매우 간단한 테스트 응답 생성
            test_results = [
                {
                    "content_id": "test_search_result",
                    "content_type": "text",
                    "similarity_score": 0.95,
                    "text_content": f"검색어 '{query}'에 대한 테스트 결과입니다.",
                    "image_path": None,
                    "metadata": {"source": "test", "query": query}
                }
            ]
            
            logger.info(f"멀티모달 검색 완료: {len(test_results)}개 결과")
            return test_results
            
        except Exception as e:
            logger.error(f"멀티모달 검색 실패: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        return {
            "ocr_system": {
                "available": self.ocr_system is not None and hasattr(self.ocr_system, 'initialized') and self.ocr_system.initialized,
                "enabled": self.config.enable_ocr
            },
            "image_processor": {
                "available": self.image_processor is not None and hasattr(self.image_processor, 'initialized') and self.image_processor.initialized,
                "clip_enabled": self.config.enable_clip
            },
            "multimodal_search": {
                "available": self.multimodal_search is not None,
                "index_stats": self._get_index_stats()
            },
            "processing_stats": {
                "total_processed": len(self.processing_history),
                "successful": len([r for r in self.processing_history if r.get("success", False)]),
                "failed": len([r for r in self.processing_history if not r.get("success", False)])
            },
            "content_registry": {
                "total_content": len(self.content_registry),
                "content_types": list(set([c["content_type"] for c in self.content_registry.values()]))
            }
        }
    
    def _get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 반환"""
        try:
            if self.multimodal_search and hasattr(self.multimodal_search, 'text_index') and hasattr(self.multimodal_search, 'image_index'):
                return {
                    "text_content_count": len(self.multimodal_search.text_index),
                    "image_content_count": len(self.multimodal_search.image_index),
                    "multimodal_content_count": len(self.multimodal_search.multimodal_index),
                    "total_content_count": len(self.multimodal_search.text_index) + len(self.multimodal_search.image_index)
                }
            else:
                return {
                    "text_content_count": 0,
                    "image_content_count": 0,
                    "multimodal_content_count": 0,
                    "total_content_count": 0
                }
        except Exception as e:
            logger.error(f"인덱스 통계 조회 실패: {e}")
            return {
                "text_content_count": 0,
                "image_content_count": 0,
                "multimodal_content_count": 0,
                "total_content_count": 0
            }
    
    def clear_cache(self):
        """캐시 및 인덱스 초기화"""
        try:
            # 멀티모달 검색 인덱스 초기화
            if self.multimodal_search:
                self.multimodal_search.clear_index()
            
            # 처리 히스토리 초기화
            self.processing_history.clear()
            
            # 콘텐츠 레지스트리 초기화
            self.content_registry.clear()
            
            logger.info("멀티모달 시스템 캐시 초기화 완료")
            
        except Exception as e:
            logger.error(f"캐시 초기화 실패: {e}")
    
    def export_processing_history(self, file_path: str):
        """처리 히스토리를 JSON 파일로 내보내기"""
        try:
            history_data = []
            for result in self.processing_history:
                history_data.append({
                    "content_id": result.content_id,
                    "content_type": result.content_type,
                    "processing_time": result.processing_time,
                    "success": result.success,
                    "error_message": result.error_message,
                    "results_keys": list(result.results.keys()) if result.results else []
                })
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"처리 히스토리 내보내기 완료: {file_path}")
            
        except Exception as e:
            logger.error(f"처리 히스토리 내보내기 실패: {e}")
