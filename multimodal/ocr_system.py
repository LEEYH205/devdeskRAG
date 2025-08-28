"""
OCR (Optical Character Recognition) 시스템
이미지에서 텍스트를 추출하고 분석하는 시스템
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import easyocr
import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR 결과 데이터 클래스"""
    text: str
    confidence: float
    bbox: List[Tuple[int, int]]
    language: str
    page_number: int = 0

@dataclass
class OCRConfig:
    """OCR 설정 클래스"""
    languages: List[str] = None
    gpu: bool = False
    model_storage_directory: str = "models"
    download_enabled: bool = True
    recog_network: str = "standard"
    detect_network: str = "craft"
    enable_list: List[str] = None
    disable_list: List[str] = None

class OCRSystem:
    """OCR 시스템 메인 클래스"""
    
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.reader = None
        self.initialized = False
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """OCR 시스템 초기화"""
        try:
            if self.config.languages is None:
                self.config.languages = ['ko', 'en']  # 한국어, 영어 기본
            
            logger.info(f"OCR 시스템 초기화 중... 언어: {self.config.languages}")
            
            # EasyOCR 리더 초기화
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.gpu,
                model_storage_directory=self.config.model_storage_directory,
                download_enabled=self.config.download_enabled,
                recog_network=self.config.recog_network,
                detect_network=self.config.detect_network
            )
            
            self.initialized = True
            logger.info("OCR 시스템 초기화 완료!")
            
        except Exception as e:
            logger.error(f"OCR 시스템 초기화 실패: {e}")
            self.initialized = False
    
    def extract_text(self, image_path: str) -> List[OCRResult]:
        """이미지에서 텍스트 추출"""
        if not self.initialized:
            logger.error("OCR 시스템이 초기화되지 않았습니다.")
            return []
        
        try:
            logger.info(f"이미지에서 텍스트 추출 중: {image_path}")
            
            # EasyOCR로 텍스트 추출
            results = self.reader.readtext(image_path)
            
            ocr_results = []
            for i, (bbox, text, confidence) in enumerate(results):
                ocr_result = OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    language=self._detect_language(text),
                    page_number=0
                )
                ocr_results.append(ocr_result)
            
            logger.info(f"텍스트 추출 완료: {len(ocr_results)}개 텍스트 블록")
            return ocr_results
            
        except Exception as e:
            logger.error(f"텍스트 추출 실패: {e}")
            return []
    
    def extract_text_from_pil(self, image: Image.Image) -> List[OCRResult]:
        """PIL Image에서 텍스트 추출"""
        if not self.initialized:
            logger.error("OCR 시스템이 초기화되지 않았습니다.")
            return []
        
        try:
            # PIL Image를 numpy 배열로 변환
            image_array = np.array(image)
            
            # EasyOCR로 텍스트 추출
            results = self.reader.readtext(image_array)
            
            ocr_results = []
            for i, (bbox, text, confidence) in enumerate(results):
                ocr_result = OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    language=self._detect_language(text),
                    page_number=0
                )
                ocr_results.append(ocr_result)
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"PIL Image에서 텍스트 추출 실패: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """텍스트 언어 감지 (간단한 휴리스틱)"""
        # 한국어 문자 포함 여부로 판단
        korean_chars = set('가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허기니디리미비시이지치키티피히구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐긔늬듸리미비시이지치키티피히')
        
        if any(char in korean_chars for char in text):
            return 'ko'
        else:
            return 'en'
    
    def get_text_summary(self, ocr_results: List[OCRResult]) -> Dict[str, Any]:
        """OCR 결과 요약 정보 생성"""
        if not ocr_results:
            return {}
        
        total_text = " ".join([result.text for result in ocr_results])
        avg_confidence = sum([result.confidence for result in ocr_results]) / len(ocr_results)
        
        # 언어별 통계
        language_stats = {}
        for result in ocr_results:
            lang = result.language
            if lang not in language_stats:
                language_stats[lang] = 0
            language_stats[lang] += 1
        
        return {
            "total_text": total_text,
            "text_blocks": len(ocr_results),
            "average_confidence": round(avg_confidence, 3),
            "language_distribution": language_stats,
            "total_characters": len(total_text)
        }
    
    def is_available(self) -> bool:
        """OCR 시스템 사용 가능 여부 확인"""
        return self.initialized and self.reader is not None
