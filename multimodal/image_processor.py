"""
이미지 처리 및 분석 시스템
이미지 전처리, 특징 추출, 품질 향상을 위한 시스템
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch

logger = logging.getLogger(__name__)

@dataclass
class ImageInfo:
    """이미지 정보 데이터 클래스"""
    width: int
    height: int
    channels: int
    format: str
    size_bytes: int
    dpi: Tuple[int, int]
    mode: str

@dataclass
class ImageFeature:
    """이미지 특징 데이터 클래스"""
    feature_vector: np.ndarray
    feature_type: str
    confidence: float
    metadata: Dict[str, Any]

class ImageProcessor:
    """이미지 처리 및 분석 시스템"""
    
    def __init__(self, use_clip: bool = True):
        self.use_clip = use_clip
        self.clip_processor = None
        self.clip_model = None
        self.initialized = False
        
        if self.use_clip:
            self._initialize_clip()
    
    def _initialize_clip(self):
        """CLIP 모델 초기화"""
        try:
            logger.info("CLIP 모델 초기화 중...")
            
            # CLIP 모델 및 프로세서 로드
            model_name = "openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name)
            
            self.initialized = True
            logger.info("CLIP 모델 초기화 완료!")
            
        except Exception as e:
            logger.error(f"CLIP 모델 초기화 실패: {e}")
            self.initialized = False
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """이미지 로드"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"이미지 파일이 존재하지 않습니다: {image_path}")
                return None
            
            image = Image.open(image_path)
            logger.info(f"이미지 로드 완료: {image_path}")
            return image
            
        except Exception as e:
            logger.error(f"이미지 로드 실패: {e}")
            return None
    
    def get_image_info(self, image: Image.Image) -> ImageInfo:
        """이미지 정보 추출"""
        try:
            # DPI 정보 추출
            dpi = image.info.get('dpi', (72, 72))
            
            # 파일 크기 (메모리상)
            size_bytes = len(image.tobytes())
            
            image_info = ImageInfo(
                width=image.width,
                height=image.height,
                channels=len(image.getbands()),
                format=image.format or "Unknown",
                size_bytes=size_bytes,
                dpi=dpi,
                mode=image.mode
            )
            
            return image_info
            
        except Exception as e:
            logger.error(f"이미지 정보 추출 실패: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """이미지 전처리 (크기 조정, 정규화 등)"""
        try:
            # 크기 조정
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # RGB 모드로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            logger.info(f"이미지 전처리 완료: {target_size}")
            return image
            
        except Exception as e:
            logger.error(f"이미지 전처리 실패: {e}")
            return image
    
    def extract_clip_features(self, image: Image.Image) -> Optional[ImageFeature]:
        """CLIP을 사용한 이미지 특징 추출"""
        if not self.initialized or not self.use_clip:
            logger.warning("CLIP 모델이 초기화되지 않았습니다.")
            return None
        
        try:
            # 이미지 전처리
            processed_image = self.preprocess_image(image)
            
            # CLIP 특징 추출
            inputs = self.clip_processor(images=processed_image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                feature_vector = image_features.cpu().numpy().flatten()
            
            # 특징 정규화
            feature_vector = feature_vector / np.linalg.norm(feature_vector)
            
            image_feature = ImageFeature(
                feature_vector=feature_vector,
                feature_type="CLIP",
                confidence=1.0,
                metadata={
                    "model": "CLIP-ViT-Base",
                    "feature_dim": len(feature_vector),
                    "normalized": True
                }
            )
            
            logger.info(f"CLIP 특징 추출 완료: {len(feature_vector)}차원")
            return image_feature
            
        except Exception as e:
            logger.error(f"CLIP 특징 추출 실패: {e}")
            return None
    
    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """이미지 품질 향상"""
        try:
            # 노이즈 제거
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # 밝기 조정
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            # 채도 향상
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.2)
            
            logger.info("이미지 품질 향상 완료")
            return image
            
        except Exception as e:
            logger.error(f"이미지 품질 향상 실패: {e}")
            return image
    
    def detect_text_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """텍스트 영역 감지 (간단한 휴리스틱)"""
        try:
            # PIL Image를 OpenCV 형식으로 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 이진화
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                # 면적이 너무 작은 것은 제외
                area = cv2.contourArea(contour)
                if area < 100:  # 최소 면적 임계값
                    continue
                
                # 경계 사각형
                x, y, w, h = cv2.boundingRect(contour)
                
                # 너무 작은 영역은 제외
                if w < 20 or h < 10:
                    continue
                
                text_regions.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": area,
                    "width": w,
                    "height": h
                })
            
            logger.info(f"텍스트 영역 감지 완료: {len(text_regions)}개 영역")
            return text_regions
            
        except Exception as e:
            logger.error(f"텍스트 영역 감지 실패: {e}")
            return []
    
    def is_available(self) -> bool:
        """이미지 처리 시스템 사용 가능 여부 확인"""
        return self.initialized or True  # 기본 이미지 처리 기능은 항상 사용 가능
