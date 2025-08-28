"""
DevDesk-RAG 멀티모달 지원 시스템
이미지 처리, OCR, 멀티모달 검색을 위한 통합 패키지
"""

from .ocr_system import OCRSystem
from .image_processor import ImageProcessor
from .multimodal_search import MultimodalSearch
from .multimodal_manager import MultimodalManager, MultimodalConfig

__all__ = [
    'OCRSystem',
    'ImageProcessor', 
    'MultimodalSearch',
    'MultimodalManager',
    'MultimodalConfig'
]

__version__ = "2.5.0"
__author__ = "DevDesk-RAG Team"
