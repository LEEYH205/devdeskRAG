"""
DevDesk-RAG DoLA (Domain-Oriented Low-rank Adaptation) 시스템

DoLA는 도메인 지향적 학습으로 특정 태스크에 최적화된 LoRA 어댑터를 제공합니다.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path
from enum import Enum

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """태스크 유형을 정의하는 열거형"""
    SEARCH = "search"
    RERANK = "rerank"
    PERSONALIZATION = "personalization"
    DOCUMENT_ANALYSIS = "document_analysis"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWERING = "question_answering"

class DomainType(Enum):
    """도메인 유형을 정의하는 열거형"""
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    BUSINESS = "business"
    CREATIVE = "creative"
    SCIENTIFIC = "scientific"
    GENERAL = "general"

@dataclass
class DoLAConfig:
    """DoLA 설정을 관리하는 클래스"""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    bias: bool = False
    device: str = "auto"
    domain_aware: bool = True
    task_specific: bool = True
    adaptive_learning: bool = True
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DomainContext:
    """도메인 컨텍스트를 저장하는 클래스"""
    domain_type: DomainType
    task_type: TaskType
    keywords: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TaskProfile:
    """태스크 프로필을 저장하는 클래스"""
    task_type: TaskType
    domain_type: DomainType
    performance_metrics: Dict[str, float]
    adaptation_history: List[Dict[str, Any]]
    last_updated: datetime = field(default_factory=datetime.now)

class DomainAnalyzer:
    """도메인 특성을 분석하는 클래스"""
    
    def __init__(self, config: DoLAConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.domain_patterns = {}
        self.task_patterns = {}
        logger.info("DomainAnalyzer 초기화 완료")
    
    def analyze_domain(self, text: str, task_type: TaskType = None) -> DomainContext:
        """
        텍스트를 분석하여 도메인 컨텍스트를 추출합니다.
        
        Args:
            text: 분석할 텍스트
            task_type: 태스크 유형 (선택사항)
        
        Returns:
            DomainContext: 도메인 컨텍스트
        """
        # 키워드 추출
        keywords = self._extract_keywords(text)
        
        # 도메인 유형 분류
        domain_type = self._classify_domain(text, keywords)
        
        # 태스크 유형이 지정되지 않은 경우 자동 감지
        if task_type is None:
            task_type = self._detect_task_type(text, keywords)
        
        # 메타데이터 생성
        metadata = self._generate_metadata(text, keywords, domain_type, task_type)
        
        domain_context = DomainContext(
            domain_type=domain_type,
            task_type=task_type,
            keywords=keywords,
            metadata=metadata
        )
        
        logger.info(f"도메인 분석 완료: {domain_type.value} - {task_type.value}")
        return domain_context
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드를 추출합니다."""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용)
        words = text.lower().split()
        # 일반적인 단어 제거
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 빈도 기반 상위 키워드 선택
        from collections import Counter
        keyword_counts = Counter(keywords)
        top_keywords = [word for word, count in keyword_counts.most_common(10)]
        
        return top_keywords
    
    def _classify_domain(self, text: str, keywords: List[str]) -> DomainType:
        """텍스트를 도메인 유형으로 분류합니다."""
        text_lower = text.lower()
        
        # 기술적 도메인
        tech_keywords = ['algorithm', 'system', 'code', 'programming', 'software', 'hardware', 'database', 'network']
        if any(keyword in text_lower for keyword in tech_keywords):
            return DomainType.TECHNICAL
        
        # 학술적 도메인
        academic_keywords = ['research', 'study', 'analysis', 'theory', 'methodology', 'experiment', 'data']
        if any(keyword in text_lower for keyword in academic_keywords):
            return DomainType.ACADEMIC
        
        # 비즈니스 도메인
        business_keywords = ['business', 'market', 'strategy', 'management', 'finance', 'customer', 'product']
        if any(keyword in text_lower for keyword in business_keywords):
            return DomainType.BUSINESS
        
        # 과학적 도메인
        scientific_keywords = ['science', 'physics', 'chemistry', 'biology', 'mathematics', 'engineering']
        if any(keyword in text_lower for keyword in scientific_keywords):
            return DomainType.SCIENTIFIC
        
        # 창의적 도메인
        creative_keywords = ['creative', 'design', 'art', 'music', 'writing', 'story', 'imagination']
        if any(keyword in text_lower for keyword in creative_keywords):
            return DomainType.CREATIVE
        
        return DomainType.GENERAL
    
    def _detect_task_type(self, text: str, keywords: List[str]) -> TaskType:
        """텍스트에서 태스크 유형을 감지합니다."""
        text_lower = text.lower()
        
        # 검색 태스크
        search_keywords = ['search', 'find', 'look', 'query', 'information']
        if any(keyword in text_lower for keyword in search_keywords):
            return TaskType.SEARCH
        
        # 문서 분석 태스크
        analysis_keywords = ['analyze', 'analyze', 'examine', 'review', 'evaluate', 'assess']
        if any(keyword in text_lower for keyword in analysis_keywords):
            return TaskType.DOCUMENT_ANALYSIS
        
        # 코드 생성 태스크
        code_keywords = ['code', 'program', 'function', 'class', 'method', 'algorithm']
        if any(keyword in text_lower for keyword in code_keywords):
            return TaskType.CODE_GENERATION
        
        # 질문 답변 태스크
        qa_keywords = ['what', 'how', 'why', 'when', 'where', 'explain', 'describe']
        if any(keyword in text_lower for keyword in qa_keywords):
            return TaskType.QUESTION_ANSWERING
        
        return TaskType.SEARCH
    
    def _generate_metadata(self, text: str, keywords: List[str], domain_type: DomainType, task_type: TaskType) -> Dict[str, Any]:
        """도메인 컨텍스트에 대한 메타데이터를 생성합니다."""
        return {
            "text_length": len(text),
            "keyword_count": len(keywords),
            "domain_confidence": self._calculate_domain_confidence(text, domain_type),
            "task_confidence": self._calculate_task_confidence(text, task_type),
            "complexity_score": self._calculate_complexity_score(text),
            "specialization_level": self._calculate_specialization_level(keywords, domain_type)
        }
    
    def _calculate_domain_confidence(self, text: str, domain_type: DomainType) -> float:
        """도메인 분류 신뢰도를 계산합니다."""
        # 간단한 신뢰도 계산 (실제로는 더 정교한 방법 사용)
        domain_keywords = {
            DomainType.TECHNICAL: ['algorithm', 'system', 'code', 'programming'],
            DomainType.ACADEMIC: ['research', 'study', 'analysis', 'theory'],
            DomainType.BUSINESS: ['business', 'market', 'strategy', 'management'],
            DomainType.SCIENTIFIC: ['science', 'physics', 'chemistry', 'biology'],
            DomainType.CREATIVE: ['creative', 'design', 'art', 'music'],
            DomainType.GENERAL: []
        }
        
        relevant_keywords = domain_keywords.get(domain_type, [])
        if not relevant_keywords:
            return 0.5
        
        text_lower = text.lower()
        matches = sum(1 for keyword in relevant_keywords if keyword in text_lower)
        return min(matches / len(relevant_keywords), 1.0)
    
    def _calculate_task_confidence(self, text: str, task_type: TaskType) -> float:
        """태스크 분류 신뢰도를 계산합니다."""
        # 간단한 신뢰도 계산
        task_keywords = {
            TaskType.SEARCH: ['search', 'find', 'look', 'query'],
            TaskType.DOCUMENT_ANALYSIS: ['analyze', 'examine', 'review', 'evaluate'],
            TaskType.CODE_GENERATION: ['code', 'program', 'function', 'class'],
            TaskType.QUESTION_ANSWERING: ['what', 'how', 'why', 'explain'],
            TaskType.RERANK: ['rerank', 'reorder', 'sort', 'rank'],
            TaskType.PERSONALIZATION: ['personal', 'custom', 'individual', 'user']
        }
        
        relevant_keywords = task_keywords.get(task_type, [])
        if not relevant_keywords:
            return 0.5
        
        text_lower = text.lower()
        matches = sum(1 for keyword in relevant_keywords if keyword in text_lower)
        return min(matches / len(relevant_keywords), 1.0)
    
    def _calculate_complexity_score(self, text: str) -> float:
        """텍스트 복잡도 점수를 계산합니다."""
        # 간단한 복잡도 계산
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        unique_words = len(set(words))
        complexity = (avg_word_length * 0.3) + (unique_words / len(words) * 0.7) if words else 0
        return min(complexity, 1.0)
    
    def _calculate_specialization_level(self, keywords: List[str], domain_type: DomainType) -> float:
        """도메인 특화 수준을 계산합니다."""
        # 도메인별 전문 용어 빈도 계산
        domain_terms = {
            DomainType.TECHNICAL: ['algorithm', 'complexity', 'optimization', 'architecture'],
            DomainType.ACADEMIC: ['methodology', 'hypothesis', 'correlation', 'significance'],
            DomainType.BUSINESS: ['strategy', 'optimization', 'efficiency', 'scalability'],
            DomainType.SCIENTIFIC: ['hypothesis', 'experiment', 'variable', 'control'],
            DomainType.CREATIVE: ['aesthetic', 'composition', 'harmony', 'expression'],
            DomainType.GENERAL: []
        }
        
        relevant_terms = domain_terms.get(domain_type, [])
        if not relevant_terms:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in relevant_terms)
        return min(matches / len(relevant_terms), 1.0)

class TaskSpecializer:
    """태스크별 모델 최적화를 관리하는 클래스"""
    
    def __init__(self, config: DoLAConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.task_profiles = {}
        self.specialized_adapters = {}
        logger.info("TaskSpecializer 초기화 완료")
    
    def create_task_specific_adapter(self, task_type: TaskType, domain_type: DomainType,
                                   weight_shape: Tuple[int, ...]) -> nn.Module:
        """
        태스크별 특화 어댑터를 생성합니다.
        
        Args:
            task_type: 태스크 유형
            domain_type: 도메인 유형
            weight_shape: 가중치 형태
        
        Returns:
            nn.Module: 태스크별 특화 어댑터
        """
        # 태스크 프로필 생성 또는 업데이트
        task_key = f"{task_type.value}_{domain_type.value}"
        if task_key not in self.task_profiles:
            self.task_profiles[task_key] = TaskProfile(
                task_type=task_type,
                domain_type=domain_type,
                performance_metrics={},
                adaptation_history=[]
            )
        
        # 태스크별 특화 어댑터 생성
        adapter = DoLAAdapter(
            weight_shape=weight_shape,
            task_type=task_type,
            domain_type=domain_type,
            config=self.config
        )
        
        self.specialized_adapters[task_key] = adapter
        logger.info(f"태스크별 특화 어댑터 생성 완료: {task_key}")
        
        return adapter
    
    def get_task_adapter(self, task_type: TaskType, domain_type: DomainType) -> Optional[nn.Module]:
        """태스크별 어댑터를 가져옵니다."""
        task_key = f"{task_type.value}_{domain_type.value}"
        return self.specialized_adapters.get(task_key)
    
    def update_task_performance(self, task_type: TaskType, domain_type: DomainType,
                              metrics: Dict[str, float]):
        """태스크 성능을 업데이트합니다."""
        task_key = f"{task_type.value}_{domain_type.value}"
        if task_key in self.task_profiles:
            self.task_profiles[task_key].performance_metrics.update(metrics)
            self.task_profiles[task_key].last_updated = datetime.now()
            logger.info(f"태스크 성능 업데이트 완료: {task_key}")
    
    def get_task_profile(self, task_type: TaskType, domain_type: DomainType) -> Optional[TaskProfile]:
        """태스크 프로필을 가져옵니다."""
        task_key = f"{task_type.value}_{domain_type.value}"
        return self.task_profiles.get(task_key)
    
    def list_task_adapters(self) -> List[str]:
        """모든 태스크 어댑터 이름을 반환합니다."""
        return list(self.specialized_adapters.keys())

class AdaptiveLearning:
    """적응형 학습을 관리하는 클래스"""
    
    def __init__(self, config: DoLAConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.learning_history = []
        self.adaptation_strategies = {}
        logger.info("AdaptiveLearning 초기화 완료")
    
    def adapt_to_domain(self, adapter: nn.Module, domain_context: DomainContext,
                       performance_feedback: Dict[str, float]) -> nn.Module:
        """
        도메인 컨텍스트에 맞게 어댑터를 적응시킵니다.
        
        Args:
            adapter: 적응시킬 어댑터
            domain_context: 도메인 컨텍스트
            performance_feedback: 성능 피드백
        
        Returns:
            nn.Module: 적응된 어댑터
        """
        # 적응 전략 선택
        strategy = self._select_adaptation_strategy(domain_context, performance_feedback)
        
        # 어댑터 적응
        adapted_adapter = self._apply_adaptation_strategy(adapter, strategy, domain_context)
        
        # 학습 히스토리 기록
        self._record_adaptation(domain_context, strategy, performance_feedback)
        
        logger.info(f"도메인 적응 완료: {domain_context.domain_type.value}")
        return adapted_adapter
    
    def _select_adaptation_strategy(self, domain_context: DomainContext,
                                  performance_feedback: Dict[str, float]) -> str:
        """적응 전략을 선택합니다."""
        # 성능 기반 전략 선택
        if performance_feedback.get("accuracy", 0) < 0.7:
            return "aggressive_adaptation"
        elif performance_feedback.get("accuracy", 0) < 0.85:
            return "moderate_adaptation"
        else:
            return "conservative_adaptation"
    
    def _apply_adaptation_strategy(self, adapter: nn.Module, strategy: str,
                                 domain_context: DomainContext) -> nn.Module:
        """적응 전략을 적용합니다."""
        if strategy == "aggressive_adaptation":
            return self._aggressive_adaptation(adapter, domain_context)
        elif strategy == "moderate_adaptation":
            return self._moderate_adaptation(adapter, domain_context)
        else:
            return self._conservative_adaptation(adapter, domain_context)
    
    def _aggressive_adaptation(self, adapter: nn.Module, domain_context: DomainContext) -> nn.Module:
        """공격적 적응을 수행합니다."""
        # 학습률 증가, 더 많은 파라미터 업데이트
        for param in adapter.parameters():
            if hasattr(param, 'requires_grad') and param.requires_grad:
                param.data *= 1.2  # 가중치 증가
        
        logger.info("공격적 적응 적용 완료")
        return adapter
    
    def _moderate_adaptation(self, adapter: nn.Module, domain_context: DomainContext) -> nn.Module:
        """보통 수준의 적응을 수행합니다."""
        # 중간 수준의 파라미터 조정
        for param in adapter.parameters():
            if hasattr(param, 'requires_grad') and param.requires_grad:
                param.data *= 1.1  # 가중치 증가
        
        logger.info("보통 수준 적응 적용 완료")
        return adapter
    
    def _conservative_adaptation(self, adapter: nn.Module, domain_context: DomainContext) -> nn.Module:
        """보수적 적응을 수행합니다."""
        # 최소한의 파라미터 조정
        for param in adapter.parameters():
            if hasattr(param, 'requires_grad') and param.requires_grad:
                param.data *= 1.05  # 가중치 증가
        
        logger.info("보수적 적응 적용 완료")
        return adapter
    
    def _record_adaptation(self, domain_context: DomainContext, strategy: str,
                          performance_feedback: Dict[str, float]):
        """적응 과정을 기록합니다."""
        adaptation_record = {
            "timestamp": datetime.now(),
            "domain_type": domain_context.domain_type.value,
            "task_type": domain_context.task_type.value,
            "strategy": strategy,
            "performance_feedback": performance_feedback,
            "keywords": domain_context.keywords
        }
        
        self.learning_history.append(adaptation_record)
        logger.info(f"적응 과정 기록 완료: {strategy}")

class DoLAAdapter(nn.Module):
    """DoLA 어댑터 모듈"""
    
    def __init__(self, weight_shape: Tuple[int, ...], task_type: TaskType,
                 domain_type: DomainType, config: DoLAConfig):
        super().__init__()
        self.config = config
        self.weight_shape = weight_shape
        self.task_type = task_type
        self.domain_type = domain_type
        
        # DoLA 파라미터 초기화
        self.rank = config.rank
        self.alpha = config.alpha
        
        # A와 B 행렬 초기화
        if len(weight_shape) == 2:
            in_features, out_features = weight_shape
            self.lora_A = nn.Parameter(torch.randn(in_features, self.rank) * 0.02)
            self.lora_B = nn.Parameter(torch.zeros(self.rank, out_features))
            self.scaling = nn.Parameter(torch.ones(1))
        else:
            # 다차원 텐서의 경우 적절한 형태로 변환
            total_elements = np.prod(weight_shape)
            sqrt_elements = int(np.sqrt(total_elements))
            self.lora_A = nn.Parameter(torch.randn(sqrt_elements, self.rank) * 0.02)
            self.lora_B = nn.Parameter(torch.zeros(self.rank, sqrt_elements))
            self.scaling = nn.Parameter(torch.ones(1))
        
        # 도메인별 가중치
        self.domain_weight = nn.Parameter(torch.ones(1))
        self.task_weight = nn.Parameter(torch.ones(1))
        
        # 드롭아웃
        self.dropout = nn.Dropout(config.dropout)
        
        # 가중치 초기화
        self._init_weights()
        
        logger.info(f"DoLA 어댑터 초기화 완료: {weight_shape} -> {task_type.value} - {domain_type.value}")
    
    def _init_weights(self):
        """가중치를 초기화합니다."""
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.scaling)
        nn.init.ones_(self.domain_weight)
        nn.init.ones_(self.task_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DoLA 어댑터를 통한 순전파
        
        Args:
            x: 입력 텐서
        
        Returns:
            torch.Tensor: 적응된 출력 텐서
        """
        # LoRA 계산
        lora_output = self.dropout(x @ self.lora_A @ self.lora_B)
        
        # 도메인 및 태스크별 가중치 적용
        adapted_output = lora_output * self.scaling * self.alpha / self.rank
        adapted_output *= self.domain_weight * self.task_weight
        
        return adapted_output
    
    def get_adapted_weight(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        원본 가중치에 DoLA 어댑터를 적용한 가중치를 반환합니다.
        
        Args:
            original_weight: 원본 가중치 텐서
        
        Returns:
            torch.Tensor: 적응된 가중치 텐서
        """
        # DoLA 어댑터 계산
        adapter_weight = self.lora_A @ self.lora_B
        adapted_weight = original_weight + adapter_weight * self.scaling * self.alpha / self.rank
        adapted_weight *= self.domain_weight * self.task_weight
        
        return adapted_weight
    
    def get_adaptation_strength(self) -> Dict[str, float]:
        """적응 강도를 반환합니다."""
        return {
            "overall": float(self.scaling.item() * self.alpha / self.rank),
            "domain": float(self.domain_weight.item()),
            "task": float(self.task_weight.item())
        }

class DoLASystem:
    """DoLA 시스템의 메인 클래스"""
    
    def __init__(self, config: DoLAConfig = None):
        if config is None:
            config = DoLAConfig()
        
        self.config = config
        self.domain_analyzer = DomainAnalyzer(config)
        self.task_specializer = TaskSpecializer(config)
        self.adaptive_learning = AdaptiveLearning(config)
        
        logger.info("DoLA 시스템 초기화 완료")
    
    def create_dola_adapter(self, name: str, weight_shape: Tuple[int, ...],
                           task_type: TaskType = None, domain_type: DomainType = None) -> DoLAAdapter:
        """
        DoLA 어댑터를 생성합니다.
        
        Args:
            name: 어댑터 이름
            weight_shape: 가중치 형태
            task_type: 태스크 유형 (선택사항)
            domain_type: 도메인 유형 (선택사항)
        
        Returns:
            DoLAAdapter: 생성된 DoLA 어댑터
        """
        # 기본값 설정
        if task_type is None:
            task_type = TaskType.SEARCH
        if domain_type is None:
            domain_type = DomainType.GENERAL
        
        # 태스크별 특화 어댑터 생성
        adapter = self.task_specializer.create_task_specific_adapter(
            task_type, domain_type, weight_shape
        )
        
        logger.info(f"DoLA 어댑터 '{name}' 생성 완료: {task_type.value} - {domain_type.value}")
        return adapter
    
    def analyze_and_adapt(self, text: str, adapter: DoLAAdapter,
                         performance_feedback: Dict[str, float] = None) -> DoLAAdapter:
        """
        텍스트를 분석하고 어댑터를 적응시킵니다.
        
        Args:
            text: 분석할 텍스트
            adapter: 적응시킬 어댑터
            performance_feedback: 성능 피드백 (선택사항)
        
        Returns:
            DoLAAdapter: 적응된 어댑터
        """
        # 도메인 분석
        domain_context = self.domain_analyzer.analyze_domain(text, adapter.task_type)
        
        # 성능 피드백이 없는 경우 기본값 설정
        if performance_feedback is None:
            performance_feedback = {"accuracy": 0.8, "latency": 0.1}
        
        # 어댑터 적응
        adapted_adapter = self.adaptive_learning.adapt_to_domain(
            adapter, domain_context, performance_feedback
        )
        
        logger.info(f"텍스트 분석 및 어댑터 적응 완료: {domain_context.domain_type.value}")
        return adapted_adapter
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태를 반환합니다."""
        return {
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "device": str(self.config.device),
                "domain_aware": self.config.domain_aware,
                "task_specific": self.config.task_specific,
                "adaptive_learning": self.config.adaptive_learning
            },
            "task_adapters": {
                "total": len(self.task_specializer.specialized_adapters),
                "names": self.task_specializer.list_task_adapters()
            },
            "task_profiles": {
                "total": len(self.task_specializer.task_profiles),
                "types": [profile.task_type.value for profile in self.task_specializer.task_profiles.values()]
            },
            "learning_history": {
                "total": len(self.adaptive_learning.learning_history),
                "recent_adaptations": len([h for h in self.adaptive_learning.learning_history 
                                        if (datetime.now() - h["timestamp"]).days < 7])
            }
        }

# 전역 DoLA 시스템 인스턴스
dola_system = DoLASystem()

def get_dola_system() -> DoLASystem:
    """전역 DoLA 시스템 인스턴스를 반환합니다."""
    return dola_system

if __name__ == "__main__":
    # 테스트 코드
    print("DoLA 시스템 테스트 시작...")
    
    # 시스템 초기화
    system = DoLASystem()
    
    # 가상 텍스트로 도메인 분석
    test_text = "DevDesk-RAG 시스템의 성능을 최적화하고 검색 알고리즘을 개선하는 방법"
    
    # DoLA 어댑터 생성
    adapter = system.create_dola_adapter(
        name="test_dola",
        weight_shape=(100, 200),
        task_type=TaskType.SEARCH,
        domain_type=DomainType.TECHNICAL
    )
    
    # 텍스트 분석 및 적응
    adapted_adapter = system.analyze_and_adapt(test_text, adapter)
    
    # 시스템 상태 출력
    status = system.get_system_status()
    print(f"시스템 상태: {json.dumps(status, indent=2, default=str)}")
    
    print("DoLA 시스템 테스트 완료!")
