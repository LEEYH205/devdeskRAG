"""
DevDesk-RAG 통합 LoRA 관리 시스템

DoRA와 DoLA 시스템을 통합하여 효율적으로 관리하고, 사용자별 최적화된 모델을 제공합니다.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path

# DoRA 및 DoLA 시스템 import
from .dora_system import DoRASystem, DoRAConfig, get_dora_system
from .dola_system import DoLASystem, DoLAConfig, get_dola_system, TaskType, DomainType

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegratedLoRAConfig:
    """통합 LoRA 설정을 관리하는 클래스"""
    # DoRA 설정
    dora_rank: int = 8
    dora_alpha: float = 16.0
    dora_dropout: float = 0.1
    
    # DoLA 설정
    dola_rank: int = 8
    dola_alpha: float = 16.0
    dola_dropout: float = 0.1
    
    # 통합 설정
    auto_adaptation: bool = True
    performance_threshold: float = 0.8
    adaptation_frequency: int = 100  # N번 요청마다 적응
    device: str = "auto"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class AdapterProfile:
    """어댑터 프로필을 저장하는 클래스"""
    adapter_id: str
    adapter_type: str  # "dora" or "dola"
    task_type: Optional[str] = None
    domain_type: Optional[str] = None
    weight_shape: Tuple[int, ...] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class UserModelProfile:
    """사용자별 모델 프로필을 저장하는 클래스"""
    user_id: str
    preferred_adapters: Dict[str, str]  # task_type -> adapter_id
    performance_history: List[Dict[str, Any]]
    adaptation_preferences: Dict[str, Any]
    last_updated: datetime = field(default_factory=datetime.now)

class AdapterRegistry:
    """어댑터 레지스트리를 관리하는 클래스"""
    
    def __init__(self):
        self.adapters: Dict[str, nn.Module] = {}
        self.adapter_profiles: Dict[str, AdapterProfile] = {}
        self.user_profiles: Dict[str, UserModelProfile] = {}
        logger.info("AdapterRegistry 초기화 완료")
    
    def register_adapter(self, adapter_id: str, adapter: nn.Module, 
                        adapter_type: str, **kwargs) -> AdapterProfile:
        """
        새로운 어댑터를 레지스트리에 등록합니다.
        
        Args:
            adapter_id: 어댑터 고유 ID
            adapter: 등록할 어댑터
            adapter_type: 어댑터 유형 ("dora" 또는 "dola")
            **kwargs: 추가 정보 (task_type, domain_type, weight_shape 등)
        
        Returns:
            AdapterProfile: 생성된 어댑터 프로필
        """
        # 어댑터 등록
        self.adapters[adapter_id] = adapter
        
        # 프로필 생성
        profile = AdapterProfile(
            adapter_id=adapter_id,
            adapter_type=adapter_type,
            task_type=kwargs.get('task_type'),
            domain_type=kwargs.get('domain_type'),
            weight_shape=kwargs.get('weight_shape'),
            created_at=datetime.now()
        )
        
        self.adapter_profiles[adapter_id] = profile
        logger.info(f"어댑터 등록 완료: {adapter_id} ({adapter_type})")
        
        return profile
    
    def get_adapter(self, adapter_id: str) -> Optional[nn.Module]:
        """어댑터 ID로 어댑터를 가져옵니다."""
        return self.adapters.get(adapter_id)
    
    def get_adapter_profile(self, adapter_id: str) -> Optional[AdapterProfile]:
        """어댑터 ID로 프로필을 가져옵니다."""
        return self.adapter_profiles.get(adapter_id)
    
    def list_adapters(self, adapter_type: str = None) -> List[str]:
        """등록된 어댑터 목록을 반환합니다."""
        if adapter_type:
            return [aid for aid, profile in self.adapter_profiles.items() 
                   if profile.adapter_type == adapter_type]
        return list(self.adapters.keys())
    
    def update_adapter_performance(self, adapter_id: str, metrics: Dict[str, float]):
        """어댑터 성능을 업데이트합니다."""
        if adapter_id in self.adapter_profiles:
            self.adapter_profiles[adapter_id].performance_metrics.update(metrics)
            self.adapter_profiles[adapter_id].last_used = datetime.now()
            logger.info(f"어댑터 성능 업데이트 완료: {adapter_id}")
    
    def increment_usage(self, adapter_id: str):
        """어댑터 사용 횟수를 증가시킵니다."""
        if adapter_id in self.adapter_profiles:
            self.adapter_profiles[adapter_id].usage_count += 1
            self.adapter_profiles[adapter_id].last_used = datetime.now()
    
    def get_user_profile(self, user_id: str) -> UserModelProfile:
        """사용자 프로필을 가져오거나 생성합니다."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserModelProfile(
                user_id=user_id,
                preferred_adapters={},
                performance_history=[],
                adaptation_preferences={}
            )
        return self.user_profiles[user_id]
    
    def update_user_preferences(self, user_id: str, task_type: str, adapter_id: str):
        """사용자 선호도를 업데이트합니다."""
        profile = self.get_user_profile(user_id)
        profile.preferred_adapters[task_type] = adapter_id
        profile.last_updated = datetime.now()
        logger.info(f"사용자 선호도 업데이트 완료: {user_id} - {task_type}")

class PerformanceOptimizer:
    """성능 최적화를 관리하는 클래스"""
    
    def __init__(self, config: IntegratedLoRAConfig):
        self.config = config
        self.optimization_history = []
        logger.info("PerformanceOptimizer 초기화 완료")
    
    def optimize_adapter_selection(self, user_id: str, task_type: str, 
                                 domain_context: Dict[str, Any],
                                 available_adapters: List[str]) -> str:
        """
        사용자와 태스크에 최적화된 어댑터를 선택합니다.
        
        Args:
            user_id: 사용자 ID
            task_type: 태스크 유형
            domain_context: 도메인 컨텍스트
            available_adapters: 사용 가능한 어댑터 목록
        
        Returns:
            str: 선택된 어댑터 ID
        """
        # 사용자 선호도 확인
        user_profile = self._get_user_profile(user_id)
        preferred_adapter = user_profile.preferred_adapters.get(task_type)
        
        if preferred_adapter and preferred_adapter in available_adapters:
            return preferred_adapter
        
        # 성능 기반 최적화
        best_adapter = self._select_best_performing_adapter(
            available_adapters, task_type, domain_context
        )
        
        # 사용자 선호도 업데이트
        user_profile.preferred_adapters[task_type] = best_adapter
        
        logger.info(f"어댑터 선택 최적화 완료: {user_id} - {task_type} -> {best_adapter}")
        return best_adapter
    
    def _get_user_profile(self, user_id: str) -> UserModelProfile:
        """사용자 프로필을 가져옵니다."""
        # 실제로는 AdapterRegistry에서 가져와야 함
        return UserModelProfile(
            user_id=user_id,
            preferred_adapters={},
            performance_history=[],
            adaptation_preferences={}
        )
    
    def _select_best_performing_adapter(self, available_adapters: List[str],
                                      task_type: str, domain_context: Dict[str, Any]) -> str:
        """성능이 가장 좋은 어댑터를 선택합니다."""
        # 간단한 선택 로직 (실제로는 더 정교한 방법 사용)
        if not available_adapters:
            return "default"
        
        # 첫 번째 사용 가능한 어댑터 반환
        return available_adapters[0]
    
    def record_optimization(self, optimization_data: Dict[str, Any]):
        """최적화 과정을 기록합니다."""
        record = {
            "timestamp": datetime.now(),
            **optimization_data
        }
        self.optimization_history.append(record)
        logger.info("최적화 과정 기록 완료")

class IntegratedLoRAManager:
    """DoRA와 DoLA 시스템을 통합하여 관리하는 메인 클래스"""
    
    def __init__(self, config: IntegratedLoRAConfig = None):
        if config is None:
            config = IntegratedLoRAConfig()
        
        self.config = config
        self.dora_system = get_dora_system()
        self.dola_system = get_dola_system()
        self.adapter_registry = AdapterRegistry()
        self.performance_optimizer = PerformanceOptimizer(config)
        
        # 사용 통계
        self.request_count = 0
        self.adaptation_count = 0
        
        logger.info("IntegratedLoRAManager 초기화 완료")
    
    def create_specialized_model(self, user_id: str, task_type: str,
                               domain_type: str = None, weight_shape: Tuple[int, ...] = None) -> str:
        """
        사용자별, 태스크별 특화 모델을 생성합니다.
        
        Args:
            user_id: 사용자 ID
            task_type: 태스크 유형
            domain_type: 도메인 유형 (선택사항)
            weight_shape: 가중치 형태 (선택사항)
        
        Returns:
            str: 생성된 어댑터 ID
        """
        # 기본 가중치 형태 설정
        if weight_shape is None:
            weight_shape = (100, 200)  # 기본값
        
        # DoRA 어댑터 생성
        dora_adapter_id = f"dora_{user_id}_{task_type}"
        dora_adapter = self.dora_system.create_dora_adapter(
            name=dora_adapter_id,
            weight_shape=weight_shape,
            domain=domain_type
        )
        
        # DoLA 어댑터 생성
        dola_adapter_id = f"dola_{user_id}_{task_type}"
        dola_adapter = self.dola_system.create_dola_adapter(
            name=dola_adapter_id,
            weight_shape=weight_shape,
            task_type=TaskType(task_type) if task_type else None,
            domain_type=DomainType(domain_type) if domain_type else None
        )
        
        # 레지스트리에 등록
        self.adapter_registry.register_adapter(
            dora_adapter_id, dora_adapter, "dora",
            task_type=task_type, domain_type=domain_type, weight_shape=weight_shape
        )
        
        self.adapter_registry.register_adapter(
            dola_adapter_id, dola_adapter, "dola",
            task_type=task_type, domain_type=domain_type, weight_shape=weight_shape
        )
        
        logger.info(f"사용자별 특화 모델 생성 완료: {user_id} - {task_type}")
        
        return dora_adapter_id  # 기본적으로 DoRA 반환
    
    def get_optimized_adapter(self, user_id: str, task_type: str,
                            domain_context: Dict[str, Any] = None) -> nn.Module:
        """
        사용자와 태스크에 최적화된 어댑터를 가져옵니다.
        
        Args:
            user_id: 사용자 ID
            task_type: 태스크 유형
            domain_context: 도메인 컨텍스트 (선택사항)
        
        Returns:
            nn.Module: 최적화된 어댑터
        """
        # 사용 가능한 어둸터 목록
        available_adapters = self.adapter_registry.list_adapters()
        
        # 최적화된 어댑터 선택
        selected_adapter_id = self.performance_optimizer.optimize_adapter_selection(
            user_id, task_type, domain_context or {}, available_adapters
        )
        
        # 어댑터 가져오기
        adapter = self.adapter_registry.get_adapter(selected_adapter_id)
        if adapter is None:
            # 기본 어댑터 생성
            adapter = self._create_default_adapter(user_id, task_type)
        
        # 사용 통계 업데이트
        self.request_count += 1
        self.adapter_registry.increment_usage(selected_adapter_id)
        
        # 자동 적응 확인
        if self.config.auto_adaptation and self.request_count % self.config.adaptation_frequency == 0:
            adapter = self._auto_adapt_adapter(adapter, user_id, task_type, domain_context)
        
        return adapter
    
    def _create_default_adapter(self, user_id: str, task_type: str) -> nn.Module:
        """기본 어댑터를 생성합니다."""
        logger.info(f"기본 어댑터 생성: {user_id} - {task_type}")
        
        # DoRA 기본 어댑터 생성
        default_adapter = self.dora_system.create_dora_adapter(
            name=f"default_{user_id}_{task_type}",
            weight_shape=(100, 200)
        )
        
        # 레지스트리에 등록
        adapter_id = f"default_{user_id}_{task_type}"
        self.adapter_registry.register_adapter(
            adapter_id, default_adapter, "dora",
            task_type=task_type, weight_shape=(100, 200)
        )
        
        return default_adapter
    
    def _auto_adapt_adapter(self, adapter: nn.Module, user_id: str, task_type: str,
                           domain_context: Dict[str, Any]) -> nn.Module:
        """
        어댑터를 자동으로 적응시킵니다.
        
        Args:
            adapter: 적응시킬 어댑터
            user_id: 사용자 ID
            task_type: 태스크 유형
            domain_context: 도메인 컨텍스트
        
        Returns:
            nn.Module: 적응된 어댑터
        """
        try:
            # DoLA 시스템을 사용한 적응
            if hasattr(adapter, 'task_type') and hasattr(adapter, 'domain_type'):
                # DoLA 어댑터인 경우
                adapted_adapter = self.dola_system.analyze_and_adapt(
                    text=domain_context.get('text', ''),
                    adapter=adapter,
                    performance_feedback=domain_context.get('performance', {})
                )
            else:
                # DoRA 어댑터인 경우
                adapted_adapter = adapter
            
            self.adaptation_count += 1
            logger.info(f"자동 적응 완료: {user_id} - {task_type} (총 {self.adaptation_count}회)")
            
            return adapted_adapter
            
        except Exception as e:
            logger.error(f"자동 적응 실패: {e}")
            return adapter
    
    def update_performance_metrics(self, adapter_id: str, metrics: Dict[str, float]):
        """어댑터 성능 메트릭을 업데이트합니다."""
        self.adapter_registry.update_adapter_performance(adapter_id, metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태를 반환합니다."""
        return {
            "config": {
                "dora_rank": self.config.dora_rank,
                "dola_rank": self.config.dola_rank,
                "auto_adaptation": self.config.auto_adaptation,
                "performance_threshold": self.config.performance_threshold,
                "adaptation_frequency": self.config.adaptation_frequency,
                "device": str(self.config.device)
            },
            "usage_statistics": {
                "total_requests": self.request_count,
                "total_adaptations": self.adaptation_count,
                "adaptation_rate": self.adaptation_count / max(self.request_count, 1)
            },
            "adapter_registry": {
                "total_adapters": len(self.adapter_registry.adapters),
                "dora_adapters": len(self.adapter_registry.list_adapters("dora")),
                "dola_adapters": len(self.adapter_registry.list_adapters("dola"))
            },
            "dora_system": self.dora_system.get_system_status(),
            "dola_system": self.dola_system.get_system_status()
        }
    
    def export_adapters(self, export_path: str):
        """등록된 모든 어댑터를 파일로 내보냅니다."""
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        for adapter_id, adapter in self.adapter_registry.adapters.items():
            adapter_path = export_dir / f"{adapter_id}.pth"
            torch.save(adapter.state_dict(), adapter_path)
        
        # 프로필 정보도 내보내기
        profiles_path = export_dir / "adapter_profiles.json"
        with open(profiles_path, 'w') as f:
            json.dump([profile.__dict__ for profile in self.adapter_registry.adapter_profiles.values()], 
                     f, default=str, indent=2)
        
        logger.info(f"어댑터 내보내기 완료: {export_path}")
    
    def import_adapters(self, import_path: str):
        """파일에서 어댑터를 가져옵니다."""
        import_dir = Path(import_path)
        
        if not import_dir.exists():
            logger.error(f"가져올 경로가 존재하지 않습니다: {import_path}")
            return
        
        # 프로필 정보 로드
        profiles_path = import_dir / "adapter_profiles.json"
        if profiles_path.exists():
            with open(profiles_path, 'r') as f:
                profiles_data = json.load(f)
            
            for profile_data in profiles_data:
                # 어댑터 파일 로드
                adapter_path = import_dir / f"{profile_data['adapter_id']}.pth"
                if adapter_path.exists():
                    # 어댑터 타입에 따라 적절한 클래스로 로드
                    if profile_data['adapter_type'] == 'dora':
                        adapter = self.dora_system.low_rank_adapter.get_adapter(profile_data['adapter_id'])
                    else:  # dola
                        adapter = self.dola_system.task_specializer.get_task_adapter(
                            TaskType(profile_data['task_type']),
                            DomainType(profile_data['domain_type'])
                        )
                    
                    if adapter:
                        self.adapter_registry.adapters[profile_data['adapter_id']] = adapter
        
        logger.info(f"어댑터 가져오기 완료: {import_path}")

# 전역 통합 LoRA 관리자 인스턴스
integrated_lora_manager = IntegratedLoRAManager()

def get_integrated_lora_manager() -> IntegratedLoRAManager:
    """전역 통합 LoRA 관리자 인스턴스를 반환합니다."""
    return integrated_lora_manager

if __name__ == "__main__":
    # 테스트 코드
    print("통합 LoRA 관리 시스템 테스트 시작...")
    
    # 시스템 초기화
    manager = IntegratedLoRAManager()
    
    # 사용자별 특화 모델 생성
    adapter_id = manager.create_specialized_model(
        user_id="user_123",
        task_type="search",
        domain_type="technical"
    )
    
    # 최적화된 어댑터 가져오기
    adapter = manager.get_optimized_adapter(
        user_id="user_123",
        task_type="search",
        domain_context={"text": "DevDesk-RAG 성능 최적화"}
    )
    
    # 시스템 상태 출력
    status = manager.get_system_status()
    print(f"시스템 상태: {json.dumps(status, indent=2, default=str)}")
    
    print("통합 LoRA 관리 시스템 테스트 완료!")
