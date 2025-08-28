"""
DevDesk-RAG DoRA (Weight-Decomposed Low-Rank Adaptation) 시스템

DoRA는 가중치 분해를 통한 고급 LoRA로, 더 효율적이고 정확한 모델 적응을 제공합니다.
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DoRAConfig:
    """DoRA 설정을 관리하는 클래스"""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    bias: bool = False
    device: str = "auto"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class WeightDecomposition:
    """가중치 분해 결과를 저장하는 클래스"""
    original_weight: torch.Tensor
    decomposed_weights: List[torch.Tensor]
    rank: int
    decomposition_method: str
    timestamp: datetime = field(default_factory=datetime.now)

class WeightDecomposer:
    """가중치 분해를 수행하는 클래스"""
    
    def __init__(self, config: DoRAConfig):
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"WeightDecomposer 초기화 완료 (device: {self.device})")
    
    def decompose_weight(self, weight: torch.Tensor, method: str = "svd") -> WeightDecomposition:
        """
        가중치 텐서를 분해합니다.
        
        Args:
            weight: 분해할 가중치 텐서
            method: 분해 방법 ("svd", "pca", "random")
        
        Returns:
            WeightDecomposition: 분해 결과
        """
        if method == "svd":
            return self._svd_decomposition(weight)
        elif method == "pca":
            return self._pca_decomposition(weight)
        elif method == "random":
            return self._random_decomposition(weight)
        else:
            raise ValueError(f"지원하지 않는 분해 방법: {method}")
    
    def _svd_decomposition(self, weight: torch.Tensor) -> WeightDecomposition:
        """SVD를 사용한 가중치 분해"""
        try:
            # SVD 수행
            U, S, V = torch.svd(weight)
            
            # 상위 rank개 성분만 선택
            rank = min(self.config.rank, len(S))
            U_rank = U[:, :rank]
            S_rank = S[:rank]
            V_rank = V[:, :rank]
            
            # 분해된 가중치 생성
            decomposed = [
                U_rank * S_rank[0],
                V_rank.T * S_rank[0]
            ]
            
            logger.info(f"SVD 분해 완료: {weight.shape} -> rank {rank}")
            return WeightDecomposition(
                original_weight=weight,
                decomposed_weights=decomposed,
                rank=rank,
                decomposition_method="svd"
            )
            
        except Exception as e:
            logger.error(f"SVD 분해 실패: {e}")
            return self._fallback_decomposition(weight)
    
    def _pca_decomposition(self, weight: torch.Tensor) -> WeightDecomposition:
        """PCA를 사용한 가중치 분해"""
        try:
            # PCA 수행 (공분산 행렬의 고유값 분해)
            weight_centered = weight - weight.mean(dim=0)
            cov_matrix = torch.mm(weight_centered.T, weight_centered)
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
            
            # 상위 rank개 성분 선택
            rank = min(self.config.rank, len(eigenvals))
            top_indices = torch.argsort(eigenvals, descending=True)[:rank]
            top_eigenvecs = eigenvecs[:, top_indices]
            top_eigenvals = eigenvals[top_indices]
            
            # 분해된 가중치 생성
            decomposed = [
                weight_centered @ top_eigenvecs,
                top_eigenvecs.T * torch.sqrt(top_eigenvals)
            ]
            
            logger.info(f"PCA 분해 완료: {weight.shape} -> rank {rank}")
            return WeightDecomposition(
                original_weight=weight,
                decomposed_weights=decomposed,
                rank=rank,
                decomposition_method="pca"
            )
            
        except Exception as e:
            logger.error(f"PCA 분해 실패: {e}")
            return self._fallback_decomposition(weight)
    
    def _random_decomposition(self, weight: torch.Tensor) -> WeightDecomposition:
        """랜덤 분해 (fallback용)"""
        rank = min(self.config.rank, min(weight.shape))
        
        # 랜덤 행렬 생성
        A = torch.randn(weight.shape[0], rank, device=self.device) * 0.1
        B = torch.randn(rank, weight.shape[1], device=self.device) * 0.1
        
        decomposed = [A, B]
        
        logger.info(f"랜덤 분해 완료: {weight.shape} -> rank {rank}")
        return WeightDecomposition(
            original_weight=weight,
            decomposed_weights=decomposed,
            rank=rank,
            decomposition_method="random"
        )
    
    def _fallback_decomposition(self, weight: torch.Tensor) -> WeightDecomposition:
        """분해 실패 시 fallback 분해"""
        logger.warning("분해 실패로 fallback 분해 사용")
        return self._random_decomposition(weight)

class LowRankAdapter:
    """저차원 어댑터를 관리하는 클래스"""
    
    def __init__(self, config: DoRAConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.adapters = {}
        logger.info(f"LowRankAdapter 초기화 완료 (device: {self.device})")
    
    def create_adapter(self, name: str, weight_shape: Tuple[int, ...]) -> nn.Module:
        """
        새로운 저차원 어댑터를 생성합니다.
        
        Args:
            name: 어댑터 이름
            weight_shape: 가중치 텐서의 형태
        
        Returns:
            nn.Module: 생성된 어댑터
        """
        adapter = DoRAAdapter(
            weight_shape=weight_shape,
            config=self.config
        )
        self.adapters[name] = adapter
        logger.info(f"어댑터 '{name}' 생성 완료: {weight_shape}")
        return adapter
    
    def get_adapter(self, name: str) -> Optional[nn.Module]:
        """이름으로 어댑터를 가져옵니다."""
        return self.adapters.get(name)
    
    def list_adapters(self) -> List[str]:
        """모든 어댑터 이름을 반환합니다."""
        return list(self.adapters.keys())
    
    def save_adapter(self, name: str, path: str):
        """어댑터를 파일로 저장합니다."""
        if name in self.adapters:
            torch.save(self.adapters[name].state_dict(), path)
            logger.info(f"어댑터 '{name}' 저장 완료: {path}")
        else:
            logger.error(f"어댑터 '{name}'을 찾을 수 없습니다.")
    
    def load_adapter(self, name: str, path: str):
        """파일에서 어댑터를 로드합니다."""
        if os.path.exists(path):
            adapter = DoRAAdapter(weight_shape=(1, 1), config=self.config)
            adapter.load_state_dict(torch.load(path, map_location=self.device))
            self.adapters[name] = adapter
            logger.info(f"어댑터 '{name}' 로드 완료: {path}")
        else:
            logger.error(f"어댑터 파일을 찾을 수 없습니다: {path}")

class DoRAAdapter(nn.Module):
    """DoRA 어댑터 모듈"""
    
    def __init__(self, weight_shape: Tuple[int, ...], config: DoRAConfig):
        super().__init__()
        self.config = config
        self.weight_shape = weight_shape
        
        # DoRA 파라미터 초기화
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
        
        # 드롭아웃
        self.dropout = nn.Dropout(config.dropout)
        
        # 가중치 초기화
        self._init_weights()
        
        logger.info(f"DoRA 어댑터 초기화 완료: {weight_shape} -> rank {self.rank}")
    
    def _init_weights(self):
        """가중치를 초기화합니다."""
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.scaling)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DoRA 어댑터를 통한 순전파
        
        Args:
            x: 입력 텐서
        
        Returns:
            torch.Tensor: 적응된 출력 텐서
        """
        # LoRA 계산
        lora_output = self.dropout(x @ self.lora_A @ self.lora_B)
        
        # 스케일링 적용
        adapted_output = lora_output * self.scaling * self.alpha / self.rank
        
        return adapted_output
    
    def get_adapted_weight(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        원본 가중치에 DoRA 어댑터를 적용한 가중치를 반환합니다.
        
        Args:
            original_weight: 원본 가중치 텐서
        
        Returns:
            torch.Tensor: 적응된 가중치 텐서
        """
        # DoRA 어댑터 계산
        adapter_weight = self.lora_A @ self.lora_B
        adapted_weight = original_weight + adapter_weight * self.scaling * self.alpha / self.rank
        
        return adapted_weight
    
    def get_adaptation_strength(self) -> float:
        """적응 강도를 반환합니다."""
        return float(self.scaling.item() * self.alpha / self.rank)

class DomainSpecializer:
    """도메인 특화를 관리하는 클래스"""
    
    def __init__(self, config: DoRAConfig):
        self.config = config
        self.domain_adapters = {}
        self.domain_configs = {}
        logger.info("DomainSpecializer 초기화 완료")
    
    def create_domain_adapter(self, domain_name: str, base_adapter: DoRAAdapter) -> DoRAAdapter:
        """
        특정 도메인에 맞는 어댑터를 생성합니다.
        
        Args:
            domain_name: 도메인 이름
            base_adapter: 기본 어댑터
        
        Returns:
            DoRAAdapter: 도메인 특화 어댑터
        """
        # 도메인별 설정 로드
        domain_config = self._load_domain_config(domain_name)
        
        # 도메인 특화 어댑터 생성
        domain_adapter = DoRAAdapter(
            weight_shape=base_adapter.weight_shape,
            config=domain_config
        )
        
        # 도메인별 가중치 초기화
        self._initialize_domain_weights(domain_adapter, domain_name)
        
        self.domain_adapters[domain_name] = domain_adapter
        logger.info(f"도메인 어댑터 '{domain_name}' 생성 완료")
        
        return domain_adapter
    
    def _load_domain_config(self, domain_name: str) -> DoRAConfig:
        """도메인별 설정을 로드합니다."""
        # 기본 설정
        config = DoRAConfig()
        
        # 도메인별 설정 파일 확인
        config_path = f"configs/{domain_name}_dora.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                domain_settings = json.load(f)
                for key, value in domain_settings.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        return config
    
    def _initialize_domain_weights(self, adapter: DoRAAdapter, domain_name: str):
        """도메인별 가중치를 초기화합니다."""
        # 도메인별 사전 훈련된 가중치가 있다면 로드
        weight_path = f"weights/{domain_name}_dora.pth"
        if os.path.exists(weight_path):
            try:
                adapter.load_state_dict(torch.load(weight_path, map_location=self.config.device))
                logger.info(f"도메인 가중치 로드 완료: {weight_path}")
            except Exception as e:
                logger.warning(f"도메인 가중치 로드 실패: {e}")
    
    def get_domain_adapter(self, domain_name: str) -> Optional[DoRAAdapter]:
        """도메인별 어댑터를 가져옵니다."""
        return self.domain_adapters.get(domain_name)
    
    def list_domains(self) -> List[str]:
        """지원하는 도메인 목록을 반환합니다."""
        return list(self.domain_adapters.keys())

class DoRASystem:
    """DoRA 시스템의 메인 클래스"""
    
    def __init__(self, config: DoRAConfig = None):
        if config is None:
            config = DoRAConfig()
        
        self.config = config
        self.weight_decomposer = WeightDecomposer(config)
        self.low_rank_adapter = LowRankAdapter(config)
        self.domain_specializer = DomainSpecializer(config)
        
        logger.info("DoRA 시스템 초기화 완료")
    
    def create_dora_adapter(self, name: str, weight_shape: Tuple[int, ...], 
                           domain: str = None) -> DoRAAdapter:
        """
        DoRA 어댑터를 생성합니다.
        
        Args:
            name: 어댑터 이름
            weight_shape: 가중치 형태
            domain: 도메인 이름 (선택사항)
        
        Returns:
            DoRAAdapter: 생성된 DoRA 어댑터
        """
        # 기본 어댑터 생성
        base_adapter = self.low_rank_adapter.create_adapter(name, weight_shape)
        
        # 도메인 특화가 요청된 경우
        if domain:
            domain_adapter = self.domain_specializer.create_domain_adapter(
                domain, base_adapter
            )
            return domain_adapter
        
        return base_adapter
    
    def decompose_and_adapt(self, weight: torch.Tensor, method: str = "svd") -> Tuple[WeightDecomposition, DoRAAdapter]:
        """
        가중치를 분해하고 DoRA 어댑터를 생성합니다.
        
        Args:
            weight: 분해할 가중치
            method: 분해 방법
        
        Returns:
            Tuple[WeightDecomposition, DoRAAdapter]: 분해 결과와 어댑터
        """
        # 가중치 분해
        decomposition = self.weight_decomposer.decompose_weight(weight, method)
        
        # DoRA 어댑터 생성
        adapter = self.create_dora_adapter(
            name=f"dora_{method}_{decomposition.rank}",
            weight_shape=weight.shape
        )
        
        logger.info(f"가중치 분해 및 DoRA 어댑터 생성 완료: {weight.shape}")
        
        return decomposition, adapter
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태를 반환합니다."""
        return {
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "device": str(self.config.device)
            },
            "adapters": {
                "total": len(self.low_rank_adapter.adapters),
                "names": self.low_rank_adapter.list_adapters()
            },
            "domains": {
                "total": len(self.domain_specializer.domain_adapters),
                "names": self.domain_specializer.list_domains()
            }
        }

# 전역 DoRA 시스템 인스턴스
dora_system = DoRASystem()

def get_dora_system() -> DoRASystem:
    """전역 DoRA 시스템 인스턴스를 반환합니다."""
    return dora_system

if __name__ == "__main__":
    # 테스트 코드
    print("DoRA 시스템 테스트 시작...")
    
    # 시스템 초기화
    system = DoRASystem()
    
    # 가상 가중치 생성
    test_weight = torch.randn(100, 200)
    
    # 가중치 분해 및 어댑터 생성
    decomposition, adapter = system.decompose_and_adapt(test_weight, "svd")
    
    # 시스템 상태 출력
    status = system.get_system_status()
    print(f"시스템 상태: {json.dumps(status, indent=2, default=str)}")
    
    print("DoRA 시스템 테스트 완료!")
