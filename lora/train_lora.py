#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DevDesk-RAG 특화 LoRA 모델 훈련 스크립트
EXAONE 모델을 기반으로 DevDesk-RAG 시스템에 특화된 LoRA 어댑터를 훈련합니다.
"""

import os
import torch
import json
from typing import List, Dict
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DevDeskRAGLoRATrainer:
    """DevDesk-RAG 특화 LoRA 훈련기"""
    
    def __init__(self, model_name: str = "exaone3.5:7.8b"):
        self.model_name = model_name
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        
    def _setup_device(self) -> torch.device:
        """훈련 디바이스 설정"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🚀 Apple Silicon MPS (Metal) 사용")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("🚀 NVIDIA GPU CUDA 사용")
        else:
            device = torch.device("cpu")
            logger.info("💻 CPU 사용 (느릴 수 있음)")
        
        return device
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        logger.info(f"📥 모델 로딩 중: {self.model_name}")
        
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                device_map="auto" if self.device.type != "cpu" else None
            )
            
            # 모델을 디바이스로 이동
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("✅ 모델과 토크나이저 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def setup_lora_config(self):
        """LoRA 설정 구성"""
        logger.info("🔧 LoRA 설정 구성 중...")
        
        self.lora_config = LoraConfig(
            r=16,  # LoRA 랭크
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        logger.info("✅ LoRA 설정 완료")
    
    def apply_lora_to_model(self):
        """모델에 LoRA 적용"""
        logger.info("🔗 LoRA를 모델에 적용 중...")
        
        try:
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
            logger.info("✅ LoRA 적용 완료")
            
        except Exception as e:
            logger.error(f"❌ LoRA 적용 실패: {e}")
            raise
    
    def load_training_data(self, filename: str = "devdesk_rag_training.jsonl") -> Dataset:
        """훈련 데이터 로드"""
        logger.info(f"📚 훈련 데이터 로드 중: {filename}")
        
        try:
            data = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            # 데이터셋 생성
            dataset = Dataset.from_list(data)
            logger.info(f"✅ {len(dataset)}개 샘플 로드 완료")
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ 훈련 데이터 로드 실패: {e}")
            raise
    
    def tokenize_function(self, examples):
        """데이터 토크나이징"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """데이터셋 전처리"""
        logger.info("🔧 데이터셋 전처리 중...")
        
        # 토크나이징
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info("✅ 데이터셋 전처리 완료")
        return tokenized_dataset
    
    def setup_training_args(self, output_dir: str = "./devdesk_rag_lora") -> TrainingArguments:
        """훈련 인자 설정"""
        logger.info("⚙️ 훈련 인자 설정 중...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",  # wandb 비활성화
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            fp16=self.device.type != "cpu",
            bf16=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
            data_seed=42,
            group_by_length=True,
            length_column_name="length",
            ignore_data_skip=False,
            dataloader_drop_last=False,
            eval_delay=0,
            save_on_each_node=False,
            auto_find_batch_size=False,
            full_determinism=False,
            ddp_find_unused_parameters=None,
            dataloader_pin_memory_device="",
            torch_compile=False,
            torch_compile_backend="inductor",
            torch_compile_mode="default",
            dataloader_prefetch_factor=None,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
            dataloader_persistent_workers=False,
        )
        
        logger.info("✅ 훈련 인자 설정 완료")
        return training_args
    
    def train(self, dataset: Dataset, output_dir: str = "./devdesk_rag_lora"):
        """LoRA 모델 훈련 실행"""
        logger.info("🚀 LoRA 훈련 시작!")
        
        try:
            # 1. 모델과 토크나이저 로드
            self.load_model_and_tokenizer()
            
            # 2. LoRA 설정
            self.setup_lora_config()
            
            # 3. LoRA 적용
            self.apply_lora_to_model()
            
            # 4. 데이터셋 전처리
            processed_dataset = self.prepare_dataset(dataset)
            
            # 5. 훈련 인자 설정
            training_args = self.setup_training_args(output_dir)
            
            # 6. 데이터 콜레이터 설정
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # 7. 훈련기 생성 및 훈련 실행
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            logger.info("🔥 훈련 실행 중...")
            trainer.train()
            
            # 8. 모델 저장
            logger.info(f"💾 모델 저장 중: {output_dir}")
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info("🎉 LoRA 훈련 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 훈련 실패: {e}")
            raise

def main():
    """메인 실행 함수"""
    logger.info("🚀 DevDesk-RAG 특화 LoRA 훈련 시작!")
    
    try:
        # 1. 훈련기 초기화
        trainer = DevDeskRAGLoRATrainer()
        
        # 2. 훈련 데이터 로드
        dataset = trainer.load_training_data()
        
        # 3. LoRA 훈련 실행
        success = trainer.train(dataset)
        
        if success:
            logger.info("🎉 DevDesk-RAG 특화 LoRA 훈련이 성공적으로 완료되었습니다!")
            logger.info("📁 훈련된 모델: ./devdesk_rag_lora/")
            logger.info("🚀 이제 Ollama에 적용할 수 있습니다!")
        
    except Exception as e:
        logger.error(f"❌ 훈련 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
