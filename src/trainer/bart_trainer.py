#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BART 모델 트레이너

ESConv 데이터셋에 대한 BART 모델 학습 및 평가를 위한 트레이너 클래스를 제공합니다.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
import numpy as np
import logging
import time
from datetime import datetime
from transformers import BartTokenizer, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BartTrainer")

class BartTrainer:
    """BART 모델 학습 및 평가 클래스"""
    
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer_name: str = "facebook/bart-base",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.1,
        output_dir: str = "./outputs",
        device: str = None
    ):
        """
        Args:
            model: BART 모델
            train_dataloader: 학습 데이터 로더
            val_dataloader: 검증 데이터 로더
            tokenizer_name: 토크나이저 이름
            learning_rate: 학습률
            weight_decay: 가중치 감쇠
            gradient_accumulation_steps: 그래디언트 누적 단계
            max_grad_norm: 최대 그래디언트 노름
            warmup_ratio: 워밍업 비율
            output_dir: 출력 디렉토리
            device: 학습 장치 ('cuda' 또는 'cpu')
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        
        # 출력 디렉토리 생성
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 모델 장치 설정
        self.model = self.model.to(self.device)
        
        # 최적화 설정
        self._setup_optimizer()
        
        # ROUGE 평가 도구 초기화
        self.rouge = Rouge()
    
    def _setup_optimizer(self):
        """최적화 설정"""
        # 매개변수 그룹 설정
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # 최적화 초기화
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # 스케줄러 설정
        total_steps = len(self.train_dataloader) * 1  # 1 epoch
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    
    def train(self, num_epochs: int = 3, save_steps: int = 500) -> Dict[str, List[float]]:
        """
        모델을 학습합니다.
        
        Args:
            num_epochs: 학습 에폭 수
            save_steps: 모델 저장 단계
            
        Returns:
            학습 기록
        """
        logger.info(f"Starting training on {self.device}...")
        logger.info(f"Number of training examples: {len(self.train_dataloader.dataset)}")
        logger.info(f"Number of validation examples: {len(self.val_dataloader.dataset)}")
        
        # 학습 기록 초기화
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_bleu": [],
            "val_rouge": []
        }
        
        # 전체 단계 계산
        total_steps = len(self.train_dataloader) * num_epochs
        global_step = 0
        best_val_bleu = 0.0
        
        # 시간 측정 시작
        start_time = time.time()
        
        # 에폭별 학습
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 학습 모드 설정
            self.model.train()
            epoch_loss = 0.0
            
            # 진행 표시줄 설정
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Training (Epoch {epoch+1})",
                leave=True
            )
            
            # 배치별 학습
            for step, batch in enumerate(progress_bar):
                # 배치 데이터 준비
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 순전파 및 손실 계산
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 그래디언트 누적
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # 그래디언트 누적 단계 완료 시 최적화
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # 최적화 단계
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 모델 저장
                    if global_step % save_steps == 0:
                        self._save_checkpoint(global_step)
                
                # 손실 누적
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                
                # 진행 표시줄 업데이트
                progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
            
            # 에폭 평균 손실 계산
            epoch_loss /= len(self.train_dataloader)
            history["train_loss"].append(epoch_loss)
            
            logger.info(f"Epoch {epoch+1} - Train loss: {epoch_loss:.4f}")
            
            # 검증
            val_metrics = self.evaluate()
            val_loss = val_metrics["loss"]
            val_bleu = val_metrics["bleu"]
            val_rouge = val_metrics["rouge"]["rouge-l"]["f"]
            
            history["val_loss"].append(val_loss)
            history["val_bleu"].append(val_bleu)
            history["val_rouge"].append(val_rouge)
            
            logger.info(f"Epoch {epoch+1} - Val loss: {val_loss:.4f}, Val BLEU: {val_bleu:.4f}, Val ROUGE-L: {val_rouge:.4f}")
            
            # 최고 성능 모델 저장
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                self._save_best_model()
                logger.info(f"New best model saved with BLEU: {val_bleu:.4f}")
        
        # 학습 시간 계산
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # 최종 모델 저장
        self._save_final_model()
        logger.info("Final model saved")
        
        return history
    
    def evaluate(self, log_examples: int = 5) -> Dict[str, Any]:
        """
        모델을 평가합니다.
        
        Args:
            log_examples: 결과를 로깅할 예제 수
            
        Returns:
            평가 지표
        """
        logger.info("Starting evaluation...")
        
        # 평가 모드 설정
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        # 예제 로깅을 위한 카운터
        logged_examples = 0
        
        # 진행 표시줄 설정
        progress_bar = tqdm(
            self.val_dataloader, 
            desc="Evaluating",
            leave=True
        )
        
        with torch.no_grad():
            for step, batch in enumerate(progress_bar):
                # 배치 데이터 준비
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 순전파 및 손실 계산
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 손실 누적
                total_loss += loss.item()
                
                # 예측 생성
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                target_ids = batch["target_ids"]
                
                # 전략과 감정 ID 추출 (있는 경우)
                strategy_ids = batch.get("strategy_ids", None)
                emotion_ids = batch.get("emotion_ids", None)
                
                # 생성 수행
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    strategy_ids=strategy_ids,
                    emotion_ids=emotion_ids,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
                # 디코딩
                predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                targets = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
                
                # 예측 및 타겟 누적
                all_preds.extend(predictions)
                all_targets.extend(targets)
                
                # 예제 로깅
                if logged_examples < log_examples:
                    batch_size = input_ids.size(0)
                    for i in range(min(batch_size, log_examples - logged_examples)):
                        input_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                        logger.info(f"Example {logged_examples + 1}:")
                        logger.info(f"Input: {input_text[:100]}...")
                        logger.info(f"Target: {targets[i]}")
                        logger.info(f"Prediction: {predictions[i]}")
                        logger.info("---")
                        logged_examples += 1
        
        # 평균 손실 계산
        val_loss = total_loss / len(self.val_dataloader)
        
        # BLEU 점수 계산
        bleu_scores = []
        smoothie = SmoothingFunction().method1
        for pred, target in zip(all_preds, all_targets):
            pred_tokens = pred.split()
            target_tokens = target.split()
            
            if len(pred_tokens) > 0 and len(target_tokens) > 0:
                bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothie)
                bleu_scores.append(bleu)
        
        # 평균 BLEU 점수 계산
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # ROUGE 점수 계산
        try:
            rouge_scores = self.rouge.get_scores(all_preds, all_targets, avg=True)
        except:
            # ROUGE 계산 실패 시 기본값 사용
            rouge_scores = {
                "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
            }
        
        # 결과 반환
        results = {
            "loss": val_loss,
            "bleu": avg_bleu,
            "rouge": rouge_scores,
            "num_examples": len(all_preds)
        }
        
        return results
    
    def _save_checkpoint(self, step: int) -> None:
        """
        체크포인트를 저장합니다.
        
        Args:
            step: 현재 단계
        """
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
        self.model.save_model(checkpoint_path)
        
        # 최적화 상태 저장
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # 스케줄러 상태 저장
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _save_best_model(self) -> None:
        """최고 성능 모델을 저장합니다."""
        best_model_dir = os.path.join(self.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        best_model_path = os.path.join(best_model_dir, "model.pt")
        self.model.save_model(best_model_path)
        
        logger.info(f"Best model saved to {best_model_dir}")
    
    def _save_final_model(self) -> None:
        """최종 모델을 저장합니다."""
        final_model_dir = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        
        final_model_path = os.path.join(final_model_dir, "model.pt")
        self.model.save_model(final_model_path)
        
        logger.info(f"Final model saved to {final_model_dir}") 