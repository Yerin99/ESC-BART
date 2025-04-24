#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESConv 데이터셋용 BART 모델

이 모듈은 BART 모델을 확장하여 ESConv 데이터셋에 대한 응답을 생성하기 위한 클래스를 제공합니다.
전략(strategy)과 감정(emotion) 정보를 벡터로 주입하여 응답 생성 품질을 향상시킵니다.
"""

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from typing import Dict, List, Tuple, Optional, Union, Any

class BartForESConv(nn.Module):
    """
    ESConv 데이터셋을 위한 BART 모델
    
    전략과 감정 정보를 벡터로 주입하여 응답을 생성합니다.
    4가지 모드 지원:
    1. 전략만 사용
    2. 감정만 사용 
    3. 전략과 감정 모두 사용
    4. 전략과 감정 모두 사용하지 않음 (기본 BART)
    """
    
    def __init__(
        self, 
        model_name_or_path: str = "facebook/bart-base",
        strategy_embedding_dim: int = 64,
        emotion_embedding_dim: int = 32,
        num_strategies: int = 8,  # ESConv 데이터셋의 전략 수
        num_emotions: int = 6,    # ESConv 데이터셋의 감정 수 
        use_strategy: bool = True,
        use_emotion: bool = True,
        device: str = None
    ):
        """
        Args:
            model_name_or_path: BART 모델 이름 또는 경로
            strategy_embedding_dim: 전략 임베딩 차원
            emotion_embedding_dim: 감정 임베딩 차원
            num_strategies: 전략 종류의 수
            num_emotions: 감정 종류의 수
            use_strategy: 전략 벡터 사용 여부
            use_emotion: 감정 벡터 사용 여부
            device: 모델 실행 장치 ('cuda' 또는 'cpu')
        """
        super().__init__()
        
        # 기본 설정
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_strategy = use_strategy
        self.use_emotion = use_emotion
        
        # 기본 BART 모델 로드
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        
        # 모델 차원 가져오기
        self.hidden_dim = self.model.config.d_model
        
        # 전략 임베딩 레이어
        if self.use_strategy:
            self.strategy_embeddings = nn.Embedding(num_strategies, strategy_embedding_dim)
            self.strategy_projection = nn.Linear(strategy_embedding_dim, self.hidden_dim)
        
        # 감정 임베딩 레이어
        if self.use_emotion:
            self.emotion_embeddings = nn.Embedding(num_emotions, emotion_embedding_dim)
            self.emotion_projection = nn.Linear(emotion_embedding_dim, self.hidden_dim)
            
        # 최종 투사 레이어 (선택적)
        additional_dim = 0
        if self.use_strategy:
            additional_dim += self.hidden_dim
        if self.use_emotion:
            additional_dim += self.hidden_dim
            
        if additional_dim > 0:
            self.final_projection = nn.Linear(self.hidden_dim + additional_dim, self.hidden_dim)
        
        self.to(self.device)
    
    def _prepare_inputs_for_bart(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy_ids: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """BART 모델의 입력을 준비합니다"""
        
        # 기본 BART 인코더 실행
        encoder_outputs = self.model.get_encoder()(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            return_dict=True
        )
        
        # 인코더 hidden states 가져오기
        hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # 벡터 변환 여부 확인
        should_transform = False
        
        # 전략 임베딩 처리
        if self.use_strategy and strategy_ids is not None:
            strategy_embeds = self.strategy_embeddings(strategy_ids)  # [batch_size, strategy_emb_dim]
            strategy_hidden = self.strategy_projection(strategy_embeds)  # [batch_size, hidden_dim]
            # 전략 벡터를 인코더 출력과 결합할 준비
            strategy_hidden_expanded = strategy_hidden.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            hidden_states = torch.cat([hidden_states, strategy_hidden_expanded], dim=-1)
            should_transform = True
            
        # 감정 임베딩 처리
        if self.use_emotion and emotion_ids is not None:
            emotion_embeds = self.emotion_embeddings(emotion_ids)  # [batch_size, emotion_emb_dim]
            emotion_hidden = self.emotion_projection(emotion_embeds)  # [batch_size, hidden_dim]
            # 감정 벡터를 인코더 출력과 결합할 준비
            emotion_hidden_expanded = emotion_hidden.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            hidden_states = torch.cat([hidden_states, emotion_hidden_expanded], dim=-1)
            should_transform = True
            
        # 결합된 벡터를 원래 차원으로 변환
        if should_transform:
            hidden_states = self.final_projection(hidden_states)
            
        # 수정된 인코더 출력 반환
        encoder_outputs.last_hidden_state = hidden_states
        
        return {
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            **kwargs
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        strategy_ids: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        모델 순전파 함수
        
        Args:
            input_ids: 입력 시퀀스의 토큰 ID
            attention_mask: 어텐션 마스크
            decoder_input_ids: 디코더 입력 ID
            strategy_ids: 전략 ID
            emotion_ids: 감정 ID
            labels: 레이블
            
        Returns:
            BART 모델의 출력 (손실, logits 등)
        """
        # 입력 준비
        inputs = self._prepare_inputs_for_bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=strategy_ids,
            emotion_ids=emotion_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs
        )
        
        # BART 모델 호출
        outputs = self.model(**inputs)
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy_ids: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> List[List[int]]:
        """
        입력에 대한 응답을 생성합니다.
        
        Args:
            input_ids: 입력 시퀀스의 토큰 ID
            attention_mask: 어텐션 마스크
            strategy_ids: 전략 ID
            emotion_ids: 감정 ID
            
        Returns:
            생성된 텍스트의 토큰 ID 목록
        """
        # 입력 준비
        inputs = self._prepare_inputs_for_bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=strategy_ids,
            emotion_ids=emotion_ids
        )
        
        # 생성 매개변수 설정
        generation_kwargs = {
            "max_length": kwargs.pop("max_length", 128),
            "min_length": kwargs.pop("min_length", 10),
            "num_beams": kwargs.pop("num_beams", 5),
            "no_repeat_ngram_size": kwargs.pop("no_repeat_ngram_size", 3),
            "early_stopping": kwargs.pop("early_stopping", True),
            **kwargs
        }
        
        # 텍스트 생성
        outputs = self.model.generate(
            encoder_outputs=inputs["encoder_outputs"],
            attention_mask=inputs["attention_mask"],
            **generation_kwargs
        )
        
        return outputs
    
    def save_model(self, save_path: str) -> None:
        """
        모델을 저장합니다.
        
        Args:
            save_path: 모델을 저장할 경로
        """
        # 모델 상태 저장
        model_dict = {
            "bart_state_dict": self.model.state_dict(),
            "model_config": self.model.config.to_dict(),
            "use_strategy": self.use_strategy,
            "use_emotion": self.use_emotion,
        }
        
        # 전략 임베딩 저장
        if self.use_strategy:
            model_dict["strategy_embeddings"] = self.strategy_embeddings.state_dict()
            model_dict["strategy_projection"] = self.strategy_projection.state_dict()
            
        # 감정 임베딩 저장
        if self.use_emotion:
            model_dict["emotion_embeddings"] = self.emotion_embeddings.state_dict()
            model_dict["emotion_projection"] = self.emotion_projection.state_dict()
            
        # 최종 투사 레이어 저장
        if hasattr(self, "final_projection"):
            model_dict["final_projection"] = self.final_projection.state_dict()
            
        torch.save(model_dict, save_path)
        
    @classmethod
    def load_model(cls, load_path: str, device: str = None) -> 'BartForESConv':
        """
        저장된 모델을 로드합니다.
        
        Args:
            load_path: 모델이 저장된 경로
            device: 모델 실행 장치
            
        Returns:
            로드된 BartForESConv 모델
        """
        # 장치 설정
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # 모델 상태 로드
        model_dict = torch.load(load_path, map_location=device)
        
        # 모델 구성 복원
        use_strategy = model_dict.get("use_strategy", True)
        use_emotion = model_dict.get("use_emotion", True)
        
        # BART 모델 초기화
        model = cls(
            model_name_or_path="facebook/bart-base",  # 기본 모델 사용
            use_strategy=use_strategy,
            use_emotion=use_emotion,
            device=device
        )
        
        # BART 모델 상태 복원
        model.model.load_state_dict(model_dict["bart_state_dict"])
        
        # 전략 임베딩 복원
        if use_strategy and "strategy_embeddings" in model_dict:
            model.strategy_embeddings.load_state_dict(model_dict["strategy_embeddings"])
            model.strategy_projection.load_state_dict(model_dict["strategy_projection"])
            
        # 감정 임베딩 복원
        if use_emotion and "emotion_embeddings" in model_dict:
            model.emotion_embeddings.load_state_dict(model_dict["emotion_embeddings"])
            model.emotion_projection.load_state_dict(model_dict["emotion_projection"])
            
        # 최종 투사 레이어 복원
        if "final_projection" in model_dict:
            model.final_projection.load_state_dict(model_dict["final_projection"])
            
        return model 