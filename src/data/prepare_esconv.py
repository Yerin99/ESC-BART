#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESConv 데이터셋 전처리 유틸리티

이 모듈은 ESConv 데이터셋을 BART 모델 학습에 적합한 형식으로 변환하는 함수들을 제공합니다.
대화 기록, 전략, 감정 정보 등을 추출하고 처리합니다.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import BartTokenizer

from src.data.load_esconv import load_esconv_dataset, parse_example_data

class ESConvProcessor:
    """ESConv 데이터셋 처리 클래스"""
    
    # 전략 ID 매핑 (ESConv 데이터셋 기준)
    STRATEGY_MAP = {
        "Question": 0,
        "Restatement or Paraphrasing": 1,
        "Reflection of Feelings": 2,
        "Self-disclosure": 3,
        "Affirmation and Reassurance": 4,
        "Providing Suggestions": 5,
        "Information": 6,
        "Others": 7
    }
    
    # 감정 ID 매핑 (ESConv 데이터셋 기준)
    EMOTION_MAP = {
        "anger": 0,
        "anxiety": 1,
        "depression": 2,
        "sadness": 3,
        "fear": 4,
        "other": 5
    }
    
    def __init__(self, tokenizer_name: str = "facebook/bart-base", max_source_length: int = 512, max_target_length: int = 128):
        """
        Args:
            tokenizer_name: 토크나이저 이름
            max_source_length: 최대 소스 텍스트 길이
            max_target_length: 최대 타겟 텍스트 길이
        """
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # 역방향 매핑 생성
        self.id_to_strategy = {v: k for k, v in self.STRATEGY_MAP.items()}
        self.id_to_emotion = {v: k for k, v in self.EMOTION_MAP.items()}
    
    def extract_dialog_turns(self, dialog_data: List[Dict]) -> List[Dict]:
        """
        대화에서 턴별 정보를 추출합니다.
        
        Args:
            dialog_data: 대화 데이터 목록
            
        Returns:
            턴별 정보가 포함된 목록
        """
        turns = []
        history = []
        
        for i, utterance in enumerate(dialog_data):
            speaker = utterance.get("speaker", "")
            text = utterance.get("text", "")
            
            # 시스템 응답인 경우 (대상)
            if speaker == "sys":
                strategy = utterance.get("strategy", "Others")
                strategy_id = self.STRATEGY_MAP.get(strategy, 7)  # 기본값은 "Others"
                
                # 대화 기록이 있는 경우에만 추가
                if history:
                    turns.append({
                        "history": history.copy(),  # 이전까지의 대화 기록
                        "response": text,           # 시스템 응답
                        "strategy": strategy,       # 사용된 전략
                        "strategy_id": strategy_id  # 전략 ID
                    })
            
            # 대화 기록 갱신
            history.append({
                "speaker": speaker,
                "text": text
            })
            
        return turns
    
    def prepare_input_for_turn(
        self, 
        history: List[Dict], 
        emotion: Optional[str] = None,
        emotion_id: Optional[int] = None,
        situation: Optional[str] = None
    ) -> str:
        """
        대화 턴에 대한 입력 텍스트를 준비합니다.
        
        Args:
            history: 대화 기록
            emotion: 대화에서의 감정
            emotion_id: 감정 ID
            situation: 대화 상황 설명
            
        Returns:
            포맷팅된 입력 텍스트
        """
        # 상황 정보가 있는 경우 먼저 추가
        input_text = ""
        if situation:
            input_text += f"Situation: {situation}\n\n"
        
        # 감정 정보가 있는 경우 추가
        if emotion:
            input_text += f"Emotion: {emotion}\n\n"
        
        # 대화 기록 추가
        input_text += "Dialog:\n"
        for turn in history:
            speaker_name = "Seeker" if turn["speaker"] == "usr" else "Supporter"
            input_text += f"{speaker_name}: {turn['text']}\n"
        
        return input_text.strip()
    
    def prepare_dataset_features(
        self, 
        examples: List[Dict], 
        include_situation: bool = True,
        include_emotion: bool = True,
        is_eval: bool = False
    ) -> List[Dict]:
        """
        데이터셋 특성을 준비합니다.
        
        Args:
            examples: 예제 목록
            include_situation: 상황 정보 포함 여부
            include_emotion: 감정 정보 포함 여부
            is_eval: 평가 모드 여부
            
        Returns:
            모델 입력용 특성을 포함한 목록
        """
        features = []
        
        for example in examples:
            # JSON 파싱
            result = parse_example_data(example)
            if not result['success']:
                continue
                
            data = result['data']
            dialog = data.get('dialog', [])
            
            # 감정 정보 추출
            emotion = data.get('emotion_type', 'other')
            emotion_id = self.EMOTION_MAP.get(emotion, 5)  # 기본값은 "other"
            
            # 상황 정보 추출
            situation = data.get('situation', '') if include_situation else None
            
            # 대화 턴 추출
            turns = self.extract_dialog_turns(dialog)
            
            # 각 턴에 대한 특성 생성
            for turn in turns:
                history = turn['history']
                response = turn['response']
                strategy = turn['strategy']
                strategy_id = turn['strategy_id']
                
                # 입력 텍스트 준비
                input_text = self.prepare_input_for_turn(
                    history=history,
                    emotion=emotion if include_emotion else None,
                    situation=situation
                )
                
                # 특성 추가
                feature = {
                    "input_text": input_text,
                    "target_text": response,
                    "strategy": strategy,
                    "strategy_id": strategy_id,
                    "emotion": emotion,
                    "emotion_id": emotion_id
                }
                
                # 평가 모드가 아닌 경우만 추가 (훈련 데이터셋인 경우)
                if not is_eval or (is_eval and len(history) > 0):
                    features.append(feature)
        
        return features
    
    def tokenize_features(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        특성을 토큰화합니다.
        
        Args:
            features: 특성 목록
            
        Returns:
            토큰화된 특성 딕셔너리
        """
        # 입력 텍스트와 타겟 텍스트 추출
        input_texts = [f["input_text"] for f in features]
        target_texts = [f["target_text"] for f in features]
        strategy_ids = [f["strategy_id"] for f in features]
        emotion_ids = [f["emotion_id"] for f in features]
        
        # 소스 텍스트 토큰화
        source_encoding = self.tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt"
        )
        
        # 타겟 텍스트 토큰화
        target_encoding = self.tokenizer(
            target_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        )
        
        # 레이블 설정
        labels = target_encoding.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # 패딩 토큰은 무시
        
        # 결과 반환
        return {
            "input_ids": source_encoding.input_ids,
            "attention_mask": source_encoding.attention_mask,
            "labels": labels,
            "strategy_ids": torch.tensor(strategy_ids),
            "emotion_ids": torch.tensor(emotion_ids),
            "target_ids": target_encoding.input_ids,
            "target_attention_mask": target_encoding.attention_mask
        }

class ESConvDataset(Dataset):
    """ESConv 데이터셋 클래스"""
    
    def __init__(
        self, 
        tokenizer_name: str = "facebook/bart-base",
        split: str = "train",
        max_source_length: int = 512,
        max_target_length: int = 128,
        include_situation: bool = True,
        include_emotion: bool = True,
        use_strategy: bool = True,
        use_emotion: bool = True,
        sample_size: Optional[int] = None  # 샘플 크기 제한 (테스트용)
    ):
        """
        Args:
            tokenizer_name: 토크나이저 이름
            split: 데이터셋 분할 ("train", "validation", "test")
            max_source_length: 최대 소스 텍스트 길이
            max_target_length: 최대 타겟 텍스트 길이
            include_situation: 상황 정보 포함 여부
            include_emotion: 감정 정보 포함 여부
            use_strategy: 전략 정보 사용 여부
            use_emotion: 감정 정보 사용 여부
            sample_size: 샘플 크기 제한 (테스트용)
        """
        self.processor = ESConvProcessor(
            tokenizer_name=tokenizer_name,
            max_source_length=max_source_length,
            max_target_length=max_target_length
        )
        self.use_strategy = use_strategy
        self.use_emotion = use_emotion
        
        # 데이터셋 로드
        print(f"Loading ESConv {split} dataset...")
        dataset = load_esconv_dataset()
        
        if dataset is None or split not in dataset:
            raise ValueError(f"Failed to load ESConv {split} dataset")
        
        # 샘플 크기 제한 (테스트용)
        if sample_size is not None:
            examples = dataset[split][:sample_size]
        else:
            examples = dataset[split]
        
        # 특성 준비
        print(f"Preparing features for {split} dataset...")
        is_eval = split in ["validation", "test"]
        self.features = self.processor.prepare_dataset_features(
            examples=examples,
            include_situation=include_situation,
            include_emotion=include_emotion,
            is_eval=is_eval
        )
        
        # 토큰화
        print(f"Tokenizing {len(self.features)} features...")
        self.tokenized_features = self.processor.tokenize_features(self.features)
        
        print(f"Dataset prepared with {len(self.features)} examples")
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """인덱스에 해당하는 아이템 반환"""
        item = {
            "input_ids": self.tokenized_features["input_ids"][idx],
            "attention_mask": self.tokenized_features["attention_mask"][idx],
            "labels": self.tokenized_features["labels"][idx],
            "target_ids": self.tokenized_features["target_ids"][idx],
            "target_attention_mask": self.tokenized_features["target_attention_mask"][idx],
        }
        
        # 전략 정보 추가
        if self.use_strategy:
            item["strategy_ids"] = self.tokenized_features["strategy_ids"][idx]
            
        # 감정 정보 추가
        if self.use_emotion:
            item["emotion_ids"] = self.tokenized_features["emotion_ids"][idx]
            
        return item
    
    def get_original_data(self, idx: int) -> Dict:
        """원본 데이터 반환"""
        return self.features[idx]


def prepare_dataloaders(
    tokenizer_name: str = "facebook/bart-base",
    batch_size: int = 8,
    max_source_length: int = 512,
    max_target_length: int = 128,
    include_situation: bool = True,
    include_emotion: bool = True,
    use_strategy: bool = True,
    use_emotion: bool = True,
    sample_size: Optional[int] = None,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    데이터 로더를 준비합니다.
    
    Args:
        tokenizer_name: 토크나이저 이름
        batch_size: 배치 크기
        max_source_length: 최대 소스 텍스트 길이
        max_target_length: 최대 타겟 텍스트 길이
        include_situation: 상황 정보 포함 여부
        include_emotion: 감정 정보 포함 여부
        use_strategy: 전략 정보 사용 여부
        use_emotion: 감정 정보 사용 여부
        sample_size: 샘플 크기 제한 (테스트용)
        num_workers: 데이터 로딩을 위한 워커 수
    
    Returns:
        데이터 로더 딕셔너리
    """
    # 데이터셋 준비
    train_dataset = ESConvDataset(
        tokenizer_name=tokenizer_name,
        split="train",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        include_situation=include_situation,
        include_emotion=include_emotion,
        use_strategy=use_strategy,
        use_emotion=use_emotion,
        sample_size=sample_size
    )
    
    val_dataset = ESConvDataset(
        tokenizer_name=tokenizer_name,
        split="validation",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        include_situation=include_situation,
        include_emotion=include_emotion,
        use_strategy=use_strategy,
        use_emotion=use_emotion,
        sample_size=sample_size
    )
    
    # 데이터 로더 준비
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        "train": train_loader,
        "validation": val_loader
    }

# 테스트 코드
if __name__ == "__main__":
    # 프로세서 초기화
    processor = ESConvProcessor()
    
    # 데이터셋 로드
    print("Loading ESConv dataset...")
    dataset = load_esconv_dataset()
    
    if dataset is None:
        print("Failed to load ESConv dataset")
        exit(1)
    
    # 샘플 예제
    print("\nProcessing a sample example...")
    example = dataset["train"][0]
    result = parse_example_data(example)
    
    if not result['success']:
        print(f"Failed to parse example: {result['error']}")
        exit(1)
    
    data = result['data']
    dialog = data.get('dialog', [])
    emotion = data.get('emotion_type', 'other')
    situation = data.get('situation', '')
    
    print(f"Emotion: {emotion}")
    print(f"Situation: {situation[:100]}...")
    
    # 대화 턴 추출
    turns = processor.extract_dialog_turns(dialog)
    print(f"\nExtracted {len(turns)} turns from the dialog")
    
    if turns:
        # 첫 번째 턴의 예시 확인
        turn = turns[0]
        print("\nSample turn:")
        print(f"Strategy: {turn['strategy']} (ID: {turn['strategy_id']})")
        print(f"Response: {turn['response']}")
        
        # 입력 텍스트 생성
        input_text = processor.prepare_input_for_turn(
            history=turn['history'],
            emotion=emotion,
            situation=situation
        )
        
        print("\nSample input text:")
        print(input_text[:500] + "..." if len(input_text) > 500 else input_text)
    
    # 데이터셋 샘플 생성
    print("\nCreating a sample dataset...")
    sample_dataset = ESConvDataset(sample_size=10)
    
    print(f"Sample dataset size: {len(sample_dataset)}")
    
    # 첫 번째 아이템 확인
    first_item = sample_dataset[0]
    first_data = sample_dataset.get_original_data(0)
    
    print("\nFirst item in the dataset:")
    print(f"Input text: {first_data['input_text'][:100]}...")
    print(f"Target text: {first_data['target_text']}")
    print(f"Strategy: {first_data['strategy']} (ID: {first_data['strategy_id']})")
    print(f"Emotion: {first_data['emotion']} (ID: {first_data['emotion_id']})")
    
    print("\nTokenized input shape:", first_item["input_ids"].shape)
    print("Tokenized labels shape:", first_item["labels"].shape)
    
    print("\nSample dataset created successfully!") 