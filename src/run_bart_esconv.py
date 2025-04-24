#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESConv 데이터셋에 대한 BART 모델 학습 및 추론 실행 스크립트

이 스크립트는 다음 기능을 제공합니다:
1. 전략만 사용하는 모델 학습
2. 감정만 사용하는 모델 학습
3. 전략과 감정 모두 사용하는 모델 학습
4. 전략과 감정 모두 사용하지 않는 모델 학습
5. 학습된 모델을 사용한 샘플 대화 생성

실행 방법:
    python src/run_bart_esconv.py --mode [train/inference] --use_strategy [0/1] --use_emotion [0/1]
"""

import os
import sys
import argparse
import logging
import torch
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 모듈 임포트
from src.models.bart_esconv import BartForESConv
from src.data.prepare_esconv import prepare_dataloaders, ESConvProcessor
from src.trainer.bart_trainer import BartTrainer
from src.data.load_esconv import load_esconv_dataset, parse_example_data

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_bart_esconv")

# 시드 설정 함수
def set_seed(seed: int = 42):
    """시드를 설정하여 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(args):
    """
    모델을 학습합니다.
    
    Args:
        args: 명령행 인수
    """
    logger.info("Starting model training...")
    logger.info(f"Using strategy: {args.use_strategy}, Using emotion: {args.use_emotion}")
    
    # 출력 디렉토리 설정
    model_type = get_model_type(args.use_strategy, args.use_emotion)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로더 준비
    dataloaders = prepare_dataloaders(
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        include_situation=args.include_situation,
        include_emotion=args.include_emotion,
        use_strategy=args.use_strategy,
        use_emotion=args.use_emotion,
        sample_size=args.sample_size if args.sample_size > 0 else None,
        num_workers=args.num_workers
    )
    
    # 모델 생성
    model = BartForESConv(
        model_name_or_path=args.model_name,
        strategy_embedding_dim=args.strategy_embedding_dim,
        emotion_embedding_dim=args.emotion_embedding_dim,
        use_strategy=args.use_strategy,
        use_emotion=args.use_emotion,
        device=args.device
    )
    
    # 트레이너 초기화
    trainer = BartTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["validation"],
        tokenizer_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        output_dir=output_dir,
        device=args.device
    )
    
    # 학습 실행
    history = trainer.train(num_epochs=args.num_epochs)
    
    logger.info("Training completed!")

def run_inference(args):
    """
    학습된 모델을 사용하여 추론을 실행합니다.
    
    Args:
        args: 명령행 인수
    """
    logger.info("Starting model inference...")
    logger.info(f"Using strategy: {args.use_strategy}, Using emotion: {args.use_emotion}")
    
    if not args.model_path:
        logger.error("Model path not specified. Please provide --model_path.")
        return
    
    # 모델 로드
    try:
        model = BartForESConv.load_model(args.model_path, device=args.device)
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
    # 프로세서 초기화
    processor = ESConvProcessor(
        tokenizer_name=args.model_name,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
    
    # 토크나이저 설정
    tokenizer = model.tokenizer
    
    # 샘플 예제 추론
    if args.sample_inference:
        run_sample_inference(model, processor, tokenizer, args)
    
    # 대화형 모드
    if args.interactive:
        run_interactive_mode(model, processor, tokenizer, args)

def run_sample_inference(model, processor, tokenizer, args):
    """
    샘플 예제에 대한 추론을 실행합니다.
    
    Args:
        model: BART 모델
        processor: ESConv 프로세서
        tokenizer: 토크나이저
        args: 명령행 인수
    """
    logger.info("Running inference on sample examples...")
    
    # 데이터셋 로드
    dataset = load_esconv_dataset()
    if dataset is None:
        logger.error("Failed to load ESConv dataset")
        return
    
    # 검증 데이터셋에서 샘플 선택
    split = "validation"
    if split not in dataset:
        logger.error(f"{split} split not found in dataset")
        return
    
    # 샘플 수 설정
    num_samples = min(5, len(dataset[split]))
    
    for i in range(num_samples):
        logger.info(f"\n--- Sample {i+1} ---")
        
        # 예제 파싱
        example = dataset[split][i]
        result = parse_example_data(example)
        
        if not result["success"]:
            logger.error(f"Failed to parse example: {result['error']}")
            continue
        
        data = result["data"]
        dialog = data.get("dialog", [])
        emotion = data.get("emotion_type", "other")
        situation = data.get("situation", "")
        
        # 대화 턴 추출
        turns = processor.extract_dialog_turns(dialog)
        
        if not turns:
            logger.warning("No turns found in the dialog")
            continue
        
        # 첫 번째 턴 선택
        turn = turns[0]
        history = turn["history"]
        target_response = turn["response"]
        strategy = turn["strategy"]
        strategy_id = turn["strategy_id"]
        emotion_id = processor.EMOTION_MAP.get(emotion, 5)  # 기본값은 "other"
        
        # 입력 텍스트 준비
        input_text = processor.prepare_input_for_turn(
            history=history,
            emotion=emotion if args.include_emotion else None,
            situation=situation if args.include_situation else None
        )
        
        logger.info(f"Situation: {situation[:100]}..." if len(situation) > 100 else f"Situation: {situation}")
        logger.info(f"Emotion: {emotion} (ID: {emotion_id})")
        logger.info(f"Strategy: {strategy} (ID: {strategy_id})")
        logger.info(f"Input: {input_text[:200]}..." if len(input_text) > 200 else f"Input: {input_text}")
        logger.info(f"Target response: {target_response}")
        
        # 입력 토큰화
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True
        )
        
        # 전략 및 감정 ID 준비
        strategy_ids = torch.tensor([strategy_id]).to(args.device) if args.use_strategy else None
        emotion_ids = torch.tensor([emotion_id]).to(args.device) if args.use_emotion else None
        
        # 장치 설정
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        
        # 생성 실행
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                emotion_ids=emotion_ids,
                max_length=args.max_target_length,
                num_beams=4,
                early_stopping=True
            )
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.info(f"Generated response: {generated_text}")
        logger.info("---")

def run_interactive_mode(model, processor, tokenizer, args):
    """
    대화형 모드로 모델을 실행합니다.
    
    Args:
        model: BART 모델
        processor: ESConv 프로세서
        tokenizer: 토크나이저
        args: 명령행 인수
    """
    logger.info("\n--- Interactive Mode ---")
    logger.info("Type 'exit' to quit the interactive mode.")
    
    # 상황 설정
    print("\nEnter the situation (context):")
    situation = input("> ")
    
    # 감정 설정
    print("\nSelect emotion from: anger, anxiety, depression, sadness, fear, other")
    emotion = input("Emotion > ").strip().lower()
    if emotion not in processor.EMOTION_MAP:
        emotion = "other"
    emotion_id = processor.EMOTION_MAP.get(emotion, 5)
    
    # 대화 기록 초기화
    history = []
    
    while True:
        # 사용자 입력
        print("\nSeeker: (Type 'exit' to quit)")
        user_input = input("> ")
        
        if user_input.lower() == "exit":
            print("Exiting interactive mode...")
            break
        
        # 대화 기록 업데이트
        history.append({
            "speaker": "usr",
            "text": user_input
        })
        
        # 전략 선택
        if args.use_strategy:
            print("\nSelect strategy from:")
            for strategy, idx in processor.STRATEGY_MAP.items():
                print(f"{idx}: {strategy}")
            
            strategy_input = input("Strategy ID > ")
            try:
                strategy_id = int(strategy_input)
                if strategy_id not in processor.id_to_strategy:
                    strategy_id = 7  # Others
            except ValueError:
                strategy_id = 7  # Others
                
            strategy = processor.id_to_strategy[strategy_id]
            print(f"Using strategy: {strategy}")
        else:
            strategy_id = None
        
        # 입력 텍스트 준비
        input_text = processor.prepare_input_for_turn(
            history=history,
            emotion=emotion if args.include_emotion else None,
            situation=situation if args.include_situation else None
        )
        
        # 입력 토큰화
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True
        )
        
        # 전략 및 감정 ID 준비
        strategy_ids = torch.tensor([strategy_id]).to(args.device) if args.use_strategy else None
        emotion_ids = torch.tensor([emotion_id]).to(args.device) if args.use_emotion else None
        
        # 장치 설정
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        
        # 생성 실행
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                emotion_ids=emotion_ids,
                max_length=args.max_target_length,
                num_beams=4,
                early_stopping=True
            )
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"\nSupporter: {generated_text}")
        
        # 대화 기록 업데이트
        history.append({
            "speaker": "sys",
            "text": generated_text
        })

def get_model_type(use_strategy: bool, use_emotion: bool) -> str:
    """
    모델 타입을 결정합니다.
    
    Args:
        use_strategy: 전략 사용 여부
        use_emotion: 감정 사용 여부
        
    Returns:
        모델 타입 문자열
    """
    if use_strategy and use_emotion:
        return "strategy_emotion"
    elif use_strategy:
        return "strategy_only"
    elif use_emotion:
        return "emotion_only"
    else:
        return "baseline"

def parse_args():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="ESConv 데이터셋에 대한 BART 모델 학습 및 추론"
    )
    
    # 모드 설정
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="실행 모드 (train 또는 inference)"
    )
    
    # 모델 설정
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/bart-base",
        help="BART 모델 이름 또는 경로"
    )
    parser.add_argument(
        "--use_strategy",
        type=int,
        default=1,
        choices=[0, 1],
        help="전략 벡터 사용 여부 (0: 사용 안 함, 1: 사용)"
    )
    parser.add_argument(
        "--use_emotion",
        type=int,
        default=1,
        choices=[0, 1],
        help="감정 벡터 사용 여부 (0: 사용 안 함, 1: 사용)"
    )
    parser.add_argument(
        "--strategy_embedding_dim",
        type=int,
        default=64,
        help="전략 임베딩 차원"
    )
    parser.add_argument(
        "--emotion_embedding_dim",
        type=int,
        default=32,
        help="감정 임베딩 차원"
    )
    
    # 데이터 설정
    parser.add_argument(
        "--include_situation",
        type=int,
        default=1,
        choices=[0, 1],
        help="대화 상황 포함 여부 (0: 포함 안 함, 1: 포함)"
    )
    parser.add_argument(
        "--include_emotion",
        type=int,
        default=1,
        choices=[0, 1],
        help="감정 정보 포함 여부 (0: 포함 안 함, 1: 포함)"
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="최대 소스 텍스트 길이"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="최대 타겟 텍스트 길이"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="샘플 크기 (0: 전체 데이터셋 사용)"
    )
    
    # 학습 설정
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="배치 크기"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="학습 에폭 수"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="학습률"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="가중치 감쇠"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="그래디언트 누적 단계"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="최대 그래디언트 노름"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="워밍업 비율"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="데이터 로딩을 위한 워커 수"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드"
    )
    
    # 추론 설정
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="추론 시 사용할 모델 경로"
    )
    parser.add_argument(
        "--sample_inference",
        action="store_true",
        help="샘플 예제에 대한 추론 실행"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="대화형 모드 실행"
    )
    
    # 장치 설정
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="학습/추론 장치 (cuda 또는 cpu)"
    )
    
    # 불리언 인수를 간단히 처리
    args = parser.parse_args()
    args.use_strategy = bool(args.use_strategy)
    args.use_emotion = bool(args.use_emotion)
    args.include_situation = bool(args.include_situation)
    args.include_emotion = bool(args.include_emotion)
    
    # 장치 자동 설정
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 모드에 따른 실행
    if args.mode == "train":
        train_model(args)
    elif args.mode == "inference":
        run_inference(args)
    else:
        logger.error(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main() 