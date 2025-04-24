# ESC-BART: 감정 지원 대화를 위한 BART 기반 응답 생성

ESC-BART는 감정 지원 대화(Emotional Support Conversation)를 위한 BART 기반 응답 생성 시스템입니다. ESConv 데이터셋을 활용하여 전략과 감정 정보를 벡터로 주입하여 응답을 생성합니다.

## 주요 기능

- ESConv 데이터셋을 사용한 대화 응답 생성
- 전략(Strategy)과 감정(Emotion) 정보를 벡터로 모델에 주입
- 4가지 실험 설정 지원:
  1. 전략만 사용
  2. 감정만 사용
  3. 전략과 감정 모두 사용
  4. 전략과 감정 모두 사용하지 않음 (기본 BART)
- 대화형 추론 모드

## 프로젝트 구조

```
ESC-BART/
├── src/                        # 소스 코드
│   ├── data/                   # 데이터 처리 관련 코드
│   │   ├── load_esconv.py      # ESConv 데이터셋 로드 유틸리티
│   │   ├── prepare_esconv.py   # ESConv 데이터셋 전처리 유틸리티
│   │   └── explore_esconv.py   # ESConv 데이터셋 탐색 도구
│   ├── models/                 # 모델 정의
│   │   └── bart_esconv.py      # BART 모델 확장 클래스
│   ├── trainer/                # 학습 관련 코드
│   │   └── bart_trainer.py     # BART 모델 트레이너
│   └── run_bart_esconv.py      # 메인 실행 스크립트
├── outputs/                    # 모델 출력 및 체크포인트
└── images/                     # 데이터셋 분석 그래프 및 시각화
```

## 설치 및 요구사항

다음 라이브러리가 필요합니다:

```bash
pip install torch transformers datasets nltk rouge tqdm
```

## 사용 방법

### 데이터셋 탐색

ESConv 데이터셋의 구조를 탐색합니다:

```bash
python src/data/explore_esconv.py
```

### 모델 학습

다양한 설정으로 모델을 학습할 수 있습니다:

#### 전략과 감정 모두 사용

```bash
python src/run_bart_esconv.py --mode train --use_strategy 1 --use_emotion 1
```

#### 전략만 사용

```bash
python src/run_bart_esconv.py --mode train --use_strategy 1 --use_emotion 0
```

#### 감정만 사용

```bash
python src/run_bart_esconv.py --mode train --use_strategy 0 --use_emotion 1
```

#### 기본 BART (전략과 감정 모두 사용하지 않음)

```bash
python src/run_bart_esconv.py --mode train --use_strategy 0 --use_emotion 0
```

### 모델 추론

학습된 모델을 사용하여 추론을 실행합니다:

#### 샘플 예제에 대한 추론

```bash
python src/run_bart_esconv.py --mode inference --model_path ./outputs/best_model/model.pt --sample_inference
```

#### 대화형 모드

```bash
python src/run_bart_esconv.py --mode inference --model_path ./outputs/best_model/model.pt --interactive
```

## 추가 설정 옵션

모델 학습 및 추론에 사용할 수 있는 추가 옵션:

- `--batch_size`: 배치 크기 (기본값: 8)
- `--num_epochs`: 학습 에폭 수 (기본값: 3)
- `--learning_rate`: 학습률 (기본값: 5e-5)
- `--sample_size`: 샘플 크기 제한 (테스트용, 기본값: 0, 전체 데이터셋 사용)
- `--max_source_length`: 최대 소스 텍스트 길이 (기본값: 512)
- `--max_target_length`: 최대 타겟 텍스트 길이 (기본값: 128)
- `--strategy_embedding_dim`: 전략 임베딩 차원 (기본값: 64)
- `--emotion_embedding_dim`: 감정 임베딩 차원 (기본값: 32)

자세한 옵션은 다음 명령으로 확인할 수 있습니다:

```bash
python src/run_bart_esconv.py --help
``` 
