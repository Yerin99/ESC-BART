from datasets import load_dataset
import pandas as pd
import json

def load_esconv_dataset():
    """
    Load the ESConv dataset from Hugging Face.

    Returns:
        dataset: The loaded ESConv dataset
    """
    # ESConv 데이터셋 로드
    # 실제 데이터셋 이름은 'thu-coai/esconv'입니다
    dataset_name = "thu-coai/esconv"
    try:
        dataset = load_dataset(dataset_name)
        print(f"Dataset '{dataset_name}' successfully loaded.")
        print(f"Available splits: {list(dataset.keys())}")
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        print("Please check the dataset name or your internet connection.")
        # 데이터셋 로드 실패 시 None 반환 또는 예외 재발생 등의 처리 가능
        return None

    return dataset

def parse_example_data(example):
    """
    JSON 데이터를 파싱하고 필요한 필드를 추출하는 유틸리티 함수.
    
    Args:
        example: 데이터셋의 예제(샘플)
        
    Returns:
        dict: 파싱된 데이터와 상태 정보를 포함한 딕셔너리
            - success (bool): 파싱 성공 여부
            - data (dict): 파싱된 데이터 (성공 시)
            - error (str): 오류 메시지 (실패 시)
    """
    result = {
        'success': False,
        'data': None,
        'error': None
    }
    
    if 'text' not in example:
        result['error'] = "Example does not have 'text' field"
        return result
    
    try:
        json_data = json.loads(example['text'])
        result['success'] = True
        result['data'] = json_data
        return result
    except json.JSONDecodeError as e:
        result['error'] = f"JSON decode error: {str(e)}"
        return result

def get_example_fields(example):
    """
    데이터셋 예제에서 주요 필드를 추출합니다.
    
    Args:
        example: 데이터셋의 예제(샘플)
        
    Returns:
        dict: 주요 필드와 값을 포함한 딕셔너리.
            파싱 실패 시 빈 딕셔너리 반환.
    """
    result = parse_example_data(example)
    if not result['success']:
        return {}
    
    json_data = result['data']
    
    fields = {}
    # 주요 필드 추출
    for field in ['dialog', 'situation', 'seeker_emotion', 'strategies', 'seeker_id']:
        if field in json_data:
            fields[field] = json_data[field]
    
    return fields

def explore_dataset(dataset, split="train", num_examples=3):
    """
    Explore the loaded dataset.

    Args:
        dataset: The loaded dataset
        split: The dataset split to explore (default: 'train')
        num_examples: Number of examples to display (default: 3)
    """
    if split not in dataset:
        print(f"Warning: {split} split is not in the dataset.")
        return
    
    # 데이터셋 기본 정보 출력
    print(f"\nSize of {split} split: {len(dataset[split])}")
    print(f"Dataset columns: {dataset[split].column_names}")
    
    # 샘플 데이터 확인
    print(f"\nFirst {num_examples} examples from {split} split:")
    for i in range(min(num_examples, len(dataset[split]))):
        print(f"\nExample {i+1}:")
        example = dataset[split][i]
        
        # 기본 컬럼 정보 출력
        for key, value in example.items():
            if key != "text":  # text는 별도로 처리
                print(f"{key}: {value}")
        
        # text 필드 파싱 및 출력
        parsed_data = parse_example_data(example)
        if parsed_data['success']:
            json_data = parsed_data['data']
            print(f"text: [JSON data with keys: {list(json_data.keys())}]")
            
            # 대화와 상황 정보만 간략히 표시
            if "dialog" in json_data:
                dialog_len = len(json_data["dialog"])
                print(f"  dialog: [Contains {dialog_len} utterances]")
                
                # 처음 2개와 마지막 2개 대화만 표시
                if dialog_len > 4:
                    first_two = json_data["dialog"][:2]
                    last_two = json_data["dialog"][-2:]
                    
                    for j, utterance in enumerate(first_two):
                        print(f"    Utterance {j}: {utterance}")
                    print(f"    ...")
                    for j, utterance in enumerate(last_two):
                        print(f"    Utterance {dialog_len-2+j}: {utterance}")
                else:
                    for j, utterance in enumerate(json_data["dialog"]):
                        print(f"    Utterance {j}: {utterance}")
                        
            if "situation" in json_data:
                print(f"  situation: {json_data['situation']}")
        else:
            print(f"text: [Failed to parse JSON: {parsed_data['error']}]")

# 메인 실행 블록 추가
if __name__ == "__main__":
    print("=== ESConv 데이터셋 로드 및 탐색 ===")
    # 데이터셋 로드
    dataset = load_esconv_dataset()
    
    if dataset:
        print("\n=== 데이터셋 로드 성공 ===")
        # 각 분할(split) 정보 출력
        for split in dataset.keys():
            print(f"\n--- {split.upper()} 분할 정보 ---")
            print(f"크기: {len(dataset[split])} 예제")
            print(f"컬럼: {dataset[split].column_names}")
        
        # train 분할의 샘플 데이터 탐색
        print("\n=== Train 분할 샘플 데이터 탐색 ===")
        explore_dataset(dataset, split="train", num_examples=1)
    else:
        print("\n=== 데이터셋 로드 실패 ===")
        print("데이터셋 로드 중 오류가 발생했습니다.") 