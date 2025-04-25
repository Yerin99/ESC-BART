import sys
import os
import pytest

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data.load_esconv import load_esconv_dataset, parse_example_data, get_example_fields

# ===== 공용 Fixtures =====

@pytest.fixture(scope="module")
def dataset():
    """
    ESConv 데이터셋을 모듈 레벨에서 한 번만 로드하는 fixture
    
    이 fixture는 모든 테스트에서 공유됨으로써 데이터 로드 시간을 크게 줄입니다.
    """
    print("\n[FIXTURE] 데이터셋 로드 중... (모듈당 한 번만 실행)")
    ds = load_esconv_dataset()
    print(f"[FIXTURE] 데이터셋 로드 완료: {list(ds.keys()) if ds else 'FAILED'}")
    return ds

@pytest.fixture(scope="module")
def train_example(dataset):
    """Train 분할에서 첫 번째 예제를 제공하는 fixture"""
    if dataset and 'train' in dataset and len(dataset['train']) > 0:
        return dataset['train'][0]
    return None

# ===== 데이터셋 로드 테스트 =====

def test_dataset_was_loaded_successfully(dataset):
    """✅ LOAD-1: 데이터셋이 성공적으로 로드되었는지 확인합니다."""
    assert dataset is not None, "데이터셋 로드 실패!"

def test_train_split_exists(dataset):
    """✅ LOAD-2: 'train' 분할이 존재하는지 확인합니다."""
    assert 'train' in dataset, "train 분할이 존재하지 않습니다!"

def test_validation_split_exists(dataset):
    """✅ LOAD-3: 'validation' 분할이 존재하는지 확인합니다."""
    assert 'validation' in dataset, "validation 분할이 존재하지 않습니다!"

def test_test_split_exists(dataset):
    """✅ LOAD-4: 'test' 분할이 존재하는지 확인합니다."""
    assert 'test' in dataset, "test 분할이 존재하지 않습니다!"

def test_splits_are_not_empty(dataset):
    """✅ LOAD-5: 각 분할이 비어있지 않은지 확인합니다."""
    for split_name in dataset:
        assert len(dataset[split_name]) > 0, f"{split_name} 분할이 비어 있습니다!"
        print(f"[정보] {split_name} 분할에 {len(dataset[split_name])} 개의 항목이 있습니다.")

# ===== 데이터 구조 및 파싱 테스트 =====

def test_train_example_is_available(train_example):
    """✅ STRUCT-1: train 예제가 사용 가능한지 확인합니다."""
    assert train_example is not None, "train 분할에서 예제를 가져올 수 없습니다!"

def test_text_field_exists(train_example):
    """✅ STRUCT-2: 예제에 'text' 필드가 존재하는지 확인합니다."""
    assert 'text' in train_example, "예제에 'text' 필드가 없습니다!"
    print(f"[정보] text 필드 길이: {len(train_example['text'])} 문자")

def test_json_parsing_works(train_example):
    """✅ PARSE-1: 'text' 필드의 JSON 파싱이 제대로 작동하는지 확인합니다."""
    parsed_result = parse_example_data(train_example)
    
    # 파싱 성공 여부 확인
    assert parsed_result['success'], f"JSON 파싱 실패: {parsed_result.get('error')}"
    
    # 파싱된 데이터가 존재하는지 확인
    assert parsed_result['data'] is not None, "파싱된 데이터가 None입니다."
    
    # 파싱된 데이터의 키 출력 (정보 제공)
    if parsed_result['success']:
        print(f"[정보] 파싱된 JSON의 키: {list(parsed_result['data'].keys())}")

def test_field_extraction_works(train_example):
    """✅ PARSE-2: 주요 필드 추출이 제대로 작동하는지 확인합니다."""
    # 필드 추출
    fields = get_example_fields(train_example)
    
    # 추출된 필드가 있는지 확인
    assert len(fields) > 0, "예제에서 필드를 추출할 수 없습니다!"
    
    # 예상 필드 중 하나 이상이 존재하는지 확인
    expected_fields = ['dialog', 'situation', 'seeker_emotion', 'strategies', 'seeker_id']
    found_fields = [field for field in expected_fields if field in fields]
    
    assert len(found_fields) > 0, f"예상 필드 중 어느 것도 찾을 수 없습니다: {expected_fields}"
    print(f"[정보] 찾은 필드들: {found_fields}")
    
    # 추가적인 필드 정보 제공
    for field in found_fields:
        field_type = type(fields[field]).__name__
        if field == 'dialog' and isinstance(fields[field], list):
            print(f"[정보] 'dialog' 필드에는 {len(fields[field])} 개의 대화가 있습니다.")
        elif field == 'situation':
            print(f"[정보] 'situation' 필드: {fields[field][:50]}{'...' if len(fields[field]) > 50 else ''}") 