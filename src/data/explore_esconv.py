#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESConv 데이터셋 구조 탐색기

이 스크립트는 ESConv 데이터셋의 구조와 내용을 상세하게 분석하고 출력합니다.
데이터셋의 구조를 이해하기 위한 탐색 도구로 사용할 수 있습니다.

실행 방법:
    python src/data/explore_esconv.py
"""

import sys
import os
import json
from collections import Counter
from typing import Dict, List, Any, Optional
import statistics
from tabulate import tabulate  # pip install tabulate
import matplotlib.pyplot as plt  # pip install matplotlib
import numpy as np

# 그래프 생성을 위한 경로 설정
IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../images'))
os.makedirs(IMAGES_DIR, exist_ok=True)

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.load_esconv import load_esconv_dataset, parse_example_data, get_example_fields

# ANSI 색상 코드 - 출력을 더 읽기 쉽게 만듭니다
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str) -> None:
    """굵은 헤더 텍스트를 출력합니다."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD} {text} {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")

def print_subheader(text: str) -> None:
    """서브 헤더를 출력합니다."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'-' * 50}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD} {text} {Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'-' * 50}{Colors.ENDC}")

def print_info(text: str) -> None:
    """정보 텍스트를 출력합니다."""
    print(f"{Colors.BLUE}ℹ️ {text}{Colors.ENDC}")

def print_success(text: str) -> None:
    """성공 텍스트를 출력합니다."""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str) -> None:
    """경고 텍스트를 출력합니다."""
    print(f"{Colors.YELLOW}⚠️ {text}{Colors.ENDC}")

def print_error(text: str) -> None:
    """오류 텍스트를 출력합니다."""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def print_json_sample(data: Dict, max_length: int = 100) -> None:
    """JSON 데이터 샘플을 읽기 쉽게 출력합니다."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    if len(json_str) > max_length:
        # 너무 긴 경우 잘라서 표시
        lines = json_str.split('\n')
        if len(lines) > 10:
            # 앞부분 5줄, 뒷부분 5줄만 표시
            shown_lines = lines[:5] + ['...'] + lines[-5:]
            json_str = '\n'.join(shown_lines)
    print(json_str)

def analyze_dataset_structure():
    """데이터셋의 기본 구조를 분석하고 출력합니다."""
    print_header("ESConv 데이터셋 구조 분석")
    
    print_info("데이터셋 로드 중...")
    dataset = load_esconv_dataset()
    
    if dataset is None:
        print_error("데이터셋을 로드할 수 없습니다.")
        return
    
    print_success("데이터셋 로드 완료!")
    
    # 기본 정보 출력
    print_subheader("데이터셋 기본 정보")
    splits = list(dataset.keys())
    split_sizes = {split: len(dataset[split]) for split in splits}
    
    # 표 형태로 출력
    table_data = []
    for split, size in split_sizes.items():
        columns = dataset[split].column_names
        table_data.append([split, size, ', '.join(columns)])
    
    headers = ["분할(Split)", "크기", "컬럼"]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    
    # 각 분할별 확인
    for split in splits:
        print_subheader(f"{split.upper()} 분할 데이터터 분석")
        
        # 첫 번째 예제
        example = dataset[split][0]
        print_info(f"첫 번째 예제의 구조:")
        for key, value in example.items():
            print(f"  - {key}: {type(value).__name__} (길이: {len(str(value))})")
        
        # JSON 파싱 확인
        result = parse_example_data(example)
        if not result['success']:
            print_error(f"JSON 파싱 실패: {result['error']}")
            continue
        
        # JSON 구조 분석
        json_data = result['data']
        print_info(f"파싱된 JSON의 키: {list(json_data.keys())}")
        
        # 필드 추출 및 분석
        fields = get_example_fields(example)
        if not fields:
            print_warning("추출된 필드가 없습니다.")
            continue
            
        print_info(f"추출된 필드: {list(fields.keys())}")
        
        # 특정 필드 상세 분석 (dialog, situation 등)
        if 'dialog' in fields:
            analyze_dialog_field(fields['dialog'])
        
        if 'situation' in fields:
            print_info(f"상황 내용: {fields['situation'][:100]}...")
        
        # 추가 필드가 있다면 내용 확인
        for field in ['seeker_emotion', 'strategies', 'seeker_id']:
            if field in fields:
                print_info(f"{field}: {fields[field]}")
        
def analyze_dialog_field(dialog_data: List[Dict]) -> None:
    """대화 필드를 상세히 분석합니다."""
    print_subheader("대화 구조 분석")
    
    print_info(f"대화 길이: {len(dialog_data)} 발화(utterances)")
    
    # 화자별 발화 수 계산
    speakers = [utterance.get('speaker', 'unknown') for utterance in dialog_data]
    speaker_counts = Counter(speakers)
    
    print_info("화자별 발화 수:")
    for speaker, count in speaker_counts.items():
        print(f"  - {speaker}: {count}회")
    
    # 첫 3개와 마지막 3개 발화 내용 표시
    print_info("대화 샘플:")
    
    table_data = []
    
    # 첫 3개 발화
    for i, utterance in enumerate(dialog_data[:3]):
        speaker = utterance.get('speaker', 'unknown')
        text = utterance.get('text', '')
        strategy = utterance.get('strategy', '')
        table_data.append([i+1, speaker, text[:50] + ('...' if len(text) > 50 else ''), strategy])
    
    # 중간 생략 표시
    if len(dialog_data) > 6:
        table_data.append(['...', '...', '...', '...'])
    
    # 마지막 3개 발화
    for i, utterance in enumerate(dialog_data[-3:]):
        idx = len(dialog_data) - 3 + i + 1
        speaker = utterance.get('speaker', 'unknown')
        text = utterance.get('text', '')
        strategy = utterance.get('strategy', '')
        table_data.append([idx, speaker, text[:50] + ('...' if len(text) > 50 else ''), strategy])
    
    headers = ["번호", "화자", "내용", "전략"]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    
    # 전략 분석 (supporter의 발화에 대해서만)
    strategies = [utterance.get('strategy', '') for utterance in dialog_data 
                 if utterance.get('speaker') == 'sys' and 'strategy' in utterance]
    
    if strategies:
        strategy_counts = Counter(strategies)
        print_info("지지자(supporter)가 사용한 전략 분포:")
        for strategy, count in strategy_counts.most_common():
            print(f"  - {strategy}: {count}회 ({count/len(strategies)*100:.1f}%)")

def show_json_structure_example(dataset, split="train"):
    """
    Print an example of the JSON structure from the dataset.
    
    Args:
        dataset: The loaded dataset
        split: The dataset split to take the example from
    """
    print_subheader("JSON Structure Example")
    
    if split not in dataset:
        print_warning(f"{split} split not found in dataset")
        return
        
    example = dataset[split][0]
    result = parse_example_data(example)
    
    if not result['success']:
        print_error(f"Failed to parse JSON: {result['error']}")
        return
        
    json_data = result['data']
    
    # Print a formatted sample of the JSON structure
    print_info("Example of JSON structure:")
    print(json.dumps(json_data, indent=2, ensure_ascii=False)[:1000] + "...")
    
    # Extract specific fields for detailed view
    print_info("\nKey fields from the example:")
    
    # Show emotion_type
    if 'emotion_type' in json_data:
        print(f"- emotion_type: \"{json_data['emotion_type']}\"")
    
    # Show problem_type
    if 'problem_type' in json_data:
        print(f"- problem_type: \"{json_data['problem_type']}\"")
    
    # Show situation (truncated)
    if 'situation' in json_data:
        situation = json_data['situation']
        print(f"- situation: \"{situation[:100]}{'...' if len(situation) > 100 else ''}\"")
    
    # Show a sample of dialog structure
    if 'dialog' in json_data and isinstance(json_data['dialog'], list) and len(json_data['dialog']) > 0:
        print("- dialog: [")
        for i, utterance in enumerate(json_data['dialog'][:3]):
            print(f"    {json.dumps(utterance, ensure_ascii=False)},")
        if len(json_data['dialog']) > 3:
            print("    ... (more utterances)")
        print("  ]")

def perform_dataset_statistics(dataset):
    """Calculate and display various statistics of the dataset."""
    print_header("ESConv Dataset Statistical Analysis")
    
    # Initialize variables for overall dataset statistics
    all_dialog_lengths = []
    all_strategy_counter = Counter()
    all_emotion_counter = Counter()
    all_problem_counter = Counter()
    total_examples = 0
    
    # Analysis for each split
    for split in dataset.keys():
        print_subheader(f"{split.upper()} Split Statistics")
        
        dialog_lengths = []
        strategy_counter = Counter()
        emotion_counter = Counter()
        problem_counter = Counter()
        
        # Data collection - analyze full dataset
        max_samples = 100  # Limit samples for speed
        num_samples = min(len(dataset[split]), max_samples)
        print_info(f"Analyzing {num_samples} samples from {split} split (out of {len(dataset[split])})...")
        
        for i in range(num_samples):
            example = dataset[split][i]
            result = parse_example_data(example)
            
            if not result['success']:
                continue
                
            json_data = result['data']
            total_examples += 1
            
            # Collect dialog lengths
            if 'dialog' in json_data and isinstance(json_data['dialog'], list):
                dialog_length = len(json_data['dialog'])
                dialog_lengths.append(dialog_length)
                all_dialog_lengths.append(dialog_length)
            
            # Collect emotion types
            if 'emotion_type' in json_data:
                emotion = json_data['emotion_type']
                emotion_counter[emotion] += 1
                all_emotion_counter[emotion] += 1
            
            # Collect problem types
            if 'problem_type' in json_data:
                problem = json_data['problem_type']
                problem_counter[problem] += 1
                all_problem_counter[problem] += 1
            
            # Collect strategies
            if 'dialog' in json_data and isinstance(json_data['dialog'], list):
                for utterance in json_data['dialog']:
                    if isinstance(utterance, dict) and 'strategy' in utterance:
                        strategy = utterance['strategy']
                        strategy_counter[strategy] += 1
                        all_strategy_counter[strategy] += 1
        
        # Dialog length statistics
        if dialog_lengths:
            print_info(f"Dialog length statistics (analyzed samples: {len(dialog_lengths)})")
            print(f"  - Mean: {statistics.mean(dialog_lengths):.1f} utterances")
            print(f"  - Median: {statistics.median(dialog_lengths)} utterances")
            print(f"  - Min: {min(dialog_lengths)} utterances")
            print(f"  - Max: {max(dialog_lengths)} utterances")
        
        # Emotion type distribution
        if emotion_counter:
            print_info("Emotion type distribution:")
            total = sum(emotion_counter.values())
            for emotion, count in emotion_counter.most_common():
                print(f"  - {emotion}: {count} ({count/total*100:.1f}%)")
            
            # Bar chart for emotion types - 파일명에 _bar 추가
            plot_bar_chart(
                emotion_counter.most_common(), 
                f"ESConv {split} Split Emotion Distribution",
                f"{split}_emotion_distribution_bar.png",
                max_items=8
            )
        
        # Problem type distribution
        if problem_counter:
            print_info("Problem type distribution:")
            total = sum(problem_counter.values())
            for problem, count in problem_counter.most_common():
                print(f"  - {problem}: {count} ({count/total*100:.1f}%)")
            
            # Bar chart for problem types - 파일명에 _bar 추가
            plot_bar_chart(
                problem_counter.most_common(), 
                f"ESConv {split} Split Problem Distribution",
                f"{split}_problem_distribution_bar.png",
                max_items=5
            )
        
        # Strategy distribution
        if strategy_counter:
            print_info("Strategy distribution:")
            total = sum(strategy_counter.values())
            for strategy, count in strategy_counter.most_common():
                print(f"  - {strategy}: {count} ({count/total*100:.1f}%)")
            
            # Bar chart for strategies - 파일명에 _bar 추가
            plot_bar_chart(
                strategy_counter.most_common(), 
                f"ESConv {split} Split Strategy Distribution",
                f"{split}_strategy_distribution_bar.png"
            )
    
    # Overall dataset statistics
    print_header("Overall Dataset Statistics")
    
    # Dialog length statistics
    if all_dialog_lengths:
        print_info(f"Dialog length statistics (entire dataset: {len(all_dialog_lengths)} samples)")
        print(f"  - Mean: {statistics.mean(all_dialog_lengths):.1f} utterances")
        print(f"  - Median: {statistics.median(all_dialog_lengths)} utterances")
        print(f"  - Min: {min(all_dialog_lengths)} utterances")
        print(f"  - Max: {max(all_dialog_lengths)} utterances")
        
        # Histogram for dialog length distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_dialog_lengths, bins=20, alpha=0.7, color='skyblue')
        plt.title('ESConv Overall Dataset Dialog Length Distribution')
        plt.xlabel('Dialog Length (utterances)')
        plt.ylabel('Number of Dialogs')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(IMAGES_DIR, 'all_dialog_length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Emotion type distribution
    if all_emotion_counter:
        print_info("Overall emotion type distribution:")
        total = sum(all_emotion_counter.values())
        for emotion, count in all_emotion_counter.most_common():
            print(f"  - {emotion}: {count} ({count/total*100:.1f}%)")
        
        # Bar chart for overall emotion types
        plot_bar_chart(
            all_emotion_counter.most_common(), 
            "ESConv Overall Dataset Emotion Distribution",
            "all_emotion_distribution_bar.png",
            max_items=8
        )
    
    # Problem type distribution
    if all_problem_counter:
        print_info("Overall problem type distribution:")
        total = sum(all_problem_counter.values())
        for problem, count in all_problem_counter.most_common():
            print(f"  - {problem}: {count} ({count/total*100:.1f}%)")
        
        # Bar chart for overall problem types
        plot_bar_chart(
            all_problem_counter.most_common(), 
            "ESConv Overall Dataset Problem Distribution",
            "all_problem_distribution_bar.png",
            max_items=5
        )
    
    # Strategy distribution
    if all_strategy_counter:
        print_info("Overall strategy distribution:")
        total = sum(all_strategy_counter.values())
        for strategy, count in all_strategy_counter.most_common():
            print(f"  - {strategy}: {count} ({count/total*100:.1f}%)")
        
        # Bar chart for overall strategies
        plot_bar_chart(
            all_strategy_counter.most_common(), 
            "ESConv Overall Dataset Strategy Distribution",
            "all_strategy_distribution_bar.png"
        )

def plot_pie_chart(data, title, filename, max_items=None):
    """
    Create and save a pie chart.
    
    Args:
        data: List of (item, count) tuples
        title: Chart title
        filename: Filename to save the chart
        max_items: Maximum number of items to display (rest grouped as 'Other')
    """
    plt.figure(figsize=(10, 8))
    
    # Prepare data
    if max_items and len(data) > max_items:
        main_items = data[:max_items]
        other_sum = sum(count for _, count in data[max_items:])
        
        labels = [item for item, _ in main_items] + ["Other"]
        sizes = [count for _, count in main_items] + [other_sum]
    else:
        labels = [item for item, _ in data]
        sizes = [count for _, count in data]
    
    # Set colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=None,  # We'll use a legend instead
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.5, edgecolor='w')  # Donut chart style
    )
    
    # Set autotext style
    plt.setp(autotexts, size=10, weight="bold", color="white")
    
    # Set title
    plt.title(title, size=16, pad=20)
    
    # Add legend
    plt.legend(
        wedges, 
        labels,
        title="Types",
        loc="center left",
        bbox_to_anchor=(0.9, 0, 0.5, 1)
    )
    
    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print_success(f"Pie chart saved: {filename}")

def plot_bar_chart(data, title, filename, max_items=None):
    """
    Create and save a bar chart.
    
    Args:
        data: List of (item, count) tuples
        title: Chart title
        filename: Filename to save the chart
        max_items: Maximum number of items to display
    """
    # Select data to display
    if max_items:
        data = data[:max_items]
    
    # Prepare data
    categories = [item for item, _ in data]
    values = [count for _, count in data]
    total = sum(values)
    percentages = [count / total * 100 for count in values]
    
    # Create graph
    plt.figure(figsize=(12, 8))
    bars = plt.bar(categories, percentages, color=plt.cm.tab20(np.linspace(0, 1, len(categories))))
    
    # Set graph style
    plt.title(title, size=16, pad=20)
    plt.xlabel('Type', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Show percentages above bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.3,
            f'{percentage:.1f}%',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print_success(f"Bar chart saved: {filename}")

def main():
    """Main function: Run the dataset analysis."""
    try:
        # Load dataset
        print_header("Loading ESConv Dataset")
        dataset = load_esconv_dataset()
        
        if not dataset:
            print_error("Failed to load dataset")
            return
            
        print_success("Dataset loaded successfully")
        
        # Show JSON structure example
        show_json_structure_example(dataset)
        
        # Dataset structure analysis
        analyze_dataset_structure()
        
        # Dataset statistics analysis
        if dataset:
            perform_dataset_statistics(dataset)
        
        print_header("Analysis Complete")
        print_success("ESConv dataset structure and content analyzed successfully.")
        print_info(f"Generated charts are saved in the {IMAGES_DIR} directory.")
        
    except Exception as e:
        print_error(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 