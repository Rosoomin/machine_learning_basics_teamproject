"""
목적 1: 최적 생성모델 조건 탐색

9개의 생성모델군 구성:
- Non-finetuned: 1개
- LoRA rank {4, 8, 16, 32} × Case {A, B}: 8개

각 모델로 클래스당 100장 생성 후 분류 성능 비교
"""

import torch
from pathlib import Path
import json
import time

from data import get_few_shot_cifar10
from augment_traditional import generate_traditional_augmentations, AugmentedDataset
from train_lora import train_lora
from generate_sd import generate_sd_augmentations, downsample_to_32x32
from train_classifier import train_and_evaluate
from resnet import get_resnet18_cifar10


def prepare_augmentations_for_lora_training():
    """LoRA 학습에 필요한 전통적 증강 데이터 생성 (클래스당 500장)"""
    print("\n" + "="*70)
    print("Preparing Traditional Augmentations for LoRA Training (500/class)")
    print("="*70)
    
    train_subset, _ = get_few_shot_cifar10(samples_per_class=100)
    
    aug_dir = './data/aug_traditional_500'
    if not Path(aug_dir).exists():
        generate_traditional_augmentations(
            train_subset,
            augments_per_image=5,
            output_dir=aug_dir
        )
    else:
        print(f"Already exists: {aug_dir}")
    
    return train_subset, aug_dir


def train_all_lora_models(train_subset, aug_dir):
    """9개 생성모델군의 LoRA 학습 (Non-finetuned 제외)"""
    print("\n" + "="*70)
    print("Training LoRA Models")
    print("="*70)
    
    ranks = [4, 8, 16, 32]
    cases = ['A', 'B']
    
    lora_paths = {}
    
    for rank in ranks:
        for case in cases:
            model_name = f"rank{rank}_case{case}"
            output_path = f"./models/lora/{model_name}.safetensors"
            
            if Path(output_path).exists():
                print(f"\nSkipping {model_name} (already exists)")
                lora_paths[model_name] = output_path
                continue
            
            if case == 'A':
                # Case A: 원본 100장만
                lora_paths[model_name] = train_lora(
                    original_dataset=train_subset,
                    aug_dir=None,
                    output_path=output_path,
                    rank=rank,
                    steps=1000,
                    batch_size=2,
                    lr=5e-5
                )
            else:
                # Case B: 원본 100장 + 전통적 증강 500장
                lora_paths[model_name] = train_lora(
                    original_dataset=train_subset,
                    aug_dir=aug_dir,
                    output_path=output_path,
                    rank=rank,
                    steps=2000,
                    batch_size=2,
                    lr=5e-5
                )
    
    # Non-finetuned는 LoRA 없음
    lora_paths['vanilla'] = None
    
    return lora_paths


def generate_all_augmentations(train_subset, lora_paths):
    """9개 생성모델군으로 각각 증강 데이터 생성 (클래스당 100장씩)"""
    print("\n" + "="*70)
    print("Generating Augmentations from All Models")
    print("="*70)
    
    aug_paths = {}
    
    for model_name, lora_path in lora_paths.items():
        output_dir = f"./data/aug_sd_{model_name}"
        output_dir_32 = f"./data/aug_sd_{model_name}_32"
        
        if Path(output_dir_32).exists():
            print(f"\nSkipping {model_name} (already exists)")
            aug_paths[model_name] = output_dir_32
            continue
        
        # 512x512 생성
        generate_sd_augmentations(
            original_dataset=train_subset,
            lora_path=lora_path,
            output_dir=output_dir,
            images_per_class=100,
            strength=0.5,
            steps=30,
            guidance=7.0
        )
        
        # 32x32로 다운샘플
        aug_paths[model_name] = downsample_to_32x32(
            input_dir=output_dir,
            output_dir=output_dir_32
        )
        
        time.sleep(1)
    
    return aug_paths


def run_baseline_experiments(train_subset, test_subset):
    """Baseline 실험 2개 실행"""
    print("\n" + "="*70)
    print("Running Baseline Experiments")
    print("="*70)

    Path('./results').mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Baseline 1: 원본 100장만
    print("\n[Baseline 1] Original only (100/class)")
    dataset1 = AugmentedDataset(train_subset, aug_dir=None)
    history1 = train_and_evaluate(
        train_dataset=dataset1,
        test_dataset=test_subset,
        model_fn=get_resnet18_cifar10,
        epochs=100,
        save_path='./results/exp1_baseline1.pth'
    )
    results['baseline1_orig100'] = {
        'train_size': len(dataset1),
        'best_acc': history1['best_acc'],
        'final_acc': history1['test_acc'][-1]
    }
    
    # Baseline 2: 원본 100 + 전통적 증강 100 (총 200/class)
    print("\n[Baseline 2] Original 100 + Traditional 100 (200/class)")
    
    # 전통적 증강 100장 생성
    aug_dir_100 = './data/aug_traditional_100'
    if not Path(aug_dir_100).exists():
        generate_traditional_augmentations(
            train_subset,
            augments_per_image=1,
            output_dir=aug_dir_100
        )
    
    dataset2 = AugmentedDataset(train_subset, aug_dir=aug_dir_100)
    history2 = train_and_evaluate(
        train_dataset=dataset2,
        test_dataset=test_subset,
        model_fn=get_resnet18_cifar10,
        epochs=100,
        save_path='./results/exp1_baseline2.pth'
    )
    results['baseline2_orig100_trad100'] = {
        'train_size': len(dataset2),
        'best_acc': history2['best_acc'],
        'final_acc': history2['test_acc'][-1]
    }
    
    return results


def run_generative_experiments(train_subset, test_subset, aug_paths):
    """9개 생성모델군 실험"""
    print("\n" + "="*70)
    print("Running Generative Augmentation Experiments")
    print("="*70)
    
    results = {}
    
    for model_name, aug_path in aug_paths.items():
        print(f"\n[{model_name}] Original 100 + Generated 100 (200/class)")
        
        dataset = AugmentedDataset(train_subset, aug_dir=aug_path)
        history = train_and_evaluate(
            train_dataset=dataset,
            test_dataset=test_subset,
            model_fn=get_resnet18_cifar10,
            epochs=100,
            save_path=f'./results/exp1_{model_name}.pth'
        )
        
        results[f'gen_{model_name}'] = {
            'train_size': len(dataset),
            'best_acc': history['best_acc'],
            'final_acc': history['test_acc'][-1]
        }
    
    return results


def summarize_results(results, output_dir='./results'):
    """결과 요약 및 저장"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # JSON 저장
    with open(Path(output_dir) / 'exp1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 콘솔 출력
    print("\n" + "="*70)
    print("Experiment 1 Results Summary")
    print("="*70)
    
    print("\n[Baselines]")
    for name, res in results.items():
        if 'baseline' in name:
            print(f"  {name:35s}: {res['train_size']:4d} imgs, "
                  f"best={res['best_acc']:.4f}")
    
    print("\n[Generative Augmentations]")
    gen_results = {k: v for k, v in results.items() if 'gen_' in k}
    
    # 최고 성능 찾기
    best_model = max(gen_results.items(), key=lambda x: x[1]['best_acc'])
    
    for name, res in gen_results.items():
        mark = " <- BEST" if name == best_model[0] else ""
        print(f"  {name:35s}: {res['train_size']:4d} imgs, "
              f"best={res['best_acc']:.4f}{mark}")
    
    print("\n" + "="*70)
    print(f"Best Model: {best_model[0]}")
    print(f"Best Accuracy: {best_model[1]['best_acc']:.4f}")
    print("="*70)
    
    return best_model[0]


def run_experiment_1():
    """목적 1 전체 실험 실행"""
    print("\n" + "#"*70)
    print("# EXPERIMENT 1: Optimal Generative Model Search")
    print("#"*70)
    
    # Step 1: 데이터 준비
    train_subset, aug_dir = prepare_augmentations_for_lora_training()
    _, test_subset = get_few_shot_cifar10(samples_per_class=100)
    
    # Step 2: LoRA 학습 (8개)
    lora_paths = train_all_lora_models(train_subset, aug_dir)
    
    # Step 3: 생성 (9개)
    aug_paths = generate_all_augmentations(train_subset, lora_paths)
    
    # Step 4: Baseline 실험
    results = run_baseline_experiments(train_subset, test_subset)
    
    # Step 5: 생성형 증강 실험
    gen_results = run_generative_experiments(train_subset, test_subset, aug_paths)
    results.update(gen_results)
    
    # Step 6: 결과 정리
    best_model = summarize_results(results)
    
    return results, best_model


if __name__ == '__main__':
    results, best_model = run_experiment_1()
