"""
목적 2: 전통적 증강의 포화점 탐색 및 생성형 증강의 추가효과 평가

1. 전통적 증강을 k=1~20까지 증가시키며 포화점 찾기
2. 포화점 이후 생성형 증강 추가 시 효과 검증
"""

import torch
from pathlib import Path
import json
import numpy as np

from data import get_few_shot_cifar10
from augment_traditional import generate_traditional_augmentations, AugmentedDataset
from generate_sd import generate_sd_augmentations, downsample_to_32x32
from train_classifier import train_and_evaluate
from resnet import get_resnet18_cifar10


def find_saturation_point(results, threshold=0.001):
    """
    포화점 판단
    
    연속 3개 스텝에서 평균 증가폭이 threshold 미만이면 포화로 판단
    """
    if len(results) < 4:
        return None
    
    accs = [r['best_acc'] for r in results]
    
    for i in range(2, len(accs)):
        # 최근 3개의 증가폭
        deltas = [accs[j] - accs[j-1] for j in range(i-1, i+2) if j < len(accs)]
        avg_delta = np.mean(deltas)
        
        if avg_delta < threshold:
            return i - 1
    
    return None


def generate_traditional_aug_batches(train_subset, max_k=20):
    """전통적 증강 데이터를 k=1~20까지 생성"""
    print("\n" + "="*70)
    print("Generating Traditional Augmentations (k=1~20)")
    print("="*70)
    
    aug_dirs = {}
    
    for k in range(1, max_k + 1):
        aug_dir = f'./data/aug_traditional_k{k:02d}'
        aug_dirs[k] = aug_dir
        
        if Path(aug_dir).exists():
            print(f"k={k:2d}: Already exists")
            continue
        
        # k * 100장 = 이미지당 k개 증강
        generate_traditional_augmentations(
            train_subset,
            augments_per_image=k,
            output_dir=aug_dir
        )
    
    return aug_dirs


def run_saturation_search(train_subset, test_subset, aug_dirs, max_k=20):
    """전통적 증강 포화점 탐색"""
    print("\n" + "="*70)
    print("Phase 1: Traditional Augmentation Saturation Search")
    print("="*70)
    
    results = []
    
    for k in range(1, max_k + 1):
        print(f"\n[k={k:2d}] Original 100 + Traditional {k*100} (total {(k+1)*100}/class)")
        
        dataset = AugmentedDataset(train_subset, aug_dir=aug_dirs[k])
        history = train_and_evaluate(
            train_dataset=dataset,
            test_dataset=test_subset,
            model_fn=get_resnet18_cifar10,
            epochs=100,
            save_path=f'./results/exp2_trad_k{k:02d}.pth',
            verbose=True
        )
        
        result = {
            'k': k,
            'train_size': len(dataset),
            'best_acc': history['best_acc'],
            'final_acc': history['test_acc'][-1]
        }
        results.append(result)
        
        # 포화점 체크
        saturation_k = find_saturation_point(results)
        if saturation_k is not None:
            print(f"\n*** Saturation detected at k={saturation_k+1} ***")
            print(f"    Accuracy delta < 0.001 for 3 consecutive steps")
            break
    
    # 포화점 없으면 마지막을 포화점으로
    if saturation_k is None:
        saturation_k = len(results) - 1
        print(f"\n*** No clear saturation, using k={saturation_k+1} as saturation point ***")
    
    return results, saturation_k


def generate_sd_aug_for_phase2(train_subset, best_model_lora_path):
    """Phase 2를 위한 생성형 증강 데이터 생성 (클래스당 1000장)"""
    print("\n" + "="*70)
    print("Generating SD Augmentations for Phase 2 (1000/class)")
    print("="*70)
    
    output_dir = './data/aug_sd_phase2'
    output_dir_32 = './data/aug_sd_phase2_32'
    
    if Path(output_dir_32).exists():
        print(f"Already exists: {output_dir_32}")
        return output_dir_32
    
    # 512x512 생성
    generate_sd_augmentations(
        original_dataset=train_subset,
        lora_path=best_model_lora_path,
        output_dir=output_dir,
        images_per_class=1000,
        strength=0.5,
        steps=30,
        guidance=7.0
    )
    
    # 32x32로 다운샘플
    return downsample_to_32x32(
        input_dir=output_dir,
        output_dir=output_dir_32
    )


def create_hybrid_datasets(train_subset, saturation_aug_dir, sd_aug_dir, max_k=10):
    """포화점 데이터 + 생성형 증강 k개 조합"""
    from torch.utils.data import ConcatDataset
    from data import CLASS_NAMES
    from PIL import Image
    import torchvision.transforms as T
    
    print("\n" + "="*70)
    print("Creating Hybrid Datasets (Saturation + SD k=1~10)")
    print("="*70)
    
    # 포화점 데이터셋
    base_dataset = AugmentedDataset(train_subset, aug_dir=saturation_aug_dir)
    base_size = len(base_dataset)
    
    hybrid_datasets = {}
    
    for k in range(1, max_k + 1):
        print(f"k={k:2d}: Base + SD {k*100}/class")
        
        # SD 증강에서 클래스당 k*100장 선택
        sd_data = []
        sd_path = Path(sd_aug_dir)
        
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = sd_path / class_name
            if not class_dir.exists():
                continue
            
            img_paths = sorted(list(class_dir.glob('*.png')))[:k*100]
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                sd_data.append((img, class_id))
        
        # SD 데이터를 데이터셋으로
        class SDSubset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
                from data import CIFAR_MEAN, CIFAR_STD
                self.transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(CIFAR_MEAN, CIFAR_STD)
                ])
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                img, label = self.data[idx]
                return self.transform(img), label
        
        sd_subset = SDSubset(sd_data)
        
        # 합치기
        hybrid_dataset = ConcatDataset([base_dataset, sd_subset])
        hybrid_datasets[k] = hybrid_dataset
        
        print(f"  Total size: {len(hybrid_dataset)} (base {base_size} + sd {len(sd_subset)})")
    
    return hybrid_datasets


def run_hybrid_experiments(test_subset, hybrid_datasets):
    """포화점 + 생성형 증강 실험"""
    print("\n" + "="*70)
    print("Phase 2: Hybrid Augmentation (Saturation + SD)")
    print("="*70)
    
    results = []
    
    for k, dataset in hybrid_datasets.items():
        print(f"\n[k={k:2d}] Saturation + SD {k*100}/class")
        
        history = train_and_evaluate(
            train_dataset=dataset,
            test_dataset=test_subset,
            model_fn=get_resnet18_cifar10,
            epochs=100,
            save_path=f'./results/exp2_hybrid_k{k:02d}.pth',
            verbose=True
        )
        
        result = {
            'k': k,
            'train_size': len(dataset),
            'best_acc': history['best_acc'],
            'final_acc': history['test_acc'][-1]
        }
        results.append(result)
    
    return results


def summarize_experiment2(phase1_results, saturation_k, phase2_results, output_dir='./results'):
    """실험 2 결과 정리"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    summary = {
        'phase1_traditional': phase1_results,
        'saturation_k': saturation_k + 1,
        'saturation_acc': phase1_results[saturation_k]['best_acc'],
        'phase2_hybrid': phase2_results
    }
    
    with open(Path(output_dir) / 'exp2_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("Experiment 2 Results Summary")
    print("="*70)
    
    print("\n[Phase 1: Traditional Augmentation]")
    for res in phase1_results:
        mark = " <- SATURATION" if res['k'] == saturation_k + 1 else ""
        print(f"  k={res['k']:2d} ({res['train_size']:5d} imgs): "
              f"best={res['best_acc']:.4f}{mark}")
    
    print(f"\nSaturation Point: k={saturation_k + 1}, acc={phase1_results[saturation_k]['best_acc']:.4f}")
    
    print("\n[Phase 2: Hybrid (Saturation + SD)]")
    for res in phase2_results:
        delta = res['best_acc'] - phase1_results[saturation_k]['best_acc']
        print(f"  k={res['k']:2d} ({res['train_size']:5d} imgs): "
              f"best={res['best_acc']:.4f} (delta={delta:+.4f})")
    
    # 최종 성능 비교
    final_hybrid = phase2_results[-1]
    final_delta = final_hybrid['best_acc'] - phase1_results[saturation_k]['best_acc']
    
    print("\n" + "="*70)
    print(f"Saturation accuracy: {phase1_results[saturation_k]['best_acc']:.4f}")
    print(f"Final hybrid accuracy: {final_hybrid['best_acc']:.4f}")
    print(f"Improvement: {final_delta:+.4f}")
    print("="*70)
    
    return summary


def run_experiment_2(best_model_name='rank8_caseB'):
    """목적 2 전체 실험 실행"""
    print("\n" + "#"*70)
    print("# EXPERIMENT 2: Hybrid Augmentation Analysis")
    print("#"*70)
    
    # Step 1: 데이터 준비
    train_subset, test_subset = get_few_shot_cifar10(samples_per_class=100)
    
    # Step 2: Phase 1 - 전통적 증강 포화점 탐색
    aug_dirs = generate_traditional_aug_batches(train_subset, max_k=20)
    phase1_results, saturation_k = run_saturation_search(
        train_subset, test_subset, aug_dirs, max_k=20
    )
    
    # Step 3: 생성형 증강 데이터 준비
    best_model_lora_path = f'./models/lora/{best_model_name}.safetensors'
    if not Path(best_model_lora_path).exists():
        print(f"\nWarning: {best_model_lora_path} not found, using vanilla SD")
        best_model_lora_path = None
    
    sd_aug_dir = generate_sd_aug_for_phase2(train_subset, best_model_lora_path)
    
    # Step 4: Phase 2 - 하이브리드 증강 실험
    saturation_aug_dir = aug_dirs[saturation_k + 1]
    hybrid_datasets = create_hybrid_datasets(
        train_subset, saturation_aug_dir, sd_aug_dir, max_k=10
    )
    phase2_results = run_hybrid_experiments(test_subset, hybrid_datasets)
    
    # Step 5: 결과 정리
    summary = summarize_experiment2(
        phase1_results, saturation_k, phase2_results
    )
    
    return summary


if __name__ == '__main__':
    summary = run_experiment_2(best_model_name='rank8_caseB')