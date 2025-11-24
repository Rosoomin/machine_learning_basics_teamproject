import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class TraditionalAugment:
    """CIFAR-10용 전통적 증강"""
    
    def __init__(self):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(32, padding=4),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    
    def __call__(self, img):
        return self.transform(img)


def generate_traditional_augmentations(dataset, augments_per_image=5, output_dir='./data/aug_traditional'):
    """
    원본 이미지 각각에 대해 N개씩 증강 생성하여 저장
    
    Args:
        dataset: few-shot CIFAR-10 dataset
        augments_per_image: 각 이미지당 생성할 증강 개수
        output_dir: 저장 경로
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    augmentor = TraditionalAugment()
    
    from data import CLASS_NAMES
    for class_name in CLASS_NAMES:
        (output_path / class_name).mkdir(exist_ok=True)
    
    class_counts = {i: 0 for i in range(10)}
    
    print(f"Generating {augments_per_image} augmentations per image...")
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        
        class_name = CLASS_NAMES[label]
        
        for aug_idx in range(augments_per_image):
            aug_img = augmentor(img)
            
            save_path = output_path / class_name / f"{class_counts[label]:05d}.png"
            aug_img.save(save_path)
            class_counts[label] += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)}")
    
    total = sum(class_counts.values())
    print(f"Saved {total} augmented images to {output_path}")
    
    return output_path


class AugmentedDataset(Dataset):
    """원본 + 증강 통합 데이터셋"""
    
    def __init__(self, original_dataset, aug_dir=None, normalize=True):
        from data import CIFAR_MEAN, CIFAR_STD, CLASS_NAMES
        
        self.data = []
        
        if normalize:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
        else:
            self.transform = T.ToTensor()
        
        # 원본 데이터 추가
        for idx in range(len(original_dataset)):
            img, label = original_dataset[idx]
            if isinstance(img, torch.Tensor):
                img = T.ToPILImage()(img)
            self.data.append((img, label))
        
        # 증강 데이터 추가
        if aug_dir is not None:
            aug_path = Path(aug_dir)
            if aug_path.exists():
                for class_id, class_name in enumerate(CLASS_NAMES):
                    class_dir = aug_path / class_name
                    if class_dir.exists():
                        for img_path in sorted(class_dir.glob('*.png')):
                            img = Image.open(img_path).convert('RGB')
                            self.data.append((img, class_id))
        
        print(f"Dataset size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img)
        return img, label


if __name__ == '__main__':
    from data import get_few_shot_cifar10
    
    train_subset, _ = get_few_shot_cifar10(samples_per_class=100)
    
    generate_traditional_augmentations(train_subset, augments_per_image=5)