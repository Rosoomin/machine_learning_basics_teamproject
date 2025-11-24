import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_few_shot_cifar10(samples_per_class=100, data_dir='./data', seed=42):
    """
    CIFAR-10에서 클래스당 N장만 샘플링
    test set은 그대로 유지
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    # train: transform 없이 로드 (나중에 증강 적용)
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )
    
    # test: normalize만 적용
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=normalize
    )
    
    # 클래스별 샘플링
    targets = np.array(trainset.targets)
    indices = []
    
    for class_id in range(10):
        class_indices = np.where(targets == class_id)[0]
        selected = np.random.choice(class_indices, samples_per_class, replace=False)
        indices.extend(selected.tolist())
    
    train_subset = Subset(trainset, indices)
    
    print(f"Train: {len(train_subset)} images ({samples_per_class} per class)")
    print(f"Test: {len(testset)} images")
    
    return train_subset, testset


def get_image_and_label(dataset, idx):
    """dataset에서 PIL 이미지와 레이블 추출"""
    img, label = dataset[idx]
    
    # Tensor면 PIL로 변환
    if isinstance(img, torch.Tensor):
        img = T.ToPILImage()(img)
    
    return img, label


if __name__ == '__main__':
    train, test = get_few_shot_cifar10(samples_per_class=100)
    print(f"Classes: {CLASS_NAMES}")