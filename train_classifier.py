import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy


def train_and_evaluate(
    train_dataset,
    test_dataset,
    model_fn,
    epochs=100,
    batch_size=128,
    lr=0.05,
    device='cuda',
    save_path=None,
    verbose=True
):
    """
    분류기 학습 및 평가
    
    Args:
        train_dataset: 학습 데이터셋
        test_dataset: 테스트 데이터셋
        model_fn: 모델 생성 함수
        epochs: 에포크 수
        batch_size: 배치 크기
        lr: 학습률
        device: 디바이스
        save_path: 체크포인트 저장 경로
        verbose: 출력 여부
    
    Returns:
        history: 학습 기록 (train_acc, test_acc, best_acc)
    """
    model = model_fn().to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
    
    best_acc = 0.0
    history = {'train_acc': [], 'test_acc': [], 'best_acc': 0.0}
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        metric = MulticlassAccuracy(num_classes=10).to(device)
        train_loss = 0.0
        train_samples = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device, dtype=torch.float16, 
                               enabled=(device=='cuda')):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * x.size(0)
            train_samples += x.size(0)
            metric.update(logits, y)
        
        train_acc = metric.compute().item()
        train_loss = train_loss / train_samples
        
        # Test
        model.eval()
        metric = MulticlassAccuracy(num_classes=10).to(device)
        test_loss = 0.0
        test_samples = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                
                test_loss += loss.item() * x.size(0)
                test_samples += x.size(0)
                metric.update(logits, y)
        
        test_acc = metric.compute().item()
        test_loss = test_loss / test_samples
        
        scheduler.step()
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['best_acc'] = best_acc
        
        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"[{epoch:03d}] train {train_loss:.4f}/{train_acc:.4f}  "
                  f"test {test_loss:.4f}/{test_acc:.4f}  best {best_acc:.4f}")
    
    return history


if __name__ == '__main__':
    from data import get_few_shot_cifar10
    from augment_traditional import AugmentedDataset
    from resnet import get_resnet18_cifar10
    
    train_subset, test_subset = get_few_shot_cifar10(samples_per_class=100)
    
    # 원본만으로 학습
    dataset = AugmentedDataset(train_subset, aug_dir=None)
    
    history = train_and_evaluate(
        train_dataset=dataset,
        test_dataset=test_subset,
        model_fn=get_resnet18_cifar10,
        epochs=100,
        save_path='./results/test_model.pth'
    )
    
    print(f"\nBest accuracy: {history['best_acc']:.4f}")