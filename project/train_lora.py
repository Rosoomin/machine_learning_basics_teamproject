import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import safetensors.torch as safetorch


class LoRATrainDataset(Dataset):
    """LoRA 파인튜닝 데이터셋"""
    
    def __init__(self, original_dataset, aug_dir=None, target_size=512):
        from data import CLASS_NAMES
        
        self.data = []
        self.target_size = target_size
        
        # 원본 100장
        for idx in range(len(original_dataset)):
            img, label = original_dataset[idx]
            if isinstance(img, torch.Tensor):
                from torchvision import transforms as T
                img = T.ToPILImage()(img)
            self.data.append((img, CLASS_NAMES[label]))
        
        # 증강 데이터 추가 (Case B)
        if aug_dir is not None:
            aug_path = Path(aug_dir)
            if aug_path.exists():
                for class_id, class_name in enumerate(CLASS_NAMES):
                    class_dir = aug_path / class_name
                    if class_dir.exists():
                        for img_path in sorted(class_dir.glob('*.png')):
                            img = Image.open(img_path).convert('RGB')
                            self.data.append((img, class_name))
        
        print(f"LoRA dataset: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, class_name = self.data[idx]
        
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)
        img = (img - 0.5) * 2.0
        
        prompt = f"a photo of {class_name}"
        
        return {'pixel_values': img, 'prompt': prompt}


def train_lora(
    original_dataset,
    aug_dir=None,
    output_path='./models/lora/rank8_caseA.safetensors',
    rank=8,
    steps=2000,
    batch_size=2,
    lr=5e-5,
    device='cuda'
):
    """
    SD LoRA 파인튜닝
    
    Args:
        original_dataset: 원본 few-shot dataset
        aug_dir: 증강 이미지 경로 (None이면 Case A, 경로 지정시 Case B)
        output_path: LoRA 가중치 저장 경로
        rank: LoRA rank
        steps: 학습 스텝
        batch_size: 배치 크기
        lr: 학습률
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    case_type = "Case A (원본만)" if aug_dir is None else "Case B (원본+증강)"
    print(f"\n{'='*60}")
    print(f"LoRA Training: rank={rank}, {case_type}")
    print(f"{'='*60}")
    
    dataset = LoRATrainDataset(original_dataset, aug_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print("Loading SD components...")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    text_encoder.requires_grad_(False)
    
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False)
    
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=0.01)
    
    print("\nTraining...")
    unet.train()
    
    global_step = 0
    loss_sum = 0.0
    pbar = tqdm(total=steps, desc="Training")
    
    while global_step < steps:
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device, dtype=torch.float16)
            prompts = batch['prompt']
            
            text_inputs = tokenizer(
                prompts, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            
            loss = F.mse_loss(model_pred.float(), noise.float())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            
            loss_sum += loss.item()
            global_step += 1
            
            pbar.update(1)
            if global_step % 50 == 0:
                avg_loss = loss_sum / 50
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                loss_sum = 0.0
            
            if global_step >= steps:
                break
    
    pbar.close()
    
    print("\nSaving LoRA weights...")
    lora_state = {k: v.cpu() for k, v in unet.state_dict().items() if 'lora' in k}
    safetorch.save_file(lora_state, output_path)
    
    print(f"Saved to {output_path}\n")
    
    return output_path


if __name__ == '__main__':
    from data import get_few_shot_cifar10
    
    train_subset, _ = get_few_shot_cifar10(samples_per_class=100)
    
    train_lora(
        original_dataset=train_subset,
        aug_dir=None,
        output_path='./models/lora/rank8_caseA.safetensors',
        rank=8,
        steps=2000
    )