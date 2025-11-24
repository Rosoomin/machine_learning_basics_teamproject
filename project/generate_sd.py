import torch
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
import safetensors.torch as safetorch


def load_lora_to_pipe(pipe, lora_path, alpha=1.0):
    """LoRA 가중치를 파이프라인에 병합"""
    lora_state = safetorch.load_file(lora_path)
    
    unet = pipe.unet
    for name, param in unet.named_parameters():
        lora_a_key = name.replace('.weight', '.lora_A.weight')
        lora_b_key = name.replace('.weight', '.lora_B.weight')
        
        if lora_a_key in lora_state and lora_b_key in lora_state:
            lora_a = lora_state[lora_a_key].to(param.device, dtype=param.dtype)
            lora_b = lora_state[lora_b_key].to(param.device, dtype=param.dtype)
            param.data += alpha * (lora_b @ lora_a)
    
    return pipe


def generate_sd_augmentations(
    original_dataset,
    lora_path=None,
    output_dir='./data/aug_sd',
    images_per_class=100,
    strength=0.5,
    steps=30,
    guidance=7.0,
    device='cuda'
):
    """
    SD img2img로 증강 이미지 생성
    
    Args:
        original_dataset: 원본 few-shot dataset
        lora_path: LoRA 가중치 경로 (None이면 vanilla SD)
        output_dir: 저장 경로
        images_per_class: 클래스당 생성할 이미지 개수
        strength: denoising strength
        steps: inference steps
        guidance: guidance scale
    """
    from data import CLASS_NAMES
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_name in CLASS_NAMES:
        (output_path / class_name).mkdir(exist_ok=True)
    
    lora_type = "vanilla SD" if lora_path is None else f"LoRA ({Path(lora_path).name})"
    print(f"\n{'='*60}")
    print(f"Generating with {lora_type}")
    print(f"{'='*60}")
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    pipe.enable_attention_slicing()
    
    if lora_path is not None and Path(lora_path).exists():
        print(f"Loading LoRA from {lora_path}")
        pipe = load_lora_to_pipe(pipe, lora_path)
    
    # 원본 이미지를 클래스별로 정리
    class_images = {i: [] for i in range(10)}
    for idx in range(len(original_dataset)):
        img, label = original_dataset[idx]
        if isinstance(img, torch.Tensor):
            from torchvision import transforms as T
            img = T.ToPILImage()(img)
        class_images[label].append(img)
    
    print(f"Generating {images_per_class} images per class...")
    
    generator = torch.Generator(device=device).manual_seed(42)
    
    for class_id in range(10):
        class_name = CLASS_NAMES[class_id]
        originals = class_images[class_id]
        
        prompt = f"a photo of {class_name}"
        neg_prompt = "low quality, blurry, deformed"
        
        saved_count = 0
        
        while saved_count < images_per_class:
            for orig_img in originals:
                if saved_count >= images_per_class:
                    break
                
                # 32x32 -> 512x512
                img_512 = orig_img.resize((512, 512), Image.LANCZOS)
                
                with torch.autocast(device):
                    result = pipe(
                        prompt=prompt,
                        image=img_512,
                        strength=strength,
                        negative_prompt=neg_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator
                    ).images[0]
                
                save_path = output_path / class_name / f"{saved_count:05d}.png"
                result.save(save_path)
                
                saved_count += 1
        
        print(f"  {class_name}: {saved_count} images")
    
    print(f"Saved to {output_path}\n")
    
    return output_path


def downsample_to_32x32(input_dir='./data/aug_sd', output_dir='./data/aug_sd_32'):
    """512x512 이미지를 32x32로 다운샘플"""
    from data import CLASS_NAMES
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downsampling 512x512 -> 32x32...")
    
    for class_name in CLASS_NAMES:
        (output_path / class_name).mkdir(exist_ok=True)
        
        class_dir = input_path / class_name
        if not class_dir.exists():
            continue
        
        count = 0
        for img_path in sorted(class_dir.glob('*.png')):
            img = Image.open(img_path).convert('RGB')
            img_32 = img.resize((32, 32), Image.LANCZOS)
            
            out_path = output_path / class_name / img_path.name
            img_32.save(out_path)
            count += 1
        
        print(f"  {class_name}: {count} images")
    
    print(f"Saved to {output_path}")
    
    return output_path


if __name__ == '__main__':
    from data import get_few_shot_cifar10
    
    train_subset, _ = get_few_shot_cifar10(samples_per_class=100)
    
    generate_sd_augmentations(
        original_dataset=train_subset,
        lora_path=None,
        images_per_class=100,
        strength=0.5
    )
    
    downsample_to_32x32()