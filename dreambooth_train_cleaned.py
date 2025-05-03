import os
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
instance_data_dir = "data/peanuts_training_cleaned"
output_dir = "peanuts_finetuned_sd_no_text"
pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"

# Load Pretrained Models
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device)

noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

# 🧠 Windows fix: Do NOT compile models on Windows (triton unavailable)
# vae = torch.compile(vae)        # ❌ Skip
# text_encoder = torch.compile(text_encoder)  # ❌ Skip
# unet = torch.compile(unet)      # ❌ Skip

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Dataset
class PeanutsDataset(Dataset):
    def __init__(self, folder, tokenizer, prompt="a peanuts comic style panel"):
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.input_ids = tokenizer(
            [prompt] * len(self.image_paths),
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except (UnidentifiedImageError, OSError):
            image = torch.zeros((3, 512, 512))
        input_id = self.input_ids[idx]
        return {"pixel_values": image, "input_ids": input_id}

# DataLoader
dataset = PeanutsDataset(instance_data_dir, tokenizer)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# Optimizer
optimizer = optim.AdamW(unet.parameters(), lr=5e-6)

# Mixed Precision
scaler = torch.cuda.amp.GradScaler()

# Training
EPOCHS = 5
global_step = 0
unet.train()

print("🔥 Starting DreamBooth fine-tuning...")

for epoch in range(EPOCHS):
    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")
    progress_bar = tqdm(train_dataloader)

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            encoder_hidden_states = text_encoder(input_ids)[0]
            latents = vae.encode(pixel_values * 2 - 1).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix(loss=loss.item())
        global_step += 1

# Save
pipeline = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    unet=unet,
    text_encoder=text_encoder,
    vae=vae,
    safety_checker=None,
)
pipeline.save_pretrained(output_dir)

print("✅ DreamBooth fine-tuning complete and model saved!")
