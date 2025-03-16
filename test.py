from huggingface_hub import model_info

model_id = "stabilityai/stable-video-diffusion-img2vid-xt"

try:
    info = model_info(model_id)
    print(f"✅ Model exists: {info.modelId}")
except Exception as e:
    print(f"❌ Model not found: {e}")


import bitsandbytes as bnb
from diffusers import StableVideoDiffusionPipeline
print(bnb.__version__)


import os
import torch
path = "cuda" if torch.cuda.is_available() else "cpu"
models_dir = "optimized_video_model"
os.makedirs(models_dir, exist_ok=True)


pipeline = StableVideoDiffusionPipeline.from_pretrained(
    models_dir,
    torch_dtype=torch.float8 if torch.cuda.is_available() else torch.float32
)




pipeline.save_pretrained(models_dir)  # Save locally

# # Load from local directory
# pipeline = StableVideoDiffusionPipeline.from_pretrained("./optimized_video_model")
