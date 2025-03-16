import torch
from PIL import Image
import numpy as np
import cv2
from diffusers import StableVideoDiffusionPipeline

def test_optimized_model(model_path="optimized_video_model", test_image_path="test_image.jpg"):
    # Load optimized model
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float8 if torch.cuda.is_available() else torch.float32
    )
    
    # Enable optimizations
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()
    if hasattr(pipeline, 'enable_attention_slicing'):
        pipeline.enable_attention_slicing()
    
    # Load test image
    image = Image.open(test_image_path)
    
    # Generate video
    video_frames = pipeline(
        image,
        num_frames=14,
        num_inference_steps=10  # Reduced steps for faster inference
    ).frames
    
    # Save video
    height, width, layers = video_frames[0].shape
    out = cv2.VideoWriter(
        'optimized_test_video.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        8,
        (width, height)
    )
    
    for frame in video_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print("Test video generated: optimized_test_video.mp4")

if __name__ == "__main__":
    test_optimized_model()