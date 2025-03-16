import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableVideoDiffusionPipeline
from datasets import load_dataset
import torch.quantization
import os
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import get_dataset_split_names

class ModelOptimizer:
    def __init__(self, cache_dir="models"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def quantize_model(self, model_id="stabilityai/stable-video-diffusion-img2vid-xt"):
        """
        Quantize the model to 8-bit precision
        """
        print(f"Loading model: {model_id}")
        
        # Load model with 8-bit quantization
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=self.cache_dir
        )
        
        # Apply memory optimizations
        print("Applying memory optimizations...")
        pipeline.enable_model_cpu_offload()  # Move model to CPU when not in use
        
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        return pipeline
    
    def prepare_dataset(self, dataset_name="sayakpaul/ucf101-subset"):
        """
        Load and prepare dataset for fine-tuning
        """
        print(f"Loading dataset: {dataset_name}")

        print("get_dataset_split_names",get_dataset_split_names("sayakpaul/ucf101-subset"))
        dataset = load_dataset(dataset_name, split="train[:10%]")
        
        # Prepare dataset for video fine-tuning
        def preprocess_video(examples):
            videos = [torch.tensor(video) for video in examples['video']]
            return {'video': videos}
        print(get_dataset_split_names("ucf101-subset"))
        dataset = load_dataset("sayakpaul/ucf101-subset", split="train", download_mode="force_redownload")
        return dataset



    def fine_tune(self, pipeline, dataset, num_epochs=1, batch_size=1, learning_rate=1e-5):
        """
        Fine-tune the quantized model
        """
        print("Starting fine-tuning...")
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=True)):
                videos = batch['video'].to(self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    loss = pipeline(videos, return_dict=True).loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"Loss: {loss.item():.4f}")
        
        return pipeline
    
    def save_optimized_model(self, pipeline, output_dir):
        """
        Save the quantized and fine-tuned model
        """
        print(f"Saving optimized model to: {output_dir}")
        pipeline.save_pretrained(output_dir)

def main():
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Step 1: Quantize model
    print("Step 1: Quantizing model...")
    pipeline = optimizer.quantize_model()
    
    # Step 2: Prepare dataset
    print("\nStep 2: Preparing dataset...")
    dataset = optimizer.prepare_dataset()
    
    # Step 3: Fine-tune
    print("\nStep 3: Fine-tuning...")
    pipeline = optimizer.fine_tune(
        pipeline=pipeline,
        dataset=dataset,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-5
    )
    
    # Step 4: Save optimized model
    print("\nStep 4: Saving optimized model...")
    optimizer.save_optimized_model(pipeline, "optimized_video_model")

if __name__ == "__main__":
    main()