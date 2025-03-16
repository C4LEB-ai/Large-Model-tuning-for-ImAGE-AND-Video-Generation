import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableVideoDiffusionPipeline
import os

class VideoGenSetup:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def setup_image_generator(self):
        """Setup Stable Diffusion for frame generation"""
        print("Downloading Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=self.models_dir
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(self.device)
        return pipe
    
    def setup_video_generator(self, optimize=True):
        """Setup Stable Video Diffusion pipeline with optional optimization"""
        print("Downloading Stable Video Diffusion model...")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float8 if torch.cuda.is_available() else torch.float32,
            variant="fp16",
            cache_dir=self.models_dir
        )
        
        if optimize:
            print("Applying optimizations...")
            # Enable memory optimizations
            pipe.enable_model_cpu_offload()
            # pipe.enable_vae_slicing()
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
            
            # Load in 8-bit if CUDA is available
            if torch.cuda.is_available():
                pipe.unet = pipe.unet.to(torch.float16)
                
        pipe.to(self.device)
        return pipe
    
    def test_setup(self, optimize=True):
        """Test the setup with a simple generation"""
        try:
            # Test image generation
            image_pipe = self.setup_image_generator()
            test_prompt1 = "A majestic fluffy white Persian cat with emerald green eyes and pink nose, lounging elegantly on a burgundy velvet cushion, illuminated by soft afternoon sunlight streaming through a window, creating a warm glow, photorealistic style, ultra-high detail, professional studio lighting, 8K resolution, shallow depth of field"
            test_prompt2 = "Dynamic abstract composition featuring swirling patterns of liquid metal merging with cosmic nebula formations. Deep indigo base transitioning through electric blues and molten golds, with fractal-like structures emerging from the chaos. High-energy movement suggesting both microscopic and astronomical scales simultaneously. Digital art style combining fluid dynamics with celestial photography, ultra-high detail, 8K resolution, inspired by Ernst Haas and NASA imagery, dramatic contrast, perfect color balance"
            test_prompt3 = "Modern sustainable office building integrated into forest landscape, featuring living walls covered in lush vegetation. Large glass facades reflecting surrounding trees, solar panels seamlessly integrated into roof design. Golden hour lighting creating warm reflections on glass surfaces, drone's-eye view showing organic flow between building and nature. Hyperrealistic architectural visualization style, perfect geometry, accurate lighting simulation, 12K resolution, LEED-certified design elements, environmental harmony theme"
            test_prompt = "Luxury watch on dark leather surface, positioned at 30-degree angle to emphasize depth. Dramatic side lighting creating metallic highlights on polished surfaces, subtle reflection in glass face. Ultra-sharp macro photography style, focus on intricate mechanical details, deep blacks, professional product photography lighting setup with main light at 90 degrees and fill card opposite, 5K resolution, perfect reflections, minimal composition, premium advertising look"
            test_prompt4 = "Ethereal floating islands suspended in a twilight sky, connected by glowing crystal bridges. Ancient stone ruins with ornate architecture dot the landscape, while bioluminescent plants emit a soft blue-green light. Multiple moons visible in the deep purple-blue sky, with aurora-like phenomena weaving between them. Photorealistic digital art style with cinematic lighting, volumetric fog, ray-traced reflections, ultra-high detail, 16K resolution, inspired by Studio Ghibli meets Syd Mead, dramatic scale, atmospheric perspective"
            
            test_prompt0 = "Young professional woman, navy suit, white blouse, pearl necklace, soft window light, shallow depth, 8K, perfect skin, direct eye contact, rule of thirds"
            test_prompt2 = "Luxury watch, dark leather, 30-degree angle, metallic highlights, ultra-sharp macro, deep blacks, professional lighting, premium look"
            test_prompt7 = "Floating islands, twilight sky, crystal bridges, bioluminescent plants, purple-blue atmosphere, cinematic lighting, ultra detail"
            test_prompt6 = "Vintage camera, wooden desk, morning light, shallow depth, perfect reflections, 5K resolution, minimal composition"
            test_prompt6 = "Liquid metal meets cosmic nebula, indigo base, electric blues, molten golds, fractal patterns, ultra detail, dramatic contrast"

            print("Testing image generation...")
            image = image_pipe(test_prompt, num_inference_steps=5).images[0]
            image.save("test_image.png")
            print("✓ Image generation test successful")
            
            # Test video model loading
            video_pipe = self.setup_video_generator(optimize=optimize)
            print("Testing video generation...")
            video_frames = video_pipe(
                image,
                num_frames=14,
                num_inference_steps=2  # Small number for testing
            ).frames
            print("✓ Video generation test successful")
            
            print("\nSetup completed successfully!")
            print("Test image saved as 'test_image.png'")
            
        except Exception as e:
            print(f"Error during setup: {str(e)}")

if __name__ == "__main__":
    # You can run setup with or without optimization
    setup = VideoGenSetup()
    setup.test_setup(optimize=True)  # Set to False if you don't want optimizations