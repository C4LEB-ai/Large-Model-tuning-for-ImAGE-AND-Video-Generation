import torch
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
import setup as s
from setup import VideoGenSetup

class VideoGenerator:
    def __init__(self, setup):
        self.setup = setup
        self.image_pipe = setup.setup_image_generator()
        self.video_pipe = setup.setup_video_generator()
        
    def generate_video(self, prompt, num_frames=14, num_steps=25, fps=8):
        """Generate a video from a text prompt"""
        try:
            print(f"Generating initial image for: {prompt}")
            
            # First generate the image from the prompt
            image = self.image_pipe(
                prompt=prompt,
                num_inference_steps=num_steps
            ).images[0]
            
            print("Generating video from the image...")
            # Generate video frames from the image
            video_frames = self.video_pipe(
                image,
                num_frames=num_frames,
                num_inference_steps=num_steps
            ).frames
            
            # Convert frames to video
            output_path = f"generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            # Get dimensions from first frame
            height, width, layers = video_frames[0].shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in video_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
            out.release()
            print(f"Video generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    setup = s()
    generator = VideoGenerator(setup)
    
    # Test with a simple prompt
    test_prompt = "A serene mountain lake with snow-capped peaks in the background"
    video_path = generator.generate_video(
        prompt=test_prompt,
        num_frames=14,    # Number of frames to generate
        num_steps=25,     # Number of denoising steps
        fps=8            # Frames per second in output video
    )