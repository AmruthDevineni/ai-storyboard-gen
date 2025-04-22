import argparse
import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Storyboard Generator")
    parser.add_argument("--story", type=str, default="A hero finds a magical sword, battles a dragon, and returns as a legend.",
                        help="Story text to convert into a storyboard")
    parser.add_argument("--num-scenes", type=int, default=3, help="Number of scenes to generate")
    parser.add_argument("--output-dir", type=str, default="outputs/images", help="Output directory")
    parser.add_argument("--style", type=str, default="comic book style, detailed, vibrant colors",
                        help="Visual style for the storyboard")
    parser.add_argument("--panel-dir", type=str, default=None, 
                        help="Directory containing comic panels (if using panel mode)")
    parser.add_argument("--input-type", type=str, choices=["text", "panels"], default="text",
                        help="Input type: text story or comic panels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def split_story_into_scenes(story_text, num_scenes):
    # Simple sentence-based splitting
    sentences = re.split(r'[.!?]+', story_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) >= num_scenes:
        # Use existing sentences
        scenes = []
        for i in range(num_scenes):
            idx = i * len(sentences) // num_scenes
            scenes.append(sentences[idx])
    else:
        # Duplicate sentences to reach num_scenes
        scenes = sentences.copy()
        while len(scenes) < num_scenes:
            scenes.append(sentences[len(scenes) % len(sentences)])
    
    return scenes

def format_scene_prompt(scene_text, style_prompt):
    return f"A comic panel showing {scene_text}, {style_prompt}"

def generate_storyboard(scenes, style_prompt, output_dir, seed=42):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Initialize Stable Diffusion
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    
    # Generate images for each scene
    generated_images = []
    for i, scene_text in enumerate(scenes):
        print(f"Generating scene {i+1}/{len(scenes)}...")
        prompt = format_scene_prompt(scene_text, style_prompt)
        
        # Generate image
        image = pipe(
            prompt,
            negative_prompt="low quality, blurry, distorted proportions",
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # Save image
        image_path = output_dir / f"scene_{i+1:02d}.png"
        image.save(image_path)
        print(f"Saved to {image_path}")
        
        generated_images.append((image, scene_text))
    
    # Create a combined storyboard image
    create_storyboard_grid(generated_images, output_dir / "storyboard.png")
    
    return generated_images

def process_panels(panel_dir, num_panels, output_dir):
    panel_dir = Path(panel_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    panel_paths = []
    
    for ext in image_extensions:
        panel_paths.extend(list(panel_dir.glob(f"**/*{ext}")))
        panel_paths.extend(list(panel_dir.glob(f"**/*{ext.upper()}")))
    
    # Sort paths and limit to num_panels
    panel_paths = sorted(panel_paths)[:num_panels]
    
    if not panel_paths:
        print(f"No panel images found in {panel_dir}")
        return []
    
    print(f"Found {len(panel_paths)} panel images")
    
    # Process panels
    processed_panels = []
    for i, path in enumerate(panel_paths):
        try:
            # Load image
            image = Image.open(path)
            
            # Save a copy to output directory
            output_path = output_dir / f"panel_{i+1:02d}.png"
            image.save(output_path)
            
            # Add to processed panels
            processed_panels.append((image, f"Panel {i+1}: {path.stem}"))
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    # Create storyboard grid
    if processed_panels:
        create_storyboard_grid(processed_panels, output_dir / "storyboard.png")
    
    return processed_panels

def create_storyboard_grid(image_scene_pairs, output_path):
    if not image_scene_pairs:
        return
    
    # Determine grid size
    num_images = len(image_scene_pairs)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    # Get image dimensions
    width, height = image_scene_pairs[0][0].size
    
    # Create a new image for the grid
    grid_width = cols * width
    grid_height = rows * (height + 50)  # Extra space for text
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Add images and text to grid
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid_image)
    
    for i, (image, scene_text) in enumerate(image_scene_pairs):
        row = i // cols
        col = i % cols
        
        # Paste image
        grid_image.paste(image, (col * width, row * (height + 50)))
        
        # Add text
        text_position = (col * width + 10, row * (height + 50) + height + 10)
        draw.text(text_position, f"Scene {i+1}: {scene_text}", fill="black", font=font)
    
    # Save the grid
    grid_image.save(output_path)
    print(f"Storyboard grid saved to {output_path}")

def main():
    args = parse_args()
    
    if args.input_type == "text":
        print(f"Story: {args.story}")
        scenes = split_story_into_scenes(args.story, args.num_scenes)
        
        print(f"Scenes:")
        for i, scene in enumerate(scenes):
            print(f"{i+1}. {scene}")
        
        generate_storyboard(scenes, args.style, args.output_dir, args.seed)
        
    else:  # panels mode
        if not args.panel_dir:
            print("Error: --panel-dir must be specified when using --input-type panels")
            return
        
        process_panels(args.panel_dir, args.num_scenes, args.output_dir)
    
    print("Storyboard generation complete!")

if __name__ == "__main__":
    main()