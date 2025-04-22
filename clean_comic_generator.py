import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import json
import re
import random
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Clean Comic Panel Generator (No Text)")
    parser.add_argument("--story", type=str, 
                        default="A hero finds a magical sword, battles a dragon, and returns as a legend.",
                        help="Story text to convert into a storyboard")
    parser.add_argument("--num-scenes", type=int, default=3, 
                        help="Number of scenes to generate")
    parser.add_argument("--output-dir", type=str, default="outputs/comic", 
                        help="Output directory")
    parser.add_argument("--style", type=str, 
                        default="comic book style, detailed, vibrant colors",
                        help="Visual style for the storyboard")
    parser.add_argument("--panel-dir", type=str, default=None, 
                        help="Directory containing comic panels (if using panel mode)")
    parser.add_argument("--input-type", type=str, choices=["text", "panels"], default="text",
                        help="Input type: text story or comic panels")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--inference-steps", type=int, default=50,  
                        help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=8.5,  
                        help="Guidance scale for text-to-image generation")
    parser.add_argument("--width", type=int, default=768,  
                        help="Image width")
    parser.add_argument("--height", type=int, default=768,  
                        help="Image height")
    parser.add_argument("--background-color", type=str, default="white",
                        help="Background color for panels (white, light-blue, etc.)")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1",
                        help="Diffusion model to use")
    return parser.parse_args()

def split_story_into_scenes(story_text, num_scenes):
    """Split story text into scenes"""
    # First try to split by sentences
    sentences = re.split(r'[.!?]+', story_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) >= num_scenes:
        # If we have enough sentences, use them
        scenes = []
        for i in range(num_scenes):
            # Distribute sentences evenly across scenes
            idx = i * len(sentences) // num_scenes
            scenes.append(sentences[idx])
    else:
        # If not enough sentences, try to split the story evenly
        words = story_text.split()
        words_per_scene = max(1, len(words) // num_scenes)
        
        scenes = []
        for i in range(num_scenes):
            start = i * words_per_scene
            end = min(start + words_per_scene, len(words))
            if start < len(words):
                scene_text = " ".join(words[start:end])
                scenes.append(scene_text)
            
        # If still not enough, duplicate scenes
        while len(scenes) < num_scenes:
            scenes.append(scenes[len(scenes) % len(scenes)])
    
    return scenes

def format_scene_prompt(scene_text, style_prompt, background_color="white"):
    """Format scene description into a prompt for image generation"""
    return f"A comic panel showing {scene_text}, with clear space around characters, {style_prompt}, clear composition with proper spacing and {background_color} background, single scene with no overlap, NO TEXT, NO SPEECH BUBBLES, NO CAPTIONS"

def generate_panel_images(scenes, style_prompt, output_dir, seed=42, 
                         inference_steps=50, guidance_scale=8.5,
                         width=768, height=768, background_color="white",
                         model_name="stabilityai/stable-diffusion-2-1"):
    """Generate comic panel images from scene descriptions"""
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
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    
    # Generate images for each scene
    generated_images = []
    
    for i, scene_text in enumerate(scenes):
        print(f"Generating panel {i+1}/{len(scenes)}...")
        
        # Format prompt to explicitly exclude text and speech bubbles
        prompt = format_scene_prompt(scene_text, style_prompt, background_color)
        
        # Enhanced negative prompt to avoid text, overlapping and smudged elements
        negative_prompt = "text, words, speech bubbles, dialogue, captions, lettering, writing, letters, low quality, blurry, distorted proportions, bad anatomy, illegible, gibberish writing, overlapping characters, smudged details, pixelated, poor composition, cluttered scene, multiple scenes, duplicate elements, multiple copies, messy image"
        
        # Generate image
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]
        
        # Save raw image
        raw_image_path = output_dir / f"panel_{i+1:02d}_raw.png"
        image.save(raw_image_path)
        print(f"Saved raw panel to {raw_image_path}")
        
        # Add to results
        generated_images.append({
            "index": i+1,
            "scene_text": scene_text,
            "prompt": prompt,
            "image_path": str(raw_image_path),
            "image": image
        })
        
        # Add a small delay to prevent rate limiting
        time.sleep(1)
    
    # Save scene data
    scene_data = [{
        "index": scene["index"],
        "scene_text": scene["scene_text"],
        "prompt": scene["prompt"],
        "image_path": scene["image_path"]
    } for scene in generated_images]
    
    with open(output_dir / "scene_data.json", "w") as f:
        json.dump(scene_data, f, indent=2)
    
    return generated_images

def process_panels(panel_dir, num_panels, output_dir, width=768, height=768):
    """Process existing comic panels"""
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
            
            # Resize for consistency
            image = image.resize((width, height), Image.LANCZOS)
            
            # Save a copy to output directory
            output_path = output_dir / f"panel_{i+1:02d}_raw.png"
            image.save(output_path)
            
            # Add to processed panels
            processed_panels.append({
                "index": i+1,
                "scene_text": f"Panel {i+1}: {path.stem}",
                "prompt": "",
                "image_path": str(output_path),
                "image": image
            })
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    # Save scene data
    scene_data = [{
        "index": panel["index"],
        "scene_text": panel["scene_text"],
        "prompt": panel["prompt"],
        "image_path": panel["image_path"]
    } for panel in processed_panels]
    
    with open(output_dir / "scene_data.json", "w") as f:
        json.dump(scene_data, f, indent=2)
    
    return processed_panels

def create_storyboard_grid(panels, output_dir):
    """Create a grid of panels for the storyboard"""
    output_dir = Path(output_dir)
    
    # Filter valid panels
    valid_panels = [p for p in panels if p.get("image_path")]
    
    if not valid_panels:
        print("No valid panels to create grid")
        return
    
    # Determine grid size
    num_panels = len(valid_panels)
    cols = min(3, num_panels)
    rows = (num_panels + cols - 1) // cols
    
    # Get image size from first panel
    first_image_path = valid_panels[0].get("image_path")
    first_image = Image.open(first_image_path)
    panel_width, panel_height = first_image.size
    
    # Create grid image
    grid_width = cols * panel_width
    grid_height = rows * (panel_height + 60)  # Extra space for captions
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Add panels to grid
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # Slightly larger font
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid_image)
    
    for i, panel in enumerate(valid_panels):
        row = i // cols
        col = i % cols
        
        # Get panel image
        image_path = panel.get("image_path")
        image = Image.open(image_path)
        
        # Add a black border around each panel
        bordered_image = Image.new('RGB', (panel_width + 4, panel_height + 4), color='black')
        bordered_image.paste(image, (2, 2))
        
        # Paste panel with border
        grid_image.paste(bordered_image, (col * panel_width, row * (panel_height + 60)))
        
        # Add caption with more contrasting background
        caption_bg = [(col * panel_width, row * (panel_height + 60) + panel_height + 4),
                     ((col + 1) * panel_width, row * (panel_height + 60) + panel_height + 60)]
        draw.rectangle(caption_bg, fill=(240, 240, 240))  # Light gray background
        
        caption = f"Scene {panel['index']}: {panel['scene_text']}"
        caption_pos = (col * panel_width + 10, row * (panel_height + 60) + panel_height + 15)
        
        # Draw text with a subtle shadow for better readability
        draw.text((caption_pos[0]+1, caption_pos[1]+1), caption, fill=(50, 50, 50), font=font)  # Shadow
        draw.text(caption_pos, caption, fill=(0, 0, 0), font=font)  # Main text
    
    # Add a title to the storyboard
    if valid_panels:
        title_text = "AI-Generated Comic Storyboard (No Text)"
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
        except:
            title_font = font
        
        # Draw title at the bottom
        title_pos = (grid_width // 2 - 200, grid_height - 40)
        draw.rectangle([(0, grid_height - 50), (grid_width, grid_height)], fill=(200, 200, 200))
        draw.text((title_pos[0]+2, title_pos[1]+2), title_text, fill=(50, 50, 50), font=title_font)  # Shadow
        draw.text(title_pos, title_text, fill=(0, 0, 0), font=title_font)  # Main text
    
    # Save grid
    grid_path = output_dir / "storyboard_raw.png"
    grid_image.save(grid_path)
    print(f"Storyboard grid saved to {grid_path}")
    
    return grid_path

def main():
    args = parse_args()
    
    if args.input_type == "text":
        print(f"Story: {args.story}")
        scenes = split_story_into_scenes(args.story, args.num_scenes)
        
        print(f"Scenes:")
        for i, scene in enumerate(scenes):
            print(f"{i+1}. {scene}")
        
        panels = generate_panel_images(
            scenes, 
            args.style, 
            args.output_dir, 
            args.seed,
            args.inference_steps,
            args.guidance_scale,
            args.width,
            args.height,
            args.background_color,
            args.model
        )
        
    else:  # panels mode
        if not args.panel_dir:
            print("Error: --panel-dir must be specified when using --input-type panels")
            return
        
        panels = process_panels(
            args.panel_dir, 
            args.num_scenes, 
            args.output_dir,
            args.width,
            args.height
        )
    
    # Create storyboard grid
    create_storyboard_grid(panels, args.output_dir)
    
    print("Clean panel generation complete!")
    print(f"Output saved to {args.output_dir}")
    print("")
    print("Now you can add dialogue using add_dialogue.py:")
    print(f"python add_dialogue.py --input-dir \"{args.output_dir}\" --theme superhero")

if __name__ == "__main__":
    main()