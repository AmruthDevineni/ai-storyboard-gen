import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import textwrap
import json
import re
import random
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Comic Storyboard Generator")
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
    parser.add_argument("--inference-steps", type=int, default=50,  # Increased from 30 to 50
                        help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=8.5,  # Increased from 7.5 to 8.5
                        help="Guidance scale for text-to-image generation")
    parser.add_argument("--width", type=int, default=768,  # Increased from 512 to 768
                        help="Image width")
    parser.add_argument("--height", type=int, default=768,  # Increased from 512 to 768
                        help="Image height")
    parser.add_argument("--add-text-bubbles", action="store_true",
                        help="Add text bubbles to panels")
    parser.add_argument("--dialogue-file", type=str, default=None,
                        help="JSON file containing dialogue for panels")
    parser.add_argument("--auto-dialogue", action="store_true",
                        help="Automatically generate dialogue based on scene content")
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
    return f"A comic panel showing {scene_text}, with clear space around characters, {style_prompt}, clear composition with proper spacing and {background_color} background, single scene with no overlap"

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
        prompt = format_scene_prompt(scene_text, style_prompt, background_color)
        
        # Generate image
        # Enhanced negative prompt to avoid overlapping and smudged elements
        image = pipe(
            prompt,
            negative_prompt="low quality, blurry, distorted proportions, bad anatomy, text, illegible, gibberish writing, overlapping characters, smudged details, pixelated, poor composition, cluttered scene, multiple scenes, duplicate elements, multiple copies, messy image",
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

def auto_generate_dialogue(panels):
    """Automatically generate dialogue for comic panels based on scene content"""
    print("Automatically generating dialogue based on scene content...")
    
    # Character names to use in dialogue
    character_names = [
        "Hero", "Captain", "Commander", "Ace", "Chief", "Star", "Doc", 
        "Spark", "Nova", "Flash", "Shadow", "Iron", "Bolt", "Thunder",
        "Agent", "Hawk", "Eagle", "Wolf", "Tiger", "Lion", "Viper"
    ]
    
    # Common dialogue templates for different scene types
    action_dialogue = [
        "Look out!",
        "We need to move, now!",
        "This ends here!",
        "Stand back, I'll handle this!",
        "Follow my lead!",
        "Is that all you've got?",
        "Watch your six!",
        "This is our only chance!",
        "Let's finish this!",
        "Get down!"
    ]
    
    dramatic_dialogue = [
        "I never thought it would come to this...",
        "So it was you all along.",
        "This changes everything.",
        "I can't believe what I'm seeing.",
        "We have to make a choice.",
        "I knew this day would come.",
        "There's something you need to know.",
        "This is just the beginning.",
        "Everything has led to this moment.",
        "This is bigger than all of us."
    ]
    
    victory_dialogue = [
        "We did it!",
        "That was close.",
        "It's finally over.",
        "We make a good team.",
        "That wasn't so hard, was it?",
        "Mission accomplished.",
        "That'll teach them.",
        "Is everyone okay?",
        "I knew we could do it!",
        "Let's head back."
    ]
    
    thought_templates = [
        "I need to be careful here...",
        "What if this is a trap?",
        "There must be another way.",
        "I can't let them see I'm worried.",
        "This reminds me of last time.",
        "I hope the others are safe.",
        "If only they knew the truth.",
        "I never expected it to be this hard.",
        "I'm running out of options.",
        "This might be my last chance."
    ]
    
    caption_templates = [
        "Meanwhile...",
        "Later that day...",
        "The battle continues...",
        "Moments earlier...",
        "As the dust settles...",
        "On the other side of town...",
        "Little did they know...",
        "The adventure continues...",
        "What happens next will shock everyone...",
        "The tide of battle shifts..."
    ]
    
    dialogue = {}
    used_names = []
    
    # Generate dialogue for each panel
    for panel in panels:
        panel_idx = panel["index"]
        scene_text = panel["scene_text"].lower()
        dialogue[str(panel_idx)] = {}
        
        # Determine number of bubbles (1-3)
        num_bubbles = random.randint(1, 3)
        
        # Determine scene type based on keywords
        scene_type = "neutral"
        if any(word in scene_text for word in ["battle", "fight", "attack", "defend", "blast", "punch", "leap"]):
            scene_type = "action"
        elif any(word in scene_text for word in ["discover", "reveal", "secret", "mystery", "find"]):
            scene_type = "dramatic"
        elif any(word in scene_text for word in ["victory", "win", "defeat", "success", "retreat", "triumph"]):
            scene_type = "victory"
        
        # Assign character names
        for i in range(num_bubbles):
            bubble_num = str(i + 1)
            
            # Choose character name that hasn't been used yet
            available_names = [name for name in character_names if name not in used_names]
            if not available_names:  # If we've used all names, reset
                used_names = []
                available_names = character_names
            
            name = random.choice(available_names)
            used_names.append(name)
            
            # Determine bubble type and content
            bubble_type = "speech"
            if i == num_bubbles - 1 and random.random() < 0.3:  # 30% chance for last bubble to be thought/caption
                if random.random() < 0.5:
                    bubble_type = "thought"
                else:
                    bubble_type = "caption"
            
            # Generate dialogue based on scene type and bubble type
            if bubble_type == "speech":
                if scene_type == "action":
                    text = random.choice(action_dialogue)
                elif scene_type == "dramatic":
                    text = random.choice(dramatic_dialogue)
                elif scene_type == "victory":
                    text = random.choice(victory_dialogue)
                else:
                    # Mix of all dialogue types
                    text = random.choice(action_dialogue + dramatic_dialogue + victory_dialogue)
                
                # Add name to dialogue sometimes
                if random.random() < 0.5:
                    other_name = random.choice([n for n in used_names if n != name])
                    text = f"{other_name}! {text}"
            
            elif bubble_type == "thought":
                text = random.choice(thought_templates)
            
            else:  # caption
                text = random.choice(caption_templates)
            
            # Calculate position - avoid center (where characters often are)
            # Position bubbles toward the top and sides
            if i == 0:
                # First bubble usually top left
                x = random.randint(int(panel["image"].width * 0.2), int(panel["image"].width * 0.4))
                y = random.randint(50, 100)
            elif i == 1:
                # Second bubble usually top right
                x = random.randint(int(panel["image"].width * 0.6), int(panel["image"].width * 0.8))
                y = random.randint(50, 150)
            else:
                # Third bubble somewhere else (often bottom)
                x = random.randint(int(panel["image"].width * 0.3), int(panel["image"].width * 0.7))
                y = random.randint(int(panel["image"].height * 0.7), int(panel["image"].height * 0.9))
            
            # Add bubble to dialogue
            dialogue[str(panel_idx)][bubble_num] = {
                "text": text,
                "type": bubble_type,
                "position": [x, y]
            }
    
    return dialogue

def add_text_bubbles(panels, dialogue_data, output_dir):
    """Add text bubbles to comic panels"""
    output_dir = Path(output_dir)
    
    # Load dialogue data
    if isinstance(dialogue_data, str):
        try:
            with open(dialogue_data, "r") as f:
                dialogue = json.load(f)
        except Exception as e:
            print(f"Error loading dialogue file: {str(e)}")
            # If no dialogue file, interactively ask for dialogue
            dialogue = create_dialogue_interactively(panels)
    else:
        dialogue = dialogue_data or create_dialogue_interactively(panels)
    
    # Save dialogue data
    with open(output_dir / "dialogue.json", "w") as f:
        json.dump(dialogue, f, indent=2)
    
    # Process each panel
    for panel in panels:
        panel_idx = panel["index"]
        panel_dialogue = dialogue.get(str(panel_idx), {})
        
        if not panel_dialogue:
            continue
        
        # Get the image
        image = panel["image"].copy()
        draw = ImageDraw.Draw(image)
        
        # Try to load font
        try:
            # Attempt to load a comic-style font
            font_path = "comic.ttf"  # Change to your comic font path if available
            font = ImageFont.truetype(font_path, 20)
            small_font = ImageFont.truetype(font_path, 14)
        except:
            # Fallback to default font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                small_font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
                small_font = font
        
        # Add text bubbles
        for bubble_idx, bubble_data in panel_dialogue.items():
            bubble_type = bubble_data.get("type", "speech")
            text = bubble_data.get("text", "")
            position = bubble_data.get("position", [50, 50])
            
            # Skip if no text
            if not text:
                continue
            
            # Add text bubble
            add_text_bubble(draw, text, position, bubble_type, font, small_font, image.size)
        
        # Save the image with text bubbles
        output_path = output_dir / f"panel_{panel_idx:02d}.png"
        image.save(output_path)
        panel["text_image_path"] = str(output_path)
    
    return panels

def add_text_bubble(draw, text, position, bubble_type="speech", 
                   font=None, small_font=None, image_size=(768, 768)):
    """Add a text bubble to an image"""
    if not text:
        return
    
    # Default font if none provided
    if font is None:
        font = ImageFont.load_default()
    
    if small_font is None:
        small_font = font
    
    # Wrap text
    width = min(300, image_size[0] - 40)
    wrapped_lines = textwrap.wrap(text, width=30)
    
    # Calculate text size
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in wrapped_lines]
    text_height = sum(line_heights) + (len(wrapped_lines) - 1) * 5
    text_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in wrapped_lines])
    
    # Calculate bubble size
    padding = 15
    bubble_width = text_width + padding * 2
    bubble_height = text_height + padding * 2
    
    # Calculate position
    x, y = position
    # Ensure bubble stays within image bounds
    x = min(max(bubble_width // 2 + 10, x), image_size[0] - bubble_width // 2 - 10)
    y = min(max(bubble_height // 2 + 10, y), image_size[1] - bubble_height // 2 - 10)
    
    # Calculate bubble coordinates
    x0 = x - bubble_width // 2
    y0 = y - bubble_height // 2
    x1 = x + bubble_width // 2
    y1 = y + bubble_height // 2
    
    # Draw bubble shape based on type
    if bubble_type.lower() in ["speech", "dialog"]:
        # Speech bubble (rounded rectangle)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=15, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        
        # Add speech pointer
        pointer_start = (x, y1)
        pointer_mid = (x - 15, y1 + 20)
        pointer_end = (x - 30, y1 + 10)
        draw.polygon([pointer_start, pointer_mid, pointer_end], fill=(255, 255, 255), outline=(0, 0, 0))
        
    elif bubble_type.lower() in ["thought", "thinking"]:
        # Thought bubble (cloud shape)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=20, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        
        # Add thought circles
        circle_sizes = [8, 6, 4]
        cx, cy = x, y1 + 15
        for size in circle_sizes:
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], 
                        fill=(255, 255, 255), outline=(0, 0, 0))
            cy += size * 2 + 2
            cx -= size
    
    elif bubble_type.lower() in ["caption", "narration"]:
        # Caption box (rectangle)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 235), outline=(0, 0, 0), width=2)
    
    else:
        # Default to speech bubble
        draw.rounded_rectangle([x0, y0, x1, y1], radius=15, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    
    # Draw text
    current_y = y0 + padding
    for i, line in enumerate(wrapped_lines):
        text_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = text_bbox[2]
        text_x = x - line_width // 2
        draw.text((text_x, current_y), line, fill=(0, 0, 0), font=font)
        if i < len(line_heights):
            current_y += line_heights[i] + 5
        else:
            current_y += 20  # Fallback if line_heights index is out of range

def create_dialogue_interactively(panels):
    """Create dialogue for panels interactively"""
    dialogue = {}
    
    print("\n=== Comic Panel Dialogue Editor ===")
    print("Add dialogue to your comic panels:\n")
    
    for panel in panels:
        panel_idx = panel["index"]
        print(f"\nPanel {panel_idx}: {panel['scene_text']}")
        dialogue[str(panel_idx)] = {}
        
        bubble_count = 1
        while True:
            print(f"\nBubble {bubble_count} for Panel {panel_idx}:")
            text = input("Enter dialogue text (or leave empty to finish this panel): ")
            
            if not text:
                break
            
            bubble_type = input("Bubble type (speech/thought/caption) [speech]: ").lower()
            if not bubble_type:
                bubble_type = "speech"
            
            print(f"Position X (0-{panel['image'].width}) [center]: (place away from main characters)")
            position_x = input(f"Position X (0-{panel['image'].width}) [center]: ")
            
            print(f"Position Y (0-{panel['image'].height}) [100]: (higher values = lower on panel)")
            position_y = input(f"Position Y (0-{panel['image'].height}) [100]: ")
            
            try:
                x = int(position_x) if position_x else panel['image'].width // 2
                y = int(position_y) if position_y else 100
                position = [x, y]
            except:
                position = [panel['image'].width // 2, 100]
            
            dialogue[str(panel_idx)][str(bubble_count)] = {
                "text": text,
                "type": bubble_type,
                "position": position
            }
            
            bubble_count += 1
    
    return dialogue

def create_storyboard_grid(panels, output_dir):
    """Create a grid of panels for the storyboard"""
    output_dir = Path(output_dir)
    
    # Filter valid panels
    valid_panels = [p for p in panels if p.get("text_image_path") or p.get("image_path")]
    
    if not valid_panels:
        print("No valid panels to create grid")
        return
    
    # Determine grid size
    num_panels = len(valid_panels)
    cols = min(3, num_panels)
    rows = (num_panels + cols - 1) // cols
    
    # Get image size from first panel
    first_image_path = valid_panels[0].get("text_image_path") or valid_panels[0].get("image_path")
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
        image_path = panel.get("text_image_path") or panel.get("image_path")
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
        title_text = "AI-Generated Comic Storyboard"
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
    grid_path = output_dir / "storyboard.png"
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
    
    # Add text bubbles
    if args.add_text_bubbles:
        if args.dialogue_file and not args.auto_dialogue:
            panels = add_text_bubbles(panels, args.dialogue_file, args.output_dir)
        else:
            # Default to auto-dialogue
            print("Generating auto-dialogue for panels...")
            dialogue = auto_generate_dialogue(panels)
            panels = add_text_bubbles(panels, dialogue, args.output_dir)
    
    # Create storyboard grid
    create_storyboard_grid(panels, args.output_dir)
    
    print("Comic generation complete!")
    print(f"Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()