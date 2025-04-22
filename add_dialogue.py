import argparse
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Add dialogue to comic panels")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing raw comic panels")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save panels with dialogue (defaults to input-dir)")
    parser.add_argument("--scene-file", type=str, default=None,
                        help="JSON file with scene data (defaults to scene_data.json in input-dir)")
    parser.add_argument("--dialogue-file", type=str, default=None,
                        help="JSON file with dialogue data (optional)")
    parser.add_argument("--theme", type=str, default="superhero",
                        choices=["superhero", "detective", "fantasy", "scifi"],
                        help="Theme of dialogue to generate")
    return parser.parse_args()

def load_scene_data(scene_file):
    """Load scene data from JSON file"""
    try:
        with open(scene_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading scene data: {e}")
        return None

def generate_dialogue(scene_data, theme):
    """Generate dialogue based on scene descriptions and theme"""
    dialogue = {}
    
    # Theme-specific dialogue templates
    themes = {
        "superhero": {
            "speech": [
                "We need to stop them before they destroy the city!",
                "I've never seen power like this before!",
                "The fate of the world depends on us!",
                "Stand together! We can win this!",
                "Is that all you've got?",
                "I'll hold them off, you get the civilians to safety!",
                "My powers are fading... I need time to recover!",
                "They're targeting our weak points!",
                "This ends now!",
                "For justice!"
            ],
            "thought": [
                "I can't let them see I'm afraid...",
                "If I use my full power, I might hurt innocent people...",
                "There has to be another way...",
                "I've trained for this my whole life.",
                "What would my mentor say now?",
                "We're outnumbered, but not outmatched."
            ],
            "caption": [
                "Meanwhile, across the city...",
                "The battle intensifies...",
                "Little did they know, a greater threat was approaching...",
                "Our heroes face their greatest challenge yet...",
                "The world holds its breath...",
                "Only minutes remain before catastrophe strikes..."
            ]
        },
        "detective": {
            "speech": [
                "The evidence doesn't add up.",
                "I've seen this pattern before.",
                "We're missing something important here.",
                "This case goes deeper than we thought.",
                "Trust no one, especially not me.",
                "Follow the money, always follow the money.",
                "This wasn't a random killing.",
                "I think our suspect has been playing us from the start.",
                "We're running out of time!",
                "I need to see those files immediately!"
            ],
            "thought": [
                "Something doesn't feel right about this...",
                "What am I not seeing?",
                "The answer is right in front of me...",
                "If I'm right about this, we're all in danger.",
                "Trust is a luxury I can't afford right now.",
                "This conspiracy goes all the way to the top."
            ],
            "caption": [
                "Later that evening...",
                "The plot thickens...",
                "As night falls on the city...",
                "24 hours earlier...",
                "The truth, hidden in plain sight...",
                "Sometimes the most obvious answer is the right one..."
            ]
        },
        "fantasy": {
            "speech": [
                "By the ancient powers, I command thee!",
                "The prophecy spoke of this day.",
                "We must reach the sacred temple before nightfall.",
                "This sword was forged in dragon fire!",
                "The dark forces gather strength with each passing hour.",
                "Only the chosen one can wield this power.",
                "The kingdom's fate rests in our hands!",
                "The elder spirits will guide us.",
                "Beware! Dark magic corrupts all it touches!",
                "Stand back, foul creature of shadow!"
            ],
            "thought": [
                "The ancient texts never prepared me for this...",
                "My magic weakens with each spell I cast...",
                "If only I had mastered the third rune...",
                "The prophecy didn't mention this part...",
                "I can feel the dark energy calling to me...",
                "What if I'm not the hero they think I am?"
            ],
            "caption": [
                "In the depths of the forbidden forest...",
                "As darkness falls upon the realm...",
                "The ancient prophecy unfolds...",
                "Magic stirs in the forgotten places...",
                "Shadows lengthen as evil awakens...",
                "The final battle approaches..."
            ]
        },
        "scifi": {
            "speech": [
                "The quantum field is destabilizing!",
                "We've never encountered this species before.",
                "The ship's AI has been compromised!",
                "Set phasers to maximum!",
                "According to my calculations, we have 15 minutes before total system failure.",
                "The wormhole is collapsing!",
                "I'm picking up strange readings from the alien artifact.",
                "Initiate the emergency protocols!",
                "The future of humanity depends on this mission!",
                "This technology shouldn't exist for another century!"
            ],
            "thought": [
                "These readings contradict everything we know about physics...",
                "The alien consciousness is trying to communicate...",
                "If I reroute power from life support to the shields...",
                "Is humanity ready for this discovery?",
                "The timeline has been altered. Nothing is certain now.",
                "What if there are parallel versions of us making the same choice?"
            ],
            "caption": [
                "Deep in uncharted space...",
                "Earth, 2157 AD...",
                "As the colony ship approaches the anomaly...",
                "Meanwhile, in the research laboratory...",
                "The countdown to extinction begins...",
                "Technology and biology merge in unexpected ways..."
            ]
        }
    }
    
    # Character names based on theme
    character_names = {
        "superhero": [
            "Captain", "Thunderbolt", "Shadow", "Titanium", "Photon", "Quantum", 
            "Steel", "Blaze", "Storm", "Vortex", "Apex", "Zenith", "Omega"
        ],
        "detective": [
            "Detective", "Inspector", "Chief", "Agent", "Commissioner", "Lieutenant", 
            "Sergeant", "Officer", "Investigator", "Marshal", "Sheriff", "Constable"
        ],
        "fantasy": [
            "Wizard", "Knight", "Ranger", "Paladin", "Druid", "Sorcerer", 
            "Bard", "Cleric", "Warrior", "Mage", "Oracle", "Slayer", "Warden"
        ],
        "scifi": [
            "Commander", "Admiral", "Lieutenant", "Doctor", "Engineer", "Specialist", 
            "Captain", "Pilot", "Navigator", "Major", "Scientist", "Technician", "Operative"
        ]
    }
    
    used_character_names = []
    theme_dialogues = themes.get(theme, themes["superhero"])
    theme_characters = character_names.get(theme, character_names["superhero"])
    
    # Generate dialogue for each scene
    for scene in scene_data:
        panel_idx = scene["index"]
        scene_text = scene["scene_text"].lower()
        
        # Initialize dialogue for this panel
        dialogue[str(panel_idx)] = {}
        
        # Determine number of speech bubbles (1-3)
        num_bubbles = random.randint(2, 3)
        
        # Generate character names for this panel
        panel_characters = []
        for _ in range(min(num_bubbles, 3)):
            available_names = [n for n in theme_characters if n not in used_character_names]
            if not available_names:
                used_character_names = []
                available_names = theme_characters
            
            name = random.choice(available_names)
            used_character_names.append(name)
            panel_characters.append(name)
        
        # Add speech bubbles
        for i in range(num_bubbles - 1):  # Reserve last one for caption or thought
            bubble_idx = str(i + 1)
            
            text = random.choice(theme_dialogues["speech"])
            
            # Add character name to dialogue
            if random.random() < 0.4:
                # Sometimes add a character name at the beginning
                char_idx = (i + 1) % len(panel_characters)
                other_char = panel_characters[char_idx]
                text = f"{other_char}! {text}"
            
            # Calculate position - avoid center (where characters often are)
            x = random.randint(150, 600) 
            y = random.randint(100, 200)
            
            if i == 1:  # Second bubble often goes on the opposite side
                x = 768 - x
                y = random.randint(150, 250)
            
            # Add to dialogue
            dialogue[str(panel_idx)][bubble_idx] = {
                "text": text,
                "type": "speech",
                "position": [x, y]
            }
        
        # Add a caption or thought bubble as the last one
        if random.random() < 0.4:
            bubble_type = "caption"
            text = random.choice(theme_dialogues["caption"])
            x = 384  # Center
            y = 700  # Bottom
        else:
            bubble_type = "thought"
            text = random.choice(theme_dialogues["thought"])
            x = 600
            y = 300
        
        dialogue[str(panel_idx)][str(num_bubbles)] = {
            "text": text,
            "type": bubble_type,
            "position": [x, y]
        }
    
    return dialogue

def add_text_bubble(draw, text, position, bubble_type="speech", 
                    font=None, small_font=None, image_size=(768, 768),
                    overlay_opacity=0.9):
    """Add a text bubble to an image with improved styling"""
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
    try:
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in wrapped_lines]
        text_height = sum(line_heights) + (len(wrapped_lines) - 1) * 5
        text_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in wrapped_lines])
    except:
        # Fallback if textbbox is not available (older PIL versions)
        line_heights = [14 for _ in wrapped_lines]  # Approximate
        text_height = sum(line_heights) + (len(wrapped_lines) - 1) * 5
        text_width = max([len(line) * 7 for line in wrapped_lines])  # Approximate width
    
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
        # Speech bubble (rounded rectangle with improved styling)
        # First draw a slightly larger black bubble for the border
        draw.rounded_rectangle([x0-2, y0-2, x1+2, y1+2], radius=20, fill=(0, 0, 0))
        # Then draw the white inner bubble
        draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(255, 255, 255, int(255 * overlay_opacity)))
        
        # Add speech pointer with better styling
        if y > image_size[1] // 2:  # Bubble is in the lower half, point up
            pointer_start = (x, y0)
            pointer_mid = (x - 20, y0 - 25)
            pointer_end = (x - 40, y0 - 15)
        else:  # Bubble is in the upper half, point down
            pointer_start = (x, y1)
            pointer_mid = (x - 20, y1 + 25)
            pointer_end = (x - 40, y1 + 15)
        
        # Draw pointer outline and fill
        draw.polygon([pointer_start, pointer_mid, pointer_end], fill=(255, 255, 255, int(255 * overlay_opacity)), outline=(0, 0, 0))
        
    elif bubble_type.lower() in ["thought", "thinking"]:
        # Thought bubble (cloud shape with improved styling)
        # First draw a slightly larger black bubble for the border
        draw.rounded_rectangle([x0-2, y0-2, x1+2, y1+2], radius=25, fill=(0, 0, 0))
        # Then draw the white inner bubble
        draw.rounded_rectangle([x0, y0, x1, y1], radius=23, fill=(255, 255, 255, int(255 * overlay_opacity)))
        
        # Add thought circles with improved styling
        circle_sizes = [10, 7, 5]
        if y > image_size[1] // 2:  # Bubble is in the lower half, circles go up
            cx, cy = x, y0 - 15
            for size in circle_sizes:
                # Draw circle outline
                draw.ellipse([cx-size-1, cy-size-1, cx+size+1, cy+size+1], fill=(0, 0, 0))
                # Draw white circle
                draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=(255, 255, 255, int(255 * overlay_opacity)))
                cy -= size * 2 + 2
                cx -= size
        else:  # Bubble is in the upper half, circles go down
            cx, cy = x, y1 + 15
            for size in circle_sizes:
                # Draw circle outline
                draw.ellipse([cx-size-1, cy-size-1, cx+size+1, cy+size+1], fill=(0, 0, 0))
                # Draw white circle
                draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=(255, 255, 255, int(255 * overlay_opacity)))
                cy += size * 2 + 2
                cx -= size
    
    elif bubble_type.lower() in ["caption", "narration"]:
        # Caption box (rectangle with improved styling)
        # Draw a caption box with a slight yellow tint
        draw.rectangle([x0-2, y0-2, x1+2, y1+2], fill=(0, 0, 0))
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 220, int(255 * overlay_opacity)))
    
    else:
        # Default to speech bubble
        draw.rounded_rectangle([x0-2, y0-2, x1+2, y1+2], radius=20, fill=(0, 0, 0))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(255, 255, 255, int(255 * overlay_opacity)))
    
    # Draw text with a slight shadow for readability
    current_y = y0 + padding
    for i, line in enumerate(wrapped_lines):
        try:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = text_bbox[2]
        except:
            # Fallback for older PIL versions
            line_width = len(line) * 7  # Approximate
        
        text_x = x - line_width // 2
        
        # Draw text shadow
        draw.text((text_x+1, current_y+1), line, fill=(100, 100, 100), font=font)
        # Draw main text
        draw.text((text_x, current_y), line, fill=(0, 0, 0), font=font)
        
        try:
            if i < len(line_heights):
                current_y += line_heights[i] + 5
            else:
                current_y += 20  # Fallback
        except:
            current_y += 20  # Fallback for older PIL versions

def add_dialogue_to_panels(scene_data, dialogue_data, input_dir, output_dir):
    """Add dialogue to comic panels"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each panel
    for scene in scene_data:
        panel_idx = scene["index"]
        panel_dialogue = dialogue_data.get(str(panel_idx), {})
        
        if not panel_dialogue:
            print(f"No dialogue found for panel {panel_idx}, skipping")
            continue
        
        # Load the raw image
        raw_image_path = os.path.join(input_dir, f"panel_{panel_idx:02d}_raw.png")
        if not os.path.exists(raw_image_path):
            print(f"Raw image not found: {raw_image_path}")
            # Try alternative pattern for filename
            raw_image_path = os.path.join(input_dir, f"panel_{panel_idx}_raw.png")
            if not os.path.exists(raw_image_path):
                print(f"Still couldn't find image, skipping panel {panel_idx}")
                continue
        
        # Open and create a copy of the image
        try:
            image = Image.open(raw_image_path).convert("RGBA")
        except Exception as e:
            print(f"Error opening image {raw_image_path}: {e}")
            continue
        
        # Create a transparent overlay for text bubbles
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Load font
        try:
            # Try to load a comic-style font
            font_path = "comic.ttf"  # Update this path to your comic font
            font = ImageFont.truetype(font_path, 20)
            small_font = ImageFont.truetype(font_path, 14)
        except:
            try:
                # Fallback to common system fonts
                if os.name == 'nt':  # Windows
                    font = ImageFont.truetype("arial.ttf", 20)
                    small_font = ImageFont.truetype("arial.ttf", 14)
                elif os.name == 'posix':  # macOS/Linux
                    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 20)
                    small_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 14)
                else:
                    font = ImageFont.load_default()
                    small_font = font
            except:
                font = ImageFont.load_default()
                small_font = font
        
        # Add text bubbles to overlay
        for bubble_idx, bubble_data in panel_dialogue.items():
            bubble_type = bubble_data.get("type", "speech")
            text = bubble_data.get("text", "")
            position = bubble_data.get("position", [image.width // 2, 100])
            
            # Skip if no text
            if not text:
                continue
            
            # Add text bubble to overlay
            add_text_bubble(draw, text, position, bubble_type, font, small_font, image.size)
        
        # Composite the overlay onto the original image
        composite = Image.alpha_composite(image, overlay)
        
        # Convert back to RGB for saving as PNG
        final_image = composite.convert("RGB")
        
        # Save the image with text bubbles
        output_path = os.path.join(output_dir, f"panel_{panel_idx:02d}.png")
        final_image.save(output_path)
        print(f"Saved panel with dialogue to {output_path}")
    
    print(f"Added dialogue to {len(scene_data)} panels")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set output directory to input directory if not specified
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Set scene file to default if not specified
    if args.scene_file is None:
        args.scene_file = os.path.join(args.input_dir, "scene_data.json")
    
    # Load scene data
    scene_data = load_scene_data(args.scene_file)
    if not scene_data:
        print("No scene data found. Please provide a valid scene data file.")
        return
    
    # Generate or load dialogue
    if args.dialogue_file:
        try:
            with open(args.dialogue_file, 'r') as f:
                dialogue_data = json.load(f)
            print(f"Loaded dialogue from {args.dialogue_file}")
        except Exception as e:
            print(f"Error loading dialogue file: {e}")
            print("Generating dialogue automatically")
            dialogue_data = generate_dialogue(scene_data, args.theme)
    else:
        print("Generating dialogue automatically")
        dialogue_data = generate_dialogue(scene_data, args.theme)
    
    # Save generated dialogue
    dialogue_path = os.path.join(args.output_dir, "dialogue.json")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(dialogue_path, 'w') as f:
            json.dump(dialogue_data, f, indent=2)
        print(f"Saved dialogue to {dialogue_path}")
    except Exception as e:
        print(f"Error saving dialogue: {e}")
    
    # Add dialogue to panels
    add_dialogue_to_panels(scene_data, dialogue_data, args.input_dir, args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()