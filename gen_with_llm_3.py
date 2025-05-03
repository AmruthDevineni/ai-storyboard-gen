import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import cv2
import numpy as np
import textwrap
import argparse
import os
import json
import math
import requests
import random
import time
from pathlib import Path

# Import our metrics module
try:
    from comic_metrics import ComicEvaluator
except ImportError:
    print("Warning: ComicEvaluator module not found. Metrics will not be available.")
    ComicEvaluator = None

class OllamaStoryGenerator:
    """
    Uses local Ollama (Llama2 or Llama3) to generate panel prompts and dialogues.
    """
    def __init__(self, model_name="llama3", host="http://localhost:11434"):
        """
        Initialize the Ollama story generator.
        
        Args:
            model_name: Name of the Ollama model (llama3 or llama2)
            host: URL for Ollama API
        """
        self.model_name = model_name
        self.host = host
        
        # Default system prompt for Llama
        self.system_prompt = """
        You are an expert comic writer for Peanuts-style comics. Your task is to convert a general story idea into a detailed comic strip description. You'll create specific prompts for each panel and the dialogue for each character.

        Guidelines:
        1. Each panel should have a clear visual prompt describing the scene, characters, and their expressions.
        2. Each panel should have no more than 2 dialogue lines.
        3. Make sure the dialogue flows naturally from panel to panel to tell a coherent story.
        4. Match dialogue to characters based on their personalities (Charlie Brown is anxious/philosophical, Snoopy is imaginative/confident, Lucy is bossy, Linus is intellectual).
        5. Ensure dialogue is properly attributed (e.g., Character: "Dialogue").
        6. Keep dialogue concise to fit speech bubbles (max 20-30 words per bubble).
        7. When characters are mentioned in the visual prompt, they should be the same ones who speak in that panel.

        ALWAYS FORMAT YOUR RESPONSE AS A VALID JSON OBJECT with the following structure:
        {
            "panels": [
                {
                    "prompt": "Detailed visual description for the panel",
                    "dialogues": [
                        {"character": "Character Name", "text": "Dialogue text"},
                        {"character": "Character Name", "text": "Dialogue text"}
                    ]
                },
                ...
            ]
        }

        Limit to at most 2 dialogues per panel for clarity.
        """
    
    def generate_story(self, theme, num_panels):
        """
        Generate panel prompts and dialogues based on a theme using Ollama.
        
        Args:
            theme: User's theme or general story idea
            num_panels: Number of panels to generate
            
        Returns:
            Dictionary with panel prompts and dialogues
        """
        # Create user prompt
        user_prompt = f"""
        Create a Peanuts comic strip with {num_panels} panels based on the following theme:
        
        Theme: "{theme}"
        
        Generate the visual prompt for each panel and the character dialogues. Remember to limit to at most 2 dialogues per panel.
        
        Format your response as a valid JSON object following this structure:
        {{
            "panels": [
                {{
                    "prompt": "Detailed visual description for the panel",
                    "dialogues": [
                        {{"character": "Character Name", "text": "Dialogue text"}},
                        {{"character": "Character Name", "text": "Dialogue text"}}
                    ]
                }},
                ...
            ]
        }}
        """
        
        try:
            # Try to call Ollama API
            return self._call_ollama(user_prompt, num_panels)
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            # Fallback to mock implementation
            return self._mock_generate_story(theme, num_panels)
    
    def _call_ollama(self, user_prompt, num_panels):
        """
        Call the Ollama API to generate the story.
        
        Args:
            user_prompt: User prompt containing theme and requirements
            num_panels: Number of panels to generate
            
        Returns:
            Dictionary with panel prompts and dialogues
        """
        api_url = f"{self.host}/api/generate"
        
        data = {
            "model": self.model_name,
            "prompt": f"{self.system_prompt}\n\n{user_prompt}",
            "stream": False
        }
        
        try:
            response = requests.post(api_url, json=data)
            response.raise_for_status()
            
            # Extract the generated response
            result = response.json()
            content = result["response"]
            
            # Try to extract JSON from the response
            try:
                # Find JSON content (it might be surrounded by markdown code blocks or other text)
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    story_data = json.loads(json_content)
                    
                    # Ensure we don't exceed requested panel count
                    if "panels" in story_data and len(story_data["panels"]) > num_panels:
                        story_data["panels"] = story_data["panels"][:num_panels]
                    
                    return story_data
                else:
                    print("No valid JSON found in Ollama response")
                    return self._mock_generate_story(user_prompt, num_panels)
            except json.JSONDecodeError:
                print("Failed to parse JSON response from Ollama. Using fallback.")
                return self._mock_generate_story(user_prompt, num_panels)
            
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return self._mock_generate_story(user_prompt, num_panels)
    
    def _mock_generate_story(self, theme, num_panels):
        """
        Generate a mock story for testing/development.
        
        Args:
            theme: User's theme or general story idea
            num_panels: Number of panels to generate
            
        Returns:
            Dictionary with panel prompts and dialogues
        """
        print("Using mock story generator (Ollama connection failed)")
        
        # Some sample themes and corresponding stories
        if "baseball" in theme.lower():
            theme_type = "baseball"
        elif "philosophy" in theme.lower() or "meaning of life" in theme.lower():
            theme_type = "philosophy"
        elif "kite" in theme.lower():
            theme_type = "kite"
        elif "friendship" in theme.lower():
            theme_type = "friendship"
        else:
            theme_type = "general"
        
        # Base templates based on theme
        templates = {
            "baseball": [
                {
                    "prompt": "Charlie Brown standing on pitcher's mound looking hopeful",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "I think this year our team might actually win a game."}
                    ]
                },
                {
                    "prompt": "Charlie Brown winding up to pitch the ball while teammates watch",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "This time I'll throw the perfect pitch!"}
                    ]
                },
                {
                    "prompt": "Charlie Brown looking disheveled after a bad pitch, with teammates in background",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "Good grief! Why can't I ever get it right?"}
                    ]
                },
                {
                    "prompt": "Charlie Brown and Linus sitting against a wall, looking contemplative",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "Will I ever win a baseball game?"},
                        {"character": "Linus", "text": "There's always next season, Charlie Brown."}
                    ]
                }
            ],
            "philosophy": [
                {
                    "prompt": "Charlie Brown and Snoopy sitting under a tree on a sunny day",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "Do you ever wonder about the meaning of life, Snoopy?"}
                    ]
                },
                {
                    "prompt": "Close-up of Snoopy looking thoughtful with his paw on his chin",
                    "dialogues": [
                        {"character": "Snoopy", "text": "..."}
                    ]
                },
                {
                    "prompt": "Charlie Brown looking worried while Snoopy watches",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "I've been thinking about this for years and still haven't figured it out."},
                        {"character": "Snoopy", "text": "*sigh*"}
                    ]
                },
                {
                    "prompt": "Charlie Brown and Snoopy walking away from the tree, side by side",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "Maybe the meaning of life is just having a friend to share your questions with."}
                    ]
                }
            ],
            "general": [
                {
                    "prompt": "Charlie Brown and Snoopy sitting together",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "Sometimes I wonder what the future holds, Snoopy."}
                    ]
                },
                {
                    "prompt": "Snoopy on top of his doghouse typing",
                    "dialogues": [
                        {"character": "Snoopy", "text": "It was a dark and stormy night..."}
                    ]
                },
                {
                    "prompt": "Charlie Brown talking to Lucy at her psychiatric booth",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "I just can't figure it out."},
                        {"character": "Lucy", "text": "That'll be five cents please."}
                    ]
                },
                {
                    "prompt": "Charlie Brown and Snoopy watching the sunset",
                    "dialogues": [
                        {"character": "Charlie Brown", "text": "In the end, it's the little moments that matter most."}
                    ]
                }
            ]
        }
        
        # Get base template for the theme
        base_template = templates.get(theme_type, templates["general"])
        
        # If we need more panels than in the template, repeat or generate variations
        panels = []
        for i in range(num_panels):
            if i < len(base_template):
                panels.append(base_template[i])
            else:
                # For additional panels, use modulo to repeat with slight variations
                base_idx = i % len(base_template)
                base_panel = base_template[base_idx]
                
                # Add slight variation to the prompt
                prompt = base_panel["prompt"] + " (variation)"
                
                # Same dialogue structure but alternate characters if possible
                dialogues = []
                for d in base_panel["dialogues"]:
                    if d["character"] == "Charlie Brown":
                        char = "Linus" if random.random() > 0.5 else "Charlie Brown"
                    elif d["character"] == "Linus":
                        char = "Lucy" if random.random() > 0.5 else "Linus"
                    elif d["character"] == "Lucy":
                        char = "Snoopy" if random.random() > 0.5 else "Lucy"
                    else:
                        char = d["character"]
                        
                    text = f"Variation of: {d['text']}"
                    dialogues.append({"character": char, "text": text})
                
                panels.append({"prompt": prompt, "dialogues": dialogues})
        
        return {"panels": panels}

class PeanutsComicGenerator:
    def __init__(self, model_path, device="cuda", ollama_model="llama3", ollama_host="http://localhost:11434"):
        """
        Initialize the comic generator with your fine-tuned model.
        
        Args:
            model_path: Path to your fine-tuned peanuts_finetuned_sd_no_text model
            device: Device to run inference on ("cuda" or "cpu")
            ollama_model: Ollama model to use (llama3 or llama2)
            ollama_host: Ollama API host
        """
        self.device = device
        print(f"Initializing model from {model_path}")
        
        # Load the model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.pipe = self.pipe.to(device)
        print("Model loaded successfully")
        
        # Initialize Ollama story generator
        self.story_generator = OllamaStoryGenerator(ollama_model, ollama_host)
        
        # Font size for speech bubbles
        self.font_size = 36  # Moderate font size for clean readability
        
        # Create directory structure
        self.panels_dir = "comic_panels"
        self.videos_dir = "comic_videos"
        os.makedirs(self.panels_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
    
    def generate_panel(self, prompt, num_inference_steps=50, guidance_scale=8.5):
        """
        Generate a single comic panel using the fine-tuned model.
        
        Args:
            prompt: Text prompt for the panel
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale (increased for stronger adherence to prompt)
            
        Returns:
            PIL Image of the generated panel
        """
        # Add specific instructions for panel generation
        enhanced_prompt = f"peanuts comic style, black and white comic strip panel, {prompt}, no speech bubbles, no dialogue bubbles, no text"
        
        # Use a stronger negative prompt
        negative_prompt = "speech bubbles, text bubbles, dialogue, words, captions, empty bubbles, white circles, white ovals"
        
        print(f"Generating panel with prompt: {enhanced_prompt}")
        
        # Generate the image with higher guidance scale for more prompt adherence
        with torch.no_grad():
            image = self.pipe(
                enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
        return image
    
    def create_smooth_speech_bubble(self, size, radius=40):
        """
        Create a smooth speech bubble mask with rounded corners.
        
        Args:
            size: (width, height) of the bubble
            radius: Corner radius
            
        Returns:
            PIL Image mask for the bubble
        """
        width, height = size
        
        # Create an image for the bubble
        bubble = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(bubble)
        
        # Draw rounded rectangle
        draw.rounded_rectangle([(0, 0), (width, height)], fill=255, radius=radius)
        
        # Apply slight blur for smoother edges
        bubble = bubble.filter(ImageFilter.GaussianBlur(radius=1))
        
        return bubble
    
    def draw_speech_bubble(self, img, text, position, panel_img_size, char_idx=0, total_chars=1):
        """
        Draw a clean speech bubble with readable text.
        
        Args:
            img: PIL Image to draw on
            text: Text to display
            position: (x, y) position for the bubble
            panel_img_size: (width, height) of the panel
            char_idx: Index of the character speaking (0 for first, 1 for second)
            total_chars: Total number of characters speaking
            
        Returns:
            Bubble bounds (x, y, w, h)
        """
        draw = ImageDraw.Draw(img)
        margin = 20
        panel_width, panel_height = panel_img_size
        
        # Try to load a font - using a clean font (no bold)
        try:
            # First try Comic Sans
            font = ImageFont.truetype("Comic_Sans_MS.ttf", self.font_size, encoding="unic")
        except IOError:
            try:
                # Try Arial
                font = ImageFont.truetype("Arial.ttf", self.font_size, encoding="unic")
            except IOError:
                try:
                    # Try DejaVu Sans
                    font = ImageFont.truetype("DejaVuSans.ttf", self.font_size, encoding="unic")
                except IOError:
                    # Fallback to default
                    font = ImageFont.load_default()
        
        # Handle special case for very short text
        if len(text) <= 10:
            # Get text size
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Create a reasonably sized bubble
            padding = 30  # Padding inside the bubble
            bubble_width = text_width + (padding * 2)
            bubble_height = text_height + (padding * 2)
            
            # Ensure minimum size for visibility
            bubble_width = max(bubble_width, 150)
            bubble_height = max(bubble_height, 80)
        else:
            # For normal text, use word wrapping
            # Adaptive chars per line
            chars_per_line = max(15, 200 // (self.font_size // 10))
            
            # Word wrap the text
            wrapper = textwrap.TextWrapper(width=chars_per_line, break_long_words=True)
            lines = wrapper.wrap(text)
            
            # Calculate text dimensions
            max_line_width = 0
            total_height = 0
            line_heights = []
            
            for line in lines:
                bbox = font.getbbox(line)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                max_line_width = max(max_line_width, width)
                line_heights.append(height)
                total_height += height
            
            # Add spacing between lines
            line_spacing = max(5, self.font_size // 6)  # Adaptive line spacing
            total_height += (len(lines) - 1) * line_spacing
            
            # Calculate bubble size with padding
            padding = max(20, self.font_size // 2)  # Adaptive padding
            bubble_width = max_line_width + (padding * 2)
            bubble_height = total_height + (padding * 2)
        
        # Position bubbles to avoid overlapping
        # For two dialogues in one panel, position them side by side
        x, y = position
        if total_chars == 2:
            if char_idx == 0:
                # First bubble (left side)
                bubble_x = panel_width - bubble_width - margin
                bubble_y = margin
            else:
                # Second bubble (right side)
                bubble_x = margin
                bubble_y = panel_height - bubble_height - margin
        else:
            # Single dialogue - center it
            bubble_x = (panel_width - bubble_width) // 2
            bubble_y = margin
        
        # Make sure bubble doesn't go off the panel
        bubble_x = max(margin, min(bubble_x, panel_width - bubble_width - margin))
        bubble_y = max(margin, min(bubble_y, panel_height // 2))
        
        # Create and draw the bubble
        bubble_mask = self.create_smooth_speech_bubble((bubble_width, bubble_height), radius=30)
        bubble_bg = Image.new("RGB", (bubble_width, bubble_height), "white")
        
        # Paste bubble
        img.paste(bubble_bg, (bubble_x, bubble_y), bubble_mask)
        
        # Draw black outline
        outline_size = 3
        outline_mask = self.create_smooth_speech_bubble(
            (bubble_width + outline_size*2, bubble_height + outline_size*2), 
            radius=30
        )
        outline_bg = Image.new("RGB", (bubble_width + outline_size*2, bubble_height + outline_size*2), "black")
        img.paste(outline_bg, (bubble_x - outline_size, bubble_y - outline_size), outline_mask)
        img.paste(bubble_bg, (bubble_x, bubble_y), bubble_mask)
        
        # Draw text
        if len(text) <= 10:
            # Simple centered text for short strings
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = bubble_x + (bubble_width - text_width) // 2
            text_y = bubble_y + (bubble_height - text_height) // 2
            draw.text((text_x, text_y), text, fill="black", font=font)
        else:
            # Multi-line text
            wrapper = textwrap.TextWrapper(width=chars_per_line, break_long_words=True)
            lines = wrapper.wrap(text)
            
            # Draw text line by line
            current_y = bubble_y + padding
            line_spacing = max(5, self.font_size // 6)
            
            for line in lines:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                
                # Center text horizontally in bubble
                line_x = bubble_x + (bubble_width - line_width) // 2
                
                # Draw text - SINGLE PASS, no overlapping
                draw.text((line_x, current_y), line, fill="black", font=font)
                
                # Move to next line with spacing
                current_y += line_height + line_spacing
        
        return (bubble_x, bubble_y, bubble_width, bubble_height)
    
    def add_dialogue_to_panel(self, panel, dialogues):
        """
        Add dialogue with clean speech bubbles to the panel image.
        
        Args:
            panel: PIL Image of the panel
            dialogues: List of dictionaries containing character and text
            
        Returns:
            PIL Image with dialogue and speech bubbles added
        """
        # Make a copy of the panel with border for better appearance
        bordered_panel = ImageOps.expand(panel, border=3, fill='black')
        result = bordered_panel.copy()
        
        # Get panel dimensions
        panel_width, panel_height = result.size
        
        # Process each dialogue
        texts_to_add = []
        for dialogue in dialogues:
            character = dialogue.get("character", "")
            text = dialogue.get("text", "")
            
            # Skip empty text
            if not text:
                continue
                
            # Add to the list of texts to add
            texts_to_add.append(text)
        
        # Position the speech bubbles to avoid overlap
        # Limit to 2 dialogues per panel for cleanliness
        texts_to_add = texts_to_add[:min(2, len(texts_to_add))]
        total_texts = len(texts_to_add)
        
        for i, text in enumerate(texts_to_add):
            # Position bubbles to avoid overlapping
            position = (20, 20)  # Default position
            
            # Draw bubble with clean text
            self.draw_speech_bubble(
                result,
                text,
                position,
                (panel_width, panel_height),
                i,  # Character index
                total_texts  # Total characters speaking
            )
        
        return result
    
    def create_video_slideshow(self, panels, output_path=None, seconds_per_panel=4):
        """
        Create a video slideshow from the panels.
        
        Args:
            panels: List of panel images
            output_path: Path to save the output video (if None, uses default path)
            seconds_per_panel: How long to display each panel
            
        Returns:
            Path to the saved video
        """
        if not panels:
            return None
        
        # If no output path specified, create one in the videos directory
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.videos_dir, f"comic_slideshow_{timestamp}.mp4")
        else:
            # Ensure output path is in the videos directory
            if not os.path.dirname(output_path):
                output_path = os.path.join(self.videos_dir, output_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get first panel dimensions
        height, width = np.array(panels[0]).shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (mp4v for .mp4)
        video = cv2.VideoWriter(output_path, fourcc, 1, (width, height))
        
        # Add each panel to the video for the specified duration
        for panel in panels:
            # Convert PIL to OpenCV format
            panel_cv = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
            
            # Add this panel for the specified number of frames (at 1 fps)
            for _ in range(seconds_per_panel):
                video.write(panel_cv)
        
        # Release the video writer
        video.release()
        
        print(f"Slideshow video saved to {output_path}")
        return output_path
    
    def create_comic_from_story(self, theme, num_panels, output_path=None, seconds_per_panel=4, run_evaluation=True):
        """
        Create a slideshow video based on a theme and story generated by Ollama.
        
        Args:
            theme: User's theme or general story idea
            num_panels: Number of panels to generate
            output_path: Path to save the output video
            seconds_per_panel: How long to display each panel
            run_evaluation: Whether to run evaluation metrics on the generated comic
            
        Returns:
            Path to the saved video
        """
        # If no output path specified, create one in the videos directory
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.videos_dir, f"comic_{timestamp}.mp4")
        elif not os.path.dirname(output_path):
            # If just a filename is provided, put it in videos directory
            output_path = os.path.join(self.videos_dir, output_path)
        
        print(f"Creating {num_panels}-panel comic based on theme: {theme}")
        
        # Generate story with Ollama
        story = self.story_generator.generate_story(theme, num_panels)
        
        if not story or "panels" not in story or not story["panels"]:
            print("Failed to generate story. Using default theme.")
            # Create a default story
            story = {
                "panels": [
                    {
                        "prompt": "Charlie Brown sitting under a tree looking thoughtful",
                        "dialogues": [
                            {"character": "Charlie Brown", "text": "Sometimes I wonder about life..."}
                        ]
                    }
                ] * num_panels
            }
        
        # Generate all panels
        panels = []
        raw_panels = []  # Store raw panels without dialogue
        
        # Limit to actual number of panels generated
        actual_panels = min(num_panels, len(story["panels"]))
        
        for i in range(actual_panels):
            panel_data = story["panels"][i]
            prompt = panel_data["prompt"]
            dialogues = panel_data.get("dialogues", [])
            
            print(f"Generating panel {i+1}/{actual_panels}: {prompt}")
            
            # Generate the panel
            panel = self.generate_panel(prompt)
            raw_panels.append(panel.copy())  # Save raw panel for evaluation
            
            # Save the raw panel first
            raw_panel_path = os.path.join(self.panels_dir, f"panel_{i+1:02d}_raw.png")
            panel.save(raw_panel_path)
            
            # Add dialogue with speech bubbles to the panel
            panel_with_dialogue = self.add_dialogue_to_panel(panel, dialogues)
            
            panels.append(panel_with_dialogue)
            
            # Save the panel with dialogue
            panel_path = os.path.join(self.panels_dir, f"panel_{i+1:02d}.png")
            panel_with_dialogue.save(panel_path)
            
            # Print dialogue for reference
            print(f"  Dialogues:")
            for d in dialogues:
                character = d.get("character", "")
                text = d.get("text", "")
                if character and text:
                    print(f"    {character}: {text}")
        
        # Save the story data for future reference
        story_path = os.path.join(self.panels_dir, "story_data.json")
        with open(story_path, 'w') as f:
            json.dump(story, f, indent=2)
        
        # Create the video slideshow
        video_path = self.create_video_slideshow(panels, output_path, seconds_per_panel)
        
        # Run evaluation if requested and metrics module is available
        if run_evaluation and ComicEvaluator is not None:
            print("\nRunning evaluation metrics...")
            evaluator = ComicEvaluator(images_dir=self.panels_dir)
            evaluation_results = evaluator.evaluate_visual_consistency()
            
            print(f"Visual consistency score: {evaluation_results.get('average_ssim', 0):.3f}")
            
            # Ask if user wants to provide feedback
            if input("\nWould you like to provide feedback on the comic? (y/n): ").lower() == 'y':
                user_feedback = evaluator.get_user_feedback()
                evaluation_results["user_feedback"] = user_feedback
                
                # Only evaluate narrative coherence if story data is provided
                coherence_ratings = evaluator.evaluate_narrative_coherence(story)
                evaluation_results["narrative_coherence"] = coherence_ratings
            
            print(f"Evaluation complete! Results saved to {evaluator.results_dir}")
        
        return video_path, story

def main():
    parser = argparse.ArgumentParser(description="Generate Peanuts comic slideshow video")
    parser.add_argument("--theme", type=str, help="Theme or general story idea")
    parser.add_argument("--panels", type=int, default=4, help="Number of panels")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    parser.add_argument("--model_path", type=str, default="./peanuts_finetuned_sd_no_text", help="Path to fine-tuned model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--ollama_model", type=str, default="llama3", help="Ollama model name (llama3 or llama2)")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--font_size", type=int, default=36, help="Font size for dialogue text")
    parser.add_argument("--seconds_per_panel", type=int, default=4, help="Seconds to display each panel in the video")
    parser.add_argument("--run_evaluation", action="store_true", help="Run evaluation metrics after generation")
    
    args = parser.parse_args()
    
    # Create the generator
    generator = PeanutsComicGenerator(
        args.model_path, 
        args.device, 
        args.ollama_model,
        args.ollama_host
    )
    
    # Set font size if provided
    if args.font_size:
        generator.font_size = args.font_size
    
    # Interactive mode if no theme provided
    if not args.theme:
        print("===== Peanuts Comic Slideshow Generator =====")
        
        # Check if Ollama is running
        try:
            test_url = f"{args.ollama_host}/api/tags"
            response = requests.get(test_url)
            if response.status_code == 200:
                available_models = response.json().get("models", [])
                model_names = [model.get("name") for model in available_models]
                
                if model_names:
                    print(f"Available Ollama models: {', '.join(model_names)}")
                    if args.ollama_model not in model_names:
                        print(f"Warning: {args.ollama_model} not found. Using built-in templates.")
                else:
                    print("No models found in Ollama. Using built-in templates.")
            else:
                print("Ollama detected but couldn't get available models. Using built-in templates.")
        except:
            print("Warning: Couldn't connect to Ollama. Using built-in templates.")
        
        theme = input("Enter the theme or story idea for your comic: ")
        
        try:
            num_panels = int(input("How many panels would you like? (1-20): ") or "4")
            num_panels = max(1, min(20, num_panels))
        except ValueError:
            num_panels = 4
            print("Using default: 4 panels")
        
        output_path = input("Output video filename (default: auto-generated): ") or args.output
        
        # Ask for font size
        try:
            font_size = int(input(f"Font size for dialogue (default: {generator.font_size}): ") or str(generator.font_size))
            if font_size > 0:
                generator.font_size = font_size
        except ValueError:
            pass  # Keep default font size
            
        # Ask for seconds per panel
        try:
            seconds_per_panel = int(input(f"Seconds to show each panel (default: {args.seconds_per_panel}): ") or str(args.seconds_per_panel))
            if seconds_per_panel > 0:
                args.seconds_per_panel = seconds_per_panel
        except ValueError:
            pass  # Keep default
        
        # Ask about evaluation
        run_evaluation = input("Run evaluation metrics after generation? (y/n): ").lower() == 'y'
    else:
        theme = args.theme
        num_panels = args.panels
        output_path = args.output
        run_evaluation = args.run_evaluation
    
    # Create the comic slideshow
    video_path, story = generator.create_comic_from_story(
        theme, num_panels, output_path, args.seconds_per_panel, run_evaluation
    )
    
    print(f"\nComic slideshow created successfully!")
    print(f"Video saved to: {video_path}")
    print(f"Individual panels saved in the '{generator.panels_dir}' folder")
    
    if run_evaluation and ComicEvaluator is not None:
        print("\nYou can run a comprehensive evaluation with:")
        print(f"  python comic_metrics.py --images_dir {generator.panels_dir} --story_file {os.path.join(generator.panels_dir, 'story_data.json')}")

if __name__ == "__main__":
    main()