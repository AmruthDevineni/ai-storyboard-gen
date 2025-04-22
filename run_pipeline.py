#!/usr/bin/env python3
"""
Main pipeline orchestrator for the AI-powered storyboard generator
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import yaml
from PIL import Image

from app.ocr.text_extraction import TextExtractor
from app.captioning.image_captioner import ImageCaptioner
from app.prompt_splitter.story_segmenter import StorySegmenter
from app.generation.image_generator import ImageGenerator
from app.evaluation.metrics import StoryboardEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('storyboard_generator.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}


def load_models_config(config_path: str = "config/models.yaml") -> Dict:
    """Load models configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading models configuration: {str(e)}")
        return {}


def init_pipeline_components(config: Dict, models_config: Dict) -> Dict:
    """Initialize pipeline components based on configuration"""
    components = {}
    
    try:
        # Initialize OCR component
        if config["pipeline"]["run_ocr"]:
            logger.info("Initializing OCR component")
            components["ocr"] = TextExtractor(
                lang=config["ocr"]["lang"],
                use_gpu=config["ocr"]["use_gpu"],
                confidence_threshold=config["ocr"]["confidence_threshold"]
            )
        
        # Initialize captioning component
        if config["pipeline"]["run_captioning"]:
            logger.info("Initializing captioning component")
            components["captioner"] = ImageCaptioner(
                model_name=models_config["captioning"]["model_name"],
                use_gpu=config["device"] == "cuda",
                max_length=models_config["captioning"]["config"]["max_length"],
                min_length=models_config["captioning"]["config"]["min_length"],
                num_beams=models_config["captioning"]["config"]["num_beams"]
            )
        
        # Initialize prompt splitter component
        if config["pipeline"]["run_prompt_breakdown"]:
            logger.info("Initializing prompt splitter component")
            components["segmenter"] = StorySegmenter(
                method=config["prompt_breakdown"]["method"],
                model_name=models_config["prompt_splitter"]["model_name"],
                min_scenes=config["prompt_breakdown"]["min_scenes"],
                max_scenes=config["prompt_breakdown"]["max_scenes"],
                scene_prompt_template=config["prompt_breakdown"]["scene_prompt_template"],
                style_prompt=config["prompt_breakdown"]["style_prompt"],
                use_gpu=config["device"] == "cuda"
            )
        
        # Initialize image generator component
        if config["pipeline"]["run_image_generation"]:
            logger.info("Initializing image generator component")
            components["generator"] = ImageGenerator(
                model_name=models_config["image_generation"]["model_name"],
                use_gpu=config["device"] == "cuda",
                output_dir=config["project"]["output_dir"],
                num_inference_steps=models_config["image_generation"]["config"]["num_inference_steps"],
                guidance_scale=models_config["image_generation"]["config"]["guidance_scale"],
                negative_prompt=models_config["image_generation"]["config"]["negative_prompt"],
                use_chaining=config["image_generation"]["use_chaining"],
                seed=config["project"]["seed"],
                image_size=tuple(config["data"]["image_size"])
            )
        
        # Initialize evaluator component
        if config["pipeline"]["run_evaluation"]:
            logger.info("Initializing evaluator component")
            components["evaluator"] = StoryboardEvaluator(
                use_gpu=config["device"] == "cuda",
                semantic_model=models_config["semantic_similarity"]["model_name"],
                metrics_file=config["evaluation"]["metrics_file"]
            )
    
    except Exception as e:
        logger.error(f"Error initializing pipeline components: {str(e)}")
        raise
    
    return components


def process_story_text(
    story_text: str,
    components: Dict,
    config: Dict
) -> Dict:
    """Process story text input"""
    results = {
        "input_type": "text",
        "story_text": story_text,
        "scenes": [],
        "storyboard": [],
        "metrics": None
    }
    
    try:
        # Segment story into scenes
        logger.info("Segmenting story into scenes")
        segmenter = components.get("segmenter")
        if not segmenter:
            raise ValueError("Story segmenter not initialized")
            
        scenes = segmenter.segment_story(story_text)
        results["scenes"] = scenes
        
        # Generate images
        logger.info("Generating storyboard images")
        generator = components.get("generator")
        if not generator:
            raise ValueError("Image generator not initialized")
            
        storyboard = generator.generate_storyboard(scenes)
        results["storyboard"] = storyboard
        
        # Evaluate storyboard
        if len(storyboard) > 1 and components.get("evaluator"):
            logger.info("Evaluating storyboard quality")
            evaluator = components.get("evaluator")
            metrics = evaluator.evaluate_storyboard(storyboard)
            results["metrics"] = metrics
            
        logger.info("Story processing complete")
        
    except Exception as e:
        logger.error(f"Error processing story text: {str(e)}")
        raise
    
    return results


def process_comic_panels(
    panel_dir: str,
    components: Dict,
    config: Dict
) -> Dict:
    """Process comic panel input"""
    results = {
        "input_type": "panels",
        "panel_dir": panel_dir,
        "extracted_texts": {},
        "captions": {},
        "scenes": [],
        "storyboard": [],
        "metrics": None
    }
    
    try:
        panel_dir_path = Path(panel_dir)
        if not panel_dir_path.exists():
            raise FileNotFoundError(f"Panel directory not found: {panel_dir}")
        
        # Find image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        panel_paths = sorted([
            str(p) for p in panel_dir_path.glob("*") 
            if p.suffix.lower() in image_extensions
        ])
        
        if not panel_paths:
            raise ValueError(f"No image files found in directory: {panel_dir}")
            
        logger.info(f"Found {len(panel_paths)} panel images")
        
        # Extract text using OCR
        ocr = components.get("ocr")
        if ocr:
            logger.info("Extracting text from panels using OCR")
            for path in panel_paths:
                panel_name = Path(path).name
                try:
                    results["extracted_texts"][panel_name] = ocr.extract_text_from_image(path)
                except Exception as e:
                    logger.error(f"Error extracting text from {panel_name}: {str(e)}")
                    results["extracted_texts"][panel_name] = []
        
        # Generate captions
        captioner = components.get("captioner")
        if captioner:
            logger.info("Generating captions for panels")
            for path in panel_paths:
                panel_name = Path(path).name
                try:
                    results["captions"][panel_name] = captioner.caption_image(path)
                except Exception as e:
                    logger.error(f"Error captioning {panel_name}: {str(e)}")
                    results["captions"][panel_name] = ""
        
        # Create scenes from captions and extracted text
        logger.info("Creating scenes from captions and extracted text")
        scenes = []
        for i, path in enumerate(panel_paths):
            panel_name = Path(path).name
            caption = results["captions"].get(panel_name, "")
            texts = results["extracted_texts"].get(panel_name, [])
            dialogue = " ".join([text["text"] for text in texts]) if texts else ""
            
            scene = {
                "scene_number": i + 1,
                "original_panel": path,
                "scene_text": caption,
                "dialogue": dialogue,
                "scene_prompt": f"A comic panel showing {caption}, comic book style"
            }
            scenes.append(scene)
        
        results["scenes"] = scenes
        
        # Generate new images
        logger.info("Generating storyboard images")
        generator = components.get("generator")
        if not generator:
            raise ValueError("Image generator not initialized")
            
        storyboard = generator.generate_storyboard(scenes)
        results["storyboard"] = storyboard
        
        # Evaluate storyboard
        if len(storyboard) > 1 and components.get("evaluator"):
            logger.info("Evaluating storyboard quality")
            evaluator = components.get("evaluator")
            
            # Use original panels as reference for FID
            reference_images = [Image.open(scene["original_panel"]) for scene in scenes]
            metrics = evaluator.evaluate_storyboard(storyboard, reference_images)
            results["metrics"] = metrics
            
        logger.info("Panel processing complete")
        
    except Exception as e:
        logger.error(f"Error processing comic panels: {str(e)}")
        raise
    
    return results


def save_results(results: Dict, output_dir: str):
    """Save processing results to disk"""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_dir}")
    
    # Save scene data
    scenes_path = output_dir_path / "scenes.json"
    with open(scenes_path, "w") as f:
        # Create serializable scene data
        serializable_scenes = []
        for scene in results["scenes"]:
            scene_copy = scene.copy()
            # Remove non-serializable fields
            if "image" in scene_copy:
                scene_copy.pop("image")
            serializable_scenes.append(scene_copy)
        
        json.dump(serializable_scenes, f, indent=2)
    
    # Save metrics if available
    if results.get("metrics"):
        metrics_path = output_dir_path / "metrics.json"
        with open(metrics_path, "w") as f:
            # Create serializable metrics data
            serializable_metrics = {}
            for key, value in results["metrics"].items():
                if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                    serializable_metrics[key] = value
                else:
                    serializable_metrics[key] = str(value)
            
            json.dump(serializable_metrics, f, indent=2)
    
    # Save input story text if available
    if results.get("story_text"):
        story_path = output_dir_path / "story.txt"
        with open(story_path, "w") as f:
            f.write(results["story_text"])
    
    logger.info("Results saved successfully")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI-Powered Storyboard Generator"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--models-config",
        type=str,
        default="config/models.yaml",
        help="Path to models configuration file"
    )
    
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["text", "panels"],
        default=None,
        help="Input type: 'text' for story text, 'panels' for comic panels"
    )
    
    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help="Story text input or path to a text file containing the story"
    )
    
    parser.add_argument(
        "--panel-dir",
        type=str,
        default=None,
        help="Directory containing comic panel images"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--min-scenes",
        type=int,
        default=None,
        help="Minimum number of scenes"
    )
    
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes"
    )
    
    parser.add_argument(
        "--style-prompt",
        type=str,
        default=None,
        help="Style prompt for image generation"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    models_config = load_models_config(args.models_config)
    
    if not config or not models_config:
        logger.error("Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # Override configuration with command line arguments
    if args.input_type:
        config["data"]["input_type"] = args.input_type
    
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    
    if args.min_scenes:
        config["prompt_breakdown"]["min_scenes"] = args.min_scenes
    
    if args.max_scenes:
        config["prompt_breakdown"]["max_scenes"] = args.max_scenes
    
    if args.style_prompt:
        config["prompt_breakdown"]["style_prompt"] = args.style_prompt
    
    # Initialize pipeline components
    try:
        components = init_pipeline_components(config, models_config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline components: {str(e)}")
        sys.exit(1)
    
    # Process input based on type
    input_type = config["data"]["input_type"]
    
    try:
        if input_type == "text":
            # Get story text
            story_text = args.story
            
            # If story is a file path, read from file
            if story_text and os.path.isfile(story_text):
                with open(story_text, 'r') as f:
                    story_text = f.read()
            
            if not story_text:
                # Use example story if none provided
                logger.info("No story text provided. Using example story.")
                story_text = "A young hero finds a magical sword, battles a dragon, and returns to his village as a legend."
            
            # Process story text
            results = process_story_text(story_text, components, config)
            
        elif input_type == "panels":
            # Get panel directory
            panel_dir = args.panel_dir or config["data"]["panel_dir"]
            
            if not panel_dir or not os.path.isdir(panel_dir):
                logger.error(f"Panel directory not found: {panel_dir}")
                sys.exit(1)
            
            # Process comic panels
            results = process_comic_panels(panel_dir, components, config)
        
        else:
            logger.error(f"Invalid input type: {input_type}")
            sys.exit(1)
        
        # Save results
        output_dir = config["project"]["output_dir"]
        save_results(results, output_dir)
        
        logger.info(f"Processing complete. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()