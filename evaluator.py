"""
Comic Evaluation Script

Evaluates comic panels with visual consistency, image quality, and narrative coherence metrics.
"""

import os
import argparse
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Try importing specialized libraries with fallbacks
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except ImportError:
    print("Warning: scikit-image not found. Using simple similarity metric instead.")
    HAVE_SKIMAGE = False

try:
    import torch
    import torchvision
    HAVE_TORCH = True
except ImportError:
    print("Warning: torch/torchvision not found. Some metrics will be unavailable.")
    HAVE_TORCH = False

class ComicEvaluator:
    """Evaluates comic panels using multiple metrics"""
    
    def __init__(self, images_dir="comic_panels", results_dir="results"):
        """
        Initialize the comic evaluator.
        
        Args:
            images_dir: Directory containing the comic panels to evaluate
            results_dir: Directory where evaluation results will be saved
        """
        self.images_dir = images_dir
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Evaluator initialized. Looking for images in {images_dir}")
        print(f"Results will be saved to {results_dir}")
    
    def load_images(self):
        """
        Load panel images from the images directory.
        
        Returns:
            List of loaded images
        """
        images = []
        image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.startswith("panel_") and f.endswith((".png", ".jpg", ".jpeg"))
        ])
        
        for file in image_files:
            try:
                image_path = os.path.join(self.images_dir, file)
                img = Image.open(image_path).convert('RGB')
                images.append(img)
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        print(f"Loaded {len(images)} panel images")
        return images
    
    def evaluate_visual_consistency(self):
        """
        Evaluate visual consistency between panels using SSIM or a simpler metric.
        
        Returns:
            Dictionary with consistency scores
        """
        print("Evaluating visual consistency...")
        
        # Load images
        images = self.load_images()
        
        if len(images) < 2:
            print("Need at least 2 images to evaluate consistency")
            return {"error": "Insufficient images"}
        
        # Convert to numpy arrays
        np_images = []
        for img in images:
            np_img = np.array(img)
            np_images.append(np_img)
        
        # Calculate similarity between consecutive panels
        similarity_scores = []
        for i in range(len(np_images)-1):
            img1 = np_images[i]
            img2 = np_images[i+1]
            
            # Resize second image if dimensions don't match
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Use SSIM if available, otherwise use simpler metric
            if HAVE_SKIMAGE:
                # Convert to grayscale for SSIM
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                score = ssim(img1_gray, img2_gray, data_range=255)
            else:
                # Simple normalized MSE as a fallback
                mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
                max_mse = 255**2  # Maximum possible MSE
                score = 1.0 - (mse / max_mse)  # Convert to similarity (0-1)
            
            similarity_scores.append(float(score))
        
        # Calculate average score
        avg_score = np.mean(similarity_scores)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(similarity_scores)+1), similarity_scores)
        plt.axhline(y=avg_score, color='r', linestyle='-', label=f'Average: {avg_score:.3f}')
        plt.xlabel('Panel Pair')
        plt.ylabel('Similarity Score')
        plt.title('Visual Consistency Between Adjacent Panels')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        visual_path = os.path.join(self.results_dir, 'visual_consistency.png')
        plt.savefig(visual_path)
        plt.close()
        
        # Save results as JSON
        results = {
            "average_similarity": float(avg_score),
            "min_similarity": float(min(similarity_scores)) if similarity_scores else 0,
            "max_similarity": float(max(similarity_scores)) if similarity_scores else 0,
            "panel_pair_scores": similarity_scores
        }
        
        with open(os.path.join(self.results_dir, 'visual_consistency.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Visual consistency: Average similarity score = {avg_score:.3f}")
        print(f"Visual consistency plot saved to {visual_path}")
        
        return results
    
    def calculate_image_quality(self):
        """
        Calculate basic image quality metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        print("Calculating image quality metrics...")
        
        # Load images
        images = self.load_images()
        
        if not images:
            return {"error": "No images found"}
        
        # Calculate contrast and sharpness for each image
        contrast_scores = []
        for img in images:
            # Convert to grayscale
            img_gray = np.array(img.convert('L'))
            
            # Calculate contrast (standard deviation)
            contrast = float(np.std(img_gray))
            contrast_scores.append(contrast)
        
        # Calculate statistics
        avg_contrast = np.mean(contrast_scores)
        
        # Save results
        results = {
            "average_contrast": float(avg_contrast),
            "contrast_scores": [float(score) for score in contrast_scores]
        }
        
        with open(os.path.join(self.results_dir, 'image_quality.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Image quality: Average contrast = {avg_contrast:.3f}")
        
        return results
    
    def get_user_feedback(self):
        """
        Collect user feedback on the comic quality.
        
        Returns:
            Dictionary with feedback scores
        """
        print("\n===== Comic Evaluation Feedback =====")
        
        # Define questions
        questions = {
            "visual_appeal": "Rate the visual appeal (1-10): ",
            "narrative_flow": "Rate the narrative flow (1-10): ",
            "character_consistency": "Rate character consistency (1-10): ",
            "dialogue_quality": "Rate dialogue quality (1-10): ",
            "overall_satisfaction": "Rate overall satisfaction (1-10): "
        }
        
        # Collect ratings
        feedback = {}
        for key, question in questions.items():
            while True:
                try:
                    rating = float(input(question))
                    if 1 <= rating <= 10:
                        feedback[key] = rating
                        break
                    else:
                        print("Please enter a value between 1 and 10")
                except ValueError:
                    print("Please enter a numerical value")
        
        # Get comments
        feedback["comments"] = input("Any additional comments? ")
        
        # Calculate average score
        scores = [v for k, v in feedback.items() if isinstance(v, (int, float))]
        feedback["average_score"] = sum(scores) / len(scores)
        
        # Save feedback
        with open(os.path.join(self.results_dir, 'user_feedback.json'), 'w') as f:
            json.dump(feedback, f, indent=2)
        
        print(f"User feedback: Average score = {feedback['average_score']:.3f}")
        
        return feedback
    
    def evaluate_narrative_coherence(self, story_data=None):
        """
        Evaluate narrative coherence based on story data and user input.
        
        Args:
            story_data: Dictionary containing panel prompts and dialogues
            
        Returns:
            Dictionary with coherence scores
        """
        if not story_data:
            print("No story data provided. Skipping narrative coherence evaluation.")
            return {"error": "No story data provided"}
        
        # Display story for evaluation
        print("\n===== Narrative Coherence Evaluation =====")
        
        panels = story_data.get("panels", [])
        for i, panel in enumerate(panels):
            print(f"\nPanel {i+1}:")
            print(f"Prompt: {panel.get('prompt', '')}")
            
            dialogues = panel.get("dialogues", [])
            if dialogues:
                print("Dialogues:")
                for dialogue in dialogues:
                    print(f"  {dialogue.get('character', '')}: {dialogue.get('text', '')}")
        
        # Collect ratings
        print("\nPlease rate the following aspects of narrative coherence (1-10):")
        
        questions = {
            "logical_flow": "Logical flow between panels: ",
            "character_consistency": "Character consistency: ",
            "dialogue_relevance": "Dialogue relevance to scenes: ",
            "story_completeness": "Story completeness: ",
            "theme_adherence": "Adherence to theme: "
        }
        
        ratings = {}
        for key, question in questions.items():
            while True:
                try:
                    rating = float(input(question))
                    if 1 <= rating <= 10:
                        ratings[key] = rating
                        break
                    else:
                        print("Please enter a value between 1 and 10")
                except ValueError:
                    print("Please enter a numerical value")
        
        # Calculate average score
        ratings["average_score"] = sum(ratings.values()) / len(ratings)
        
        # Save ratings
        with open(os.path.join(self.results_dir, 'narrative_coherence.json'), 'w') as f:
            json.dump(ratings, f, indent=2)
        
        print(f"Narrative coherence: Average score = {ratings['average_score']:.3f}")
        
        return ratings
    
    def run_all_metrics(self, real_images_dir=None, story_data=None):
        """
        Run all available metrics and combined evaluation.
        
        Args:
            real_images_dir: Directory with real comic images for comparison (optional)
            story_data: Dictionary with panel prompts and dialogues (optional)
            
        Returns:
            Dictionary with all evaluation results
        """
        # 1. Evaluate visual consistency
        consistency_results = self.evaluate_visual_consistency()
        
        # 2. Calculate image quality
        quality_results = self.calculate_image_quality()
        
        # 3. Combine results
        evaluation_results = {
            "visual_consistency": consistency_results,
            "image_quality": quality_results
        }
        
        # Save comprehensive results
        with open(os.path.join(self.results_dir, 'all_metrics.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\nEvaluation complete! Results saved to {self.results_dir}")
        
        return evaluation_results

def main():
    """Main function when running the script directly"""
    parser = argparse.ArgumentParser(description="Evaluate comic quality with multiple metrics")
    parser.add_argument("--images_dir", type=str, default="comic_panels", help="Directory with comic panels")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--real_images_dir", type=str, default=None, help="Directory with real comic images for FID")
    parser.add_argument("--story_file", type=str, default=None, help="Path to story data JSON file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ComicEvaluator(args.images_dir, args.results_dir)
    
    # Load story data if provided
    story_data = None
    if args.story_file and os.path.exists(args.story_file):
        try:
            with open(args.story_file, 'r') as f:
                story_data = json.load(f)
                print(f"Loaded story data from {args.story_file}")
        except Exception as e:
            print(f"Error loading story data: {e}")
    
    # Run evaluation
    evaluator.run_all_metrics(args.real_images_dir, story_data)
    
    # If interactive mode, get user feedback and narrative coherence
    if args.interactive:
        evaluator.get_user_feedback()
        
        if story_data:
            evaluator.evaluate_narrative_coherence(story_data)

if __name__ == "__main__":
    main()