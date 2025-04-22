#!/usr/bin/env python3
"""
Preprocessing script for extracting panels from comic book pages
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.ocr.panel_extractor import PanelExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocess.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract panels from comic book pages"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing comic book pages"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/frames",
        help="Directory to save extracted panels"
    )
    
    parser.add_argument(
        "--min-panel-size",
        type=int,
        default=150,
        help="Minimum panel size in pixels"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize panel extractor
    extractor = PanelExtractor(
        min_panel_size=args.min_panel_size,
        output_dir=args.output_dir
    )
    
    # Extract panels
    try:
        logger.info(f"Extracting panels from {args.input_dir} to {args.output_dir}")
        results = extractor.extract_panels_from_directory(
            args.input_dir,
            recursive=args.recursive,
            save_panels=True
        )
        
        # Count extracted panels
        total_panels = sum(len(panels) for panels in results.values())
        logger.info(f"Extracted {total_panels} panels from {len(results)} pages")
        
    except Exception as e:
        logger.error(f"Error during panel extraction: {str(e)}")
        sys.exit(1)
    
    logger.info("Panel extraction complete")

if __name__ == "__main__":
    main()