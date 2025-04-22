Enhanced AI Comic Generator
This enhanced version of the AI comic generator allows you to create comic panels with proper text bubbles and dialogue. It solves the problem of AI-generated gibberish text by letting you add your own dialogue to the generated panels.

Features
Generate comic panels from text stories using AI
Process existing comic panels from your dataset
Add custom speech bubbles, thought bubbles, and captions
Interactive dialogue creation mode
Support for custom dialogue files in JSON format
Customizable panel size and generation quality
Installation
Make sure you have Python 3.8+ installed
Install dependencies:
bash
pip install torch torchvision diffusers transformers pillow numpy matplotlib
Usage
Basic Usage
Generate a comic with 6 panels and interactive dialogue creation:

bash
python comic_generator.py --input-type text --story "The Avengers assemble on a rooftop as alien ships appear in the sky. Iron Man blasts an alien ship while Captain America coordinates the team's defense. Black Widow and Hawkeye fight side by side against ground troops. Thor summons lightning to destroy multiple enemies at once. Hulk leaps into the air to punch a massive alien mothership. The team stands victorious as the alien threat retreats." --num-scenes 6 --style "professional Marvel comics style, dynamic action poses, detailed facial expressions, vibrant colors, clean linework, dramatic lighting" --add-text-bubbles --output-dir "outputs/avengers_comic"
Use Pre-defined Dialogue
Create a comic with dialogue defined in a JSON file:

bash
python comic_generator.py --input-type text --story "The Avengers battle an alien invasion" --num-scenes 6 --style "marvel comics style" --add-text-bubbles --dialogue-file "sample_dialogue.json" --output-dir "outputs/avengers_comic"
Process Existing Panels
Create a comic from your existing panel dataset:

bash
python comic_generator.py --input-type panels --panel-dir "data/frames" --num-scenes 6 --add-text-bubbles --output-dir "outputs/custom_comic"
Customizing Image Generation
Fine-tune the panel generation:

bash
python comic_generator.py --input-type text --story "Sci-fi adventure" --num-scenes 4 --style "retro sci-fi comic style" --width 600 --height 800 --inference-steps 50 --guidance-scale 8.5 --output-dir "outputs/scifi_comic"
Dialogue JSON Format
The dialogue JSON file has the following structure:

json
{
  "1": {
    "1": {
      "text": "This is the dialogue for the first bubble in panel 1",
      "type": "speech",
      "position": [256, 100]
    },
    "2": {
      "text": "This is the second bubble in panel 1",
      "type": "thought",
      "position": [400, 200]
    }
  },
  "2": {
    "1": {
      "text": "This is dialogue for panel 2",
      "type": "speech",
      "position": [256, 100]
    }
  }
}
The outer keys represent panel numbers (starting from 1)
For each panel, inner keys represent bubble numbers (starting from 1)
Each bubble has:
text: The dialogue text
type: One of "speech", "thought", or "caption"
position: [x, y] coordinates for the bubble center
Interactive Dialogue Mode
If you don't provide a dialogue file, the script will enter interactive mode, asking you to input dialogue for each panel:

Enter the text for each dialogue bubble
Specify the bubble type (speech, thought, or caption)
Set the position of the bubble on the panel
Press Enter without text to finish adding bubbles to the current panel
Output
The script will create:

Raw panel images without text (panel_01_raw.png, etc.)
Panels with text bubbles (panel_01.png, etc.)
A combined storyboard image (storyboard.png)
Data files with scene and dialogue information (scene_data.json, dialogue.json)
All output is saved to the specified output directory.

