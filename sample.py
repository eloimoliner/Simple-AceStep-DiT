"""
Simplified ACE-Step 1.5 inference for NVIDIA GPUs (>16GB VRAM).

Generates music from text prompts with different models and configurations.
Models are auto-downloaded on first run.

Usage:
    python sample_simple.py [model] [prompt_index]

    Models: turbo (default), base
    Prompts: 0-5 (default: 0)

Examples:
    python sample_simple.py                  # Turbo + Ambient Piano
    python sample_simple.py turbo 0         # Turbo + Ambient Piano
    python sample_simple.py base 1          # Base + Upbeat Electronic
    python sample_simple.py turbo -l        # List all available prompts

Prompts:
    0: Ambient Piano
    1: Upbeat Electronic
    2: Jazz Trio
    3: Classical Orchestra
    4: Indie Folk
    5: Rock Ballad
"""

import os
import sys
import torch

import os
import torch


# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(SAVE_DIR, exist_ok=True)

# Verify NVIDIA GPU
DEVICE = "cuda"
if not torch.cuda.is_available():
    print("Error: NVIDIA GPU required but not detected.")
    sys.exit(1)

mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"Using device: {DEVICE} ({mem_gb:.1f}GB VRAM)\n")

checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# Test prompts
TEST_PROMPTS = [
    {"name": "Ambient Piano", "caption": "calm ambient music with soft piano and gentle strings"},
    {"name": "Upbeat Electronic", "caption": "upbeat electronic dance music with heavy bass and synthesizer leads"},
    {"name": "Jazz Trio", "caption": "jazz trio with piano, upright bass, and brush drums"},
    {"name": "Classical Orchestra", "caption": "epic cinematic orchestral arrangement with strings and brass"},
    {"name": "Indie Folk", "caption": "melancholic indie folk with acoustic guitar and soft vocals"},
    {"name": "Rock Ballad", "caption": "emotional rock ballad with electric guitar and atmospheric strings"},
]

# Parse arguments
model_choice = "base"
prompt_arg = sys.argv[1] if len(sys.argv) > 1 else "0"


from processor import ACEStepProcessor

# Get available models

# Validate model
config_path = "acestep-v15-base"

# Validate prompt index
try:
    prompt_index = int(prompt_arg)
except ValueError:
    print(f"✗ Invalid prompt index: {prompt_arg}")
    sys.exit(1)

if prompt_index < 0 or prompt_index >= len(TEST_PROMPTS):
    print(f"✗ Invalid prompt index: {prompt_index}")
    print(f"   Available: 0-{len(TEST_PROMPTS)-1}")
    sys.exit(1)

prompt = TEST_PROMPTS[prompt_index]

print(f"Model: {config_path} ({model_choice})")
print(f"Prompt: {prompt['name']}")
print(f"Caption: {prompt['caption']}\n")

# Initialize model with selected config
print("Initializing model...")
model = ACEStepProcessor(
    project_root=PROJECT_ROOT,
    config_path=config_path,
    device=DEVICE,
    batch_size=2,
)



print("saving at",SAVE_DIR)
print("Generating music...")
model.generate_music(
    audio_duration=30,
    captions=prompt["caption"],
    infer_steps=64,
    guidance_scale=2.0,
    save_dir=SAVE_DIR,
    mode="conditional"
)

