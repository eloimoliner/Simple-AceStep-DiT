"""
Simplified ACE-Step 1.5 inference for NVIDIA GPUs (>16GB VRAM).

Generates music from text prompts with different models and configurations.
Models are auto-downloaded on first run.

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

# Parse arguments
prompt = sys.argv[1]


from processor import ACEStepProcessor

# Get available models

# Validate model
config_path = "acestep-v15-base"


print(f"Model: {config_path} ")
print(f"Caption: {prompt}\n")

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
    captions=prompt,
    infer_steps=64,
    guidance_scale=2.0,
    save_dir=SAVE_DIR,
    mode="conditional"
)

