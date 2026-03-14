#!/bin/bash

# ACE-Step 1.5 Simple Inference Script
# Generates music from text prompts using NVIDIA GPU

set -e

echo "================================"
echo "ACE-Step 1.5 - Simple Inference"
echo "================================"
echo ""

# Get GPU ID from argument or environment variable
GPU_ID="${1:-${CUDA_VISIBLE_DEVICES:-0}}"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "GPU Configuration:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ Error: nvidia-smi not found. NVIDIA GPU drivers required."
    exit 1
fi

echo "GPU Status:"
nvidia-smi -i "$GPU_ID" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || {
    echo "✗ Error: GPU ID $GPU_ID not found or invalid."
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    exit 1
}
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Check Python
if ! command -v python &> /dev/null; then
    echo "✗ Error: Python not found."
    exit 1
fi

echo "Python: $(python --version)"
echo ""

# Create output directory
mkdir -p output

# Run the inference script
echo "Starting inference..."
echo ""

#python sample.py "calm ambient music with soft piano and gentle strings"
#python sample.py "upbeat electronic dance music with heavy bass and synthesizer leads"
python sample.py "jazz trio with piano, upright bass, and brush drums"
#python sample.py "epic cinematic orchestral arrangement with strings and brass"
#python sample.py "melancholic indie folk with acoustic guitar and soft vocals"
#python sample.py "emotional rock ballad with electric guitar and atmospheric strings"

echo ""
echo "✓ Complete! Check ./output/ for generated audio files."

