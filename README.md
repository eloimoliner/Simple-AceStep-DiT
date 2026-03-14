
# ACE-step v1.5 Simple Implementation

This repository provides a **very much simplified implementation** of the open-source music generation model ACE-step v1.5. It is designed for research and experimentation, and supports both conditional (text-to-music) and unconditional music generation using the "base" model weights.

## Project Purpose
This project allows researchers to easily generate music from text prompts or without prompts, using the original ACE-step v1.5 weights. All model weights are loaded from Hugging Face. **I do not own the ACE-step model.**

## Features
- Loads ACE-step v1.5 "base" model weights from Hugging Face automatically
- Conditional mode: generate music from text prompts (prompt-conditioned audio)
- Unconditional mode: generate music without a prompt
- Simple interface for prompt selection and output
- Designed for NVIDIA GPUs (requires >16GB VRAM)

## Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support (required)
- PyTorch, Transformers, Hugging Face Hub, Loguru, SoundFile, Einops

## Usage
Run the main script:

```bash
python sample.py [model] [prompt_index]
```

- `model`: "base" (default) or "turbo" (not included here)
- `prompt_index`: 0-5 (default: 0)
- Use `-l` to list all available prompts

Examples:
- `python sample.py` (Base + Ambient Piano)
- `python sample.py base 1` (Base + Upbeat Electronic)
- `python sample.py -l` (List prompts)

### Available Prompts
0: Ambient Piano
1: Upbeat Electronic
2: Jazz Trio
3: Classical Orchestra
4: Indie Folk
5: Rock Ballad

### Output
Generated audio files are saved in the `output/` directory.

## Main Components
- `sample.py`: Entry point, handles prompt selection, device check, and output.
- `processor.py`: Loads the ACE-step model and runs inference.
- `conditioning_processor.py`: Prepares text conditioning for prompt-based generation.
- `modeling_acestep_v15_base.py`: Defines the ACE-step v1.5 base model architecture.
- `configuration_acestep_v15.py`: Model configuration.

## Model Ownership & Disclaimer
- **I do not own the ACE-step model.**
- All model weights and original code are credited to the ACE-step authors and Hugging Face.
- This repository is for research and educational purposes only.

## References
- [ACE-step v1.5 on Hugging Face](https://huggingface.co/ACE-Step/Ace-Step1.5)
- [Original ACE-step repository](https://github.com/ACE-Step/ACE-Step)

## License
Please refer to the original ACE-step license for terms of use. This simplified implementation is provided as-is for research purposes.

---
For questions or contributions, please open an issue or pull request.
