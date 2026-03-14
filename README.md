
# ACE-step v1.5 Simple Implementation

This repository is a **very much simplified implementation** of the open-source ACE-step v1.5 music generation model. It is designed for research and experimentation, supporting both conditional (text-to-music) and unconditional music generation using the "base" model weights.

## Purpose
Generate music from text prompts or without prompts, using original ACE-step v1.5 weights loaded from Hugging Face. Results may differ from the original codebase.

## Features
- Loads ACE-step v1.5 "base" model weights from Hugging Face
- Conditional mode: text-to-music generation
- Unconditional mode: music generation without prompt
- Simple prompt selection and output
- Requires NVIDIA GPU (>16GB VRAM)

## Usage
Run:

```bash
python sample.py "your prompt here"
```
Example:
```bash
python sample.py "jazz trio with piano, upright bass, and brush drums"
```

Output files are saved in the `output/` directory.

## Components
- sample.py: Entry point, prompt selection, device check, output
- processor.py: Loads model, runs inference
- conditioning_processor.py: Prepares text conditioning
- modeling_acestep_v15_base.py: Model architecture
- configuration_acestep_v15.py: Model configuration

## Disclaimer
- All model weights and original code are credited to ACE-step authors and Hugging Face
- This repository is for research and educational purposes only
- I do not own the ACE-step model

## References
- [ACE-step v1.5 on Hugging Face](https://huggingface.co/ACE-Step/Ace-Step1.5)
- [Original ACE-step repository](https://github.com/ACE-Step/ACE-Step)

## License
See the original ACE-step license for terms of use. This simplified implementation is provided as-is for research purposes.

