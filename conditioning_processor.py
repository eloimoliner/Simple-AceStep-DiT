
import os
import os
import sys
from typing import Optional, Dict, Any, List, Union,  Tuple

import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

class ConditioningPreprocessor:
    def __init__(self,  checkpoint_dir: str, encoder: Any, silence_latent: torch.Tensor, batch_size: int= 1, device="cuda", dtype=torch.float32):

        self.batch_size = batch_size
        self.encoder=encoder
        self.device=device
        self.dtype=dtype
        self.silence_latent= silence_latent
        self._load_text_encoder(checkpoint_dir=checkpoint_dir)

    def _load_text_encoder(self, *, checkpoint_dir: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        te_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")

        if not os.path.exists(te_path):
            os.makedirs(te_path)
            snapshot_download(
                repo_id="ACE-Step/Ace-Step1.5",
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False,
                token=None,
            )

        self.text_tokenizer = AutoTokenizer.from_pretrained(te_path)
        self.text_encoder = AutoModel.from_pretrained(te_path)

        self.text_encoder = self.text_encoder.to(self.device).to(self.dtype)
        self.text_encoder.eval()

    def prepare_conditioners(self, captions, audio_duration):
        """Prepare conditioning inputs for generation."""
        captions = [captions] * self.batch_size

        calculated_duration = float(audio_duration)

        metadata_dict: Dict[str, Union[str, int]] = self._build_metadata_dict(
            None, "", "", calculated_duration
        )
        metas = [metadata_dict.copy() for _ in range(self.batch_size)]
        metas = self._parse_metas(metas)

        lyrics=[""]*self.batch_size

        vocal_languages = ["en"] * self.batch_size

        instructions = [""]*self.batch_size

        (
            padded_text_token_idss,
            padded_text_attention_masks,
            padded_lyric_token_idss,
            padded_lyric_attention_masks,
        ) = self._prepare_text_conditioning_inputs(
            self.batch_size,
            instructions,
            captions,
            lyrics,
            metas,
            vocal_languages,
        )

        text_token_idss= padded_text_token_idss.to(self.device)
        text_attention_masks= padded_text_attention_masks.to(self.device)
        lyric_token_idss= padded_lyric_token_idss.to(self.device)
        lyric_attention_masks= padded_lyric_attention_masks.to(self.device)

        text_hidden_states = self.infer_text_embeddings(text_token_idss)

        lyric_hidden_states = self.infer_lyric_embeddings(lyric_token_idss)

        refer_audios = [[torch.zeros(2, 30 * 48000)] for _ in range(self.batch_size)]

        for ii, refer_audio_list in enumerate(refer_audios):
            if isinstance(refer_audio_list, list):
                for idx, _ in enumerate(refer_audio_list):
                    refer_audio_list[idx] = refer_audio_list[idx].to(self.device).to(self.dtype)
            elif isinstance(refer_audio_list, torch.Tensor):
                refer_audios[ii] = refer_audios[ii].to(self.device)

        refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask = self.infer_refer_latent(
            refer_audios
        )
        refer_audio_acoustic_hidden_states_packed = refer_audio_acoustic_hidden_states_packed.to(self.dtype)

        with torch.no_grad():
            encoder_hidden_states, encoder_attention_mask = self.encoder(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_masks,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_masks,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask=refer_audio_order_mask,
            )

        return {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }


    def _build_metadata_dict(
        self,
        bpm: Optional[Union[int, str]],
        key_scale: str,
        time_signature: str,
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build metadata dictionary with defaults for missing fields."""
        metadata_dict: Dict[str, Any] = {}
        metadata_dict["bpm"] = bpm if bpm else "N/A"
        metadata_dict["keyscale"] = key_scale if key_scale.strip() else "N/A"
        if time_signature.strip() and time_signature != "N/A" and time_signature:
            metadata_dict["timesignature"] = time_signature
        else:
            metadata_dict["timesignature"] = "N/A"
        if duration is not None:
            metadata_dict["duration"] = f"{int(duration)} seconds"
        return metadata_dict



    def _parse_metas(self, metas: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """Parse and normalize metadata values with safe fallbacks."""
        parsed_metas = []
        for meta in metas:
            if meta is None:
                parsed_meta = self._create_default_meta()
            elif isinstance(meta, str):
                parsed_meta = meta
            elif isinstance(meta, dict):
                parsed_meta = self._dict_to_meta_string(meta)
            else:
                parsed_meta = self._create_default_meta()
            parsed_metas.append(parsed_meta)
        return parsed_metas

    def _get_vae_dtype(self, device: Optional[str] = None) -> torch.dtype:
        """Get VAE dtype based on target device and GPU tier."""
        target_device = device or self.device
        if target_device in ["cuda", "xpu"]:
            return torch.bfloat16
        if target_device == "mps":
            return torch.float16
        if target_device == "cpu":
            return torch.float32
        return self.dtype

    def _dict_to_meta_string(self, meta_dict: Dict[str, Any]) -> str:
        """Convert metadata dict to formatted string."""
        bpm = meta_dict.get("bpm", meta_dict.get("tempo", "N/A"))
        timesignature = meta_dict.get("timesignature", meta_dict.get("time_signature", "N/A"))
        keyscale = meta_dict.get("keyscale", meta_dict.get("key", meta_dict.get("scale", "N/A")))
        duration = meta_dict.get("duration", meta_dict.get("length", 30))

        if isinstance(duration, (int, float)):
            duration = f"{int(duration)} seconds"
        elif not isinstance(duration, str):
            duration = "30 seconds"

        return (
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration}\n"
        )


    def _prepare_text_conditioning_inputs(
        self,
        batch_size: int,
        instructions: List[str],
        captions: List[str],
        lyrics: List[str],
        parsed_metas: List[str],
        vocal_languages: List[str],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Tokenize caption/lyric prompts and optional non-cover branch prompts."""
        actual_captions, actual_languages = self._extract_caption_and_language(parsed_metas, captions, vocal_languages)

        # Detect is_lego_sft from the loaded model config (set on the SFT-stems checkpoint).

        text_token_idss = []
        text_attention_masks = []
        lyric_token_idss = []
        lyric_attention_masks = []

        for i in range(batch_size):
            instruction = self._format_instruction(
                instructions[i] if i < len(instructions) else DEFAULT_DIT_INSTRUCTION
            )
            actual_caption = actual_captions[i]
            actual_language = actual_languages[i]

            text_prompt = SFT_GEN_PROMPT.format(instruction, actual_caption, parsed_metas[i])

            text_inputs_dict = self.text_tokenizer(
                text_prompt,
                padding="longest",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            text_token_ids = text_inputs_dict.input_ids[0]
            text_attention_mask = text_inputs_dict.attention_mask[0].bool()

            lyrics_text = self._format_lyrics(lyrics[i], actual_language)
            lyrics_inputs_dict = self.text_tokenizer(
                lyrics_text,
                padding="longest",
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            lyric_token_ids = lyrics_inputs_dict.input_ids[0]
            lyric_attention_mask = lyrics_inputs_dict.attention_mask[0].bool()

            text_token_idss.append(text_token_ids)
            text_attention_masks.append(text_attention_mask)
            lyric_token_idss.append(lyric_token_ids)
            lyric_attention_masks.append(lyric_attention_mask)

        max_text_length = max(len(seq) for seq in text_token_idss)
        padded_text_token_idss = self._pad_sequences(text_token_idss, max_text_length, self.text_tokenizer.pad_token_id)
        padded_text_attention_masks = self._pad_sequences(text_attention_masks, max_text_length, 0)

        max_lyric_length = max(len(seq) for seq in lyric_token_idss)
        padded_lyric_token_idss = self._pad_sequences(lyric_token_idss, max_lyric_length, self.text_tokenizer.pad_token_id)
        padded_lyric_attention_masks = self._pad_sequences(lyric_attention_masks, max_lyric_length, 0)

        return (
            padded_text_token_idss,
            padded_text_attention_masks,
            padded_lyric_token_idss,
            padded_lyric_attention_masks,
        )


    def _extract_caption_and_language(
        self,
        metas: List[Union[str, Dict[str, Any]]],
        captions: List[str],
        vocal_languages: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Extract caption/language values from metas with fallback values."""
        actual_captions = list(captions)
        actual_languages = list(vocal_languages)

        for i, meta in enumerate(metas):
            if i >= len(actual_captions):
                break

            meta_dict = None
            if isinstance(meta, str):
                parsed = self._parse_metas([meta])
                if parsed and isinstance(parsed[0], dict):
                    meta_dict = parsed[0]
            elif isinstance(meta, dict):
                meta_dict = meta

            if meta_dict:
                if "caption" in meta_dict and meta_dict["caption"]:
                    actual_captions[i] = str(meta_dict["caption"])
                if "language" in meta_dict and meta_dict["language"]:
                    actual_languages[i] = str(meta_dict["language"])
        return actual_captions, actual_languages

    def _format_instruction(self, instruction: str) -> str:
        """Ensure instruction ends with a colon."""
        if not instruction.endswith(":"):
            instruction = instruction + ":"
        return instruction
    
    def _format_lyrics(self, lyrics: str, language: str) -> str:
        """Format lyrics text with language header."""
        return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    def _pad_sequences(
        self, sequences: List[torch.Tensor], max_length: int, pad_value: int = 0
    ) -> torch.Tensor:
        """Pad sequence tensors to the same length."""
        return torch.stack(
            [
                torch.nn.functional.pad(seq, (0, max_length - len(seq)), "constant", pad_value)
                for seq in sequences
            ]
        )

    def infer_refer_latent(self, refer_audioss: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer packed reference-audio latents and order mask."""
        refer_audio_order_mask = []
        refer_audio_latents = []

        self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

        def _normalize_audio_2d(a: torch.Tensor) -> torch.Tensor:
            if not isinstance(a, torch.Tensor):
                raise TypeError(f"refer_audio must be a torch.Tensor, got {type(a)!r}")
            if a.dim() == 3 and a.shape[0] == 1:
                a = a.squeeze(0)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if a.dim() != 2:
                raise ValueError(f"refer_audio must be 1D/2D/3D(1,2,T); got shape={tuple(a.shape)}")
            if a.shape[0] == 1:
                a = torch.cat([a, a], dim=0)
            return a[:2]

        def _ensure_latent_3d(z: torch.Tensor) -> torch.Tensor:
            if z.dim() == 4 and z.shape[0] == 1:
                z = z.squeeze(0)
            if z.dim() == 2:
                z = z.unsqueeze(0)
            return z

        refer_encode_cache: Dict[int, torch.Tensor] = {}
        for batch_idx, refer_audios in enumerate(refer_audioss):
            if len(refer_audios) == 1 and torch.all(refer_audios[0] == 0.0):
                refer_audio_latent = _ensure_latent_3d(self.silence_latent[:, :750, :])
                refer_audio_latents.append(refer_audio_latent)
                refer_audio_order_mask.append(batch_idx)
            else:
                for refer_audio in refer_audios:
                    cache_key = refer_audio.data_ptr()
                    if cache_key in refer_encode_cache:
                        refer_audio_latent = refer_encode_cache[cache_key].clone()
                    else:
                        refer_audio = _normalize_audio_2d(refer_audio)
                        with torch.inference_mode():
                            refer_audio_latent = self.tiled_encode(refer_audio, offload_latent_to_cpu=True)
                        refer_audio_latent = refer_audio_latent.to(self.device).to(self.dtype)
                        if refer_audio_latent.dim() == 2:
                            refer_audio_latent = refer_audio_latent.unsqueeze(0)
                        refer_audio_latent = _ensure_latent_3d(refer_audio_latent.transpose(1, 2))
                        refer_encode_cache[cache_key] = refer_audio_latent
                    refer_audio_latents.append(refer_audio_latent)
                    refer_audio_order_mask.append(batch_idx)

        refer_audio_latents = torch.cat(refer_audio_latents, dim=0)
        refer_audio_order_mask = torch.tensor(refer_audio_order_mask, device=self.device, dtype=torch.long)
        return refer_audio_latents, refer_audio_order_mask

    def infer_text_embeddings(self, text_token_idss):
        """Infer text-token embeddings via text encoder."""
        with torch.inference_mode():
            return self.text_encoder(input_ids=text_token_idss, lyric_attention_mask=None).last_hidden_state

    def infer_lyric_embeddings(self, lyric_token_ids):
        """Infer lyric-token embeddings via text encoder embedding table."""
        with torch.inference_mode():
            return self.text_encoder.embed_tokens(lyric_token_ids)

    def is_silence(self, audio: torch.Tensor) -> bool:
        """Return True when audio is effectively silent."""
        return bool(torch.all(audio.abs() < 1e-6))



