import soundfile as sf
import os
import os
from typing import Optional,  Any
from einops import rearrange
from modeling_acestep_v15_base import AceStepSimple
from conditioning_processor import ConditioningPreprocessor
from huggingface_hub import snapshot_download

import torch
from loguru import logger



class ACEStepProcessor():
    """DiT + VAE + text-encoder loader for CUDA/Linux.

    All model state from AceStepHandler lives here. Models are downloaded
    and loaded inside __init__ — no separate initialize_service call needed.

    Args:
        project_root:       Project root containing the ``checkpoints/`` dir.
                            Defaults to cwd / ACESTEP_PROJECT_ROOT env var.
        config_path:        Checkpoint sub-directory, e.g. "acestep-v15-turbo".
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        config_path: str = "acestep-v15-base",
        device="cuda",
        batch_size: int = 1,
    ):
        super().__init__()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required but not available.")

        # ------------------------------------------------------------------ #
        # State — mirrors AceStepHandler.__init__                             #
        # ------------------------------------------------------------------ #
        self.device = device
        self.dtype = torch.float32

        self.sample_rate = 48000
        self.batch_size = batch_size

        # ------------------------------------------------------------------ #
        # Resolve paths                                                        #
        # ------------------------------------------------------------------ #
        base_root = project_root
        checkpoint_dir = os.path.join(base_root, "checkpoints")

        # ------------------------------------------------------------------ #
        # Load DiT                                                             #
        # ------------------------------------------------------------------ #



        model_checkpoint_path = os.path.join(checkpoint_dir, "acestep-v15-base")
        if not os.path.exists(model_checkpoint_path):
            #download checkpoints from huggingface
            os.makedirs(model_checkpoint_path)
            snapshot_download(
                repo_id="ACE-Step/acestep-v15-base",
                local_dir=model_checkpoint_path,
                local_dir_use_symlinks=False,
                token=None,
            )


        self.model = AceStepSimple.from_pretrained(
            model_checkpoint_path,
            attn_implementation="sdpa",
            dtype=self.dtype,

        )

        self.model.config._attn_implementation = "sdpa"
        self.config = self.model.config

        self.model = self.model.to(self.device).to(self.dtype)

        silence_path = os.path.join(model_checkpoint_path, "silence_latent.pt")
        if not os.path.exists(silence_path):
            raise FileNotFoundError(f"Silence latent not found: {silence_path}")

        self.silence_latent = (
            torch.load(silence_path, weights_only=True)
            .transpose(1, 2)
            .to(self.device)
            .to(self.dtype)
        )

        # ------------------------------------------------------------------ #
        # Load VAE                                                             #
        # ------------------------------------------------------------------ #
        self._load_vae(checkpoint_dir=checkpoint_dir)

        # ------------------------------------------------------------------ #
        # Load text encoder + tokenizer                                        #
        # ------------------------------------------------------------------ #
        self.conditioning_preprocessor = ConditioningPreprocessor(
            checkpoint_dir=checkpoint_dir,
            batch_size=self.batch_size,
            encoder=self.model.encoder,
            device=self.device,
            dtype=self.dtype,
            silence_latent=self.silence_latent,
        )


        logger.info(
            f"[ACEStepSimpleModel] Ready — device={self.device} dtype={self.dtype} model={config_path}"
        )

    def _load_vae(self, *, checkpoint_dir: str) -> None:
        from diffusers.models import AutoencoderOobleck

        vae_path = os.path.join(checkpoint_dir, "vae")
        if not os.path.exists(vae_path):
            os.makedirs(vae_path)
            snapshot_download(
                repo_id="ACE-Step/Ace-Step1.5",
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False,
                token=None,
            )

        self.vae = AutoencoderOobleck.from_pretrained(vae_path)

        self.vae = self.vae.to(self.device).to(self.dtype)


    def v_predict(self, x, t, guidance_scale=1.0, mode="conditional", conditioners=None):


        silence_latent_tiled = self.silence_latent[0, :x.shape[1], :]
        src_latents = silence_latent_tiled.unsqueeze(0).expand(x.shape[0], -1, -1)
        chunk_mask= torch.ones_like(src_latents, dtype=self.dtype, device=self.device)
        src_latents= src_latents.to(self.device).to(self.dtype)
        context_latents = torch.cat([src_latents, chunk_mask], dim=-1)
        attention_mask = torch.ones(x.shape[0], x.shape[1], device=self.device, dtype=self.dtype)

        # main task condition
        if mode=="conditional":
            assert conditioners is not None, "Conditioners must be provided for conditional mode"

            encoder_hidden_states = conditioners["encoder_hidden_states"]
            encoder_attention_mask = conditioners["encoder_attention_mask"]

            if guidance_scale > 1.0:
                encoder_hidden_states = torch.cat([encoder_hidden_states, self.model.null_condition_emb.expand_as(encoder_hidden_states)], dim=0)
                encoder_attention_mask = torch.cat([encoder_attention_mask, encoder_attention_mask], dim=0)

                context_latents = torch.cat([context_latents, context_latents], dim=0)
                attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

                x = torch.cat([x, x], dim=0) 

        elif mode=="unconditional":
            encoder_hidden_states = self.model.null_condition_emb.expand(x.shape[0], -1, -1)
            encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:-1], device=x.device, dtype=torch.bool)
            context_latents = context_latents
            attention_mask = attention_mask


        t = t * torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)

        decoder_outputs = self.model.decoder(
                    hidden_states=x,
                    timestep=t,
                    timestep_r=t,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                    use_cache=False,
                )
                
        vt = decoder_outputs[0]

        if mode=="conditional" and guidance_scale > 1.0:
            pred_cond, pred_null_cond = vt.chunk(2)
            vt=pred_cond + (guidance_scale-1.0) * (pred_cond - pred_null_cond)

        return vt


    def generate_music(self, 
                          captions: Any,
                            audio_duration: float= 20,
                          infer_steps: int = 8,
                            guidance_scale: float = 7.0,
                        save_dir: Optional[str] = None,
                        mode="conditional"
                        ):
        """Generate music from text caption.

        Args:
            params: GenerationParams with caption, lyrics, duration, etc.
            config: GenerationConfig with batch_size, seeds, audio_format, etc.
            save_dir: Optional directory to save generated audio files

        Returns:
            GenerationResult with generated audios, metadata, and status
        """


        latent_length = int(audio_duration * self.sample_rate) // 1920
        self.shape = (self.batch_size, latent_length, 64)

        if mode=="conditional":
            conditioners=self.conditioning_preprocessor.prepare_conditioners(captions, audio_duration)
        else:
            conditioners=None

        t = torch.linspace(1.0, 0.0, infer_steps + 1, device=self.device, dtype=self.dtype)

        xt = torch.randn(self.shape, device=self.device, dtype=self.dtype)

        for step_idx, (t_curr, t_prev) in enumerate(zip(t[:-1], t[1:])):
                
            with torch.no_grad():
                vt=self.v_predict(xt, t_curr, guidance_scale=guidance_scale, mode=mode, conditioners=conditioners)

            #simple Euler ODE step
            dt = t_curr - t_prev
            dt_tensor = dt * torch.ones((self.batch_size,), device=self.device, dtype=self.dtype).unsqueeze(-1).unsqueeze(-1)
            xt = xt - vt * dt_tensor
        
        pred_latents = xt

        pred_latents_for_decode = pred_latents.transpose(1, 2).contiguous().to(self.vae.dtype)
        with torch.no_grad():
            pred_wavs=self.vae.decode(pred_latents_for_decode).sample
        pred_wavs = pred_wavs.detach().cpu()

        self.save_audio(pred_wavs, save_dir=save_dir)

    def save_audio(self, pred_wavs, save_dir: Optional[str] = None):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        for index in range(self.batch_size):
            audio_tensor = pred_wavs[index].cpu().float()

            audio_file = os.path.join(save_dir, f"audio_{index}.wav")

            audio_np = audio_tensor.numpy().T
            sf.write(audio_file, audio_np, self.sample_rate, subtype='FLOAT', format='WAV')

            print(f"✓ Saved audio {index} to: {audio_file}")


