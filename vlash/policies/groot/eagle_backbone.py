"""Eagle VLM backbone for GR00T (Eagle2 for N1.5, Eagle3 for N1.6).

Ported from Isaac-GR00T / lerobot. The Eagle backbone processes images
and language together via the Eagle VLM (SigLIP/SigLIP2 + Qwen3),
then projects hidden states to the action head's expected dimension.

The forward method takes explicit tensor arguments (PI05-style) to be
compile-friendly and avoid CUDA graph breaks from dict unpacking.

Attention layers in both the Qwen3 LLM and SigLIP vision tower are replaced
with vlash's compile-friendly Attention class, eliminating graph breaks from
transformers' flash_attention_2 wrapper (data-dependent branching in
modeling_flash_attention_utils.py:515).
"""

import math
from pathlib import Path
from shutil import copytree
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel

from vlash.layers.attention import Attention
from vlash.layers.rope import apply_rotary_emb

_GROOT_DIR = Path(__file__).resolve().parent
DEFAULT_EAGLE2_PATH = str((_GROOT_DIR / "eagle2_hg_model").resolve())
DEFAULT_EAGLE3_PATH = str((_GROOT_DIR / "eagle3_hg_model").resolve())
DEFAULT_VENDOR_EAGLE_PATH = DEFAULT_EAGLE2_PATH
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"
HF_LEROBOT_HOME = Path.home() / ".cache" / "lerobot"


def ensure_eagle_cache_ready(vendor_dir: Path, cache_dir: Path, assets_repo: str) -> None:
    """Populate the Eagle processor directory in cache and ensure tokenizer assets exist."""
    cache_dir = Path(cache_dir)
    vendor_dir = Path(vendor_dir)

    try:
        copytree(vendor_dir, cache_dir, dirs_exist_ok=True)
    except Exception as exc:
        print(f"[GROOT] Warning: Failed to copy vendor Eagle files to cache: {exc}")

    required_assets = [
        "vocab.json", "merges.txt", "added_tokens.json", "chat_template.json",
        "special_tokens_map.json", "config.json", "generation_config.json",
        "preprocessor_config.json", "processor_config.json", "tokenizer_config.json",
    ]

    for fname in required_assets:
        dst = cache_dir / fname
        if not dst.exists():
            hf_hub_download(
                repo_id=assets_repo, filename=fname, repo_type="model",
                local_dir=str(cache_dir),
            )


# ---------------------------------------------------------------------------
# vlash-compatible attention replacements
# ---------------------------------------------------------------------------

class VlashQwen3Attention(nn.Module):
    """Drop-in replacement for Qwen3Attention using vlash's compile-friendly attention.

    Reuses the original Q/K/V/O projections and QK norms. Replaces the
    flash_attention_2 dispatch with vlash's plain matmul-based attention,
    eliminating data-dependent graph breaks.
    """

    def __init__(self, original_attn: nn.Module):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm

        cfg = original_attn.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = original_attn.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.attn = Attention(scale=original_attn.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE: position_embeddings = (cos_full, sin_full) from Qwen3Model.rotary_emb
        # cos_full/sin_full shape: [B, L, D] (full head_dim, duplicated halves)
        # vlash's apply_rotary_emb expects half-dim cos/sin
        cos_full, sin_full = position_embeddings
        cos = cos_full[..., : self.head_dim // 2].unsqueeze(1)  # [B, 1, L, D/2]
        sin = sin_full[..., : self.head_dim // 2].unsqueeze(1)  # [B, 1, L, D/2]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # GQA: replicate K/V heads to match Q head count
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = self.attn(q, k, v, attention_mask)

        # [B, H, L, D] -> [B, L, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class VlashSiglipAttention(nn.Module):
    """Drop-in replacement for SiglipAttention / Siglip2Attention.

    Works for both SigLIP (Eagle2/N1.5) and SigLIP2 (Eagle3/N1.6).
    SigLIP2's extra arguments (rope_freqs_cis, win_meta_list, windows_attn)
    are accepted via **kwargs but ignored since N1.6 config sets
    use_rope=false and use_windows_attn=false.
    """

    def __init__(self, original_attn: nn.Module):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.out_proj = original_attn.out_proj

        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim

        self.attn = Attention(scale=1.0 / math.sqrt(self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = self.attn(q, k, v, attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


# ---------------------------------------------------------------------------
# EagleBackbone
# ---------------------------------------------------------------------------

class EagleBackbone(nn.Module):
    """Eagle2 vision-language backbone.

    Processes pixel_values and token IDs through the Eagle2 VLM (SigLIP vision
    tower + Qwen3 language model) and projects hidden states for the action head.

    Following PI05's pattern, the forward method takes explicit tensor arguments
    rather than a dict, making it compatible with torch.compile. All attention
    layers are replaced with vlash's compile-friendly implementations.
    """

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str = DEFAULT_VENDOR_EAGLE_PATH,
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        project_to_dim: int = 1536,
    ):
        super().__init__()

        vendor_dir = eagle_path
        cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
        try:
            ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_assets_repo)
        except Exception as exc:
            print(f"[GROOT] Warning: failed to prepare Eagle cache for backbone: {exc}")

        config = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)
        # Override flash_attention_2 since we replace all attention with vlash's
        # compile-friendly implementation anyway. This also avoids errors on CPU.
        config._attn_implementation = "eager"
        if hasattr(config, "text_config"):
            config.text_config._attn_implementation = "eager"
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = "eager"
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # Remove layers beyond select_layer to save compute.
        # Negative indices are resolved relative to the current layer count.
        if select_layer < 0:
            select_layer = len(self.model.language_model.model.layers) + select_layer + 1
        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer

        # Replace flash_attention_2 with vlash's compile-friendly attention
        self._replace_attention_with_vlash()

        self.set_trainable_parameters(tune_llm, tune_visual)

        eagle_cfg = self.model.config
        vision_cfg = eagle_cfg.vision_config
        self._is_siglip2 = getattr(vision_cfg, "model_type", "") == "siglip2_vision_model"
        patch_size = vision_cfg.patch_size
        self._patch_size = patch_size
        downsample_ratio = eagle_cfg.downsample_ratio

        # Eagle2 has force_image_size; Eagle3 computes from num_patches * patch_size
        if hasattr(eagle_cfg, "force_image_size") and eagle_cfg.force_image_size:
            image_size = eagle_cfg.force_image_size
        elif hasattr(vision_cfg, "image_size") and vision_cfg.image_size:
            image_size = vision_cfg.image_size
        else:
            num_patches = getattr(vision_cfg, "num_patches", 256)
            image_size = int(num_patches ** 0.5) * patch_size

        if eagle_cfg.use_pixel_shuffle:
            self._num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        else:
            self._num_image_token = int((image_size // patch_size) ** 2)
        self._image_token_index = eagle_cfg.image_token_index
        self._eagle_image_size = image_size
        self._grid_size = image_size // patch_size
        self._cache_dir = str(cache_dir)

    def _replace_attention_with_vlash(self):
        """Replace all attention modules with vlash's compile-friendly versions.

        Walks the Qwen3 LLM decoder layers and SigLIP/SigLIP2 vision encoder
        layers, swapping each self_attn module with the corresponding vlash wrapper.
        Works for both Eagle2 (N1.5) and Eagle3 (N1.6).
        """
        for layer in self.model.language_model.model.layers:
            layer.self_attn = VlashQwen3Attention(layer.self_attn)

        # Both Eagle2 and Eagle3 share the same path to encoder layers
        siglip_encoder = self.model.vision_model.vision_model.encoder
        for layer in siglip_encoder.layers:
            layer.self_attn = VlashSiglipAttention(layer.self_attn)

    @property
    def num_image_token(self) -> int:
        return self._num_image_token

    @property
    def image_token_index(self) -> int:
        return self._image_token_index

    @property
    def eagle_image_size(self) -> int:
        return self._eagle_image_size

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_flags: torch.Tensor,
        num_vit_tokens: int = 0,
        img_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Eagle VLM and project features.

        Inlines the Eagle model forward to avoid graph breaks from:
        - image_flags boolean indexing (aten.nonzero, dynamic shape)
        - try/except fallback for token count mismatch
        - Python int arithmetic on symbolic shapes

        All inputs must be GPU tensors with fixed shapes (compile-friendly).

        Args:
            pixel_values: Image tiles [num_tiles, C, H, W].
            input_ids: Token IDs with image placeholders [B, seq_len].
            attention_mask: Token-level attention mask [B, seq_len].
            image_flags: Per-tile validity flags (unused — all tiles assumed valid).
            num_vit_tokens: Number of vision tokens per batch item (plain int).
            img_start: Position of first image token in input_ids (plain int).
                Both computed outside the compiled region to avoid graph breaks.

        Returns:
            features: Backbone features [B, seq_len, dim].
            attention_mask: Boolean attention mask [B, seq_len].
            image_mask: Boolean mask indicating image token positions [B, seq_len].
        """
        self.set_frozen_modules_to_eval_mode()

        eagle = self.model

        # 1. Embed text tokens
        input_embeds = eagle.language_model.get_input_embeddings()(input_ids)

        # 2. Extract vision features
        if self._is_siglip2:
            vit_embeds = self._extract_siglip2_features(eagle, pixel_values)
        else:
            vit_embeds = eagle.extract_feature(pixel_values)

        # 3. Replace image token positions with vision embeddings.
        b, n, c = input_embeds.shape
        vit_flat = vit_embeds.reshape(b, num_vit_tokens, c)
        input_embeds = input_embeds.clone()
        input_embeds[:, img_start:img_start + num_vit_tokens, :] = vit_flat

        # 4. Run LLM with combined embeddings
        outputs = eagle.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        eagle_features = outputs.hidden_states[self.select_layer]
        eagle_features = self.eagle_linear(eagle_features)

        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_features.device, dtype=eagle_features.dtype, requires_grad=True
            )
            for param in eagle.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_features = eagle_features + dummy_term

        image_mask = input_ids == self._image_token_index
        backbone_attention_mask = attention_mask == 1

        return eagle_features, backbone_attention_mask, image_mask

    def _extract_siglip2_features(self, eagle: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
        """Inline SigLIP2 vision feature extraction (compile-safe).

        Bypasses SigLIP2's Python-heavy windowing/dynamic-shape code with
        fixed-shape tensor ops. Only valid for dynamic_image_size=false
        (fixed resolution, single tile per image).

        Pipeline: patchify → embed → pos_embed → encoder → post_ln → pixel_unshuffle → mlp1
        """
        vision = eagle.vision_model.vision_model
        B = pixel_values.shape[0]
        ps = self._patch_size
        grid = self._grid_size  # patches per side
        num_patches = grid * grid
        C_vit = vision.embeddings.embed_dim  # 1152

        # 1. Patchify: [B, 3, H, W] → [B*num_patches, ps*ps*3]
        pv = pixel_values.reshape(B, 3, grid, ps, grid, ps)
        pv = pv.permute(0, 2, 4, 3, 5, 1)  # [B, grid, grid, ps, ps, 3]
        pv = pv.reshape(B * num_patches, ps * ps * 3)

        # 2. Patch embedding (Linear): [B*num_patches, 588] → [B*num_patches, 1152]
        target_dtype = vision.embeddings.patch_embedding.weight.dtype
        patch_embeds = vision.embeddings.patch_embedding(pv.to(dtype=target_dtype))

        # 3. Add position embedding (no interpolation for fixed size)
        patch_embeds = patch_embeds.reshape(B, num_patches, C_vit)
        pos_emb = vision.embeddings.position_embedding.weight[:num_patches].unsqueeze(0)
        hidden_states = patch_embeds + pos_emb

        # 4. Run encoder layers (vlash attention, no windowing)
        for layer in vision.encoder.layers:
            residual = hidden_states
            hidden_states = layer.layer_norm1(hidden_states)
            hidden_states, _ = layer.self_attn(hidden_states=hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.layer_norm2(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        # 5. Post layer norm
        hidden_states = vision.post_layernorm(hidden_states)

        # 6. Pixel unshuffle: [B, 256, 1152] → [B, 1152, 16, 16] → [B, 4608, 8, 8] → [B, 64, 4608]
        ds = int(1 / eagle.downsample_ratio)  # 2
        hidden_states = hidden_states.transpose(1, 2).reshape(B, C_vit, grid, grid)
        hidden_states = F.pixel_unshuffle(hidden_states, downscale_factor=ds)
        C_out = C_vit * ds * ds  # 4608
        hidden_states = hidden_states.flatten(start_dim=2).transpose(1, 2)  # [B, 64, 4608]

        # 7. MLP1 connector
        hidden_states = eagle.mlp1(hidden_states)  # [B, 64, 2048]

        # Flatten to match extract_feature output format: [B*64, 2048]
        hidden_states = hidden_states.reshape(B * hidden_states.shape[1], -1)

        return hidden_states
