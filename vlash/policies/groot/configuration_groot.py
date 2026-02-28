"""GR00T Policy Configuration (N1.5 / N1.6).

Configuration for the GR00T (Generalist Robot 0-to-0 Transfer) model
adapted for the VLASH framework. Defaults are set for N1.6 (nvidia/GR00T-N1.6-3B).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@dataclass
class GrootConfig(PreTrainedConfig):
    """Configuration for GR00T policy (N1.5 / N1.6).

    Defaults are set for N1.6 (nvidia/GR00T-N1.6-3B). For N1.5, the
    from_pretrained method auto-detects and overrides architecture params.
    """

    # === Model Source ===
    base_model_path: str = "nvidia/GR00T-N1.6-3B"
    eagle_path: str | None = None
    tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5"

    # === Eagle Backbone ===
    tune_llm: bool = False
    tune_visual: bool = False
    eagle_select_layer: int = 16
    eagle_project_to_dim: int | None = None

    # === Action Head ===
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # === Action Prediction ===
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 128
    max_action_dim: int = 128

    # === Flow Matching ===
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_steps: int = 4

    # === Multi-Embodiment ===
    max_num_embodiments: int = 32
    embodiment_tag: str = "new_embodiment"

    # === DiT Architecture (defaults match N1.6) ===
    action_head_hidden_size: int = 1024
    action_head_input_embedding_dim: int = 1536
    action_head_backbone_embedding_dim: int = 2048
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    use_vlln: bool = True

    # N1.6: AlternateVLDiT with 32 layers
    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2

    # N1.5-only: future_tokens and vl_self_attention (disabled for N1.6)
    num_target_vision_tokens: int = 0
    vl_self_attention_cfg: dict | None = None

    # State augmentation
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0

    diffusion_model_cfg: dict = field(default_factory=lambda: {
        "num_attention_heads": 32,
        "attention_head_dim": 48,
        "output_dim": 1024,
        "num_layers": 32,
        "dropout": 0.2,
        "attention_bias": True,
        "activation_fn": "gelu-approximate",
        "norm_type": "ada_norm",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-5,
        "max_num_positional_embeddings": 512,
        "final_dropout": True,
        "positional_embeddings": None,
        "interleave_self_attention": True,
    })

    # === Image Processing ===
    image_size: tuple[int, int] = (448, 448)

    # === Tokenization ===
    tokenizer_max_length: int = 200

    # === Training ===
    use_bf16: bool = True
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None
    dtype: str = "bfloat16"

    # === Normalization ===
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # === Optimizer ===
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0

    # === Scheduler ===
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 10_000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features for GR00T."""
        image_features = [
            key for key, feat in self.input_features.items()
            if feat.type == FeatureType.VISUAL
        ]
        if not image_features:
            raise ValueError(
                "GR00T requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


__all__ = ["GrootConfig"]
