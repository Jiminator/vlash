"""GR00T N1.5 Model Implementation.

This module implements the GR00T model for robot control, adapted for the
VLASH framework. Follows the same preprocessing pattern as PI05:
  - prepare_images(): GPU resize + normalize (no CPU roundtrip)
  - prepare_language(): tokenize + insert image tokens (CPU tokenizer, then GPU)
  - Model forward takes explicit tensor arguments (compile-friendly)

Architecture:
    GrootPolicy (wrapper)
    └── GrootModel (core model)
        ├── EagleBackbone (Eagle2 VLM: SigLIP + Qwen2 → features)
        └── FlowmatchingActionHead (DiT + flow matching → actions)

"""

import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoTokenizer

from lerobot.configs.policies import PreTrainedConfig, T
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from vlash.policies.normalize import Normalize, Unnormalize
from vlash.policies.groot.configuration_groot import GrootConfig
from vlash.policies.groot.eagle_backbone import EagleBackbone, DEFAULT_VENDOR_EAGLE_PATH
from vlash.policies.groot.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)


def _pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the last dimension of a vector to a target size."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


class GrootModel(nn.Module):
    """Core GR00T N1.5 model: Eagle backbone + flow matching action head.

    Unlike PI0.5 which uses shared transformer layers between VLM and action
    expert, GR00T uses a separate DiT that cross-attends to Eagle backbone
    features. All forward methods take explicit tensor arguments for
    torch.compile compatibility.
    """

    def __init__(self, config: GrootConfig):
        super().__init__()
        self.config = config

        eagle_path = config.eagle_path or DEFAULT_VENDOR_EAGLE_PATH
        self.backbone = EagleBackbone(
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.eagle_select_layer,
            eagle_path=eagle_path,
            tokenizer_assets_repo=config.tokenizer_assets_repo,
            project_to_dim=config.eagle_project_to_dim,
        )

        action_head_cfg = FlowmatchingActionHeadConfig(
            add_pos_embed=config.add_pos_embed,
            diffusion_model_cfg=config.diffusion_model_cfg,
            input_embedding_dim=config.action_head_input_embedding_dim,
            backbone_embedding_dim=config.action_head_backbone_embedding_dim,
            hidden_size=config.action_head_hidden_size,
            max_seq_len=config.max_seq_len,
            action_dim=config.max_action_dim,
            action_horizon=config.chunk_size,
            noise_beta_alpha=config.noise_beta_alpha,
            noise_beta_beta=config.noise_beta_beta,
            noise_s=config.noise_s,
            num_timestep_buckets=config.num_timestep_buckets,
            num_inference_timesteps=config.num_inference_steps,
            max_num_embodiments=config.max_num_embodiments,
            max_state_dim=config.max_state_dim,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            use_vlln=config.use_vlln,
            vl_self_attention_cfg=config.vl_self_attention_cfg,
            num_target_vision_tokens=config.num_target_vision_tokens,
        )
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

    def _num_vit_tokens(self, num_images: int) -> int:
        """Compute total vision tokens per batch item (plain int, compile-safe)."""
        return num_images * self.backbone.num_image_token

    def forward(self, pixel_values, input_ids, attention_mask, image_flags,
                state, action, action_mask, embodiment_id,
                num_images: int = 1, img_start: int = 0):
        """Training forward: compute flow matching loss.

        All arguments are explicit GPU tensors (compile-friendly).
        num_images and img_start are plain Python ints computed outside the compiled region.
        """
        num_vit_tokens = self._num_vit_tokens(num_images)
        backbone_features, backbone_attention_mask = self.backbone(
            pixel_values, input_ids, attention_mask, image_flags,
            num_vit_tokens=num_vit_tokens, img_start=img_start,
        )
        loss = self.action_head(
            backbone_features, backbone_attention_mask,
            state, action, action_mask, embodiment_id,
        )
        return loss

    @torch.no_grad()
    def sample_actions(self, pixel_values, input_ids, attention_mask, image_flags,
                       state, embodiment_id,
                       num_images: int = 1, img_start: int = 0):
        """Inference: generate actions via iterative denoising.

        All arguments are explicit GPU tensors (compile-friendly).
        num_images and img_start are plain Python ints computed outside the compiled region.
        """
        num_vit_tokens = self._num_vit_tokens(num_images)
        backbone_features, backbone_attention_mask = self.backbone(
            pixel_values, input_ids, attention_mask, image_flags,
            num_vit_tokens=num_vit_tokens, img_start=img_start,
        )
        actions = self.action_head.get_action(
            backbone_features, backbone_attention_mask,
            state, embodiment_id,
        )
        return actions

    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        """Convert model to bfloat16, keeping critical params in float32."""
        params_to_keep_float32 = [
            "vision_model.embeddings.patch_embedding.weight",
            "vision_model.embeddings.patch_embedding.bias",
            "vision_model.embeddings.position_embedding.weight",
        ]

        if precision == "bfloat16":
            for name, param in self.named_parameters():
                if any(selector in name for selector in params_to_keep_float32):
                    continue
                param.data = param.data.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)

    @classmethod
    def from_pretrained(cls, config: GrootConfig, pretrained_path: str):
        """Load a GrootModel from a pretrained GR00T checkpoint."""
        try:
            local_model_path = snapshot_download(pretrained_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(f"[GROOT] Loading from local path: {pretrained_path}")
            local_model_path = pretrained_path

        import json
        config_path = os.path.join(local_model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                checkpoint_config = json.load(f)

            if "backbone_cfg" in checkpoint_config:
                bb_cfg = checkpoint_config["backbone_cfg"]
                config.eagle_project_to_dim = bb_cfg.get(
                    "project_to_dim", config.eagle_project_to_dim)
                config.eagle_select_layer = bb_cfg.get(
                    "select_layer", config.eagle_select_layer)
                config.tune_llm = bb_cfg.get("tune_llm", config.tune_llm)
                config.tune_visual = bb_cfg.get("tune_visual", config.tune_visual)

            if "action_head_cfg" in checkpoint_config:
                ah_cfg = checkpoint_config["action_head_cfg"]
                config.diffusion_model_cfg = ah_cfg.get(
                    "diffusion_model_cfg", config.diffusion_model_cfg)
                config.vl_self_attention_cfg = ah_cfg.get(
                    "vl_self_attention_cfg", config.vl_self_attention_cfg)
                config.action_head_hidden_size = ah_cfg.get(
                    "hidden_size", config.action_head_hidden_size)
                config.action_head_input_embedding_dim = ah_cfg.get(
                    "input_embedding_dim", config.action_head_input_embedding_dim)
                config.action_head_backbone_embedding_dim = ah_cfg.get(
                    "backbone_embedding_dim", config.action_head_backbone_embedding_dim)
                config.max_action_dim = ah_cfg.get(
                    "action_dim", config.max_action_dim)
                config.chunk_size = ah_cfg.get(
                    "action_horizon", config.chunk_size)
                config.num_inference_steps = ah_cfg.get(
                    "num_inference_timesteps", config.num_inference_steps)
                config.max_num_embodiments = ah_cfg.get(
                    "max_num_embodiments", config.max_num_embodiments)
                config.max_state_dim = ah_cfg.get(
                    "max_state_dim", config.max_state_dim)
                config.add_pos_embed = ah_cfg.get(
                    "add_pos_embed", config.add_pos_embed)
                config.use_vlln = ah_cfg.get(
                    "use_vlln", config.use_vlln)
                config.num_target_vision_tokens = ah_cfg.get(
                    "num_target_vision_tokens", config.num_target_vision_tokens)

            if "action_horizon" in checkpoint_config:
                config.chunk_size = checkpoint_config["action_horizon"]
            if "action_dim" in checkpoint_config:
                config.max_action_dim = checkpoint_config["action_dim"]

        model = cls(config)

        from safetensors.torch import load_file
        import glob

        single_file = os.path.join(local_model_path, "model.safetensors")
        if os.path.exists(single_file):
            safetensors_files = [single_file]
        else:
            safetensors_files = sorted(glob.glob(os.path.join(local_model_path, "*.safetensors")))
            if not safetensors_files:
                raise FileNotFoundError(
                    f"No safetensors weights found in {local_model_path}")

        original_state_dict = {}
        for sf in safetensors_files:
            original_state_dict.update(load_file(sf))

        mapped_sd = {}
        for key, value in original_state_dict.items():
            mapped_sd[key] = value

        incompatible = model.load_state_dict(mapped_sd, strict=False)
        if incompatible.unexpected_keys:
            print(f"[GROOT] Unexpected keys (ignored): {len(incompatible.unexpected_keys)}")
        if incompatible.missing_keys:
            print(f"[GROOT] Missing keys: {len(incompatible.missing_keys)}")
            for k in incompatible.missing_keys[:10]:
                print(f"  {k}")

        return model


class GrootPolicy(PreTrainedPolicy):
    """GR00T N1.5 Policy wrapper for training and inference.

    Follows the same preprocessing pattern as PI05Policy:
    - prepare_images(): resize + normalize on GPU (no CPU roundtrip)
    - prepare_language(): tokenize with Eagle tokenizer (CPU), move to GPU
    - Model forward takes explicit GPU tensors (compile-friendly)
    """

    config_class = GrootConfig
    name = "groot"

    def __init__(
        self,
        config: GrootConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Setup normalization modules
        norm_map: dict[FeatureType, NormalizationMode] = {}
        for ft_name, mode in config.normalization_mapping.items():
            norm_map[FeatureType(ft_name)] = mode

        self.normalize_inputs = Normalize(
            features=config.input_features,
            norm_map=norm_map,
            stats=dataset_stats,
        )
        self.normalize_targets = Normalize(
            features=config.output_features,
            norm_map=norm_map,
            stats=dataset_stats,
        )
        self.unnormalize_outputs = Unnormalize(
            features=config.output_features,
            norm_map=norm_map,
            stats=dataset_stats,
        )

        # Create model
        self.model = GrootModel(config)

        # Load Eagle tokenizer from cache directory
        cache_dir = self.model.backbone.cache_dir
        self.eagle_tokenizer = AutoTokenizer.from_pretrained(
            cache_dir, trust_remote_code=True)

        # Eagle image processing constants (same normalization as SigLIP: [0,1]→[-1,1])
        self.register_buffer(
            "_img_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "_img_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)

        if config.use_bf16:
            self.model.to_bfloat16_for_selected_params("bfloat16")

        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ):
        """Load pretrained GR00T policy."""
        if config is None:
            config = GrootConfig()

        if isinstance(pretrained_name_or_path, Path):
            pretrained_name_or_path = str(pretrained_name_or_path)

        config.base_model_path = pretrained_name_or_path
        dataset_stats = kwargs.pop("dataset_stats", None)

        # Without dataset stats, force identity normalization to avoid shape mismatches
        if dataset_stats is None:
            config.normalization_mapping = {
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.IDENTITY,
                "ACTION": NormalizationMode.IDENTITY,
            }

        # Load checkpoint config first — this updates max_action_dim, max_state_dim, etc.
        # We need these values before setting feature shapes.
        pretrained_model = GrootModel.from_pretrained(config, pretrained_name_or_path)

        # Now set feature shapes using the (potentially updated) config
        from lerobot.configs.types import PolicyFeature
        if not config.input_features:
            config.input_features = {
                "observation.images.default": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 224, 224)),
            }
        if OBS_STATE not in config.input_features:
            config.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(config.max_state_dim,))
        if not config.output_features:
            config.output_features = {
                ACTION: PolicyFeature(
                    type=FeatureType.ACTION, shape=(config.max_action_dim,)),
            }

        instance = cls(config, dataset_stats=dataset_stats)
        instance.model = pretrained_model

        if config.use_bf16:
            instance.model.to_bfloat16_for_selected_params("bfloat16")

        if config.device:
            instance.to(config.device)
        instance.eval()
        return instance

    def reset(self):
        """Reset action queue. Call when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    # -------------------------
    # Input preparation (PI05-style)
    # -------------------------

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for Eagle VLM on GPU.

        Matches GROOT's image processing: direct resize to 448x448 with bicubic
        interpolation (no aspect ratio preservation, no padding, 1 tile per image).
        Normalization: [0,1] -> [-1,1] via (x - 0.5) / 0.5.

        Args:
            batch: Input batch with image tensors.

        Returns:
            images: List of processed image tensors [B, C, H, W].
            img_masks: List of boolean masks [B] indicating valid images.
        """
        images: list[Tensor] = []
        img_masks: list[Tensor] = []

        eagle_size = self.model.backbone.eagle_image_size
        present_img_keys = [key for key in self.config.image_features if key in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            # Direct resize to Eagle tile size (matches GROOT: 1 tile, no padding)
            img = F.interpolate(img, size=(eagle_size, eagle_size),
                                mode="bicubic", align_corners=False).clamp(0, 1)

            # Normalize: [0, 1] → [-1, 1]
            img = (img - self._img_mean) / self._img_std

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, int]:
        """Tokenize task description using GROOT's chat template format.

        Matches GROOT's Eagle processor: constructs the same prompt format with
        image tokens at the correct positions in the sequence (after chat
        template prefix, not prepended at position 0).

        Args:
            batch: Input batch with 'task' field.

        Returns:
            input_ids: Token IDs with image placeholders [B, seq_len].
            attention_mask: Attention mask [B, seq_len].
            img_start: Position of first image token (plain int, compile-safe).
        """
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        num_image_token = self.model.backbone.num_image_token
        num_images = len([k for k in self.config.image_features if k in batch])

        # Build GROOT-format prompts matching Eagle's chat template exactly:
        # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        # <|im_start|>user\n<image 1><img><IMG_CONTEXT>×256</img>['task']<|im_end|>\n
        # <|im_start|>assistant\n
        IMG_CONTEXT = "<IMG_CONTEXT>"
        prompts = []
        for task in tasks:
            img_sections = []
            for i in range(num_images):
                img_sections.append(
                    f"<image {i + 1}><img>{IMG_CONTEXT * num_image_token}</img>"
                )
            img_str = "".join(img_sections)
            lang_str = str([task])
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{img_str}{lang_str}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)

        tokenized = self.eagle_tokenizer(
            prompts,
            padding="longest",
            return_tensors="pt",
            padding_side="right",
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Find where image tokens start (deterministic for a given template).
        # Computed as a plain int outside compile boundary — no graph breaks.
        image_token_index = self.model.backbone.image_token_index
        first_row = input_ids[0]
        img_positions = (first_row == image_token_index).nonzero(as_tuple=True)[0]
        img_start = img_positions[0].item() if len(img_positions) > 0 else 0

        return input_ids, attention_mask, img_start

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Pad state to max_state_dim."""
        return _pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Pad action to max_action_dim."""
        return _pad_vector(batch[ACTION], self.config.max_action_dim)

    # Matches GROOT's GrootPackInputsStep.embodiment_mapping
    EMBODIMENT_MAPPING = {
        "new_embodiment": 31, "oxe_droid": 17, "agibot_genie1": 26,
        "gr1": 24, "so100": 2, "unitree_g1": 3,
    }

    def prepare_embodiment_id(self, batch: dict[str, Tensor]) -> Tensor:
        """Get or create embodiment IDs from config's embodiment_tag."""
        if "embodiment_id" in batch:
            return batch["embodiment_id"]
        emb_id = self.EMBODIMENT_MAPPING.get(self.config.embodiment_tag, 0)
        bsz = batch[OBS_STATE].shape[0]
        device = batch[OBS_STATE].device
        return torch.full((bsz,), emb_id, dtype=torch.long, device=device)

    def prepare_action_mask(self, batch: dict[str, Tensor], action: Tensor) -> Tensor:
        """Create action mask for loss computation.

        Masks both:
        - Padded action dimensions (actual_dim..max_action_dim)
        - Temporally padded steps (beyond episode end)
        Matches GROOT's GrootPackInputsStep action_mask format.
        """
        if "action_mask" in batch:
            return batch["action_mask"]

        # Mask padded dimensions: only actual action dims contribute to loss
        actual_action_dim = self.config.output_features[ACTION].shape[0]
        mask = torch.zeros_like(action)
        mask[..., :actual_action_dim] = 1.0

        # Mask temporally padded steps
        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            mask = mask * in_episode_bound.unsqueeze(-1)
        return mask

    def _build_pixel_values_and_flags(
        self, images: list[Tensor], img_masks: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Stack images into pixel_values and build image_flags tensors.

        Args:
            images: List of [B, C, H, W] image tensors.
            img_masks: List of [B] boolean masks.

        Returns:
            pixel_values: [B * num_images, C, H, W] for Eagle vision tower.
            image_flags: [B, num_images] validity flags.
        """
        # Stack: list of [B, C, H, W] → [B, num_images, C, H, W]
        bsz = images[0].shape[0]
        stacked = torch.stack(images, dim=1)  # [B, N, C, H, W]
        pixel_values = stacked.view(-1, *stacked.shape[2:])  # [B*N, C, H, W]

        flags = torch.stack(img_masks, dim=1).long()  # [B, N]
        return pixel_values, flags

    # -------------------------
    # Forward passes
    # -------------------------

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass.

        Follows PI05's pattern: prepare inputs → model forward with explicit tensors.
        """
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        input_ids, attention_mask, img_start = self.prepare_language(batch)
        state = self.prepare_state(batch)
        action = self.prepare_action(batch)
        embodiment_id = self.prepare_embodiment_id(batch)
        action_mask = self.prepare_action_mask(batch, action)
        pixel_values, image_flags = self._build_pixel_values_and_flags(images, img_masks)
        num_images = len(images)

        device = next(self.parameters()).device
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        ):
            loss = self.model(
                pixel_values, input_ids, attention_mask, image_flags,
                state, action, action_mask, embodiment_id,
                num_images=num_images, img_start=img_start,
            )

        loss_dict = {"loss": loss.item()}
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions for inference.

        All preprocessing is GPU-resident; the full forward path is compile-friendly.
        """
        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        input_ids, attention_mask, img_start = self.prepare_language(batch)
        state = self.prepare_state(batch)
        embodiment_id = self.prepare_embodiment_id(batch)
        pixel_values, image_flags = self._build_pixel_values_and_flags(images, img_masks)
        num_images = len(images)

        device = next(self.parameters()).device
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        ):
            actions = self.model.sample_actions(
                pixel_values, input_ids, attention_mask, image_flags,
                state, embodiment_id,
                num_images=num_images, img_start=img_start,
            )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"action": actions})["action"]
        return actions[:, :self.config.n_action_steps, :]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action using action chunking."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1)[:self.config.n_action_steps])
        return self._action_queue.popleft()
