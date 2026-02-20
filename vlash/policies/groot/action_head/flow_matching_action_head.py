"""Flow matching action head for GR00T.

Ported from Isaac-GR00T / lerobot. This module contains the core action
prediction logic: multi-embodiment encoders, flow matching training, and
iterative denoising for inference.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from vlash.policies.groot.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from vlash.policies.groot.action_head.cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    """Per-embodiment linear layer with separate weights per category."""

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        # TODO(cudagraph): advanced indexing with cat_ids may break graphs
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Per-embodiment 2-layer MLP."""

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Encodes noisy actions with per-embodiment weights and sinusoidal time encoding."""

    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        Args:
            actions:   (B, T, action_dim)
            timesteps: (B,)
            cat_ids:   (B,)
        Returns:
            (B, T, hidden_size)
        """
        b, t, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == b:
            timesteps = timesteps.unsqueeze(1).expand(-1, t)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig:
    """Configuration for the flow matching action head."""

    add_pos_embed: bool = True
    model_dtype: str = "float32"
    diffusion_model_cfg: dict = field(default_factory=dict)
    input_embedding_dim: int = 1536
    backbone_embedding_dim: int = 1536
    hidden_size: int = 1024
    max_seq_len: int = 1024
    action_dim: int | None = None
    action_horizon: int | None = None
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int | None = None
    max_num_embodiments: int = 32
    max_state_dim: int = 64
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    load_pretrained_det_decode_layer_path: str | None = None
    detection_coeff: float = 1.0
    freeze_decode_layer: bool = False
    expand_batch: int | None = None
    use_vlln: bool = True
    vl_self_attention_cfg: dict | None = None
    num_target_vision_tokens: int = 32


class FlowmatchingActionHead(nn.Module):
    """Flow matching action head with DiT cross-attention and multi-embodiment support."""

    supports_gradient_checkpointing = True

    def __init__(self, config: FlowmatchingActionHeadConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # TODO(cudagraph): Beta distribution sampling may break graphs
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        # TODO(cudagraph): Beta.sample() is not graph-safe
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def process_backbone_output(self, backbone_features, backbone_attention_mask):
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        return backbone_features, backbone_attention_mask

    def forward(self, backbone_features, backbone_attention_mask, state, action,
                action_mask, embodiment_id):
        """Training forward: compute flow matching loss.

        Args:
            backbone_features: VLM features [B, S, D].
            backbone_attention_mask: [B, S].
            state: Robot state [B, state_dim].
            action: Ground truth actions [B, T, action_dim].
            action_mask: Per-element mask [B, T, action_dim].
            embodiment_id: Per-sample embodiment index [B].

        Returns:
            loss: Scalar flow matching loss.
        """
        self.set_frozen_modules_to_eval_mode()

        vl_embs, vl_attn_mask = self.process_backbone_output(
            backbone_features, backbone_attention_mask)
        device = vl_embs.device

        # State is [B, state_dim] -> [B, 1, state_dim] for CategorySpecificLinear (bmm)
        state_features = self.state_encoder(state.unsqueeze(1), embodiment_id)

        noise = torch.randn(action.shape, device=action.device, dtype=action.dtype)
        t = self.sample_time(action.shape[0], device=action.device, dtype=action.dtype)
        t = t[:, None, None]  # (B, 1, 1)

        noisy_trajectory = (1 - t) * noise + t * action
        velocity = action - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -action.shape[1]:]

        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        return loss

    @torch.no_grad()
    def get_action(self, backbone_features, backbone_attention_mask, state, embodiment_id):
        """Inference: iterative denoising to predict actions.

        Args:
            backbone_features: VLM features [B, S, D].
            backbone_attention_mask: [B, S].
            state: Robot state [B, state_dim].
            embodiment_id: Per-sample embodiment index [B].

        Returns:
            actions: Predicted actions [B, T, action_dim].
        """
        vl_embs, _ = self.process_backbone_output(backbone_features, backbone_attention_mask)

        state_features = self.state_encoder(state.unsqueeze(1), embodiment_id)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # TODO(cudagraph): Python for-loop prevents full graph capture.
        # Consider unrolling or using torch.compile with dynamic=True.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]

            actions = actions + dt * pred_velocity

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
