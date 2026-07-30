import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from enfold.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .gradient_checkpointing import gradient_checkpoint_forward
from .scheduler import ContinuousFlowMatchScheduler
from .transformer import SelfAttention, modulate, precompute_freqs_cis, sinusoidal_embedding_1d

logger = get_logger(__name__)






def _dino_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _dino_patch_center_coordinates(
    num_patches_h: int,
    num_patches_w: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device) / num_patches_h
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device) / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    return 2.0 * coords.flatten(0, 1) - 1.0


def _augment_dino_patch_coordinates(
    coords: torch.Tensor,
    shift: Optional[float] = None,
    jitter: Optional[float] = None,
    rescale: Optional[float] = None,
) -> torch.Tensor:
    if shift is not None:
        coords = coords + torch.empty((1, 2), device=coords.device, dtype=coords.dtype).uniform_(-float(shift), float(shift))
    if jitter is not None:
        jitter_range = math.log(float(jitter))
        jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype).uniform_(-jitter_range, jitter_range).exp()
        coords = coords * jitter_hw
    if rescale is not None:
        rescale_range = math.log(float(rescale))
        rescale_factor = torch.empty(1, device=coords.device, dtype=coords.dtype).uniform_(-rescale_range, rescale_range).exp()
        coords = coords * rescale_factor
    return coords


class LocalDINOv3RopePositionEmbedding(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.base = float(cfg.get("rope_theta", 100.0))
        self.head_dim = int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])
        inv_freq = 1.0 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        patch_size = int(self.cfg["patch_size"])
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            coords = _dino_patch_center_coordinates(
                num_patches_h=num_patches_h,
                num_patches_w=num_patches_w,
                dtype=torch.float32,
                device=device,
            )
            if self.training:
                coords = _augment_dino_patch_coordinates(
                    coords,
                    shift=self.cfg.get("pos_embed_shift"),
                    jitter=self.cfg.get("pos_embed_jitter"),
                    rescale=self.cfg.get("pos_embed_rescale"),
                )
            angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2).tile(2)
            cos = torch.cos(angles)
            sin = torch.sin(angles)
        return cos.to(dtype=pixel_values.dtype), sin.to(dtype=pixel_values.dtype)


class LocalDINOv3Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, proj_bias: bool = True, key_bias: bool = False, value_bias: bool = True):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bool(proj_bias))
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bool(key_bias))
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bool(value_bias))
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        num_patches = int(sin.shape[-2])
        num_prefix_tokens = int(q.shape[-2] - num_patches)
        if num_prefix_tokens < 0:
            raise ValueError(
                f"DINOv3 RoPE received more patch positions ({num_patches}) than sequence tokens ({q.shape[-2]})."
            )
        q_prefix, q_patch = q.split((num_prefix_tokens, num_patches), dim=-2)
        k_prefix, k_patch = k.split((num_prefix_tokens, num_patches), dim=-2)
        cos = cos.to(device=q.device, dtype=q.dtype).view(1, 1, num_patches, self.head_dim)
        sin = sin.to(device=q.device, dtype=q.dtype).view(1, 1, num_patches, self.head_dim)
        q_patch = (q_patch * cos) + (_dino_rotate_half(q_patch) * sin)
        k_patch = (k_patch * cos) + (_dino_rotate_half(k_patch) * sin)
        q = torch.cat([q_prefix, q_patch], dim=-2)
        k = torch.cat([k_prefix, k_patch], dim=-2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.o_proj(x)


class LocalDINOv3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        use_gated_mlp: bool = False,
        hidden_act: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        self.use_gated_mlp = bool(use_gated_mlp)
        self.hidden_act = str(hidden_act).lower()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bool(bias))
        self.gate_proj = (
            nn.Linear(hidden_size, intermediate_size, bias=bool(bias))
            if self.use_gated_mlp
            else None
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bool(bias))

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_act in {"silu", "swish"}:
            return F.silu(x)
        if self.hidden_act in {"gelu", "gelu_pytorch_tanh"}:
            return F.gelu(x, approximate="tanh" if self.hidden_act == "gelu_pytorch_tanh" else "none")
        raise ValueError(f"Unsupported DINOv3 MLP activation: {self.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.up_proj(x)
        if self.gate_proj is not None:
            x_up = x_up * self._activation(self.gate_proj(x))
        else:
            x_up = self._activation(x_up)
        return self.down_proj(x_up)


class LocalDINOv3LayerScale(nn.Module):
    def __init__(self, hidden_size: int, init_value: float):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size) * float(init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1.to(device=x.device, dtype=x.dtype)


class LocalDINOv3Block(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        self.norm1 = nn.LayerNorm(hidden_size, eps=float(cfg.get("layer_norm_eps", 1e-5)))
        self.attention = LocalDINOv3Attention(
            hidden_size=hidden_size,
            num_heads=int(cfg["num_attention_heads"]),
            proj_bias=bool(cfg.get("query_bias", True)),
            key_bias=bool(cfg.get("key_bias", False)),
            value_bias=bool(cfg.get("value_bias", True)),
        )
        self.layer_scale1 = LocalDINOv3LayerScale(hidden_size, float(cfg.get("layerscale_value", 1.0)))
        self.norm2 = nn.LayerNorm(hidden_size, eps=float(cfg.get("layer_norm_eps", 1e-5)))
        self.mlp = LocalDINOv3MLP(
            hidden_size,
            int(cfg["intermediate_size"]),
            use_gated_mlp=bool(cfg.get("use_gated_mlp", False)),
            hidden_act=str(cfg.get("hidden_act", "gelu")),
            bias=bool(cfg.get("mlp_bias", True)),
        )
        self.layer_scale2 = LocalDINOv3LayerScale(hidden_size, float(cfg.get("layerscale_value", 1.0)))

    def forward(self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x), position_embeddings=position_embeddings))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class LocalDINOv3ViTModel(nn.Module):
    """Local DINOv3 ViT-H/16+ loader for the fixed Enfold student."""

    def __init__(self, model_dir: str | Path):
        super().__init__()
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "Enfold requires a local DINOv3 checkpoint directory containing "
                f"config.json and model.safetensors: {model_dir}"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.config = cfg
        hidden_size = int(cfg["hidden_size"])
        patch_size = int(cfg["patch_size"])
        num_channels = int(cfg.get("num_channels", 3))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, int(cfg.get("num_register_tokens", 0)), hidden_size))
        self.patch_embeddings = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.rope_embeddings = LocalDINOv3RopePositionEmbedding(cfg)
        self.layer = nn.ModuleList([LocalDINOv3Block(cfg) for _ in range(int(cfg["num_hidden_layers"]))])
        self.norm = nn.LayerNorm(hidden_size, eps=float(cfg.get("layer_norm_eps", 1e-5)))
        state = load_file(str(weights_path), device="cpu")
        mapped = {}
        for key, value in state.items():
            if key == "embeddings.cls_token":
                mapped["cls_token"] = value
            elif key == "embeddings.mask_token":
                mapped["mask_token"] = value
            elif key == "embeddings.register_tokens":
                mapped["register_tokens"] = value
            elif key.startswith("embeddings.patch_embeddings."):
                mapped[key.replace("embeddings.", "", 1)] = value
            else:
                mapped[key] = value
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if unexpected:
            logger.warning("Unexpected DINOv3 checkpoint keys ignored: %s", unexpected[:10])
        ignored_missing = [key for key in missing if key != "register_tokens" or self.register_tokens.numel() > 0]
        if ignored_missing:
            logger.warning("Missing DINOv3 checkpoint keys initialized randomly: %s", ignored_missing[:10])

    def forward(
        self,
        pixel_values: torch.Tensor,
        injected_text_tokens: torch.Tensor,
        text_injection_layer: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        if len(self.layer) <= 31:
            raise ValueError("Enfold requires a 32-layer DINOv3 ViT-H/16+ backbone.")
        text_injection_layer = int(text_injection_layer)
        if not 1 <= text_injection_layer <= len(self.layer):
            raise ValueError(
                f"text_concat_start_layer must be in [1, {len(self.layer)}], got {text_injection_layer}."
            )
        if injected_text_tokens.ndim != 3:
            raise ValueError(
                f"injected_text_tokens must be [B,L,D], got {tuple(injected_text_tokens.shape)}"
            )

        patch_tokens = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2).contiguous()
        batch_size = patch_tokens.shape[0]
        if injected_text_tokens.shape[0] != batch_size:
            raise ValueError(
                "Batch mismatch between pixel values and injected text tokens: "
                f"{batch_size} vs {injected_text_tokens.shape[0]}"
            )
        if injected_text_tokens.shape[-1] != patch_tokens.shape[-1]:
            raise ValueError(
                f"injected_text_tokens hidden dim must be {patch_tokens.shape[-1]}, "
                f"got {injected_text_tokens.shape[-1]}"
            )

        cls = self.cls_token.to(dtype=patch_tokens.dtype, device=patch_tokens.device).expand(batch_size, -1, -1)
        registers = self.register_tokens.to(dtype=patch_tokens.dtype, device=patch_tokens.device).expand(
            batch_size, -1, -1
        )
        prefix_len = 1 + self.register_tokens.shape[1]
        x = torch.cat([cls, registers, patch_tokens], dim=1)
        position_embeddings = self.rope_embeddings(pixel_values)
        hidden_states: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.layer, start=1):
            if layer_idx == text_injection_layer:
                x = torch.cat(
                    [x[:, :prefix_len], injected_text_tokens.to(device=x.device, dtype=x.dtype), x[:, prefix_len:]],
                    dim=1,
                )
            x = block(x, position_embeddings=position_embeddings)
            hidden_states.append(x)
        return self.norm(x), hidden_states, int(injected_text_tokens.shape[1])


class DinoVisionTokenEncoder(nn.Module):
    """DINOv3 token encoder loaded exclusively from a local checkpoint directory."""

    def __init__(self, model_id: str, output_dim: int):
        super().__init__()
        if not model_id:
            raise FileNotFoundError("Enfold requires a local DINOv3 checkpoint directory in student_config.model_id.")
        model_path = Path(model_id)
        self.model_id = str(model_path)
        self.output_dim = int(output_dim)
        self.backbone = LocalDINOv3ViTModel(model_path)
        self.backbone_dim = int(self.backbone.config["hidden_size"])
        self.num_prefix_tokens = 1 + int(self.backbone.config.get("num_register_tokens", 0))
        self.proj = (
            nn.Linear(self.backbone_dim, self.output_dim)
            if self.backbone_dim != self.output_dim
            else nn.Identity()
        )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"image must be 4D [B,3,H,W], got {tuple(image.shape)}")
        if image.shape[1] != 3:
            raise ValueError(f"image channel dimension must be 3, got {image.shape[1]}")
        image = image.to(dtype=next(self.backbone.parameters()).dtype)
        image = image.add(1.0).mul(0.5).clamp(0.0, 1.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    def _select_patch_tokens(self, tokens: torch.Tensor, text_token_count: int) -> torch.Tensor:
        patch_start = self.num_prefix_tokens + int(text_token_count)
        if tokens.shape[1] <= patch_start:
            raise ValueError(
                f"Cannot select DINO patch tokens after {self.num_prefix_tokens} prefix and "
                f"{text_token_count} text tokens from sequence length {tokens.shape[1]}."
            )
        return tokens[:, patch_start:]

    def forward_with_feature_states(
        self,
        image: torch.Tensor,
        injected_text_tokens: torch.Tensor,
        text_injection_layer: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        final_tokens, hidden_states, text_token_count = self.backbone(
            self._normalize_image(image),
            injected_text_tokens=injected_text_tokens,
            text_injection_layer=text_injection_layer,
        )
        return (
            self._select_patch_tokens(final_tokens, text_token_count),
            [
                self._select_patch_tokens(
                    tokens,
                    text_token_count if layer_idx >= text_injection_layer else 0,
                )
                for layer_idx, tokens in enumerate(hidden_states, start=1)
            ],
        )


class DinoTextImageStudent(nn.Module):
    """Enfold DINOv3 student with configurable text injection and prediction features."""

    def __init__(
        self,
        model_id: str,
        text_dim: int,
        hidden_dim: int,
        text_dropout_prob: float = 0.1,
        observation_horizon: int = 1,
        add_frame_embedding: bool = True,
        text_projection_type: str = "linear",
        text_concat_start_layer: int = 8,
        deep_supervision: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.text_dropout_prob = float(text_dropout_prob)
        self.observation_horizon = int(observation_horizon)
        if self.observation_horizon <= 0:
            raise ValueError(f"observation_horizon must be positive, got {self.observation_horizon}")
        self.text_concat_start_layer = int(text_concat_start_layer)
        if self.text_concat_start_layer <= 0:
            raise ValueError(
                f"text_concat_start_layer must be positive, got {self.text_concat_start_layer}."
            )
        self.text_projection_type = str(text_projection_type).lower()
        if self.text_projection_type not in {"linear", "mlp"}:
            raise ValueError(
                f"text_projection_type must be linear or mlp, got {text_projection_type!r}"
            )

        self.image_encoder = DinoVisionTokenEncoder(
            model_id=model_id,
            output_dim=self.hidden_dim,
        )
        if len(self.image_encoder.backbone.layer) <= 31:
            raise ValueError("Enfold requires a 32-layer DINOv3 ViT-H/16+ backbone.")

        self.frame_embedding = (
            nn.Parameter(torch.zeros(self.observation_horizon, self.hidden_dim))
            if self.observation_horizon > 1 and bool(add_frame_embedding)
            else None
        )
        if self.text_projection_type == "linear":
            self.text_proj = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, self.image_encoder.backbone_dim),
            )
        else:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, self.image_encoder.backbone_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(self.image_encoder.backbone_dim, self.image_encoder.backbone_dim),
            )

        feature_config = {} if deep_supervision is None else dict(deep_supervision)
        self.deep_supervision_enabled = bool(feature_config.get("enabled", True))
        self.deep_supervision_feature_layers = tuple(
            int(layer) for layer in feature_config.get("feature_layers", (7, 15, 23, 31))
        )
        if self.deep_supervision_enabled:
            if not self.deep_supervision_feature_layers:
                raise ValueError("deep_supervision.feature_layers must not be empty when enabled.")
            max_layer = len(self.image_encoder.backbone.layer) - 1
            if min(self.deep_supervision_feature_layers) < 0 or max(self.deep_supervision_feature_layers) > max_layer:
                raise ValueError(
                    f"deep_supervision.feature_layers must be in [0, {max_layer}], "
                    f"got {self.deep_supervision_feature_layers}."
                )
            self.prediction_condition_norm = nn.LayerNorm(
                int(self.image_encoder.backbone_dim) * len(self.deep_supervision_feature_layers)
            )
        else:
            self.prediction_condition_norm = None
        self.norm = nn.LayerNorm(self.hidden_dim)

    @property
    def prediction_condition_dim(self) -> int:
        if self.deep_supervision_enabled:
            return int(self.image_encoder.backbone_dim) * len(self.deep_supervision_feature_layers)
        return self.hidden_dim

    def _encode_single_frame_head_tokens(
        self,
        first_frame: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_patch_tokens, feature_patch_tokens = self.image_encoder.forward_with_feature_states(
            first_frame,
            injected_text_tokens=text_tokens,
            text_injection_layer=self.text_concat_start_layer,
        )
        action_tokens = self.image_encoder.proj(final_patch_tokens)
        if not self.deep_supervision_enabled:
            return action_tokens, action_tokens
        if self.prediction_condition_norm is None:
            raise RuntimeError("Prediction feature normalization is not initialized.")
        last_layer = len(feature_patch_tokens) - 1
        prediction_features = [
            final_patch_tokens if layer_idx == last_layer else feature_patch_tokens[layer_idx]
            for layer_idx in self.deep_supervision_feature_layers
        ]
        prediction_tokens = self.prediction_condition_norm(torch.cat(prediction_features, dim=-1))
        return action_tokens, prediction_tokens

    def _encode_image_head_tokens(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim == 4:
            return self._encode_single_frame_head_tokens(image, text_tokens=text_tokens)
        if image.ndim != 5:
            raise ValueError(f"image must be [B,3,H,W], [B,3,T,H,W], or [B,T,3,H,W], got {tuple(image.shape)}")
        if image.shape[1] == 3:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        elif image.shape[2] != 3:
            raise ValueError(f"image must have a 3-channel dimension, got {tuple(image.shape)}")

        batch_size, num_frames, channels, height, width = image.shape
        if num_frames > self.observation_horizon:
            image = image[:, : self.observation_horizon]
            num_frames = self.observation_horizon
        flat_image = image.view(batch_size * num_frames, channels, height, width)
        flat_text_tokens = text_tokens[:, None].expand(batch_size, num_frames, -1, -1)
        flat_text_tokens = flat_text_tokens.reshape(
            batch_size * num_frames, text_tokens.shape[1], text_tokens.shape[2]
        )
        action_tokens, prediction_tokens = self._encode_single_frame_head_tokens(
            flat_image,
            text_tokens=flat_text_tokens,
        )

        tokens_per_frame = action_tokens.shape[1]
        action_tokens = action_tokens.view(batch_size, num_frames, tokens_per_frame, action_tokens.shape[2])
        prediction_tokens = prediction_tokens.view(
            batch_size, num_frames, tokens_per_frame, prediction_tokens.shape[2]
        )
        if self.frame_embedding is not None:
            frame_embed = self.frame_embedding[:num_frames].to(
                device=action_tokens.device, dtype=action_tokens.dtype
            ).view(1, num_frames, 1, -1)
            action_tokens = action_tokens + frame_embed
            prediction_tokens = prediction_tokens + frame_embed
        return action_tokens.flatten(1, 2), prediction_tokens.flatten(1, 2)

    def forward(
        self,
        first_frame: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        drop_text: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del context_mask
        if context.ndim != 3:
            raise ValueError(f"context must be 3D [B,L,D], got {tuple(context.shape)}")
        image_for_dtype = first_frame[:, :, 0] if first_frame.ndim == 5 and first_frame.shape[1] == 3 else first_frame
        text_tokens = self.text_proj(context.to(dtype=image_for_dtype.dtype))

        if drop_text is None:
            drop_text = self.training and torch.rand((), device=context.device).item() < self.text_dropout_prob
        if drop_text:
            text_tokens = torch.zeros_like(text_tokens)

        action_image_tokens, prediction_image_tokens = self._encode_image_head_tokens(
            first_frame,
            text_tokens=text_tokens,
        )
        action_image_tokens = self.norm(action_image_tokens)
        if not self.deep_supervision_enabled:
            prediction_image_tokens = self.norm(prediction_image_tokens)
        image_token_mask = torch.ones(
            action_image_tokens.shape[:2], dtype=torch.bool, device=action_image_tokens.device
        )
        return (
            action_image_tokens,
            image_token_mask,
            action_image_tokens,
            image_token_mask,
            prediction_image_tokens,
            image_token_mask,
        )


class ConcatConditionDiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def forward(self, x, _context, t_mod, freqs, context_mask=None, self_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class TokenDiTHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        condition_dim: int,
        ffn_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        max_seq_len: int = 4096,
        use_3d_rope: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.freq_dim = int(freq_dim)
        self.condition_dim = int(condition_dim)
        self.query = nn.Parameter(torch.randn(1, int(max_seq_len), self.hidden_dim) / self.hidden_dim**0.5)
        self.use_3d_rope = bool(use_3d_rope)
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                ConcatConditionDiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=int(attn_head_dim),
                    num_heads=int(num_heads),
                    ffn_dim=int(ffn_dim),
                    eps=float(eps),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        self.attn_head_dim = int(attn_head_dim)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=int(max_seq_len))
        h_rope_dim = max(2, (self.attn_head_dim // 3) // 2 * 2)
        w_rope_dim = h_rope_dim
        t_rope_dim = self.attn_head_dim - h_rope_dim - w_rope_dim
        if t_rope_dim <= 0:
            raise ValueError(f"`attn_head_dim` is too small for 3D RoPE: {self.attn_head_dim}")
        if t_rope_dim % 2 != 0:
            t_rope_dim += 1
            h_rope_dim -= 2
        if h_rope_dim <= 0 or w_rope_dim <= 0 or t_rope_dim <= 0:
            raise ValueError(
                f"Invalid 3D RoPE split for attn_head_dim={self.attn_head_dim}: "
                f"t={t_rope_dim}, h={h_rope_dim}, w={w_rope_dim}"
            )
        if t_rope_dim + h_rope_dim + w_rope_dim != self.attn_head_dim:
            raise ValueError(
                f"3D RoPE split must sum to attn_head_dim={self.attn_head_dim}, "
                f"got t={t_rope_dim}, h={h_rope_dim}, w={w_rope_dim}"
            )
        self.rope_3d_dims = (t_rope_dim, h_rope_dim, w_rope_dim)
        self.freqs_3d = (
            precompute_freqs_cis(t_rope_dim, end=int(max_seq_len)),
            precompute_freqs_cis(h_rope_dim, end=int(max_seq_len)),
            precompute_freqs_cis(w_rope_dim, end=int(max_seq_len)),
        )
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def _build_freqs(
        self,
        seq_len: int,
        device: torch.device,
        grid_size: Optional[tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if grid_size is None or not self.use_3d_rope:
            return self.freqs[:seq_len].view(seq_len, 1, -1).to(device)
        f, h, w = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
        if f <= 0 or h <= 0 or w <= 0:
            raise ValueError(f"`grid_size` entries must be positive, got {grid_size}")
        if f * h * w != seq_len:
            raise ValueError(f"`grid_size` product {f*h*w} must equal seq_len {seq_len}, got {grid_size}")
        return torch.cat(
            [
                self.freqs_3d[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs_3d[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs_3d[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1).to(device)

    def forward(
        self,
        seq_len: int,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        grid_size: Optional[tuple[int, int, int]] = None,
        return_full_sequence: bool = False,
    ) -> torch.Tensor:
        del condition_mask
        if seq_len <= 0:
            raise ValueError(f"`seq_len` must be positive, got {seq_len}")
        if seq_len > self.query.shape[1]:
            raise ValueError(f"`seq_len`={seq_len} exceeds `max_seq_len`={self.query.shape[1]}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got {tuple(timestep.shape)}")
        batch_size = condition.shape[0]
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` batch mismatch: {timestep.shape[0]} vs {batch_size}")
        x = self.query[:, :seq_len].to(device=condition.device, dtype=condition.dtype).expand(batch_size, -1, -1)
        context = self.condition_proj(condition)
        condition_len = int(context.shape[1])
        x = torch.cat([context, x], dim=1)
        total_len = condition_len + seq_len
        if total_len > self.freqs.shape[0]:
            raise ValueError(
                f"`condition_len + query_len`={total_len} exceeds max_seq_len={self.freqs.shape[0]}"
            )
        freqs = self._build_freqs(seq_len=total_len, device=x.device, grid_size=grid_size)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=x.dtype, device=x.device)
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)
        if not return_full_sequence:
            x = x[:, condition_len:]
        return self.head(self.norm(x))


class EnfoldTrainable(nn.Module):
    def __init__(
        self,
        teacher_video_expert: Optional[nn.Module],
        student: nn.Module,
        prediction_head: Optional[nn.Module],
        action_head: Optional[nn.Module],
        student_to_wan_context: Optional[nn.Module] = None,
        proprio_to_action_context: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.teacher_video_expert = teacher_video_expert
        self.student = student
        self.prediction_head = prediction_head
        self.action_head = action_head
        self.student_to_wan_context = student_to_wan_context
        self.proprio_to_action_context = proprio_to_action_context


class EnfoldModelBase(nn.Module):
    """Shared DINO student, representation distillation, and action-denoising logic."""

    @staticmethod
    def _resolve_prediction_head_config(
        prediction_head_config: dict[str, Any],
        student: DinoTextImageStudent,
    ) -> dict[str, Any]:
        resolved = dict(prediction_head_config)
        resolved["condition_dim"] = int(student.prediction_condition_dim)
        return resolved

    def __init__(
        self,
        teacher_video_expert: Optional[nn.Module],
        vae: Optional[nn.Module],
        student: DinoTextImageStudent,
        prediction_head: Optional[TokenDiTHead],
        action_head: Optional[ActionDiT],
        proprio_dim: int,
        text_encoder=None,
        tokenizer=None,
        teacher_feature_layers: Sequence[int] = (29,),
        text_dim: int = 4096,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_num_train_timesteps: int = 1000,
        action_uniform_sigma_sampling: bool = False,
        loss_lambda_video: float = 1.0,
        loss_lambda_repr: float = 1.0,
        loss_lambda_action: float = 1.0,
        repr_distillation: Optional[dict[str, Any]] = None,
        disable_action_head: bool = False,
    ):
        super().__init__()
        self.disable_action_head = bool(disable_action_head)
        student_to_wan_context = nn.Sequential(
            nn.LayerNorm(student.hidden_dim),
            nn.Linear(student.hidden_dim, int(text_dim)),
        )
        self.proprio_condition_dim = int(proprio_dim)
        if self.proprio_condition_dim <= 0:
            raise ValueError(f"proprio_dim must be positive, got {self.proprio_condition_dim}.")
        proprio_to_action_context = nn.Sequential(
            nn.LayerNorm(self.proprio_condition_dim),
            nn.Linear(self.proprio_condition_dim, student.hidden_dim),
        )
        repr_distillation = {} if repr_distillation is None else dict(repr_distillation)
        self.repr_exclude_conditional_frames = bool(repr_distillation.get("exclude_conditional_frames", False))
        self.repr_first_frame_weight = float(repr_distillation.get("first_frame_weight", 1.0))
        self.repr_other_frames_weight = float(repr_distillation.get("other_frames_weight", 1.0))
        self.teacher_video_expert = teacher_video_expert
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_dim = int(text_dim)
        self.teacher_feature_layers = tuple(int(i) for i in teacher_feature_layers)
        self.dit = EnfoldTrainable(
            teacher_video_expert,
            student,
            prediction_head,
            action_head,
            student_to_wan_context=student_to_wan_context,
            proprio_to_action_context=proprio_to_action_context,
        )
        self.teacher_video_expert = self.dit.teacher_video_expert
        self.student = self.dit.student
        self.prediction_head = self.dit.prediction_head
        self.action_head = self.dit.action_head
        self.student_to_wan_context = self.dit.student_to_wan_context
        self.proprio_to_action_context = self.dit.proprio_to_action_context
        self.train_video_scheduler = ContinuousFlowMatchScheduler(video_num_train_timesteps, video_train_shift)
        self.infer_video_scheduler = ContinuousFlowMatchScheduler(video_num_train_timesteps, video_infer_shift)
        self.train_action_scheduler = ContinuousFlowMatchScheduler(
            action_num_train_timesteps,
            action_train_shift,
            uniform_sigma_sampling=action_uniform_sigma_sampling,
        )
        self.infer_action_scheduler = ContinuousFlowMatchScheduler(action_num_train_timesteps, action_infer_shift)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_repr = float(loss_lambda_repr)
        self.loss_lambda_action = float(loss_lambda_action)
        if self.vae is not None:
            self.vae.eval()
            self.vae.requires_grad_(False)
        self.to(device=self.device, dtype=self.torch_dtype)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.teacher_video_expert is not None:
            self.teacher_video_expert.to(*args, **kwargs)
        if self.vae is not None:
            self.vae.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if self.vae is None:
            raise RuntimeError("Video latent encoding is unavailable in action-only eval mode.")
        return self.vae.encode(video_tensor, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if self.vae is None:
            raise RuntimeError("Video latent decoding is unavailable in action-only eval mode.")
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        return [Image.fromarray(video_tensor[:, t].permute(1, 2, 0).numpy()) for t in range(video_tensor.shape[1])]

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb.to(device=self.device, dtype=self.torch_dtype), torch.ones_like(mask)

    def _prepare_context(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("Training requires cached `context` and `context_mask`.")
        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}")
        return context, context_mask

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    def _prepare_infer_context(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is None or context_mask is None:
            if prompt is None:
                raise ValueError("Either `prompt` or cached `context/context_mask` must be provided.")
            context, context_mask = self.encode_prompt(prompt)
        context = context.unsqueeze(0) if context.ndim == 2 else context
        context_mask = context_mask.unsqueeze(0) if context_mask.ndim == 1 else context_mask
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}")
        return (
            context.to(device=self.device, dtype=self.torch_dtype),
            context_mask.to(device=self.device, dtype=torch.bool),
        )

    def _student_condition(
        self,
        input_image: torch.Tensor,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        *,
        drop_text: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context, context_mask = self._prepare_infer_context(prompt, context, context_mask)
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim == 4:
            valid_shape = input_image.shape[0] == 1 and input_image.shape[1] == 3
        elif input_image.ndim == 5:
            valid_shape = input_image.shape[0] == 1 and (input_image.shape[1] == 3 or input_image.shape[2] == 3)
        else:
            valid_shape = False
        if not valid_shape:
            raise ValueError(
                "`input_image` must have shape [3,H,W], [1,3,H,W], [1,3,T,H,W], "
                f"or [1,T,3,H,W], got {tuple(input_image.shape)}"
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        student_outputs = self.student(input_image, context, context_mask, drop_text=drop_text)
        return student_outputs[0], student_outputs[1], student_outputs[2], student_outputs[3]

    @staticmethod
    def _slice_repr_teacher_features(
        teacher_features: torch.Tensor,
        grid_size: tuple[int, int, int],
        num_conditional_frames: int,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        time_len, height_len, width_len = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
        tokens_per_frame = height_len * width_len
        cond_frames = int(num_conditional_frames)
        if cond_frames <= 0:
            return teacher_features, (time_len, height_len, width_len)
        cond_tokens = cond_frames * tokens_per_frame
        if cond_tokens >= teacher_features.shape[1]:
            raise ValueError(
                f"Cannot exclude {cond_frames} conditional frame(s) from teacher features with "
                f"length {teacher_features.shape[1]} and grid {grid_size}."
            )
        future_features = teacher_features[:, cond_tokens:]
        future_grid = (time_len - cond_frames, height_len, width_len)
        return future_features, future_grid

    def _compute_repr_loss(
        self,
        pred_features: torch.Tensor,
        teacher_features: torch.Tensor,
        grid_size: tuple[int, int, int],
        *,
        normalize_teacher: bool = True,
    ) -> torch.Tensor:
        target = teacher_features
        if normalize_teacher:
            target = F.layer_norm(teacher_features.float(), (teacher_features.shape[-1],))
        if (
            self.repr_first_frame_weight == 1.0
            and self.repr_other_frames_weight == 1.0
        ):
            return F.smooth_l1_loss(pred_features, target)
        time_len, height_len, width_len = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
        tokens_per_frame = height_len * width_len
        if pred_features.shape[1] != time_len * tokens_per_frame:
            raise ValueError(
                f"Repr loss grid mismatch: pred_len={pred_features.shape[1]}, "
                f"expected {time_len * tokens_per_frame} for grid {grid_size}."
            )
        per_token = F.smooth_l1_loss(pred_features.float(), target.float(), reduction="none").mean(dim=-1)
        frame_loss = per_token.view(pred_features.shape[0], time_len, tokens_per_frame).mean(dim=-1)
        weights = torch.full(
            (time_len,),
            float(self.repr_other_frames_weight),
            device=frame_loss.device,
            dtype=frame_loss.dtype,
        )
        if time_len > 0:
            weights[0] = float(self.repr_first_frame_weight)
        weight_sum = weights.sum().clamp(min=1e-6)
        weighted_per_sample = (frame_loss * weights.view(1, -1)).sum(dim=-1) / weight_sum
        return weighted_per_sample.mean()

    def _compute_repr_branch(
        self,
        *,
        teacher_features: torch.Tensor,
        teacher_grid_size: tuple[int, int, int],
        video_timestep: torch.Tensor,
        prediction_condition: torch.Tensor,
        prediction_condition_mask: Optional[torch.Tensor],
        num_conditional_frames: int,
        normalize_teacher: bool = True,
    ) -> torch.Tensor:
        if self.prediction_head is None:
            raise RuntimeError("Prediction head is not loaded.")
        query_teacher_features, query_grid_size = teacher_features, teacher_grid_size
        return_full_sequence = False
        if self.repr_exclude_conditional_frames and num_conditional_frames > 0:
            query_teacher_features, query_grid_size = self._slice_repr_teacher_features(
                teacher_features,
                teacher_grid_size,
                num_conditional_frames,
            )
            return_full_sequence = True
        _, height_len, width_len = (
            int(teacher_grid_size[0]),
            int(teacher_grid_size[1]),
            int(teacher_grid_size[2]),
        )
        teacher_tokens_per_frame = height_len * width_len
        if teacher_tokens_per_frame <= 0:
            raise ValueError(f"Invalid teacher grid spatial size: {teacher_grid_size}")
        condition_tokens = int(prediction_condition.shape[1])
        if condition_tokens % teacher_tokens_per_frame != 0:
            student = getattr(self, "student", None)
            student_grid = getattr(student, "patch_grid_hw", None) if student is not None else None
            raise ValueError(
                f"Prediction condition length {condition_tokens} is not divisible by "
                f"teacher tokens_per_frame={teacher_tokens_per_frame} (grid {teacher_grid_size}). "
                f"DINO patch grid={student_grid}; align teacher and student resolution."
            )
        condition_frames = condition_tokens // teacher_tokens_per_frame
        if return_full_sequence:
            rope_grid_size = teacher_grid_size
        else:
            rope_grid_size = (
                condition_frames + int(query_grid_size[0]),
                height_len,
                width_len,
            )
        pred_features = self.prediction_head(
            seq_len=query_teacher_features.shape[1],
            timestep=video_timestep,
            condition=prediction_condition,
            condition_mask=prediction_condition_mask,
            grid_size=rope_grid_size,
            return_full_sequence=return_full_sequence,
        )
        repr_teacher_features = teacher_features if return_full_sequence else query_teacher_features
        repr_grid_size = teacher_grid_size if return_full_sequence else query_grid_size
        return self._compute_repr_loss(
            pred_features,
            repr_teacher_features,
            repr_grid_size,
            normalize_teacher=normalize_teacher,
        )

    def _append_proprio_to_action_condition(
        self,
        action_condition: torch.Tensor,
        action_condition_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.proprio_to_action_context is None:
            raise RuntimeError("Proprio projection is not initialized.")
        if proprio is None:
            raise ValueError("A proprio tensor is required.")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be [B,D] or [D], got {tuple(proprio.shape)}")
        if proprio.shape[0] != action_condition.shape[0]:
            raise ValueError(
                f"Batch mismatch between action condition and proprio: "
                f"{action_condition.shape[0]} vs {proprio.shape[0]}."
            )
        if proprio.shape[1] != self.proprio_condition_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_condition_dim}, got {proprio.shape[1]}.")
        proprio_token = self.proprio_to_action_context(
            proprio.to(device=action_condition.device, dtype=action_condition.dtype)
        ).unsqueeze(1)
        if action_condition_mask is None:
            action_condition_mask = torch.ones(
                action_condition.shape[:2],
                dtype=torch.bool,
                device=action_condition.device,
            )
        else:
            action_condition_mask = action_condition_mask.to(device=action_condition.device, dtype=torch.bool)
        proprio_mask = torch.ones((action_condition.shape[0], 1), dtype=torch.bool, device=action_condition.device)
        return (
            torch.cat([action_condition, proprio_token], dim=1),
            torch.cat([action_condition_mask, proprio_mask], dim=1),
        )

    def _build_teacher_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        student_condition: Optional[torch.Tensor] = None,
        student_condition_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.student_to_wan_context is None:
            raise RuntimeError("Teacher condition projection is not initialized.")
        if student_condition is None:
            raise ValueError("Student condition tokens are required for teacher conditioning.")
        student_context = self.student_to_wan_context(student_condition.detach().to(dtype=context.dtype))
        if student_condition_mask is None:
            student_condition_mask = torch.ones(
                student_context.shape[:2],
                dtype=torch.bool,
                device=student_context.device,
            )
        else:
            student_condition_mask = student_condition_mask.to(device=student_context.device, dtype=torch.bool)
        return (
            torch.cat([context, student_context], dim=1),
            torch.cat([context_mask.to(device=student_context.device, dtype=torch.bool), student_condition_mask], dim=1),
        )

    def _teacher_video_prediction(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        return self.teacher_video_expert(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

    def _teacher_video_prediction_and_features(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        pre_state = self.teacher_video_expert.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = self.teacher_video_expert.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.teacher_video_expert.video_attention_mask_mode != "bidirectional" else None

        selected = []
        max_layer = len(self.teacher_video_expert.blocks) - 1
        selected_layers = {layer if layer >= 0 else max_layer + 1 + layer for layer in self.teacher_feature_layers}
        for layer_idx, block in enumerate(self.teacher_video_expert.blocks):
            if self.teacher_video_expert.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.teacher_video_expert.use_gradient_checkpointing,
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )
            else:
                x_tokens = block(
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )
            if layer_idx in selected_layers:
                selected.append(x_tokens.detach())
        if not selected:
            raise ValueError(f"No teacher layers selected from {self.teacher_feature_layers}.")
        pred_video = self.teacher_video_expert.post_dit(x_tokens, pre_state)
        grid_size = tuple(int(v) for v in pre_state["meta"]["grid_size"])
        return pred_video, torch.cat(selected, dim=-1), grid_size

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action=None,
        action_horizon=None,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})")
        if checked_t != num_frames:
            raise ValueError(f"`num_frames` must satisfy T % 4 == 1, got {num_frames}")

        context_pos, mask_pos = self._prepare_infer_context(prompt, context, context_mask)
        context_neg = None
        mask_neg = None
        if text_cfg_scale != 1.0:
            context_neg, mask_neg = self._prepare_infer_context("" if negative_prompt is None else negative_prompt, None, None)
        _, _, student_pos, student_mask_pos = self._student_condition(
            input_image=input_image,
            prompt=None,
            context=context_pos,
            context_mask=mask_pos,
            drop_text=False,
        )
        context_pos, mask_pos = self._build_teacher_context(context_pos, mask_pos, student_pos, student_mask_pos)
        if context_neg is not None:
            _, _, student_neg, student_mask_neg = self._student_condition(
                input_image=input_image,
                prompt=None,
                context=context_neg,
                context_mask=mask_neg,
                drop_text=False,
            )
            context_neg, mask_neg = self._build_teacher_context(context_neg, mask_neg, student_neg, student_mask_neg)

        latent_t = (num_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self.vae.encode([input_image[0].unsqueeze(1)], device=self.device, tiled=tiled)
        if isinstance(first_frame_latents, list):
            first_frame_latents = first_frame_latents[0].unsqueeze(0)
        latents[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.teacher_video_expert, "fuse_vae_embedding_in_latents", False))
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(device=self.device, dtype=latents.dtype)
            pred_pos = self._teacher_video_prediction(latents, timestep, context_pos, mask_pos, fuse_flag)
            pred = pred_pos
            if context_neg is not None:
                pred_neg = self._teacher_video_prediction(latents, timestep, context_neg, mask_neg, fuse_flag)
                pred = pred_neg + text_cfg_scale * (pred_pos - pred_neg)
            latents = self.infer_video_scheduler.step(pred, step_delta, latents)
            latents[:, :, 0:1] = first_frame_latents.clone()
        return {"video": self._decode_latents(latents, tiled=tiled)}

    def _resolve_student_training_input(self, sample, video: torch.Tensor) -> torch.Tensor:
        observation_video = sample.get("observation_video")
        if observation_video is None:
            return video[:, :, 0]
        if observation_video.ndim != 5:
            raise ValueError(
                f"`sample['observation_video']` must be [B,3,T,H,W], got {tuple(observation_video.shape)}"
            )
        return observation_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

    def _compute_action_branch_loss(
        self,
        sample,
        action_condition: torch.Tensor,
        action_condition_mask: torch.Tensor,
        current_proprio: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}")
        action_noise = torch.randn_like(action)
        action_timestep = self.train_action_scheduler.sample_training_t(action.shape[0], self.device, action.dtype)
        noised_action = self.train_action_scheduler.add_noise(action, action_noise, action_timestep)
        action_target = self.train_action_scheduler.training_target(action, action_noise, action_timestep)
        action_head_condition = action_condition.detach()
        action_head_condition_mask = action_condition_mask
        action_head_condition, action_head_condition_mask = self._append_proprio_to_action_condition(
            action_head_condition,
            action_head_condition_mask,
            current_proprio,
        )
        pred_action = self.action_head(
            action_tokens=noised_action,
            timestep=action_timestep,
            context=action_head_condition,
            context_mask=action_head_condition_mask,
        )
        action_mask = sample.get("action_is_pad")
        if action_mask is not None:
            action_mask = action_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            valid = (~action_mask).to(dtype=pred_action.dtype).unsqueeze(-1)
            denom = valid.sum().clamp(min=1.0) * pred_action.shape[-1]
            return ((pred_action.float() - action_target.float()).pow(2) * valid).sum() / denom
        return F.mse_loss(pred_action.float(), action_target.float())

    def _training_loss_action_only(self, sample, tiled: bool = False):
        del tiled
        if self.disable_action_head or self.action_head is None:
            raise RuntimeError("Action-only training requires an action head.")
        action = sample.get("action")
        if action is None:
            raise ValueError("`sample['action']` is required for action-only Enfold training.")
        video = sample["video"]
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
        context, context_mask = self._prepare_context(sample)
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("sample proprio is required.")
        if proprio.ndim != 3:
            raise ValueError(f"sample proprio must be [B,T,D], got {tuple(proprio.shape)}")
        current_proprio = proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        student_input = self._resolve_student_training_input(sample, input_video)
        (
            _student_tokens,
            _student_mask,
            action_condition,
            action_condition_mask,
            _prediction_condition,
            _prediction_condition_mask,
        ) = self.student(student_input, context, context_mask)
        head_condition, head_condition_mask = self._compress_temporal_condition(
            action_condition,
            action_condition_mask,
        )
        loss_action = self._compute_action_branch_loss(
            sample,
            head_condition,
            head_condition_mask,
            current_proprio,
        )
        loss = self.loss_lambda_action * loss_action
        return loss, {
            "loss_video": 0.0,
            "loss_repr": 0.0,
            "loss_action": float(loss_action.detach().item()),
        }

    def training_loss(self, sample, tiled: bool = False):
        if self.teacher_video_expert is None:
            return self._training_loss_action_only(sample, tiled=tiled)
        video = sample["video"]
        action = sample.get("action")
        if action is None and not self.disable_action_head:
            raise ValueError("`sample['action']` is required for Enfold training.")
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
        if action is not None and action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}")
        context, context_mask = self._prepare_context(sample)
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("sample proprio is required.")
        if proprio.ndim != 3:
            raise ValueError(f"sample proprio must be [B,T,D], got {tuple(proprio.shape)}")
        current_proprio = proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        noise = torch.randn_like(input_latents)
        video_timestep = self.train_video_scheduler.sample_training_t(input_latents.shape[0], self.device, input_latents.dtype)
        noised_latents = self.train_video_scheduler.add_noise(input_latents, noise, video_timestep)
        video_target = self.train_video_scheduler.training_target(input_latents, noise, video_timestep)
        fuse_flag = bool(getattr(self.teacher_video_expert, "fuse_vae_embedding_in_latents", False))
        if fuse_flag:
            noised_latents[:, :, 0:1] = input_latents[:, :, 0:1]
        student_input = self._resolve_student_training_input(sample, input_video)
        (
            student_tokens,
            student_mask,
            action_condition,
            action_condition_mask,
            prediction_condition,
            prediction_condition_mask,
        ) = self.student(student_input, context, context_mask)
        head_condition, head_condition_mask = self._compress_temporal_condition(
            action_condition,
            action_condition_mask,
        )
        prediction_head_condition, prediction_head_condition_mask = self._pool_prediction_condition(
            prediction_condition,
            prediction_condition_mask,
        )
        video_context, video_context_mask = self._build_teacher_context(
            context,
            context_mask,
            action_condition,
            action_condition_mask,
        )
        pred_video, teacher_features, teacher_grid_size = self._teacher_video_prediction_and_features(
            noised_latents,
            video_timestep,
            video_context,
            video_context_mask,
            fuse_flag,
        )
        if fuse_flag:
            pred_video = pred_video[:, :, 1:]
            video_target = video_target[:, :, 1:]
        loss_video_per_sample = F.mse_loss(pred_video.float(), video_target.float(), reduction="none").mean(dim=(1, 2, 3, 4))
        video_weight = self.train_video_scheduler.training_weight(video_timestep).to(
            device=loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        num_conditional_frames = 1 if fuse_flag and self.repr_exclude_conditional_frames else 0
        loss_repr = self._compute_repr_branch(
            teacher_features=teacher_features,
            teacher_grid_size=teacher_grid_size,
            video_timestep=video_timestep,
            prediction_condition=prediction_head_condition,
            prediction_condition_mask=prediction_head_condition_mask,
            num_conditional_frames=num_conditional_frames,
            normalize_teacher=False,
        )

        if self.disable_action_head:
            loss_action = loss_repr.new_zeros(())
        else:
            loss_action = self._compute_action_branch_loss(
                sample,
                head_condition,
                head_condition_mask,
                current_proprio,
            )
        loss = self.loss_lambda_video * loss_video + self.loss_lambda_repr * loss_repr + self.loss_lambda_action * loss_action
        return loss, {
            "loss_video": float(loss_video.detach().item()),
            "loss_repr": float(loss_repr.detach().item()),
            "loss_action": float(loss_action.detach().item()),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        if self.disable_action_head or self.action_head is None:
            raise RuntimeError("Cannot run action inference because this model was built without an action head.")
        self.eval()
        _, _, action_condition, action_condition_mask = self._student_condition(
            input_image=input_image,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            drop_text=False,
        )
        action_condition = action_condition.detach()
        action_condition, action_condition_mask = self._append_proprio_to_action_condition(
            action_condition,
            action_condition_mask,
            proprio,
        )
        action_condition_uncond = None
        action_condition_mask_uncond = None
        if text_cfg_scale != 1.0:
            _, _, action_condition_uncond, action_condition_mask_uncond = self._student_condition(
                input_image=input_image,
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                drop_text=True,
            )
            action_condition_uncond = action_condition_uncond.detach()
            action_condition_uncond, action_condition_mask_uncond = self._append_proprio_to_action_condition(
                action_condition_uncond,
                action_condition_mask_uncond,
                proprio,
            )
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn(
            (1, int(action_horizon), self.action_head.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(device=self.device, dtype=action.dtype)
            pred = self.action_head(action, timestep, action_condition, action_condition_mask)
            if action_condition_uncond is not None:
                pred_uncond = self.action_head(action, timestep, action_condition_uncond, action_condition_mask_uncond)
                pred = pred_uncond + text_cfg_scale * (pred - pred_uncond)
            action = self.infer_action_scheduler.step(pred, step_delta, action)
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs) -> dict[str, Any]:
        num_video_frames = kwargs.pop("num_video_frames", None)
        if num_video_frames is None:
            raise ValueError("`num_video_frames` is required for `infer_joint`.")
        video_kwargs = dict(kwargs)
        video_kwargs["num_frames"] = int(num_video_frames)
        video_kwargs.pop("action_horizon", None)
        video = self.infer(**video_kwargs)["video"]
        action = self.infer_action(*args, num_video_frames=int(num_video_frames), **kwargs)["action"]
        return {"video": video, "action": action}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {"dit": self.dit.state_dict(), "step": step, "torch_dtype": str(self.torch_dtype)}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        key = "dit" if "dit" in payload else "model"
        checkpoint_state = payload[key]
        model_state = self.dit.state_dict()
        compatible_state = {}
        skipped_mismatch = []
        for name, tensor in checkpoint_state.items():
            current = model_state.get(name)
            if current is not None and current.shape != tensor.shape:
                skipped_mismatch.append((name, tuple(tensor.shape), tuple(current.shape)))
                continue
            compatible_state[name] = tensor
        if skipped_mismatch:
            logger.warning(
                "Skipped %d checkpoint tensors with incompatible shapes, e.g. %s",
                len(skipped_mismatch),
                skipped_mismatch[:5],
            )
        self.dit.load_state_dict(compatible_state, strict=False)
        self.to(device=self.device, dtype=self.torch_dtype)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
