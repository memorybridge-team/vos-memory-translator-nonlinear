"""Spatial-context Transformer memory translator (SAM 2.1 Small -> Base+).

Each memory frame ``M`` (``[C,H,W] = [64,64,64]``) is translated on its own.
Self-attention runs only among the patch tokens of one frame (``16x16 = 256``
tokens for the default 4x4 patch). Nothing attends across records, objects or
time, and no operation mixes the frame axis ``N``. The spatial output is
``M_hat = M + alpha * Delta`` with ``alpha`` initialised to zero, so an untrained
translator is exactly the identity. The object pointer is translated by a
separate residual MLP, ``p_hat = p + MLP(p)``, whose last layer starts at zero.

Shape symbols: ``N`` valid records (frames), ``C`` memory channels, ``d``
``d_model``, ``L`` tokens per frame, ``p`` patch size, ``h`` fusion hidden width.

Run ``python -m vos_memory_inspector.transformer_translator --preset base`` for a
dummy-forward summary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, replace
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .state_schema import CanonicalState, StateSpec
from .translators import (
    DirectCopyTranslator,
    LinearStateTranslator,
    MomentMatchedCopyTranslator,
    ResidualMLPStateTranslator,
    _LearnedStateTranslator,
)


# Runtime contract v1.1: spatial [B,O,K,64,64,64] bf16, pointer [B,O,K,256] fp32.
SAM21_MEMORY_SPEC = StateSpec(feature_channels=64, height=64, width=64, pointer_dim=256)

PATCH_EMBEDS = ("conv", "avgpool")
POS_EMBEDS = ("learnable", "sincos", "none")
LOCAL_FUSIONS = ("linear", "mlp", "none")
OUTPUT_DTYPES = ("source", "float32")
PAYLOAD_SCHEMA = "cmmt.spatial_context_transformer_translator.v1"

_POSITIVE_INT_FIELDS = (
    "channels",
    "height",
    "width",
    "d_model",
    "num_heads",
    "num_layers",
    "patch_size",
    "fusion_hidden_dim",
    "pointer_dim",
    "pointer_hidden_dim",
)
_BOOL_FIELDS = ("spatial_context", "use_attention", "use_residual")
_CONTEXT_MAC_KEYS = ("patch_embed", "attention", "ffn", "context_conv", "context_proj")
_PIXEL_MAC_KEYS = ("local_in", "local_out")


@dataclass(frozen=True)
class SpatialTransformerConfig:
    """Architecture of one translator; defaults are the ``base`` preset."""

    # Sizes.
    channels: int = 64
    height: int = 64
    width: int = 64
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    patch_size: int = 4
    # Structure.
    patch_embed: str = "conv"  # "conv" (k=p, s=p) | "avgpool" (avgpool p + 1x1 conv)
    pos_embed: str = "learnable"  # "learnable" | "sincos" | "none"
    spatial_context: bool = True
    use_attention: bool = True  # False: blocks keep only the token-wise FFN
    local_fusion: str = "linear"  # "linear" | "mlp" | "none"
    fusion_hidden_dim: int = 128
    use_residual: bool = True
    residual_scale_init: float = 0.0
    # Pointer head and handoff output.
    pointer_dim: int = 256
    pointer_hidden_dim: int = 512
    output_dtype: str = "source"  # "source" | "float32"

    def __post_init__(self) -> None:
        for name in _POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        for name in _BOOL_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name, allowed in (
            ("patch_embed", PATCH_EMBEDS),
            ("pos_embed", POS_EMBEDS),
            ("local_fusion", LOCAL_FUSIONS),
            ("output_dtype", OUTPUT_DTYPES),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(float(self.residual_scale_init)):
            raise ValueError("residual_scale_init must be finite")
        ffn = float(self.d_model) * float(self.mlp_ratio)
        if not self.mlp_ratio > 0 or ffn < 1 or not ffn.is_integer():
            raise ValueError("d_model * mlp_ratio must be a positive integer")
        if self.height % self.patch_size or self.width % self.patch_size:
            raise ValueError(
                f"height/width ({self.height}x{self.width}) must be divisible by "
                f"patch_size {self.patch_size}"
            )
        if self.d_model % self.num_heads:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by num_heads {self.num_heads}"
            )
        if self.pos_embed == "sincos" and self.d_model % 4:
            raise ValueError("sincos positional embedding needs d_model divisible by 4")
        if not self.spatial_context and self.local_fusion == "none":
            raise ValueError(
                "spatial_context=False with local_fusion='none' leaves no branch to learn"
            )

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.height // self.patch_size, self.width // self.patch_size

    @property
    def num_tokens(self) -> int:
        grid_h, grid_w = self.grid_size
        return grid_h * grid_w

    @property
    def ffn_hidden_dim(self) -> int:
        return int(self.d_model * self.mlp_ratio)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SpatialTransformerConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown SpatialTransformerConfig fields: {unknown}")
        return cls(**dict(values))


def sincos_2d_embedding(
    grid_h: int,
    grid_w: int,
    dim: int,
    *,
    temperature: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Fixed 2-D sine-cosine embedding ``[grid_h*grid_w, dim]`` in row-major order.

    The first half of ``dim`` encodes the row, the second half the column; each
    half is ``[sin, cos]`` over ``dim/4`` geometric frequencies.
    """

    if grid_h < 1 or grid_w < 1:
        raise ValueError("grid sizes must be positive")
    if dim < 4 or dim % 4:
        raise ValueError(f"dim must be a positive multiple of 4, got {dim}")
    quarter = dim // 4
    omega = 1.0 / temperature ** (torch.arange(quarter, dtype=torch.float64) / quarter)
    rows, cols = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float64),
        torch.arange(grid_w, dtype=torch.float64),
        indexing="ij",
    )
    row = rows.reshape(-1, 1) * omega  # [L, dim/4]
    col = cols.reshape(-1, 1) * omega  # [L, dim/4]
    embedding = torch.cat((row.sin(), row.cos(), col.sin(), col.cos()), dim=1)
    return embedding.to(device=device, dtype=dtype)  # [L, dim]


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over the tokens of each frame separately."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        frames, length, width = tokens.shape  # [N, L, d]
        head_dim = width // self.num_heads
        qkv = self.qkv(tokens).view(frames, length, 3, self.num_heads, head_dim)  # [N,L,3,h,d/h]
        query, key, value = qkv.permute(2, 0, 3, 1, 4)  # each [N, heads, L, d/heads]
        # Attention matrix per (frame, head) is [L, L]; N is a batch axis only.
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0
        )  # [N, heads, L, d/heads]
        return self.proj(attended.transpose(1, 2).reshape(frames, length, width))  # [N,L,d]


class SpatialTransformerBlock(nn.Module):
    """Pre-LN block: ``x + Attn(LN(x))`` then ``x + FFN(LN(x))`` (GELU FFN).

    With ``use_attention=False`` the attention sublayer and its LayerNorm are
    absent and the block is a token-wise FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        if hidden < 1:
            raise ValueError("d_model * mlp_ratio must be positive")
        self.use_attention = use_attention
        if use_attention:
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = SpatialSelfAttention(d_model, num_heads, dropout)
        else:
            self.norm1 = None
            self.attn = None
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [N, L, d] -> [N, L, d]
        if self.attn is not None:
            tokens = tokens + self.drop(self.attn(self.norm1(tokens)))
        hidden = F.gelu(self.fc1(self.norm2(tokens)))  # [N, L, d*mlp_ratio]
        return tokens + self.drop(self.fc2(hidden))


class SpatialMemoryTranslator(nn.Module):
    """``T: [N,C,H,W] -> [N,C,H,W]``, applied independently to each frame.

    Context branch: patch embed -> +PE -> Transformer blocks -> LN -> Conv3x3 at
    the token grid. Fusion (efficient form of a 1x1 conv on ``[M || Up(ctx)]``):

    - ``linear``: ``Delta = W_l(M) + Up(W_c(ctx))``
    - ``mlp``: ``Delta = W_2(GELU(W_l(M) + Up(W_c(ctx))))``
    - ``none``: ``Delta = Up(ctx)`` (context only)

    Bilinear upsampling is linear and acts per channel, so it commutes with the
    bias-free 1x1 ``W_c``; running ``W_c`` at the token grid saves MACs.
    Without spatial context ``Delta`` is the position-wise head alone.
    """

    def __init__(self, config: SpatialTransformerConfig | None = None):
        super().__init__()
        config = SpatialTransformerConfig() if config is None else config
        self.config = config
        channels, width = config.channels, config.d_model
        fusion = config.local_fusion
        hidden = config.fusion_hidden_dim
        fusion_channels = hidden if fusion == "mlp" else channels
        if config.spatial_context:
            patch = config.patch_size
            if config.patch_embed == "conv":
                self.patch_embed: nn.Module | None = nn.Conv2d(
                    channels, width, kernel_size=patch, stride=patch
                )
            else:
                self.patch_embed = nn.Sequential(
                    nn.AvgPool2d(patch), nn.Conv2d(channels, width, kernel_size=1)
                )
            grid_h, grid_w = config.grid_size
            if config.pos_embed == "learnable":
                self.pos_embed = nn.Parameter(torch.zeros(1, config.num_tokens, width))
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
            elif config.pos_embed == "sincos":
                self.register_buffer(
                    "pos_embed",
                    sincos_2d_embedding(grid_h, grid_w, width).unsqueeze(0),
                    persistent=False,
                )
            else:
                self.pos_embed = None
            self.blocks = nn.ModuleList(
                SpatialTransformerBlock(
                    width,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    use_attention=config.use_attention,
                )
                for _ in range(config.num_layers)
            )
            self.norm: nn.Module | None = nn.LayerNorm(width)
            context_out = channels if fusion == "none" else width
            self.context_conv: nn.Module | None = nn.Conv2d(
                width, context_out, kernel_size=3, padding=1
            )
            self.context_proj: nn.Module | None = (
                None
                if fusion == "none"
                else nn.Conv2d(width, fusion_channels, kernel_size=1, bias=False)
            )
        else:
            self.patch_embed = None
            self.pos_embed = None
            self.blocks = nn.ModuleList()
            self.norm = None
            self.context_conv = None
            self.context_proj = None
        if fusion == "linear":
            self.local_in: nn.Module | None = nn.Conv2d(channels, channels, kernel_size=1)
            self.local_out: nn.Module | None = None
        elif fusion == "mlp":
            self.local_in = nn.Conv2d(channels, hidden, kernel_size=1)
            self.local_out = nn.Conv2d(hidden, channels, kernel_size=1)
        else:
            self.local_in = None
            self.local_out = None
        # ReZero-style gate: only alpha starts at zero, so the branch receives
        # gradient as soon as alpha moves (both zero would stall learning).
        self.alpha = (
            nn.Parameter(torch.full((1, channels, 1, 1), float(config.residual_scale_init)))
            if config.use_residual
            else None
        )

    def _parameter_reference(self) -> torch.Tensor:
        return next(self.parameters())

    def _upsample(self, tensor: torch.Tensor) -> torch.Tensor:
        # [N, *, H/p, W/p] -> [N, *, H, W]
        return F.interpolate(
            tensor,
            size=(self.config.height, self.config.width),
            mode="bilinear",
            align_corners=False,
        )

    def _context(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = frames.shape[0]
        grid_h, grid_w = self.config.grid_size
        tokens = self.patch_embed(frames)  # [N, d, H/p, W/p]
        tokens = tokens.flatten(2).transpose(1, 2)  # [N, L, d]
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed  # + [1, L, d]
        for block in self.blocks:
            tokens = block(tokens)  # [N, L, d]; attention stays inside each frame
        tokens = self.norm(tokens)  # [N, L, d]
        grid = tokens.transpose(1, 2).reshape(count, self.config.d_model, grid_h, grid_w)
        return self.context_conv(grid), tokens  # ctx [N, d|C, H/p, W/p]

    def _delta(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = None
        context = None
        if self.config.spatial_context:
            ctx, tokens = self._context(frames)
            if self.context_proj is None:
                return self._upsample(ctx), tokens  # context_only: [N, C, H, W]
            context = self._upsample(self.context_proj(ctx))  # c_up [N, C|h, H, W]
        fused = self.local_in(frames)  # [N, C|h, H, W]
        if context is not None:
            fused = fused + context
        if self.local_out is not None:
            fused = self.local_out(F.gelu(fused))  # [N, C, H, W]
        return fused, tokens

    def forward(
        self, frames: torch.Tensor, *, return_tokens: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """``[N,C,H,W]`` or ``[B,O,K,C,H,W]`` -> same shape, in parameter dtype.

        Every leading index is treated as an independent frame. With
        ``return_tokens`` also returns the normalized context tokens
        ``[N, L, d]`` (leading axes restored), or ``None`` without context.
        """

        if frames.ndim == 6:
            leading = tuple(frames.shape[:3])
            flat = frames.reshape(-1, *frames.shape[3:])
        elif frames.ndim == 4:
            leading = None
            flat = frames
        else:
            raise ValueError(
                f"frames must be [N,C,H,W] or [B,O,K,C,H,W], got {tuple(frames.shape)}"
            )
        expected = (self.config.channels, self.config.height, self.config.width)
        if tuple(flat.shape[1:]) != expected:
            # Fail closed: the token grid and learnable PE are tied to the config.
            raise ValueError(
                f"frame [C,H,W]={tuple(flat.shape[1:])} does not match the configured "
                f"{expected}; resample outside the translator"
            )
        reference = self._parameter_reference()
        memory = flat.to(device=reference.device, dtype=reference.dtype)  # M [N, C, H, W]
        delta, tokens = self._delta(memory)
        output = memory + self.alpha * delta if self.alpha is not None else delta
        if leading is not None:
            output = output.reshape(*leading, *output.shape[1:])
            if tokens is not None:
                tokens = tokens.reshape(*leading, *tokens.shape[1:])
        return (output, tokens) if return_tokens else output

    def macs_breakdown(self) -> dict[str, int]:
        """Analytic multiply-accumulates per frame by module.

        Counted: convolutions, linear layers and the two attention matmuls
        (``QK^T`` and ``AV``). Not counted: LayerNorm, softmax, GELU, bilinear
        upsampling, average pooling, PE/residual additions and the alpha gate.
        """

        config = self.config
        channels, width = config.channels, config.d_model
        positions = config.height * config.width
        fusion = config.local_fusion
        hidden = config.fusion_hidden_dim
        macs: dict[str, int] = {}
        if config.spatial_context:
            tokens = config.num_tokens
            kernel = config.patch_size**2 if config.patch_embed == "conv" else 1
            macs["patch_embed"] = tokens * width * channels * kernel
            if config.use_attention:
                macs["attention"] = config.num_layers * (
                    tokens * width * 3 * width  # qkv
                    + 2 * tokens * tokens * width  # QK^T and AV over all heads
                    + tokens * width * width  # output projection
                )
            macs["ffn"] = config.num_layers * 2 * tokens * width * config.ffn_hidden_dim
            context_out = channels if fusion == "none" else width
            macs["context_conv"] = tokens * context_out * width * 9
            if fusion != "none":
                macs["context_proj"] = tokens * width * (hidden if fusion == "mlp" else channels)
        if fusion == "linear":
            macs["local_in"] = positions * channels * channels
        elif fusion == "mlp":
            macs["local_in"] = positions * channels * hidden
            macs["local_out"] = positions * hidden * channels
        return macs

    def macs_per_frame(self) -> int:
        return sum(self.macs_breakdown().values())

    def context_branch_macs(self) -> int:
        breakdown = self.macs_breakdown()
        return sum(breakdown.get(key, 0) for key in _CONTEXT_MAC_KEYS)

    def per_pixel_macs(self) -> int:
        breakdown = self.macs_breakdown()
        return sum(breakdown.get(key, 0) for key in _PIXEL_MAC_KEYS)

    def parameter_breakdown(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            key = name.split(".", 1)[0]
            groups[key] = groups.get(key, 0) + parameter.numel()
        return groups

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ResidualPointerTranslator(nn.Module):
    """``p_hat = p + fc2(GELU(fc1(p)))``; ``fc2`` starts at zero so ``p_hat = p``."""

    def __init__(self, dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        if dim < 1 or hidden_dim < 1:
            raise ValueError("pointer dims must be positive")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, pointers: torch.Tensor) -> torch.Tensor:  # [..., D] -> [..., D]
        if pointers.shape[-1] != self.dim:
            raise ValueError(f"pointer dim {pointers.shape[-1]} != {self.dim}")
        pointers = pointers.to(device=self.fc1.weight.device, dtype=self.fc1.weight.dtype)
        return pointers + self.fc2(F.gelu(self.fc1(pointers)))

    def macs_per_record(self) -> int:
        return 2 * self.dim * self.hidden_dim

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def describe_config(config: SpatialTransformerConfig) -> str:
    if not config.spatial_context:
        head = "linear" if config.local_fusion == "linear" else "MLP"
        return f"position-wise {head} head, no spatial context"
    family = "Spatial-Context Transformer" if config.use_attention else "Spatial-Context (no attention)"
    fusion = {"linear": "linear fusion", "mlp": "MLP fusion", "none": "context only"}
    return f"{family}, {fusion[config.local_fusion]}"


class TransformerStateTranslator(_LearnedStateTranslator):
    """CanonicalState translator: spatial-context memory + residual pointer MLP.

    ``translate`` gathers valid records ``[B,O,K,...] -> [N,...]``, translates
    each frame independently, casts to the source dtype (bf16 spatial, fp32
    pointer by default) and scatters back. Padding records keep the source.
    ``translate_tensors`` is the gradient-preserving fp32 training API.
    """

    name = "spatial_context_transformer"
    grid_adapter = "none"
    # The context branch reads the whole frame, so single positions cannot be sampled.
    supports_position_sampling = False
    _spatial_module_prefix = "spatial"
    _pointer_module_prefix = "pointer"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        config: SpatialTransformerConfig | None = None,
        preset: str | None = None,
    ):
        config = SpatialTransformerConfig() if config is None else config
        super().__init__(source_spec, target_spec, output_dtype=config.output_dtype)
        expected = StateSpec(
            config.channels, config.height, config.width, config.pointer_dim
        )
        for role, spec in (("source", source_spec), ("target", target_spec)):
            actual = (spec.feature_channels, spec.height, spec.width, spec.pointer_dim)
            wanted = (
                expected.feature_channels,
                expected.height,
                expected.width,
                expected.pointer_dim,
            )
            if actual != wanted:
                raise ValueError(
                    f"{role} spec [C,H,W,D]={actual} does not match the translator "
                    f"config {wanted}"
                )
        self.config = config
        self.preset = preset
        self.spatial = SpatialMemoryTranslator(config)
        self.pointer = ResidualPointerTranslator(config.pointer_dim, config.pointer_hidden_dim)

    def _translate_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.spatial(frames)  # [N, C, H, W]

    def _translate_pointers(self, pointers: torch.Tensor) -> torch.Tensor:
        return self.pointer(pointers)  # [N, D]

    def macs_per_frame(self) -> int:
        return self.spatial.macs_per_frame()

    def pointer_macs_per_record(self) -> int:
        return self.pointer.macs_per_record()

    @property
    def description(self) -> str:
        return describe_config(self.config)

    def architecture_summary(self) -> dict[str, Any]:
        """JSON-safe architecture, parameter and analytic-cost summary."""

        config = self.config
        attention = None
        if config.spatial_context and config.use_attention:
            attention = {
                "scope": "per_frame",
                "matrix": [config.num_tokens, config.num_tokens],
                "heads": config.num_heads,
                "cross_frame": False,
            }
        return {
            "translator": self.name,
            "preset": self.preset,
            "description": self.description,
            "config": config.to_dict(),
            "token_grid": list(config.grid_size) if config.spatial_context else None,
            "tokens_per_frame": config.num_tokens if config.spatial_context else 0,
            "attention": attention,
            "parameters": self.parameter_breakdown(),
            "spatial_parameters_by_module": self.spatial.parameter_breakdown(),
            "macs_per_frame": self.macs_per_frame(),
            "macs_by_module": self.spatial.macs_breakdown(),
            "context_branch_macs": self.spatial.context_branch_macs(),
            "per_pixel_macs": self.spatial.per_pixel_macs(),
            "pointer_macs_per_record": self.pointer_macs_per_record(),
            "identity_at_init": config.use_residual and config.residual_scale_init == 0.0,
        }

    def _translation_metadata(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "architecture": self.description,
            "attention_scope": (
                "per_frame_tokens"
                if self.config.spatial_context and self.config.use_attention
                else "none"
            ),
            "tokens_per_frame": self.config.num_tokens if self.config.spatial_context else 0,
            "target_positional_encoding": "regenerated_by_target",
        }

    def to_payload(self) -> dict[str, Any]:
        """Plain dict with CPU tensors; loadable with ``torch.load(weights_only=True)``."""

        return {
            "schema_version": PAYLOAD_SCHEMA,
            "translator": self.name,
            "preset": self.preset,
            "config": self.config.to_dict(),
            "source_spec": self.source_spec.to_dict(),
            "target_spec": self.target_spec.to_dict(),
            "state_dict": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TransformerStateTranslator":
        if payload.get("schema_version") != PAYLOAD_SCHEMA:
            raise ValueError("unsupported spatial-context transformer payload")
        config_values = payload.get("config")
        if not isinstance(config_values, Mapping):
            raise TypeError("payload config must be a mapping")
        missing = sorted({item.name for item in fields(SpatialTransformerConfig)} - set(config_values))
        if missing:
            raise ValueError(f"payload config is missing fields: {missing}")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("payload state_dict must be a mapping")
        preset = payload.get("preset")
        translator = cls(
            StateSpec(**dict(payload["source_spec"])),
            StateSpec(**dict(payload["target_spec"])),
            config=SpatialTransformerConfig.from_dict(config_values),
            preset=None if preset is None else str(preset),
        )
        translator.load_state_dict(dict(state_dict), strict=True)
        translator.eval()
        return translator


@dataclass(frozen=True)
class TranslatorPreset:
    """Registry entry. ``config`` holds overrides of ``SpatialTransformerConfig()``."""

    name: str
    kind: str  # "spatial_transformer" | "linear" | "residual_mlp" | "direct" | "moment_match"
    group: str
    description: str
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    training: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def learned(self) -> bool:
        return self.kind in {"spatial_transformer", "linear", "residual_mlp"}


def _preset(
    name: str,
    kind: str,
    group: str,
    description: str,
    *,
    config: Mapping[str, Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
) -> TranslatorPreset:
    return TranslatorPreset(
        name,
        kind,
        group,
        description,
        MappingProxyType(dict(config or {})),
        MappingProxyType(dict(kwargs or {})),
        MappingProxyType(dict(training or {})),
    )


_ST = "spatial_transformer"
_AVG_SINCOS = {"patch_embed": "avgpool", "pos_embed": "sincos"}
_NO_CONTEXT_MLP = {"spatial_context": False, "local_fusion": "mlp"}

PRESETS: Mapping[str, TranslatorPreset] = MappingProxyType(
    {
        preset.name: preset
        for preset in (
            # Capacity ladder (repository classes).
            _preset("direct", "direct", "ladder", "Direct Copy"),
            _preset("moment_match", "moment_match", "ladder", "Moment-Matched Copy (fit split only)"),
            _preset("linear", "linear", "ladder", "position-shared Linear 64->64, pointer 256->256"),
            _preset(
                "residual_mlp",
                "residual_mlp",
                "ladder",
                "Residual MLP 64->128->64, pointer 256->512->256",
                kwargs={"hidden_dim": 128, "pointer_hidden_dim": 512},
            ),
            # Main 2x2: local head {linear, MLP} x spatial context {off, on}.
            _preset("linear_local", _ST, "2x2", "linear local head, no context", config={"spatial_context": False}),
            _preset("base", _ST, "2x2", "linear local head + spatial context"),
            _preset("no_context", _ST, "2x2", "MLP local head (128), no context", config=_NO_CONTEXT_MLP),
            _preset("base_mlp", _ST, "2x2", "MLP local head + spatial context", config={"local_fusion": "mlp"}),
            # Ablations of base (one change each unless noted).
            _preset("context_only", _ST, "structure", "Delta = Up(Conv3x3(ctx))", config={"local_fusion": "none"}),
            _preset("no_attn_context", _ST, "structure", "attention sublayers removed", config={"use_attention": False}),
            _preset("res8", _ST, "A_resolution", "avgpool 8 + 1x1, sincos PE", config={**_AVG_SINCOS, "patch_size": 8}),
            _preset("res16", _ST, "A_resolution", "avgpool 4 + 1x1, sincos PE", config={**_AVG_SINCOS, "patch_size": 4}),
            _preset("res32", _ST, "A_resolution", "avgpool 2 + 1x1, sincos PE", config={**_AVG_SINCOS, "patch_size": 2}),
            _preset("grid8", _ST, "A_prime_grid", "conv patch 8, learnable PE", config={"patch_size": 8}),
            _preset("grid32", _ST, "A_prime_grid", "conv patch 2, learnable PE", config={"patch_size": 2}),
            _preset("depth1", _ST, "B_depth", "1 layer", config={"num_layers": 1}),
            _preset("depth4", _ST, "B_depth", "4 layers", config={"num_layers": 4}),
            _preset("heads2", _ST, "C_heads", "2 heads", config={"num_heads": 2}),
            _preset("heads8", _ST, "C_heads", "8 heads", config={"num_heads": 8}),
            _preset("no_residual", _ST, "D_residual", "M_hat = Delta", config={"use_residual": False}),
            _preset(
                "mlp256",
                _ST,
                "E_control",
                "no_context with hidden 256 (base_mlp compute)",
                config={**_NO_CONTEXT_MLP, "fusion_hidden_dim": 256},
            ),
            _preset(
                "mlp_param",
                _ST,
                "E_control",
                "no_context with hidden 1505 (base params)",
                config={**_NO_CONTEXT_MLP, "fusion_hidden_dim": 1505},
            ),
            _preset("no_pos", _ST, "F_pe", "no positional embedding", config={"pos_embed": "none"}),
            _preset("sincos_pos", _ST, "F_pe", "fixed sincos PE", config={"pos_embed": "sincos"}),
            _preset("ptr128", _ST, "pointer", "pointer hidden 128", config={"pointer_hidden_dim": 128}),
            _preset(
                "no_cos",
                _ST,
                "loss",
                "base model trained with lambda_cos = 0 (pipeline setting)",
                training={"lambda_cos": 0.0},
            ),
        )
    }
)

_LADDER_OVERRIDES = {
    "linear": {"output_dtype"},
    "residual_mlp": {"hidden_dim", "pointer_hidden_dim", "output_dtype"},
    "moment_match": {"eps", "output_dtype"},
    "direct": set(),
}


def preset_names(*, learned: bool | None = None, kind: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, preset in PRESETS.items()
        if (learned is None or preset.learned == learned) and (kind is None or preset.kind == kind)
    )


def build_translator(
    name: str,
    source_spec: StateSpec,
    target_spec: StateSpec,
    *,
    config: SpatialTransformerConfig | None = None,
    fit_pairs: Iterable[tuple[CanonicalState, CanonicalState]] | None = None,
    **overrides: Any,
) -> Any:
    """Instantiate a registered preset.

    Spatial-transformer presets accept ``config=`` (replaces the preset config)
    and ``SpatialTransformerConfig`` field overrides. Ladder translators use
    ``output_dtype="source"`` unless overridden so that handoff bytes match
    Direct Copy. ``moment_match`` requires ``fit_pairs`` from the fit split.
    """

    try:
        preset = PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from exc
    if fit_pairs is not None and preset.kind != "moment_match":
        raise ValueError("fit_pairs is only used by the moment_match preset")
    if config is not None and preset.kind != _ST:
        raise ValueError(f"config= applies only to spatial-transformer presets, not {name!r}")
    if preset.kind == _ST:
        base = config if config is not None else SpatialTransformerConfig(**dict(preset.config))
        known = {item.name for item in fields(SpatialTransformerConfig)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"unknown overrides for {name!r}: {unknown}")
        translator: Any = TransformerStateTranslator(
            source_spec, target_spec, config=replace(base, **overrides), preset=name
        )
        return translator
    unknown = sorted(set(overrides) - _LADDER_OVERRIDES[preset.kind])
    if unknown:
        raise ValueError(f"unknown overrides for {name!r}: {unknown}")
    options = {**dict(preset.kwargs), **overrides}
    if preset.kind == "direct":
        translator = DirectCopyTranslator(target_spec)
    elif preset.kind == "moment_match":
        if fit_pairs is None:
            raise ValueError(
                "moment_match needs fit_pairs (fit split only); it has no untrained form"
            )
        options.setdefault("output_dtype", "source")
        translator = MomentMatchedCopyTranslator.fit(fit_pairs, **options)
        if translator.source_spec != source_spec or translator.target_spec != target_spec:
            raise ValueError("fit_pairs specs do not match source_spec/target_spec")
    elif preset.kind == "linear":
        options.setdefault("output_dtype", "source")
        translator = LinearStateTranslator(source_spec, target_spec, **options)
    else:
        options.setdefault("output_dtype", "source")
        translator = ResidualMLPStateTranslator(source_spec, target_spec, **options)
    translator.preset = name
    return translator


# --------------------------------------------------------------------------
# Dummy-forward summary (Task 6)
# --------------------------------------------------------------------------


def _millions(value: float) -> str:
    return f"{value / 1e6:.1f}M"


# (parameter group, analytic MAC keys) in forward order.
_SUMMARY_MODULES = (
    ("patch_embed", ("patch_embed",)),
    ("pos_embed", ()),
    ("blocks", ("attention", "ffn")),
    ("norm", ()),
    ("context_conv", ("context_conv",)),
    ("context_proj", ("context_proj",)),
    ("local_in", ("local_in",)),
    ("local_out", ("local_out",)),
    ("alpha", ()),
)


def _dummy_state(
    *, batch: int, objects: int, records: int, invalid: int, seed: int, spec: StateSpec
) -> CanonicalState:
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, objects, records)
    spatial = torch.randn(
        *shape, spec.feature_channels, spec.height, spec.width, generator=generator
    ).to(torch.bfloat16)  # runtime dtype of maskmem_features
    pointer = torch.randn(*shape, spec.pointer_dim, generator=generator)
    validity = torch.ones(shape, dtype=torch.bool)
    if invalid:
        validity.view(-1)[-invalid:] = False  # padding at the end of the record order
    frames = torch.arange(records).view(1, 1, records).expand(shape).clone()
    conditioning = torch.zeros(shape, dtype=torch.bool)
    conditioning[..., 0] = True
    return CanonicalState(
        spatial_memory=spatial,
        object_pointer=pointer,
        presence_logits=torch.zeros(*shape, 1),
        frame_indices=frames,
        slot_order=frames.clone(),
        is_conditioning=conditioning,
        validity=validity,
        object_ids=tuple(range(1, objects + 1)),
        switch_frame=records - 1,
        metadata={"synthetic": True},
    ).validate()


def pytorch_flops(function: Any, *args: Any) -> int:
    """FlopCounterMode total (matmul/conv only). SDPA runs on the math backend
    so attention matmuls are counted on every device."""

    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        counter = FlopCounterMode(display=False)
        with counter:
            function(*args)
    return int(counter.get_total_flops())


def summarize_dummy_forward(
    translator: TransformerStateTranslator,
    *,
    batch: int = 1,
    objects: int = 2,
    records: int = 3,
    invalid: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    total = batch * objects * records
    if min(batch, objects, records) < 1:
        raise ValueError("batch, objects and records must be positive")
    if not 0 <= invalid <= total:
        raise ValueError(f"invalid must be in [0, {total}]")
    state = _dummy_state(
        batch=batch,
        objects=objects,
        records=records,
        invalid=invalid,
        seed=seed,
        spec=translator.source_spec,
    )
    captured: dict[str, Any] = {"frames": None, "tokens": None, "attention": []}
    hooks = [
        translator.spatial.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("frames", tuple(inputs[0].shape))
        )
    ]
    if translator.spatial.norm is not None:
        hooks.append(
            translator.spatial.norm.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__("tokens", tuple(output.shape))
            )
        )
    for block in translator.spatial.blocks:
        if block.attn is not None:
            hooks.append(
                block.attn.register_forward_pre_hook(
                    lambda _module, inputs: captured["attention"].append(tuple(inputs[0].shape))
                )
            )
    try:
        with torch.inference_mode():
            translated = translator.translate(state)
    finally:
        for hook in hooks:
            hook.remove()
    valid = state.validity
    if int(valid.sum()):
        spatial_error = (
            translated.spatial_memory[valid].float() - state.spatial_memory[valid].float()
        ).abs().max().item()
        pointer_error = (
            translated.object_pointer[valid].float() - state.object_pointer[valid].float()
        ).abs().max().item()
    else:
        spatial_error = pointer_error = 0.0
    config = translator.config
    one_frame = torch.zeros(
        1, config.channels, config.height, config.width,
        device=next(translator.parameters()).device,
    )
    return {
        "preset": translator.preset,
        "description": translator.description,
        "input_spatial": tuple(state.spatial_memory.shape),
        "input_dtype": str(state.spatial_memory.dtype).removeprefix("torch."),
        "valid_records": int(valid.sum()),
        "total_records": total,
        "frames": captured["frames"],
        "tokens": captured["tokens"],
        "attention_inputs": captured["attention"],
        "output_spatial": tuple(translated.spatial_memory.shape),
        "output_dtype": str(translated.spatial_memory.dtype).removeprefix("torch."),
        "pointer_in": tuple(state.object_pointer.shape),
        "pointer_out": tuple(translated.object_pointer.shape),
        "identity_spatial_max_abs": spatial_error,
        "identity_pointer_max_abs": pointer_error,
        "pytorch_flops_per_frame": pytorch_flops(translator.spatial, one_frame),
        "architecture": translator.architecture_summary(),
    }


def format_dummy_forward_summary(summary: Mapping[str, Any]) -> str:
    arch = summary["architecture"]
    params = arch["parameters"]
    config = arch["config"]
    frames = summary["frames"]
    lines = [
        f"preset             : {summary['preset']} ({summary['description']})",
        f"input spatial      : {summary['input_spatial']} {summary['input_dtype']}  "
        f"valid {summary['valid_records']}/{summary['total_records']} -> frames "
        + (str(frames) if frames is not None else "none (translator not called)"),
        "intermediate tokens: "
        + (
            str(summary["tokens"])
            if summary["tokens"] is not None
            else ("n/a (no spatial context)" if not config["spatial_context"] else "none (no valid record)")
        ),
        f"output spatial     : {summary['output_spatial']} {summary['output_dtype']}",
        f"object pointer     : {summary['pointer_in']} -> {summary['pointer_out']}",
        f"trainable params   : spatial {params['spatial']:,} | pointer {params['pointer']:,} "
        f"| total {params['total']:,}",
        f"MACs per frame     : ~{_millions(arch['macs_per_frame'])} (context branch "
        f"{_millions(arch['context_branch_macs'])} + per-pixel {_millions(arch['per_pixel_macs'])})",
        f"PyTorch FLOPs/frame: {_millions(summary['pytorch_flops_per_frame'])} (FlopCounterMode, "
        "matmul/conv only; ~2 FLOPs per MAC, a different unit from MACs)",
        f"identity at init   : max|T(M)-M| = {summary['identity_spatial_max_abs']:.3g} | "
        f"max|T(p)-p| = {summary['identity_pointer_max_abs']:.3g}",
    ]
    attention = arch["attention"]
    if attention is not None:
        size = attention["matrix"]
        lines.append(
            f"attention scope    : per frame, {size[0]}x{size[1]} (no cross-frame attention)"
        )
    elif config["spatial_context"]:
        lines.append("attention scope    : none (context branch without self-attention)")
    else:
        lines.append("attention scope    : none (position-wise head)")
    lines.extend(["", f"{'module':<16}{'params':>12}{'MACs/frame':>14}"])
    macs = arch["macs_by_module"]
    params_by_module = arch["spatial_parameters_by_module"]
    for module, mac_keys in _SUMMARY_MODULES:
        if module not in params_by_module and not any(key in macs for key in mac_keys):
            continue
        present = [key for key in mac_keys if key in macs]
        cell = _millions(sum(macs[key] for key in present)) if present else "-"
        note = ""
        if len(present) > 1:
            note = "  (" + " + ".join(f"{key} {_millions(macs[key])}" for key in present) + ")"
        lines.append(f"{module:<16}{params_by_module.get(module, 0):>12,}{cell:>14}{note}")
    lines.append(
        f"{'pointer':<16}{params['pointer']:>12,}"
        f"{_millions(arch['pointer_macs_per_record']):>14}  (per record)"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vos_memory_inspector.transformer_translator",
        description="Dummy forward summary of a spatial-context translator preset.",
    )
    parser.add_argument("--preset", default="base", choices=preset_names(kind=_ST))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--objects", type=int, default=2)
    parser.add_argument("--records", type=int, default=3)
    parser.add_argument("--invalid", type=int, default=1, help="padding records (end of order)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    translator = build_translator(args.preset, SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).eval()
    try:
        summary = summarize_dummy_forward(
            translator,
            batch=args.batch,
            objects=args.objects,
            records=args.records,
            invalid=args.invalid,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(format_dummy_forward_summary(summary))
    return 0


if __name__ == "__main__":
    # Run the package-qualified module so its classes are the ones other
    # modules import.
    from vos_memory_inspector.transformer_translator import main as _main

    raise SystemExit(_main())
