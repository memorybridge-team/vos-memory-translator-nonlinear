"""Shared spatial backbone, fusion options, pointer head and configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..state_schema import StateSpec
from .affine import make_skip


# Runtime contract v1.1: spatial [B,O,K,64,64,64] bf16, pointer [B,O,K,256] fp32.
SAM21_MEMORY_SPEC = StateSpec(feature_channels=64, height=64, width=64, pointer_dim=256)

PATCH_EMBEDS = ("conv", "avgpool")
POS_EMBEDS = ("learnable", "sincos", "none")
LOCAL_FUSIONS = ("linear", "mlp", "additive_mlp", "none")
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
_PIXEL_MAC_KEYS = ("local_in", "local_out", "linear_skip")


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
    local_fusion: str = "linear"  # "linear" | "mlp" | "additive_mlp" | "none"
    fusion_hidden_dim: int = 128
    use_residual: bool = True
    residual_scale_init: float = 0.0
    # Pointer head and handoff output.
    pointer_dim: int = 256
    pointer_hidden_dim: int = 512
    output_dtype: str = "source"  # "source" | "float32"
    skip_mode: str = "identity"

    def __post_init__(self) -> None:
        if self.skip_mode not in ("identity", "affine"):
            raise ValueError("unknown skip_mode")
        if self.skip_mode == "affine" and self.local_fusion != "additive_mlp":
            raise ValueError("affine skip is supported only for additive_mlp context models")
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
        if self.local_fusion == "additive_mlp" and not self.use_residual:
            raise ValueError("additive_mlp requires use_residual=True")

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
    - ``additive_mlp``: ``M_hat = M + W_2(GELU(W_l(M))) + alpha * Up(W_c(ctx))``
    - ``none``: ``Delta = Up(ctx)`` (context only)

    Bilinear upsampling is linear and acts per channel, so it commutes with the
    bias-free 1x1 ``W_c``; running ``W_c`` at the token grid saves MACs.
    Without spatial context ``Delta`` is the position-wise head alone.
    """

    def __init__(self, config: SpatialTransformerConfig | None = None):
        super().__init__()
        config = SpatialTransformerConfig() if config is None else config
        self.config = config
        self.linear_skip = make_skip(config.channels, config.channels) if config.skip_mode == "affine" else None
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
        elif fusion in ("mlp", "additive_mlp"):
            self.local_in = nn.Conv2d(channels, hidden, kernel_size=1)
            self.local_out = nn.Conv2d(hidden, channels, kernel_size=1)
            if fusion == "additive_mlp":
                # 로컬 보정은 gate 없이 더한다. 마지막 층만 0으로 초기화한다.
                nn.init.zeros_(self.local_out.weight)
                nn.init.zeros_(self.local_out.bias)
        else:
            self.local_in = None
            self.local_out = None
        # additive_mlp에서 alpha는 문맥만 조절한다. 문맥 projection은 일반
        # 초기화를 유지해 gate와 projection이 동시에 0이 되는 것을 피한다.
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
        if self.config.local_fusion == "additive_mlp":
            local = self.local_out(F.gelu(fused))
            return (local if context is None else local + self.alpha * context), tokens
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
        if self.config.local_fusion == "additive_mlp":
            skip = memory if self.linear_skip is None else self.linear_skip(memory.movedim(1, -1)).movedim(-1, 1)
            output = skip + delta  # alpha was applied only to context in _delta.
        else:
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
        elif fusion in ("mlp", "additive_mlp"):
            macs["local_in"] = positions * channels * hidden
            macs["local_out"] = positions * hidden * channels
        if config.skip_mode == "affine":
            macs["linear_skip"] = positions * channels * channels
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

    def __init__(self, dim: int = 256, hidden_dim: int = 512, skip_mode: str = "identity"):
        super().__init__()
        if dim < 1 or hidden_dim < 1:
            raise ValueError("pointer dims must be positive")
        self.linear_skip = make_skip(dim, dim) if skip_mode == "affine" else None
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
        skip = pointers if self.linear_skip is None else self.linear_skip(pointers)
        return skip + self.fc2(F.gelu(self.fc1(pointers)))

    def macs_per_record(self) -> int:
        return 2 * self.dim * self.hidden_dim + (self.dim ** 2 if self.linear_skip is not None else 0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def describe_config(config: SpatialTransformerConfig) -> str:
    if not config.spatial_context:
        head = "linear" if config.local_fusion == "linear" else "MLP"
        return f"position-wise {head} head, no spatial context"
    family = "Spatial-Context Transformer" if config.use_attention else "Spatial-Context (no attention)"
    fusion = {"linear": "linear fusion", "mlp": "MLP fusion", "none": "context only",
              "additive_mlp": "independent residual MLP + gated context"}
    return f"{family}, {fusion[config.local_fusion]}"
