"""Building blocks and shared wrapper of the MLP and the Transformer translator.

One record is translated at a time. Its spatial memory ``M`` is read as
``H*W`` tokens of ``C`` channels, in the order of the memory grid::

    Z     = S(M)                  linear stitching layer, initialised by ridge
    Z     = Z + Attn(LN(Z))       Transformer model only
    M_hat = Z + FFN(LN(Z))        both models

The object pointer is mapped by one linear layer, ``p_hat = A p + c``, also
initialised by ridge. The two models share every module and differ only in
the attention sublayer. ``docs/translator_model_design.md`` lists the source
of each part and how strong the match is; the class docstrings name the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..state_schema import StateSpec
from ._base import _LearnedStateTranslator
from .affine import copy_affine


# Runtime contract v1.1: spatial [B,O,K,64,64,64] bf16, pointer [B,O,K,256] fp32.
# SAM 2.1 Small, Base+ and Large all use this shape (checked in the configs and
# on stored Small, Base+ and Large states), so one spec serves the three pairs.
SAM21_MEMORY_SPEC = StateSpec(feature_channels=64, height=64, width=64, pointer_dim=256)

OUTPUT_DTYPES = ("source", "float32")
PAYLOAD_SCHEMA = "cmmt.stitched_translator.v1"
RIDGE_SCHEMA = "cmmt.streaming_ridge.v1"


@dataclass(frozen=True)
class TranslatorConfig:
    """Structure of one translator. The same values serve every model pair.

    ``attention`` is the only switch between the two models and is set by the
    model class, not by the caller. ``num_heads`` and ``sr_ratio`` are used only
    when ``attention`` is true.
    """

    # Number of blocks. One block is the smallest form; Denoising Vision
    # Transformers (Yang et al., ECCV 2024) also correct a frozen model's
    # feature map with a single Transformer block.
    num_layers: int = 1
    # Hidden width of the feed-forward sublayer over the token width. 4 is the
    # ratio of Vaswani et al. (2017, 512 -> 2048) and of ViT.
    mlp_ratio: float = 4.0
    attention: bool = False
    # One head at 64 channels follows stage 1 of the Pyramid Vision Transformer
    # (Wang et al., ICCV 2021); SAM 2's memory attention is single-head too.
    num_heads: int = 1
    # Spatial reduction of keys and values (PVT). 4 gives a 16x16 key grid for
    # the 64x64 memory. The value was not tuned. PVT uses 8 at its 64-channel
    # stage and 4 at the next; MViTv2 (Li et al., CVPR 2022) pools keys and
    # values with stride 4 at its highest-resolution stage.
    sr_ratio: int = 4
    output_dtype: str = "source"  # "source" | "float32"

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_heads", "sr_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if not isinstance(self.attention, bool):
            raise ValueError("attention must be a bool")
        if not math.isfinite(float(self.mlp_ratio)) or self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive and finite")
        if self.output_dtype not in OUTPUT_DTYPES:
            raise ValueError(f"output_dtype must be one of {OUTPUT_DTYPES}")

    def hidden_dim(self, channels: int) -> int:
        hidden = float(channels) * float(self.mlp_ratio)
        if hidden < 1 or not hidden.is_integer():
            raise ValueError("channels * mlp_ratio must be a positive integer")
        return int(hidden)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TranslatorConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown TranslatorConfig fields: {unknown}")
        return cls(**dict(values))


def sine_position_encoding(
    height: int, width: int, dim: int, temperature: float = 10000.0
) -> torch.Tensor:
    """Fixed 2D sine encoding ``[height*width, dim]`` in row-major order.

    The form is the one of DETR (Carion et al., ECCV 2020), which extends the
    encoding of Vaswani et al. (2017) to two axes: half of the channels encode
    ``y`` and half ``x``. Coordinates are cell centres scaled to ``(0, 2*pi)``,
    so grids of different resolution share one coordinate frame.
    """

    if dim % 4:
        raise ValueError("sine position encoding needs a channel count divisible by 4")
    half = dim // 2
    exponent = torch.arange(half, dtype=torch.float32) // 2 * 2 / half
    scale = temperature ** exponent  # [half]
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height * (2 * math.pi)
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width * (2 * math.pi)

    def encode(values: torch.Tensor) -> torch.Tensor:
        angles = values[:, None] / scale  # [n, half]
        return torch.stack((angles[:, 0::2].sin(), angles[:, 1::2].cos()), dim=2).flatten(1)

    pos_y = encode(ys)[:, None, :].expand(height, width, half)
    pos_x = encode(xs)[None, :, :].expand(height, width, half)
    return torch.cat((pos_y, pos_x), dim=2).reshape(height * width, dim)


class StitchingLayer(nn.Module):
    """``Z = W M + b``, the same affine map at every position (a 1x1 convolution).

    Source: the stitching layer of model stitching. Lenc & Vedaldi (CVPR 2015)
    introduced a learned bank of linear filters between two networks; Bansal et
    al. (NeurIPS 2021) and Csiszarik et al. (NeurIPS 2021) use a 1x1 convolution
    between frozen networks (Bansal et al. add BatchNorm around it, which this
    layer does not). Csiszarik et al. start the layer from the least-squares
    solution before training it, and Stitchable Neural Networks (Pan et al.,
    CVPR 2023) do the same while also updating the networks;
    ``initialize_from_ridge`` does that first step here. Before it the layer is
    the identity when both sides have the same width, a neutral placeholder.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if in_dim == out_dim:
            with torch.no_grad():
                nn.init.eye_(self.linear.weight)
                self.linear.bias.zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [..., Cs] -> [..., Ct]
        return self.linear(tokens)


class FeedForward(nn.Module):
    """Position-wise feed-forward sublayer ``W2 GELU(W1 x)``.

    Source: Vaswani et al. (NeurIPS 2017, section 3.3), "applied to each
    position separately and identically", with inner width 4x; GELU as in ViT
    (Dosovitskiy et al., ICLR 2021; Hendrycks & Gimpel, 2016). The same form
    (two linear layers, GELU, hidden width 4x) maps tokens into the feature
    space of another model in UNIC (Sariyildiz et al., ECCV 2024), and a
    two-layer GELU MLP replaces a linear vision-language projection in
    LLaVA-1.5 (Liu et al., CVPR 2024). The last layer starts at zero so that
    the block starts as the identity (rule 1 of Fixup, Zhang et al., ICLR
    2019; Goyal et al., 2017).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [..., d] -> [..., d]
        return self.fc2(F.gelu(self.fc1(tokens)))


class SpatialReductionAttention(nn.Module):
    """Self-attention inside one record, with spatially reduced keys and values.

    Every position is a query. Keys and values come from the same tokens after
    a strided convolution (kernel = stride = ``sr_ratio``) and a LayerNorm, as
    in the spatial-reduction attention of the Pyramid Vision Transformer (Wang
    et al., ICCV 2021), which was introduced to make attention affordable on
    high-resolution feature maps. The paper writes the reduction as a reshape
    followed by a linear projection and a norm; its official code uses this
    strided convolution, which is the same map. The attention itself is the
    scaled dot-product attention of Vaswani et al. (2017). EdgeTAM (Zhou et
    al., CVPR 2025) also applies attention to each stored SAM 2 memory map
    separately.

    The fixed sine position encoding is added to queries and keys only, at the
    attention, as in DETR (Carion et al., ECCV 2020). Values and the residual
    stream carry no position signal, so the feed-forward path is the same
    function in both models. Placing the coordinates at cell centres, so that
    the query grid and the reduced key grid share one frame, is a detail of
    this code. The output projection starts at zero (Fixup).
    """

    def __init__(self, dim: int, grid: tuple[int, int], *, num_heads: int, sr_ratio: int):
        super().__init__()
        height, width = grid
        if dim % num_heads:
            raise ValueError(f"channels {dim} must be divisible by num_heads {num_heads}")
        if height % sr_ratio or width % sr_ratio:
            raise ValueError(
                f"grid {height}x{width} must be divisible by sr_ratio {sr_ratio}"
            )
        self.dim = dim
        self.grid = (height, width)
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if sr_ratio > 1:
            self.sr: nn.Module | None = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm: nn.Module | None = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.sr_norm = None
        reduced = (height // sr_ratio, width // sr_ratio)
        self.reduced_grid = reduced
        self.register_buffer(
            "pos_query", sine_position_encoding(height, width, dim), persistent=False
        )
        self.register_buffer(
            "pos_key", sine_position_encoding(*reduced, dim), persistent=False
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [N, H*W, d] -> [N, H*W, d]
        count, positions, dim = tokens.shape
        height, width = self.grid
        if positions != height * width:
            raise ValueError(f"expected {height * width} tokens, got {positions}")
        if self.sr is None:
            reduced = tokens
        else:
            grid = tokens.transpose(1, 2).reshape(count, dim, height, width)
            reduced = self.sr_norm(self.sr(grid).flatten(2).transpose(1, 2))  # [N, P/R^2, d]
        query = self.q(tokens + self.pos_query.to(tokens.dtype))
        key = self.k(reduced + self.pos_key.to(tokens.dtype))
        value = self.v(reduced)

        def split(tensor: torch.Tensor) -> torch.Tensor:  # [N, L, d] -> [N, heads, L, d/heads]
            return tensor.reshape(count, -1, self.num_heads, dim // self.num_heads).transpose(1, 2)

        mixed = F.scaled_dot_product_attention(split(query), split(key), split(value))
        return self.proj(mixed.transpose(1, 2).reshape(count, positions, dim))

    def macs(self) -> int:
        positions = self.grid[0] * self.grid[1]
        keys = self.reduced_grid[0] * self.reduced_grid[1]
        dim = self.dim
        reduction = keys * dim * dim * self.sr_ratio ** 2 if self.sr is not None else 0
        projections = (2 * positions + 2 * keys) * dim * dim  # q, proj; k, v
        scores_and_mix = 2 * positions * keys * dim
        return reduction + projections + scores_and_mix


class TranslatorBlock(nn.Module):
    """One Pre-LN Transformer encoder block; the MLP model omits the attention.

    Source: the encoder block of Vaswani et al. (2017) in the Pre-LN order of
    ViT (Dosovitskiy et al., ICLR 2021; Xiong et al., ICML 2020; LayerNorm from
    Ba et al., 2016). Each sublayer is a residual branch (He et al., CVPR 2016).
    The block acts on the stitched memory, which is already a linear estimate
    of the target memory, so each branch learns a correction to that estimate.
    """

    def __init__(
        self,
        dim: int,
        grid: tuple[int, int],
        *,
        mlp_ratio: float,
        attention: bool,
        num_heads: int,
        sr_ratio: int,
    ):
        super().__init__()
        hidden = float(dim) * float(mlp_ratio)
        if hidden < 1 or not hidden.is_integer():
            raise ValueError("channels * mlp_ratio must be a positive integer")
        if attention:
            self.norm_attn: nn.Module | None = nn.LayerNorm(dim)
            self.attn: SpatialReductionAttention | None = SpatialReductionAttention(
                dim, grid, num_heads=num_heads, sr_ratio=sr_ratio
            )
        else:
            self.norm_attn = None
            self.attn = None
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(hidden))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            tokens = tokens + self.attn(self.norm_attn(tokens))
        return tokens + self.mlp(self.norm_mlp(tokens))


class SpatialTranslator(nn.Module):
    """``T: [N,Cs,H,W] -> [N,Ct,H,W]``, applied to each record on its own.

    A stitching layer followed by ``num_layers`` blocks. There is no final
    LayerNorm: the output is a memory feature map whose scale the target model
    reads, not the input of a prediction head.
    """

    def __init__(
        self,
        source_channels: int,
        target_channels: int,
        grid: tuple[int, int],
        config: TranslatorConfig,
    ):
        super().__init__()
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.grid = (int(grid[0]), int(grid[1]))
        self.attention = config.attention
        self.hidden_dim = config.hidden_dim(target_channels)
        self.stitch = StitchingLayer(source_channels, target_channels)
        self.blocks = nn.ModuleList(
            TranslatorBlock(
                target_channels,
                self.grid,
                mlp_ratio=config.mlp_ratio,
                attention=config.attention,
                num_heads=config.num_heads,
                sr_ratio=config.sr_ratio,
            )
            for _ in range(config.num_layers)
        )

    def forward_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[..., Cs] -> [..., Ct]`` for the model without attention.

        Without attention every operation is position-wise, so single positions
        may be translated without their record.
        """

        if self.attention:
            raise ValueError("the attention model needs whole records, not single positions")
        tokens = self.stitch(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N,C,H,W], got {tuple(frames.shape)}")
        count, channels, height, width = frames.shape
        if channels != self.source_channels or (height, width) != self.grid:
            raise ValueError(
                f"frame [C,H,W]={(channels, height, width)} does not match the configured "
                f"{(self.source_channels, *self.grid)}; the translator does not resample"
            )
        tokens = self.stitch(frames.flatten(2).transpose(1, 2))  # [N, H*W, Ct]
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(count, self.target_channels, height, width)

    def macs_breakdown(self) -> dict[str, int]:
        """Multiply-accumulates per record: linear layers, the reduction
        convolution and the two attention products. LayerNorm, GELU, softmax,
        the position encoding and additions are not counted."""

        positions = self.grid[0] * self.grid[1]
        macs = {
            "stitch": positions * self.source_channels * self.target_channels,
            "feed_forward": len(self.blocks) * positions * 2 * self.target_channels * self.hidden_dim,
            "attention": sum(block.attn.macs() for block in self.blocks if block.attn is not None),
        }
        macs["total"] = sum(macs.values())
        return macs

    def parameter_breakdown(self) -> dict[str, int]:
        def count(module: nn.Module | None) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        result = {
            "stitch": count(self.stitch),
            "feed_forward": sum(count(b.mlp) + count(b.norm_mlp) for b in self.blocks),
            "attention": sum(count(b.attn) + count(b.norm_attn) for b in self.blocks),
        }
        result["total"] = sum(result.values())
        return result


class LinearPointerTranslator(nn.Module):
    """``p_hat = A p + c`` for one object pointer.

    Source: a linear map fitted on paired vectors maps one independently
    trained vector space into another (Mikolov et al., 2013; Lahner & Moeller,
    UniReps 2024). SAM 2 takes the pointer from the mask decoder's output token
    and the spatial memory from the memory encoder (Ravi et al., ICLR 2025), and
    their sizes differ, so the pointer has its own head; UNIC (Sariyildiz et
    al., ECCV 2024) likewise uses dedicated projectors for the global token and
    the patch tokens. Initialised by ridge; identity before that when sizes
    match.
    """

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        if source_dim < 1 or target_dim < 1:
            raise ValueError("pointer dims must be positive")
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.linear = nn.Linear(source_dim, target_dim)
        if source_dim == target_dim:
            with torch.no_grad():
                nn.init.eye_(self.linear.weight)
                self.linear.bias.zero_()

    def forward(self, pointers: torch.Tensor) -> torch.Tensor:  # [..., Ds] -> [..., Dt]
        if pointers.shape[-1] != self.source_dim:
            raise ValueError(f"pointer dim {pointers.shape[-1]} != {self.source_dim}")
        return self.linear(pointers)

    def macs_per_record(self) -> int:
        return self.source_dim * self.target_dim


class StitchedStateTranslator(_LearnedStateTranslator):
    """Shared wrapper: ``CanonicalState`` handling, ridge start, save and load.

    Subclasses fix ``attention`` and nothing else. ``translate`` gathers the
    valid records ``[B,O,K,...] -> [N,...]``, translates each record on its
    own and scatters the result back; padding records keep the source.
    """

    name = "stitched"
    attention = False
    grid_adapter = "none"
    _spatial_module_prefix = "spatial"
    _pointer_module_prefix = "pointer"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        config: TranslatorConfig | None = None,
        preset: str | None = None,
    ):
        config = TranslatorConfig() if config is None else config
        config = replace(config, attention=type(self).attention)
        super().__init__(source_spec, target_spec, output_dtype=config.output_dtype)
        if (source_spec.height, source_spec.width) != (target_spec.height, target_spec.width):
            raise ValueError(
                "source and target memory grids must be equal; the translator does not resample"
            )
        self.config = config
        self.preset = preset
        self.ridge_initialization: dict[str, Any] | None = None
        self.spatial = SpatialTranslator(
            source_spec.feature_channels,
            target_spec.feature_channels,
            (target_spec.height, target_spec.width),
            config,
        )
        self.pointer = LinearPointerTranslator(source_spec.pointer_dim, target_spec.pointer_dim)

    # The legacy sampled trainer draws single positions; only the model without
    # attention can translate them.
    @property
    def supports_position_sampling(self) -> bool:  # type: ignore[override]
        return not self.config.attention

    def initialize_from_ridge(self, ridge: Mapping[str, Any]) -> None:
        """Copy the closed-form affine maps into the stitching layer and the
        pointer head (least-squares start of Pan et al., CVPR 2023)."""

        if ridge.get("schema_version") != RIDGE_SCHEMA:
            raise ValueError("streaming ridge artifact required")
        if (
            ridge["source_spec"] != self.source_spec.to_dict()
            or ridge["target_spec"] != self.target_spec.to_dict()
        ):
            raise ValueError("ridge specs mismatch")
        provenance = ridge.get("provenance", {})
        if not provenance.get("artifact_sha256") or not provenance.get("data_identity"):
            raise ValueError("ridge hash and data identity required")
        copy_affine(self.spatial.stitch.linear, ridge["spatial"])
        copy_affine(self.pointer.linear, ridge["pointer"])
        self.ridge_initialization = dict(provenance)

    def _feature_head(self, tensor: torch.Tensor) -> torch.Tensor:
        reference = self._compute_reference()
        return self.spatial.forward_positions(
            tensor.to(device=reference.device, dtype=reference.dtype)
        )

    def _pointer_head(self, tensor: torch.Tensor) -> torch.Tensor:
        reference = self._compute_reference()
        return self.pointer(tensor.to(device=reference.device, dtype=reference.dtype))

    def _translate_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.spatial(frames)  # [N, Ct, H, W]

    def _translate_pointers(self, pointers: torch.Tensor) -> torch.Tensor:
        return self.pointer(pointers)  # [N, Dt]

    def macs_per_frame(self) -> int:
        return self.spatial.macs_breakdown()["total"]

    def pointer_macs_per_record(self) -> int:
        return self.pointer.macs_per_record()

    @property
    def description(self) -> str:
        config = self.config
        blocks = f"{config.num_layers} Pre-LN block" + ("s" if config.num_layers > 1 else "")
        if config.attention:
            parts = (
                f"spatial-reduction attention ({config.num_heads} head, ratio {config.sr_ratio})"
                f" + feed-forward (ratio {config.mlp_ratio:g})"
            )
        else:
            parts = f"feed-forward only (ratio {config.mlp_ratio:g})"
        return f"linear stitching layer + {blocks}: {parts}; linear pointer"

    def architecture_summary(self) -> dict[str, Any]:
        """JSON-safe structure, parameter and analytic-cost summary."""

        config = self.config
        attention = None
        if config.attention:
            first = self.spatial.blocks[0].attn
            attention = {
                "scope": "per_record",
                "queries": first.grid[0] * first.grid[1],
                "keys": first.reduced_grid[0] * first.reduced_grid[1],
                "heads": config.num_heads,
                "sr_ratio": config.sr_ratio,
                "position_encoding": "fixed sine on queries and keys",
                "cross_record": False,
            }
        return {
            "translator": self.name,
            "preset": self.preset,
            "description": self.description,
            "config": config.to_dict(),
            "attention": attention,
            "parameters": self.parameter_breakdown(),
            "spatial_parameters_by_module": self.spatial.parameter_breakdown(),
            "macs_per_frame": self.macs_per_frame(),
            "macs_by_module": self.spatial.macs_breakdown(),
            "pointer_macs_per_record": self.pointer_macs_per_record(),
            "start": "ridge" if self.ridge_initialization else "identity_placeholder",
        }

    def _translation_metadata(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "architecture": self.description,
            "attention_scope": "per_record_tokens" if self.config.attention else "none",
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
            "ridge_initialization": self.ridge_initialization,
            "state_dict": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StitchedStateTranslator":
        schema = payload.get("schema_version")
        if schema != PAYLOAD_SCHEMA:
            raise ValueError(
                f"unsupported translator payload {schema!r}: models saved before the "
                "translator redesign (residual_mlp v2/v3, spatial_context_transformer v1/v2) "
                "cannot be loaded; use a checkout from before the redesign, or retrain"
            )
        if payload.get("translator") != cls.name:
            raise ValueError(
                f"payload holds a {payload.get('translator')!r} model, not {cls.name!r}"
            )
        config_values = payload.get("config")
        if not isinstance(config_values, Mapping):
            raise TypeError("payload config must be a mapping")
        missing = sorted({item.name for item in fields(TranslatorConfig)} - set(config_values))
        if missing:
            raise ValueError(f"payload config is missing fields: {missing}")
        config = TranslatorConfig.from_dict(config_values)
        if config.attention != cls.attention:
            raise ValueError("payload attention setting does not match the model class")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("payload state_dict must be a mapping")
        preset = payload.get("preset")
        translator = cls(
            StateSpec(**dict(payload["source_spec"])),
            StateSpec(**dict(payload["target_spec"])),
            config=config,
            preset=None if preset is None else str(preset),
        )
        ridge = payload.get("ridge_initialization")
        translator.ridge_initialization = None if ridge is None else dict(ridge)
        translator.load_state_dict(dict(state_dict), strict=True)
        translator.eval()
        return translator
