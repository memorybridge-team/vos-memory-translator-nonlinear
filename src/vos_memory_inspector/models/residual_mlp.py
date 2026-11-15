"""MLP translator: linear stitching layer + one Pre-LN feed-forward block.

    Z     = S(M)                 per-position affine map, ridge start
    M_hat = Z + FFN(LN(Z))       position-wise two-layer GELU network
    p_hat = A p + c              linear pointer map, ridge start

This is the Transformer translator without its attention sublayer; every
module is shared with it (see ``_stitched.py`` for the parts and their
sources, and ``docs/translator_model_design.md``). No operation looks at a
neighbouring position, at another record or at another object.
"""

from __future__ import annotations

from ._stitched import StitchedStateTranslator


class ResidualMLPStateTranslator(StitchedStateTranslator):
    """Position-wise model: the stitching layer of model stitching (Bansal et
    al., NeurIPS 2021; Pan et al., CVPR 2023) followed by the feed-forward
    sublayer of a Transformer block (Vaswani et al., 2017)."""

    name = "residual_mlp"
    attention = False
