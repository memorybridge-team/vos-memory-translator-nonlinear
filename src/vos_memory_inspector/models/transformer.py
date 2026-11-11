"""Transformer translator: linear stitching layer + one Pre-LN encoder block.

    Z     = S(M)                 per-position affine map, ridge start
    Z     = Z + Attn(LN(Z))      self-attention inside the record
    M_hat = Z + FFN(LN(Z))       position-wise two-layer GELU network
    p_hat = A p + c              linear pointer map, ridge start

The only difference to the MLP translator is the attention sublayer, so a
comparison of the two measures what spatial context adds (see ``_stitched.py``
for the parts and their sources, and ``docs/translator_model_design.md``).
Attention stays inside one record; records and objects never mix.
"""

from __future__ import annotations

from ._stitched import StitchedStateTranslator


class TransformerStateTranslator(StitchedStateTranslator):
    """Stitching layer followed by a standard Pre-LN Transformer encoder block
    (Vaswani et al., 2017; Dosovitskiy et al., ICLR 2021) whose attention uses
    spatially reduced keys and values (Wang et al., ICCV 2021)."""

    name = "transformer"
    attention = True
