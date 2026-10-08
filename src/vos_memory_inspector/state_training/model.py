"""모델팀 고정 body 재사용. 새 학습은 fresh init이며 tensor 계약을 바꾸지 않는다."""
from pathlib import Path
import hashlib
import subprocess
import torch
from torch import nn
from ..transformer_translator import TransformerStateTranslator, SpatialTransformerConfig, SAM21_MEMORY_SPEC
from .common import require, digest

MODEL_REVISION = "746ea3e7d84c366c2d7ac06159e90a1f684bca56"
FROZEN_HASHES = {
    "transformer_translator.py": "60d56b6785e83c10d2dc313986189d93e8d223ebc031cc6054706986b63f4421",
    "frozen_tensor_api.py": "6e647775fa96726b26ca6f56e9ac4ef144ccd617c28d334e3baf664e7ad8e0ff",
}


def text_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def lock():
    package = Path(__file__).resolve().parents[1]
    require(all(text_sha(package / n) == h for n, h in FROZEN_HASHES.items()), "FROZEN_MODEL_SOURCE_CHANGED")
    value = dict(model_revision=MODEL_REVISION, source_hashes=FROZEN_HASHES,
                 factory="vos_memory_inspector.transformer_translator:TransformerStateTranslator", preset="base",
                 config=SpatialTransformerConfig().to_dict(), source_spec=SAM21_MEMORY_SPEC.to_dict(), target_spec=SAM21_MEMORY_SPEC.to_dict())
    return dict(value, digest=digest(value))


def fresh_model():
    contract = lock()
    return TransformerStateTranslator(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC,
                                      config=SpatialTransformerConfig.from_dict(contract["config"]), preset="base")


class TensorModule(nn.Module):
    def __init__(self, translator):
        super().__init__()
        self.translator = translator

    def forward(self, spatial, pointer):
        s, p = self.translator.translate_tensors(spatial[None, None].float(), pointer[None, None].float())
        return s[0, 0], p[0, 0]


def optimizer_groups(model, weight_decay):
    norms = {id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters(recurse=False)}
    decay, excluded = [], []
    for name, p in model.named_parameters():
        no_decay = name.endswith(".bias") or id(p) in norms or name.endswith(".alpha") or "pos_embed" in name
        (excluded if no_decay else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": excluded, "weight_decay": 0.0}]


def provenance():
    package = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(package)): text_sha(p) for p in sorted(package.rglob("*.py"))}
    try:
        result = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        revision = result.stdout.strip() if result.returncode == 0 else "UNKNOWN_SOURCE_BUNDLE"
    except (OSError, subprocess.TimeoutExpired):
        revision = "UNKNOWN_SOURCE_BUNDLE"
    return dict(revision=revision, package_source_sha256=digest(hashes), source_hashes=hashes)
