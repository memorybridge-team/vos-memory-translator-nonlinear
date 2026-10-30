"""Losses and optimizer loop for learned canonical-state translators."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import torch
from torch.nn import functional as F

from .device import resolve_device
from .state_schema import CanonicalState
from .models._base import _LearnedStateTranslator
from .translators import _pair_guard, _resample_spatial

DEFAULT_EPOCHS = 8
DEFAULT_LEARNING_RATE = 1e-3


def state_mse_loss(prediction: CanonicalState, target: CanonicalState) -> torch.Tensor:
    return state_mse_components(prediction, target)["state_loss"]


def state_mse_components(
    prediction: CanonicalState, target: CanonicalState
) -> dict[str, torch.Tensor]:
    valid = _pair_guard(prediction, target)
    if not valid.any():
        raise ValueError("paired state has no valid records")
    # 패딩의 NaN이 역전파에 들어가지 않도록 빼기 전에 유효 기록을 고른다.
    spatial_loss = _component_mse(prediction.spatial_memory, target.spatial_memory, valid)
    pointer_loss = _component_mse(prediction.object_pointer, target.object_pointer, valid)
    return dict(spatial_mse=spatial_loss, pointer_mse=pointer_loss,
                state_loss=spatial_loss + pointer_loss)


def _component_mse(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    values = prediction[valid.to(prediction.device)].float()
    truth = target[valid.to(target.device)].to(device=values.device, dtype=values.dtype)
    return F.mse_loss(values, truth)


def evaluate_state_pairs(
    translator: Any, pairs: Iterable[tuple[CanonicalState, CanonicalState]]
) -> dict[str, float]:
    """FP32, whole-state component MSEs; equal pair weights, no handoff cast."""
    rows = []
    training = translator.training if isinstance(translator, torch.nn.Module) else None
    if training is not None:
        translator.eval()
    try:
        with torch.no_grad():
            for source, target in pairs:
                if isinstance(translator, _LearnedStateTranslator):
                    if source.spec != translator.source_spec or target.spec != translator.target_spec:
                        raise ValueError("paired state specs do not match the translator")
                    spatial, pointer = translator.translate_tensors(
                        source.spatial_memory, source.object_pointer, source.validity
                    )
                    prediction = source.with_continuous(
                        spatial_memory=spatial, object_pointer=pointer,
                        presence_logits=source.presence_logits,
                    )
                else:
                    # Direct/Ridge도 bf16 입력에서 계산하지 않는다.
                    fp32_source = source.with_continuous(
                        spatial_memory=source.spatial_memory.float(),
                        object_pointer=source.object_pointer.float(),
                        presence_logits=source.presence_logits,
                    )
                    prediction = translator.translate(fp32_source)
                metrics = state_mse_components(prediction, target)
                if not all(torch.isfinite(value) for value in metrics.values()):
                    raise ValueError("non-finite evaluation loss")
                rows.append({key: float(value.cpu()) for key, value in metrics.items()})
    finally:
        if training is not None:
            translator.train(training)
    if not rows:
        raise ValueError("at least one evaluation pair is required")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def fit_validated_translator(
    translator: _LearnedStateTranslator,
    pairs: Iterable[tuple[CanonicalState, CanonicalState]],
    validation_pairs: Iterable[tuple[CanonicalState, CanonicalState]],
    **fit_options: Any,
) -> dict[str, Any]:
    """Select the minimum post-update validation loss; ties keep the earlier epoch."""
    validation_pairs = list(validation_pairs)
    if not validation_pairs:
        raise ValueError("non-empty validation_pairs are required for checkpoint selection")
    # 모든 validation 계약을 첫 optimizer update 전에 검사한다.
    evaluate_state_pairs(translator, validation_pairs)
    validation_history: list[dict[str, Any]] = []
    best_weights: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch = math.inf, 0

    def validate_epoch(epoch: int) -> None:
        nonlocal best_weights, best_loss, best_epoch
        metrics = evaluate_state_pairs(translator, validation_pairs)
        validation_history.append({"epoch": epoch, **metrics})
        if metrics["state_loss"] < best_loss:
            best_loss, best_epoch = metrics["state_loss"], epoch
            best_weights = {key: value.detach().cpu().clone()
                            for key, value in translator.state_dict().items()}

    history = fit_gradient_translator(
        translator, pairs, on_epoch_end=validate_epoch, **fit_options
    )
    assert best_weights is not None
    translator.load_state_dict(best_weights, strict=True)
    translator.eval()
    return dict(train_loss_history=history, validation_history=validation_history,
                selected_epoch=best_epoch, selected_validation_loss=best_loss,
                train_history_phase="before_epoch_update",
                validation_history_phase="after_epoch_update")


def _whole_state_mse_loss(
    translator: _LearnedStateTranslator, source: CanonicalState, target: CanonicalState
) -> torch.Tensor:
    # Handoff의 bf16 변환을 거치지 않고 FP32 출력으로 학습한다.
    spatial, pointer = translator.translate_tensors(
        source.spatial_memory, source.object_pointer, source.validity
    )
    valid = source.validity
    return _component_mse(spatial, target.spatial_memory, valid) + _component_mse(
        pointer, target.object_pointer, valid
    )


def fit_gradient_translator(
    translator: _LearnedStateTranslator,
    pairs: Iterable[tuple[CanonicalState, CanonicalState]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str | torch.device | None = None,
    spatial_samples_per_pair: int | None = None,
    on_epoch_end: Callable[[int], None] | None = None,
) -> list[float]:
    device = resolve_device(device)
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("at least one paired state is required")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    for source, target in pair_list:
        valid = _pair_guard(source, target)
        if not valid.any():
            raise ValueError("paired state has no valid records")
        if source.spec != translator.source_spec or target.spec != translator.target_spec:
            raise ValueError("paired state specs do not match the translator")
    if spatial_samples_per_pair is not None and spatial_samples_per_pair < 1:
        raise ValueError("spatial_samples_per_pair must be positive")
    if spatial_samples_per_pair is not None:
        _require_position_sampling(translator)
    translator.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(translator.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(torch.initial_seed())
    history: list[float] = []
    translator.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for source, target in pair_list:
            if spatial_samples_per_pair is None:
                loss = _whole_state_mse_loss(translator, source, target)
            else:
                loss = _sampled_state_mse_loss(
                    translator,
                    source,
                    target,
                    spatial_samples=spatial_samples_per_pair,
                    generator=generator,
                )
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            # Pair마다 그래프를 해제하면서 기존 pair 평균 기울기를 유지한다.
            (loss / len(pair_list)).backward()
            epoch_loss += float(loss.detach().cpu()) / len(pair_list)
        optimizer.step()
        history.append(epoch_loss)
        if on_epoch_end is not None:
            on_epoch_end(epoch)
    translator.eval()
    return history


def _require_position_sampling(translator: _LearnedStateTranslator) -> None:
    if not getattr(translator, "supports_position_sampling", True):
        raise ValueError(
            f"{type(translator).__name__} reads spatial context across a frame, so "
            "spatial_samples_per_pair (independent position sampling) cannot train "
            "it; train on whole frames with translate_tensors instead"
        )


def _sampled_state_mse_loss(
    translator: _LearnedStateTranslator,
    source: CanonicalState,
    target: CanonicalState,
    *,
    spatial_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Estimate component-balanced state loss without materializing full MLP output."""

    _require_position_sampling(translator)
    valid = _pair_guard(source, target)
    if not valid.any():
        raise ValueError("paired state has no valid records")
    source_spatial = _resample_spatial(
        source.spatial_memory,
        translator.target_spec.height,
        translator.target_spec.width,
    )
    source_vectors = source_spatial.movedim(3, -1)[valid.to(source_spatial.device)].reshape(
        -1, translator.source_spec.feature_channels
    )
    target_vectors = target.spatial_memory.movedim(3, -1)[
        valid.to(target.spatial_memory.device)
    ].reshape(-1, translator.target_spec.feature_channels)
    sample_count = min(spatial_samples, source_vectors.shape[0])
    indices = torch.randperm(
        source_vectors.shape[0], generator=generator
    )[:sample_count]
    predicted_spatial = translator._feature_head(
        source_vectors[indices.to(source_vectors.device)]
    )
    spatial_loss = F.mse_loss(
        predicted_spatial,
        target_vectors[indices.to(target_vectors.device)].to(
            device=predicted_spatial.device, dtype=predicted_spatial.dtype
        ),
    )

    source_pointer = source.object_pointer[valid.to(source.object_pointer.device)]
    predicted_pointer = translator._pointer_head(source_pointer)
    pointer_loss = F.mse_loss(
        predicted_pointer,
        target.object_pointer[valid.to(target.object_pointer.device)].to(
            device=predicted_pointer.device, dtype=predicted_pointer.dtype
        ),
    )
    return spatial_loss + pointer_loss
