"""validation R² 선정 모델과 마지막 재개 checkpoint를 별도 전달한다."""
from pathlib import Path
import shutil
import tarfile
import torch
from ..transformer_translator import TransformerStateTranslator
from .common import require, read_json, read_bound, sha256, write_json, atomic
from .checkpoint import load, publish, equal


def load_translator(delivery):
    root = Path(delivery)
    ready = read_json(root / "DELIVERY_READY.json")
    require(ready["status"] == "DELIVERY_READY" and ready["selection_metric"] == "validation_r2_score", "R2_DELIVERY_REQUIRED")
    for n in ("best_validation_r2_weights.pth", "model_config.json", "normalization.json", "binding.json"):
        require(sha256(root / n) == ready["files"][n]["sha256"], "DELIVERY_FILE_CHANGED:" + n)
    weights, _ = load(root / "best_validation_r2_weights.pth")
    model = TransformerStateTranslator.from_payload(weights)
    config = read_json(root / "model_config.json")
    require(weights["config"] == config["config"] and read_json(root / "normalization.json")["scope"] == "train_only", "DELIVERY_CONFIG_NORMALIZATION")
    return model


def export_delivery(run, index_path, stats_root):
    run = Path(run)
    root = run / "delivery-final-validation-r2"
    require(not root.exists(), "FINAL_DELIVERY_EXISTS_PRESERVE_USE_NEW_NAMESPACE")
    best = read_json(run / "best_validation_r2.json")
    last = read_json(run / "last_complete.json")
    payload, marker = load(best["checkpoint"]["path"])
    final, _ = load(last["path"], payload["identity"])
    history = read_json(run / "history.json")
    require(best["scope"] == "full_validation" and marker["sha256"] == best["checkpoint"]["sha256"] and
            history and len(history) == final["epoch"] and [h["epoch"] for h in history] == list(range(1, final["epoch"] + 1)), "DELIVERY_FULL_HISTORY_BINDING")
    scores = [h["validation"]["validation_r2_score"] for h in history]
    tolerance = read_bound(Path(stats_root) / "r2_spec.json")["tie_tolerance"]
    maximum = max(scores)
    selected = next(i + 1 for i, s in enumerate(scores) if s >= maximum - tolerance)
    require(best["epoch"] == selected and best["score"] == scores[selected - 1], "DELIVERY_BEST_SELECTION")
    root.mkdir(parents=True)
    weights = payload["export"]
    publish(root / "best_validation_r2_weights.pth", weights,
            lambda saved: require(equal(TransformerStateTranslator.from_payload(saved).to_payload(), weights), "EXPORT_STRICT_RELOAD"))
    for name in ("normalization.json", "r2_spec.json", "validation_targets.json"):
        shutil.copyfile(Path(stats_root) / name, root / name)
    write_json(root / "model_config.json", payload["model_lock"], immutable=True)
    write_json(root / "training_config.json", payload["configuration"], immutable=True)
    write_json(root / "binding.json", dict(identity=payload["identity"], source=payload["source"], selected=best, last_complete=last,
               fresh_training_contract=True, inference_normalization="none; RMS is training loss scale only"), immutable=True)
    for source, name in ((index_path, "index.json"), (Path(index_path).parent / "used_excluded.json", "used_excluded.json"),
                         (run / "history.json", "history.json"), (run / "RUN.json", "RUN.json"), (run / "STATUS.json", "STATUS.json"),
                         (run / "epochs.jsonl", "epochs.jsonl"), (last["path"], "last_complete.pt")):
        shutil.copyfile(source, root / name)
    # 복사 이름에 맞춰 marker path만 바꾼다. SHA는 원 checkpoint와 동일하다.
    write_json(root / "last_complete.pt.complete.json", dict(last, path="last_complete.pt"), immutable=True)
    split_root = Path(read_bound(index_path)["identity"]["split_root"])
    for n in ("original_inventory.json", "original_inventory.json.sha256", "split_manifest.json", "split_manifest.json.sha256", "split_manifest.sha256", "SPLIT_FROZEN.json", "split_report.json"):
        shutil.copyfile(split_root / n, root / n)
    report = f"""# Translator 최종 전달 보고서

- 실제 완료 epoch: {final['epoch']}, optimizer step: {final['optimizer_step']}
- 종료 사유: {read_json(run / 'STATUS.json')['stop_reason']}
- 선정 epoch: {selected}, validation R² score: {best['score']:.12g}
- 선정 checkpoint SHA256: {marker['sha256']}
- 마지막 checkpoint SHA256: {last['sha256']}
- source revision: {payload['source']['revision']}
- split/index/normalization/target-statistics binding: `binding.json`
- 마지막 checkpoint는 재현·재개용이며 선정 weights와 구분한다.
- J&F/benchmark 실행: 없음. 실제 tensor·case 수는 `index.json`의 statistics를 확인한다.

본 모델은 validation state R² 기준으로 선정되었으며,
J&F 및 외부 benchmark 평가는 별도 담당자가 수행한다.

## 로드

```python
from vos_memory_inspector.state_training.delivery import load_translator
translator = load_translator('/workspace/<RUN>/delivery-final-validation-r2')
# canonical Small state의 discrete field를 유지한다. 입력 정규화는 적용하지 않는다.
translated_state = translator.translate(source_state)
```

원본 데이터/cache, 과거 run, 인증정보는 이 전달물에 포함하지 않는다.
새 계약의 재개는 원본 run의 새 계약 index/statistics와 같은 source에서만 수행한다.
"""
    atomic(root / "FINAL_REPORT_KO.md", report.encode("utf-8"), immutable=True)
    files = {str(p.relative_to(root)): dict(sha256=sha256(p), bytes=p.stat().st_size)
             for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root / "SHA256.json", files, immutable=True)
    write_json(root / "DELIVERY_READY.json", dict(status="DELIVERY_READY", selection_metric="validation_r2_score",
               best_epoch=selected, last_epoch=final["epoch"], files=files, strict_reload=True,
               JF_best=False, test_best=False), immutable=True)
    load_translator(root)
    archive = run / "translator_validation_r2_delivery.tar.gz"
    require(not archive.exists(), "ARCHIVE_EXISTS")
    partial = Path(str(archive) + ".partial")
    with tarfile.open(partial, "w:gz") as tar:
        tar.add(root, arcname=root.name)
    partial.replace(archive)
    atomic(str(archive) + ".sha256", (sha256(archive) + "  " + archive.name + "\n").encode(), immutable=True)
    write_json(run / "DELIVERY_ARCHIVE_READY.json", dict(status="DELIVERY_ARTIFACTS_READY", archive=str(archive),
               sha256=sha256(archive), bytes=archive.stat().st_size, directory=str(root)), immutable=True)
    return root
