"""RunPod CPU용 작은 synthetic tensor 검사. 모델 forward/gradient 실험을 반복하지 않는다."""
from pathlib import Path
from unittest.mock import patch
import torch

from vos_memory_inspector.state_training.common import write_json
from vos_memory_inspector.state_training.r2 import Moments, summarize
from vos_memory_inspector.state_training.statistics import tensor_moments, prepare, read_statistics
from vos_memory_inspector.state_training.policy import Config, WarmupPlateau
from vos_memory_inspector.state_training.checkpoint import publish, load


def test_small_tensor_spatial_channels_and_pointer_coordinates():
    spatial = torch.tensor([[[[1., 3.]], [[2., 6.]]], [[[5., 7.]], [[8., 12.]]]])
    pointer = torch.tensor([[1., 2.], [5., 8.]])
    for tensor, branch, axes in ((spatial, "spatial", (0, 2, 3)), (pointer, "pointer", (0,))):
        count, mean, m2 = tensor_moments(tensor, branch)
        x = tensor.double()
        assert torch.allclose(torch.tensor(mean, dtype=torch.float64), x.mean(axes))
        expected = ((x - (x.mean(axes)[None, :, None, None] if branch == "spatial" else x.mean(axes)[None, :])) ** 2).sum(axes)
        assert torch.allclose(torch.tensor(m2, dtype=torch.float64), expected)
        assert summarize(dict(count=count, mean=mean, m2=m2), [0.] * len(mean))["r2"] == 1


def test_tensor_batch_shard_chan_merge_matches_full():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(13, 3, 2, 2, generator=generator)
    full = Moments(3); full.merge(*tensor_moments(x, "spatial"))
    merged = Moments(3)
    for part in (x[:1], x[1:5], x[5:]):
        merged.merge(*tensor_moments(part, "spatial"))
    assert full.count == merged.count
    assert torch.allclose(torch.tensor(full.m2), torch.tensor(merged.m2))


def test_nonfinite_tensor_statistics_blocked():
    x = torch.tensor([[float("inf"), 2.], [1., 4.]])
    import pytest
    with pytest.raises(ValueError, match="NONFINITE_TARGET_STATISTICS"):
        Moments(2).merge(*tensor_moments(x, "pointer"))


def fake_entry(dataset, role, name):
    return dict(case=dict(dataset=dataset, training_role=role, case_id=name), records=2, valid_indices=[[0, 0, 0], [0, 0, 1]])


def fake_loader(entries, cfg):
    for e in entries:
        train = e["case"]["training_role"] == "train"
        base = 2 if train else 1000
        spatial = torch.arange(128, dtype=torch.float32).reshape(2, 64, 1, 1) + base
        pointer = torch.arange(512, dtype=torch.float32).reshape(2, 256) + base
        yield dict(tensors=dict(target_spatial=spatial, target_pointer=pointer),
                   refs=[[e["case"]["case_id"], *x] for x in e["valid_indices"]])


def test_statistics_train_only_and_immutable_index_binding(tmp_path):
    entries = [fake_entry(d, r, d + r) for d in ("MOSEv2", "LVOSv2") for r in ("train", "validation")]
    value = dict(entries=entries, identity=dict(split_sha256="a" * 64))
    index_path = tmp_path / "index.json"; write_json(index_path, value)
    with patch("vos_memory_inspector.state_training.statistics.read_index", return_value=value), \
         patch("vos_memory_inspector.state_training.statistics.outside_originals"), \
         patch("vos_memory_inspector.state_training.statistics.loader", side_effect=fake_loader):
        norm, targets, spec = prepare(index_path, tmp_path / "stats", Config(workers=0))
        assert norm["scope"] == "train_only" and norm["train_records"] == 4
        train_x = torch.arange(128).double() + 2
        assert abs(norm["scales"]["spatial"] ** 2 - float(train_x.square().mean())) < 1e-9
        assert targets["targets"]["MOSEv2"]["spatial"]["mean"][0] == 1032
        assert prepare(index_path, tmp_path / "stats", Config(workers=0)) == (norm, targets, spec)
        import pytest
        changed = dict(norm["binding"], index_sha256="b" * 64)
        with pytest.raises(ValueError, match="STATISTICS_BINDING_CHANGED"):
            read_statistics(tmp_path / "stats", changed)


def test_scheduler_max_absolute_and_resume_preserves_lr():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = WarmupPlateau(optimizer, 2)
    scheduler.after_update(); scheduler.after_update(); scheduler.after_validation(-1)
    for _ in range(4):
        scheduler.after_validation(-1)
    assert optimizer.param_groups[0]["lr"] == 1.5e-4
    state = scheduler.state_dict(); opt = optimizer.state_dict()
    restored_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1.]))], lr=3e-4)
    restored = WarmupPlateau(restored_optimizer, 2)
    restored_optimizer.load_state_dict(opt); restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored_optimizer.param_groups[0]["lr"] == 1.5e-4


def test_new_checkpoint_marker_requires_verifier_success(tmp_path):
    # 새 metadata commit 절차만 검사한다. 고정 모델 reload 증빙은 재사용한다.
    import pytest
    p = tmp_path / "transaction.pt"
    value = dict(schema_version="cmmt.r2_training_checkpoint.v1", epoch=1,
                 identity=dict(split="s", index="i", normalization="n"), tensor=torch.tensor([1., 2.]))
    def rejected(saved):
        raise ValueError("TEST_VERIFIER_REJECTED")
    with pytest.raises(ValueError, match="TEST_VERIFIER_REJECTED"):
        publish(p, value, rejected)
    assert not p.exists() and not Path(str(p) + ".complete.json").exists()
    publish(p, value, lambda saved: None)
    saved, _ = load(p, value["identity"])
    assert torch.equal(saved["tensor"], value["tensor"])
    for key in ("split", "index", "normalization"):
        wrong = dict(value["identity"], **{key: "changed"})
        with pytest.raises(ValueError, match="OLD_OR_DIFFERENT_CONTRACT"):
            load(p, wrong)
