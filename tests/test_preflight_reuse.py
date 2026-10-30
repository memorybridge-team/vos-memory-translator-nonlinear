"""Full verification belongs before torchrun, not inside a rank-0 collective."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import socket
import signal
import sys
import time

import pytest
import torch

from test_state_training import sample_index, cpu_config
from vos_memory_inspector.state_training import preflight
from vos_memory_inspector.state_training.data import load_index, load_pair
from vos_memory_inspector.state_training.storage import digest, save_json, sha256
from vos_memory_inspector.state_training.trainer import train


def test_receipt_checks_only_metadata_and_preserves_load_time_hash(sample_index, cpu_config, monkeypatch):
    index = load_index(sample_index, allow_sample=True)
    report = preflight.inspect(index, cpu_config)
    with monkeypatch.context() as scope:
        scope.setattr(preflight, 'load_pair', lambda *a, **k: pytest.fail('startup must not load tensors'))
        token = preflight.validate_readiness(index, report, allow_sample=True)
    assert token.report is report
    entry = index.entries[0]
    path = Path(entry['path'])
    stat = path.stat()
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # A stat check is intentionally not a content check; the real loader still hashes.
    preflight.validate_readiness(index, report, allow_sample=True)
    with pytest.raises(ValueError, match='checksum'):
        load_pair(entry)


def test_receipt_rejects_stale_partial_wrong_index_and_modified_rms(sample_index, cpu_config):
    index = load_index(sample_index, allow_sample=True)
    report = preflight.inspect(index, cpu_config)
    old = dict(report, schema_version='cmmt.state_training_preflight.v1')
    with pytest.raises(ValueError, match='v2 report'):
        preflight.validate_readiness(index, old, allow_sample=True)
    altered = json.loads(json.dumps(report))
    altered['normalization']['scales']['pointer'] *= 2
    with pytest.raises(ValueError, match='digest mismatch'):
        preflight.validate_readiness(index, altered, allow_sample=True)
    altered['receipt_digest'] = digest({key: altered[key] for key in preflight.RECEIPT_FIELDS})
    with pytest.raises(ValueError, match='sums/scales'):
        preflight.validate_readiness(index, altered, allow_sample=True)
    partial = preflight.inspect(index, cpu_config, sample_cases=1)
    with pytest.raises(ValueError, match='not ready'):
        preflight.validate_readiness(index, partial, allow_sample=True)
    other = replace(index, identity={**index.identity, 'index_sha256': '0' * 64})
    with pytest.raises(ValueError, match='identity mismatch'):
        preflight.validate_readiness(other, report, allow_sample=True)
    path = Path(index.entries[0]['path'])
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match='changed after preflight'):
        preflight.validate_readiness(index, report, allow_sample=True)
    # A fresh full scan can renew a report after a timestamp-only change.
    assert preflight.inspect(index, cpu_config)['sample_data_ready']


def test_train_requires_receipt_and_never_repeats_inspect(sample_index, cpu_config, tmp_path, monkeypatch):
    index = load_index(sample_index, allow_sample=True)
    report = preflight.inspect(index, cpu_config)
    monkeypatch.setattr(preflight, 'inspect', lambda *a, **k: pytest.fail('unexpected whole-data scan'))
    with pytest.raises(ValueError, match='preflight report required'):
        train(index, cpu_config, 'residual_mlp', tmp_path / 'missing', allow_sample=True)
    train(index, cpu_config, 'residual_mlp', tmp_path / 'trained', readiness=report, allow_sample=True,
          options=dict(hidden_dim=8, pointer_hidden_dim=10))
    train(index, replace(cpu_config, max_epochs=3), 'residual_mlp', tmp_path / 'trained',
          readiness=report, allow_sample=True, options=dict(hidden_dim=8, pointer_hidden_dim=10),
          resume=tmp_path / 'trained/last.ckpt')


def _official_cli_fixture(tmp_path, make_state):
    """Small official-shaped fixture, only for CPU lifecycle/CLI regression."""
    entries = []
    for i, role in enumerate(('train', 'validation')):
        state = make_state(torch.randn(1, 1, 2, 64, 64, 64).bfloat16(), torch.randn(1, 1, 2, 256))
        target = state.with_continuous(spatial_memory=(state.spatial_memory.float() * .7).bfloat16(),
                                      object_pointer=state.object_pointer * .7,
                                      presence_logits=state.presence_logits)
        path = tmp_path / 'cache' / f'case{i}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(schema_version='cmmt.prepared_handoff_case.v2', source_canonical=state,
                        target_canonical=target, metadata=dict(cache_mode='state_only',
                        source_model_id='sam2.1-small', target_model_id='sam2.1-base-plus',
                        video_id=f'video{i}', object_id=0, num_frames=3)), path)
        checksum = sha256(path)
        Path(str(path) + '.sha256').write_text(checksum)
        entries.append(dict(path=str(path), sha256=checksum, valid_indices=[[0, 0, 0], [0, 0, 1]],
            runtime_switch_frame=1, records=2, case=dict(case_id=f'case{i}', video_id=f'video{i}',
            object_id=0, dataset='LVOS v2', training_role=role, official_split='train' if i == 0 else 'valid')))
    path = tmp_path / 'official-shaped-fixture.json'
    save_json(path, dict(schema_version='cmmt.official_state_training_index.v1', state='tensor_validated',
              entries=entries, identity=dict(expected_case_ids=['case0', 'case1'])))
    config = tmp_path / 'config.json'
    save_json(config, dict(model='residual_mlp', model_options=dict(hidden_dim=8, pointer_hidden_dim=8),
              training=dict(workers=0, global_batch=2, microbatch=1, max_epochs=1)))
    return path, config


def test_actual_two_rank_cli_has_no_long_preflight_collective(tmp_path, make_state):
    index_path, config_path = _official_cli_fixture(tmp_path, make_state)
    report_path = tmp_path / 'readiness.json'
    # A deliberately slow scan takes longer than the following DDP timeout.
    # It runs in the standalone command, before a process group exists.
    slow = tmp_path / 'slow_preflight.py'
    slow.write_text('''import sys, time
from vos_memory_inspector.state_training import cli
import torch.distributed as dist
original = cli.inspect
def inspect(*args, **kwargs):
    assert not dist.is_initialized()
    time.sleep(6)
    return original(*args, **kwargs)
cli.inspect = inspect
raise SystemExit(cli.main(sys.argv[1:]))
''')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if sys.platform == 'darwin':
        env.setdefault('GLOO_SOCKET_IFNAME', 'lo0')
    started = time.monotonic()
    subprocess.run([sys.executable, str(slow), 'preflight', '--index', str(index_path),
        '--config', str(config_path), '--output', str(report_path), '--world-size', '2'],
        check=True, capture_output=True, text=True, env=env, timeout=45)
    assert time.monotonic() - started >= 6
    # All startup tensor scans are forbidden in the actual CLI invocation.
    entrypoint = tmp_path / 'train_cli.py'
    entrypoint.write_text('''import sys, time
from vos_memory_inspector.state_training import cli, trainer, preflight
def forbidden(*args, **kwargs):
    time.sleep(6)
    raise AssertionError("train must reuse preflight")
cli.inspect = forbidden
preflight.inspect = forbidden
trainer.inspect = forbidden
raise SystemExit(cli.main(sys.argv[1:]))
''')
    output = tmp_path / 'cli-run'
    def command():
        # Static loopback rendezvous avoids macOS hostname DNS outside Gloo.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        return [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
                '--master-addr=127.0.0.1', f'--master-port={port}', str(entrypoint), 'train',
                '--index', str(index_path), '--config', str(config_path), '--output', str(output),
                '--device', 'cpu', '--preflight-report', str(report_path),
                '--distributed-timeout-seconds', '5']

    first = _run_process_group(command(), env)
    assert 'epoch=1' in first.stdout
    config = json.loads(config_path.read_text())
    config['training']['max_epochs'] = 2
    save_json(config_path, config)
    resumed = _run_process_group(command() + ['--resume', str(output / 'last.ckpt')], env)
    assert 'epoch=2' in resumed.stdout
    result = json.loads((output / 'training_report.json').read_text())
    assert result['history'][-1]['step'] == 2
    assert result['identity']['world_size'] == 2
    assert not result['cuda_training_executed']


def _run_process_group(command, env):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f'torchrun timed out\n{stdout}\n{stderr}')
    assert process.returncode == 0, f'{stdout}\n{stderr}'
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
