"""신규 controller 한도/프로세스 소유권 검사. RunPod CPU에서만 실행한다."""
import os
import subprocess
import sys
import time
import pytest
from unittest.mock import patch
from pathlib import Path

from vos_memory_inspector.state_training.limits import limits, timestamp, epoch_forecast
from vos_memory_inspector.state_training.remote_run import process_identity, stop_owned_group, spawn_guard
from vos_memory_inspector.state_training.common import write_json, save_bound, stamp
from vos_memory_inspector.state_training.index import inspection_plan, build_index


def test_limit_clock_does_not_reset_after_preparation_or_resume():
    cfg = dict(approval_start_utc='2026-10-09T04:37:18Z', deadline_utc='2026-10-09T16:37:18Z',
               pod_hourly_rate=1.16, budget_usd=30, max_hours=12, artifact_reserve_seconds=600)
    start = timestamp(cfg['approval_start_utc'])
    a, b = limits(cfg, start + 3600), limits(cfg, start + 7200)
    assert a['hard_deadline_utc'] == b['hard_deadline_utc'] == cfg['deadline_utc']
    assert a['accrued_pod_usd'] == 1.16 and b['accrued_pod_usd'] == 2.32
    assert b['remaining_seconds'] == 10 * 3600
    budget_first = limits(dict(cfg, budget_usd=1.16), start)
    assert timestamp(budget_first['hard_deadline_utc']) == start + 3600


def test_first_epoch_forecast_includes_validation_and_checkpoint_and_reserves_slowdown():
    history = [dict(epoch=1, wall_seconds_including_checkpoint=100), dict(epoch=2, wall_seconds_including_checkpoint=160)]
    value = epoch_forecast(history, 60, 1000, 1250, 1.16)
    assert value['next_epoch_reserved_seconds'] == 260
    assert not value['can_start_next_epoch']
    assert value['eta_average_seconds'] == 58 * 130
    assert value['forecast_is_not_a_completion_guarantee']


@pytest.mark.skipif(sys.platform != 'linux', reason='RunPod /proc process-group safety only')
def test_owned_process_group_shutdown_preserves_unrelated_process():
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    owned = subprocess.Popen([sys.executable, '-c',
        'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True); print(p.pid,flush=True); time.sleep(60)'],
        start_new_session=True, stdout=subprocess.PIPE, text=True)
    try:
        grandchild = int(owned.stdout.readline())
        assert process_identity(grandchild)['pgid'] != owned.pid
        owner = process_identity(owned.pid)
        assert owner['pgid'] == owner['pid']
        result = stop_owned_group(owner, grace=1)
        owned.wait(timeout=5)
        assert not result['remaining'] and sentinel.poll() is None
        remaining_child = process_identity(grandchild)
        assert remaining_child is None or remaining_child['state'] == 'Z'
        with pytest.raises(ValueError, match='OWNED_SEPARATE_PROCESS_GROUP_REQUIRED'):
            stop_owned_group(dict(pid=os.getpid(), pgid=os.getpgrp(), start_ticks=0))
    finally:
        for child in (owned, sentinel):
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


@pytest.mark.skipif(sys.platform != 'linux', reason='RunPod independent deadline guard only')
def test_independent_guard_stops_blocked_owned_group():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    guard = spawn_guard(sys.executable, process_identity(child.pid), time.time() + 1)
    try:
        assert guard.wait(timeout=10) == 3
        assert child.wait(timeout=5) != 0
    finally:
        for item in (guard, child):
            if item.poll() is None:
                item.terminate()
            item.wait(timeout=5)


def test_inspection_plan_reads_metadata_without_loading_tensor(tmp_path):
    files = []
    for name in ('same', 'changed', 'sealed'):
        p = tmp_path / (name + '.pt')
        p.write_bytes(b'not a tensor; metadata-only fixture')
        Path(str(p) + '.sha256').write_text('oldsha\n')
        files.append(dict(case=dict(case_id=name, training_role='internal-test' if name == 'sealed' else 'train'), path=str(p)))
    priors = {e['path']: dict(stamp=stamp(e['path']), sha256='oldsha') for e in files}
    priors[files[1]['path']]['stamp'] = [0, 0, 0]
    cfg = tmp_path / 'config.json'; write_json(cfg, dict(exclusions=[]))
    with patch('vos_memory_inspector.state_training.index.read_split', return_value={}), \
         patch('vos_memory_inspector.state_training.index.plan', return_value=(files, [], {})), \
         patch('vos_memory_inspector.state_training.index.prior_entries', return_value=(priors, None)):
        result = inspection_plan(cfg, tmp_path)
    assert result['potential_prior_reuse'] == 1 and result['new_tensor_candidate_count'] == 1
    assert result['sealed_cases'] == 1 and result['new_tensor_candidates'][0]['case_id'] == 'changed'


def test_tensor_inspection_cap_blocks_before_read_and_does_not_reset_on_resume(tmp_path):
    cache = tmp_path / 'cache'; cache.mkdir()
    items = []
    for name in ('first', 'second'):
        p = cache / (name + '.pt'); p.write_bytes(b'not a tensor')
        Path(str(p) + '.sha256').write_text('fixture\n')
        items.append(dict(case=dict(case_id=name, training_role='train'), path=str(p)))
    cfg = tmp_path / 'config.json'; write_json(cfg, dict(datasets=[dict(cache_roots=dict(fit=str(cache)))], exclusions=[]))
    split = tmp_path / 'split'; write_json(split / 'split_manifest.json', {})
    output = tmp_path / 'new-index'
    with patch('vos_memory_inspector.state_training.index.read_split', return_value={}), \
         patch('vos_memory_inspector.state_training.index.plan', return_value=(items, [], {})), \
         patch('vos_memory_inspector.state_training.index.prior_entries', return_value=({}, None)), \
         patch('vos_memory_inspector.state_training.cache.inspect') as actual_inspect:
        with pytest.raises(PermissionError, match='NEW_TENSOR_INSPECTION_SCOPE_NOT_APPROVED'):
            build_index(cfg, split, output, operator_completed='synthetic fixture', max_new_inspections=0)
        actual_inspect.assert_not_called()
        save_bound(output / 'cases/000000.json', dict(**items[0], stamp=stamp(items[0]['path']), sha256='fixture', records=1))
        with pytest.raises(PermissionError, match='NEW_TENSOR_INSPECTION_SCOPE_NOT_APPROVED'):
            build_index(cfg, split, output, operator_completed='synthetic fixture', max_new_inspections=1)
        actual_inspect.assert_not_called()


def test_delivery_normal_and_partial_markers_remain_distinct(tmp_path):
    """순수 파일 fixture; 모델/gradient/reload 재검증 없이 종료 산출물 표지만 검사한다."""
    from vos_memory_inspector.state_training import delivery
    from vos_memory_inspector.state_training.common import read_json, sha256
    for partial in (False, True):
        base = tmp_path / str(partial)
        run, stats, split = base / 'run', base / 'stats', base / 'split'
        checkpoint = run / 'checkpoints/epoch-00001.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'fixture checkpoint')
        marker = dict(sha256=sha256(checkpoint), path=str(checkpoint))
        write_json(run / 'last_complete.json', marker)
        write_json(run / 'best_validation_r2.json', dict(scope='full_validation', epoch=1, score=0.5, checkpoint=marker))
        write_json(run / 'history.json', [dict(epoch=1, validation=dict(validation_r2_score=0.5))])
        write_json(run / 'STATUS.json', dict(stop_reason='insufficient_remaining_epoch_budget' if partial else 'max_epochs'))
        for name in ('RUN.json', 'epochs.jsonl'):
            write_json(run / name, {})
        for name in ('normalization.json', 'validation_targets.json'):
            save_bound(stats / name, {})
        save_bound(stats / 'r2_spec.json', dict(tie_tolerance=1e-12))
        index = base / 'index/index.json'
        save_bound(index, dict(identity=dict(split_root=str(split))))
        write_json(index.parent / 'used_excluded.json', [])
        for name in ('original_inventory.json', 'original_inventory.json.sha256', 'split_manifest.json', 'split_manifest.json.sha256', 'split_manifest.sha256', 'SPLIT_FROZEN.json', 'split_report.json'):
            write_json(split / name, {})
        payload = dict(epoch=1, optimizer_step=1, export={}, identity={}, model_lock={}, configuration={}, source=dict(revision='fixture'))
        def publish_fixture(path, weights, verify):
            path.write_bytes(b'fixture weights')
            return {}
        with patch.object(delivery, 'load', return_value=(payload, marker)), \
             patch.object(delivery, 'publish', side_effect=publish_fixture), \
             patch.object(delivery, 'load_translator'):
            delivery.export_delivery(run, index, stats, partial=partial)
        expected = run / ('PARTIAL_ARCHIVE_READY.json' if partial else 'DELIVERY_ARCHIVE_READY.json')
        other = run / ('DELIVERY_ARCHIVE_READY.json' if partial else 'PARTIAL_ARCHIVE_READY.json')
        assert expected.is_file() and not other.exists()
        ready = read_json(expected)
        assert ready['training_complete'] is (not partial)
        assert ready['status'] == ('PARTIAL_ARTIFACTS_READY' if partial else 'DELIVERY_ARTIFACTS_READY')
