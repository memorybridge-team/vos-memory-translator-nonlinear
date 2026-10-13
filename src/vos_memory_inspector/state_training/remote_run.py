"""Pod 전용 비-LLM controller/외부 watchdog. 다른 run이나 Pod를 제어하지 않는다."""
import argparse
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from .common import require, read_json, write_json, sha256, atomic, writer, reject_placeholders, canonical_gpu_uuid
from .limits import limits, timestamp, utc

MODULE = 'vos_memory_inspector.state_training.remote_run'


def smoke_limit(cfg):
    value = float(cfg.get('smoke_seconds', 600))
    require(0 < value <= 600, 'SMOKE_LIMIT_MUST_BE_WITHIN_600_SECONDS')
    return value


def communication_environment(cfg):
    """작은 통신 검사에서 통과한 두 옵션만 해당 자식 프로세스에 허용한다."""
    env = cfg.get('communication_env', {})
    require(set(env) <= {'NCCL_DEBUG', 'NCCL_P2P_DISABLE'} and
            all(value == ('INFO' if key == 'NCCL_DEBUG' else '1') for key, value in env.items()),
            'UNAPPROVED_NCCL_ENVIRONMENT')
    if env:
        path = cfg['communication_evidence']
        require(sha256(path) == cfg['communication_evidence_sha256'], 'COMMUNICATION_EVIDENCE_CHANGED')
        evidence = read_json(path)
        require(evidence['status'] == 'CUDA_COMMUNICATION_PASS' and evidence['selected_env'] == env,
                'COMMUNICATION_ENVIRONMENT_NOT_VERIFIED')
        rows = sorted(evidence['rank_results'], key=lambda row: row['rank'])
        require(len(rows) == 2 and all(row['rank'] == row['local_rank'] == row['current_device'] == rank
                and row['all_reduce_sum'] == 3.0 and canonical_gpu_uuid(row['physical_uuid']) == canonical_gpu_uuid(cfg['gpu_uuids'][rank])
                for rank, row in enumerate(rows)), 'COMMUNICATION_RANK_UUID_MAPPING')
    return env


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), start_ticks=int(fields[19]), state=fields[0])
    except (OSError, ValueError, IndexError):
        return None


def owned_processes(owner, adopter=None, exclude=(), tracked=None):
    """PID 시작 시각과 자식 관계로만 추적한다. process 인자/환경변수는 읽지 않는다."""
    all_processes = {int(p.name): value for p in Path('/proc').iterdir() if p.name.isdigit()
                     and (value := process_identity(int(p.name))) and value['state'] != 'Z'}
    known = dict(tracked or {})
    current = all_processes.get(owner['pid'])
    require(current is None or current['start_ticks'] == owner['start_ticks'], 'PID_REUSED_DO_NOT_SIGNAL')
    if current:
        known[current['pid']] = current
    adopted = all_processes.get(adopter['pid']) if adopter else None
    require(adopted is None or adopted['start_ticks'] == adopter['start_ticks'], 'ADOPTER_PID_REUSED')
    blocked = set(exclude) | {os.getpid()}
    pending = [pid for pid, old in known.items() if pid in all_processes and all_processes[pid]['start_ticks'] == old['start_ticks'] and pid not in blocked]
    if adopted:
        pending += [p['pid'] for p in all_processes.values() if p['ppid'] == adopted['pid'] and p['pid'] not in blocked]
    found = {}
    while pending:
        pid = pending.pop()
        if pid in found or pid in blocked:
            continue
        found[pid] = all_processes[pid]
        pending.extend(p['pid'] for p in all_processes.values() if p['ppid'] == pid and p['pid'] not in blocked)
    return found, all_processes


def stop_owned_group(owner, grace=20, *, adopter=None, exclude=(), tracked=None):
    require(owner['pid'] == owner['pgid'] and owner['pgid'] != os.getpgrp(), 'OWNED_SEPARATE_PROCESS_GROUP_REQUIRED')
    signals = []
    observed = dict(tracked or {})
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5)):
        found, processes = owned_processes(owner, adopter, exclude, observed)
        observed.update(found)
        if not found:
            break
        # torchrun 2.8은 rank마다 별도 session/group을 만든다. root group만 종료하면 부족하다.
        for pgid in {p['pgid'] for p in found.values()}:
            members = {p['pid'] for p in processes.values() if p['pgid'] == pgid}
            if pgid != os.getpgrp() and members <= set(found):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    pass
            else:
                for pid in members & set(found):
                    current = process_identity(pid)
                    if current and current['start_ticks'] == found[pid]['start_ticks']:
                        try:
                            os.kill(pid, sig)
                        except ProcessLookupError:
                            pass
        signals.append(sig.name)
        end = time.monotonic() + wait
        while time.monotonic() < end:
            found, _ = owned_processes(owner, adopter, exclude, observed)
            observed.update(found)
            if not found:
                break
            time.sleep(.25)
    found, _ = owned_processes(owner, adopter, exclude, observed)
    return dict(signals=signals, remaining=list(found.values()), observed_owned_processes=list(observed.values()))


def hard_guard(owner, end, adopter=None):
    # Network Volume에 접근하지 않으므로 controller의 I/O 대기에도 제한이 적용된다.
    tracked = {}
    while time.time() < end:
        found, _ = owned_processes(owner, adopter, tracked=tracked)
        tracked.update(found)
        if not found:
            return 0
        time.sleep(min(1, max(0, end - time.time())))
    stop_owned_group(owner, adopter=adopter, tracked=tracked)
    return 3


def spawn_guard(python, owner, end, adopter=None):
    args = [python, '-m', MODULE, 'guard', '--owner-pid', str(owner['pid']),
            '--owner-start-ticks', str(owner['start_ticks']), '--end-timestamp', str(end)]
    if adopter:
        args += ['--adopter-pid', str(adopter['pid']), '--adopter-start-ticks', str(adopter['start_ticks'])]
    return subprocess.Popen(args,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def validate_config(path):
    cfg = read_json(path)
    reject_placeholders(cfg)
    require(os.name == 'posix' and sys.platform == 'linux', 'RUNPOD_LINUX_ONLY')
    require(cfg['pod_hourly_rate'] == 1.16 and cfg['budget_usd'] == 30 and cfg['max_hours'] == 12, 'THIS_APPROVAL_LIMITS_REQUIRED')
    require(len(cfg['gpu_uuids']) == len(set(cfg['gpu_uuids'])) == 2, 'APPROVED_TWO_GPU_UUIDS_REQUIRED')
    smoke_limit(cfg)
    communication_environment(cfg)
    root = Path(cfg['operation_root']).resolve()
    require(root.is_relative_to(Path('/workspace')) and root.name.startswith('r2-'), 'ISOLATED_OPERATION_ROOT_REQUIRED')
    for key in ('python', 'index', 'statistics', 'config', 'source_root'):
        require(Path(cfg[key]).is_absolute() and Path(cfg[key]).exists(), 'MISSING_INPUT:' + key)
    require(Path(__file__).resolve().is_relative_to(Path(cfg['source_root']).resolve()), 'IMPORTED_SOURCE_OUTSIDE_ISOLATED_CHECKOUT')
    for key in ('index', 'config'):
        require(sha256(cfg[key]) == cfg[key + '_sha256'], 'INPUT_CHANGED:' + key)
    require(sha256(Path(cfg['statistics']) / 'STATISTICS_READY.json') == cfg['statistics_ready_sha256'], 'STATISTICS_READY_CHANGED')
    for key in ('config', 'index', 'statistics'):
        require(not root.is_relative_to(Path(cfg[key]).resolve()), 'OPERATION_OUTPUT_INSIDE_INPUT')
    from .model import provenance
    actual = provenance()
    require(actual['revision'] == cfg['source_revision'] and actual['package_source_sha256'] == cfg['package_source_sha256'], 'DEPLOYED_SOURCE_CHANGED')
    return cfg, root


def command(cfg, mode):
    root = Path(cfg['operation_root'])
    time_limit = limits(cfg)
    args = [cfg['python'], '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2',
            '--max_restarts=0', '-m', 'vos_memory_inspector.state_training', 'train',
            '--mode', mode, '--device', 'cuda', '--index', cfg['index'], '--statistics', cfg['statistics'],
            '--config', cfg['config'], '--output', str(root / ('smoke' if mode == 'smoke' else 'training')),
            '--max-wall-seconds', str(smoke_limit(cfg) if mode == 'smoke' else max(1, timestamp(time_limit['training_deadline_utc']) - time.time())),
            '--execute-approved', '--deadline-utc', time_limit['training_deadline_utc'],
            '--approval-start-utc', cfg['approval_start_utc'], '--pod-hourly-rate', str(cfg['pod_hourly_rate']),
            '--budget-usd', str(cfg['budget_usd'])]
    for uuid in cfg['gpu_uuids']:
        args += ['--gpu-uuid', uuid]
    if mode == 'train':
        args += ['--smoke-evidence', str(root / 'smoke/STATUS.json')]
    return args


def controller(config_path):
    cfg, root = validate_config(config_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(cfg['gpu_uuids']),
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env.pop('NCCL_P2P_DISABLE', None)
    env.update(communication_environment(cfg))
    # controller와 rank별 독립 group을 외부 실행기가 자식 관계로 추적한다.
    require(os.getpid() == os.getpgrp(), 'CONTROLLER_MUST_OWN_PROCESS_GROUP')
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10, check=True)
    rows = [line for line in query.stdout.splitlines() if any(u in line for u in cfg['gpu_uuids'])]
    require(not rows, 'APPROVED_GPU_ALREADY_IN_USE_DO_NOT_TERMINATE_OTHER_WORK')
    for mode in ('smoke', 'train'):
        timing = limits(cfg)
        require(time.time() + (smoke_limit(cfg) + 60 if mode == 'smoke' else 60) < timestamp(timing['training_deadline_utc']), 'INSUFFICIENT_REMAINING_APPROVAL')
        state = dict(status='SMOKE_RUNNING' if mode == 'smoke' else 'OFFICIAL_TRAIN_RUNNING',
                     stage=mode, controller=process_identity(os.getpid()), limits=timing, time_utc=utc())
        write_json(root / 'OPERATION_STATUS.json', state)
        with (root / (mode + '.log')).open('x', encoding='utf-8') as log:
            smoke_guard = spawn_guard(cfg['python'], process_identity(os.getpid()), min(time.time() + smoke_limit(cfg), timestamp(timing['training_deadline_utc'])), process_identity(os.getppid())) if mode == 'smoke' else None
            process = subprocess.Popen(command(cfg, mode), cwd=cfg['source_root'], env=env, stdout=log, stderr=subprocess.STDOUT)
            state['torchrun'] = process_identity(process.pid)
            write_json(root / 'OPERATION_STATUS.json', state)
            try:
                code = process.wait(timeout=min(smoke_limit(cfg), timestamp(timing['training_deadline_utc']) - time.time()) if mode == 'smoke' else None)
            except subprocess.TimeoutExpired:
                # watchdog이 이 group의 모든 rank를 회수한다. controller는 재시도하지 않는다.
                raise RuntimeError('SMOKE_EXTERNAL_DEADLINE_TIMEOUT')
            finally:
                if smoke_guard is not None:
                    if smoke_guard.poll() is None:
                        smoke_guard.terminate()
                    smoke_guard.wait(timeout=5)
        status_path = root / ('smoke/STATUS.json' if mode == 'smoke' else 'training/STATUS.json')
        require(code == 0 and status_path.is_file(), mode.upper() + '_EXIT:' + str(code))
        status = read_json(status_path)
        if mode == 'smoke':
            require(status['status'] == 'REAL_R2_DDP_SMOKE_PASS' and status['both_branches_updated']
                    and status['strict_reload'] and status['rank_parameter_max_difference'] == 0, 'REAL_SMOKE_GATE_FAILED')
            write_json(root / 'SMOKE_PASS.json', dict(status='REAL_R2_DDP_SMOKE_PASS', evidence=str(status_path),
                       sha256=sha256(status_path), time_utc=utc(), limits=limits(cfg)), immutable=True)
        else:
            completed = status['stop_reason'] in ('max_epochs', 'validation_r2_early_stopping', 'validation_r2_early_stopping_already_satisfied')
            ready = root / ('training/DELIVERY_ARCHIVE_READY.json' if completed else 'training/PARTIAL_ARCHIVE_READY.json')
            require(ready.is_file(), 'EXPECTED_DELIVERY_MARKER_MISSING')
            write_json(root / 'OPERATION_STATUS.json', dict(status='TRAINING_COMPLETED' if completed else 'LIMIT_STOPPED_AT_COMPLETE_EPOCH',
                       complete_epochs=status['complete_epochs'], optimizer_step=status['optimizer_step'],
                       stop_reason=status['stop_reason'], ready=str(ready), time_utc=utc(), limits=limits(cfg)))


def preserve_partial(root, reason, end):
    """장애 시 Torch/원본 cache를 읽지 않고 완료 marker와 운영 증거만 보존한다."""
    training = root / 'training'
    destination = root / 'partial-evidence'
    destination.mkdir(exist_ok=False)
    candidates = ['OPERATION_STATUS.json', 'OPERATION_CONFIG.json', 'WATCHDOG.json', 'controller.log', 'smoke.log', 'train.log',
                  'SMOKE_PASS.json', 'smoke/STATUS.json', 'training/LIVE_STATUS.json', 'training/STATUS.json',
                  'training/history.json', 'training/epochs.jsonl', 'training/RUN.json', 'training/FIRST_EPOCH_ETA.json',
                  'training/last_complete.json', 'training/best_validation_r2.json']
    if (training / 'last_complete.json').exists():
        last = read_json(training / 'last_complete.json')
        checkpoint = Path(last['path']).resolve()
        require(checkpoint.is_relative_to(training.resolve() / 'checkpoints'), 'PARTIAL_CHECKPOINT_OUTSIDE_RUN')
        marker = read_json(str(checkpoint) + '.complete.json')
        require(marker['sha256'] == last['sha256'], 'PARTIAL_MARKER_BINDING')
        # 원 checkpoint는 그대로 둔다. 한도 직전 대형 복사가 끊겨도 원본을 보존한다.
        candidates += [str(checkpoint.relative_to(root)), str(checkpoint.relative_to(root)) + '.complete.json']
    copied = []
    archive = root / 'partial_evidence.tar.gz'
    temporary = Path(str(archive) + '.partial')
    with tarfile.open(temporary, 'w:gz') as tar:
        for name in candidates:
            if time.time() + 10 >= end:
                break
            path = root / name
            if path.is_file():
                tar.add(path, arcname=name, recursive=False)
                copied.append(name)
    temporary.replace(archive)
    marker = dict(status='ERROR_OR_LIMIT_EVIDENCE_ONLY', reason=reason, training_complete=False,
                  archive=str(archive), sha256=sha256(archive), included_files=copied,
                  source_checkpoints_preserved=True, model_export_ready=False, time_utc=utc())
    write_json(root / 'PARTIAL_EVIDENCE_READY.json', marker, immutable=True)
    atomic(str(archive) + '.sha256', (marker['sha256'] + '  ' + archive.name + '\n').encode(), immutable=True)


def watchdog(config_path):
    cfg, root = validate_config(config_path)
    root.mkdir(parents=True, exist_ok=True)
    with writer(root):
        # 고아 rank도 이 run의 watchdog 자식으로 회수한다. cgroup/다른 프로세스 설정은 바꾸지 않는다.
        require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0, 'CHILD_SUBREAPER_REQUIRED')
        require(not (root / 'OPERATION_CONFIG.json').exists(), 'NO_AUTOMATIC_RETRY_OR_OVERWRITE')
        write_json(root / 'OPERATION_CONFIG.json', cfg, immutable=True)
        timing = limits(cfg)
        soft_end = timestamp(timing['training_deadline_utc'])
        require(time.time() < soft_end, 'APPROVAL_EXPIRED_BEFORE_START')
        with (root / 'controller.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen([cfg['python'], '-m', MODULE, 'controller', '--config', str(config_path)],
                                     cwd=cfg['source_root'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            owner = process_identity(child.pid)
            require(owner is not None and owner['pgid'] == child.pid, 'CONTROLLER_GROUP_NOT_CREATED')
            adopter = process_identity(os.getpid())
            guard = spawn_guard(cfg['python'], owner, soft_end, adopter)
            def interrupted(signum, frame):
                stop_owned_group(owner, adopter=adopter, exclude=(guard.pid,))
                if guard.poll() is None:
                    guard.terminate()
                raise SystemExit(124)
            watchdog_error = None
            timed_out = guard_failed = False
            try:
                signal.signal(signal.SIGTERM, interrupted)
                signal.signal(signal.SIGINT, interrupted)
                write_json(root / 'WATCHDOG.json', dict(status='ACTIVE', watchdog=process_identity(os.getpid()),
                           independent_no_volume_guard=process_identity(guard.pid),
                           owned_controller=owner, limits=timing, external_timeout=True, max_restarts=0), immutable=True)
                while child.poll() is None and time.time() < soft_end:
                    if guard.poll() is not None:
                        break
                    time.sleep(1)
                timed_out = child.poll() is None
                guard_failed = guard.poll() is not None and time.time() < soft_end and child.poll() is None
            except Exception as exc:
                watchdog_error = type(exc).__name__ + ':' + str(exc)
            finally:
                cleanup = stop_owned_group(owner, adopter=adopter, exclude=(guard.pid,))
                code = child.wait(timeout=5)
                if guard.poll() is None:
                    guard.terminate()
                guard.wait(timeout=30)
        prior = read_json(root / 'OPERATION_STATUS.json') if (root / 'OPERATION_STATUS.json').exists() else {}
        if timed_out or code != 0 or cleanup['remaining'] or watchdog_error:
            reason = 'WATCHDOG_ERROR:' + watchdog_error if watchdog_error else 'INDEPENDENT_GUARD_EXITED_EARLY' if guard_failed else 'EXTERNAL_DEADLINE_TIMEOUT' if timed_out else 'CONTROLLER_ERROR:' + str(code)
            write_json(root / 'OPERATION_STATUS.json', dict(status='LIMIT_STOPPED' if timed_out and not guard_failed else 'ERROR_STOPPED',
                       reason=reason, previous_status=prior, cleanup=cleanup, limits=limits(cfg), time_utc=utc(),
                       training_complete=False, automatic_retry=False))
            preserve_partial(root, reason, timestamp(timing['hard_deadline_utc']))
        write_json(root / 'WATCHDOG_EXIT.json', dict(controller_exit_code=code, timed_out=timed_out, guard_failed=guard_failed, cleanup=cleanup,
                   limits=limits(cfg), time_utc=utc(), pod_stopped=False), immutable=True)
        return 2 if guard_failed or watchdog_error else 3 if timed_out else (2 if code != 0 or cleanup['remaining'] else 0)


def main():
    p = argparse.ArgumentParser(description='승인한 단일 RunPod R² run의 독립 controller/외부 watchdog; Pod Stop 없음')
    p.add_argument('role', choices=('watchdog', 'controller', 'guard'))
    p.add_argument('--config', type=Path)
    p.add_argument('--owner-pid', type=int)
    p.add_argument('--owner-start-ticks', type=int)
    p.add_argument('--end-timestamp', type=float)
    p.add_argument('--adopter-pid', type=int)
    p.add_argument('--adopter-start-ticks', type=int)
    args = p.parse_args()
    if args.role == 'guard':
        require(args.owner_pid and args.owner_start_ticks and args.end_timestamp, 'GUARD_ARGUMENTS_REQUIRED')
        adopter = dict(pid=args.adopter_pid, start_ticks=args.adopter_start_ticks) if args.adopter_pid else None
        return hard_guard(dict(pid=args.owner_pid, pgid=args.owner_pid, start_ticks=args.owner_start_ticks), args.end_timestamp, adopter)
    require(args.config is not None, 'OPERATION_CONFIG_REQUIRED')
    return watchdog(args.config) if args.role == 'watchdog' else controller(args.config)


if __name__ == '__main__':
    raise SystemExit(main())
