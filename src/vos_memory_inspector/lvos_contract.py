"""LVOS evidence·고정 모델·완료 artifact 공통 계약."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import os
import time
import torch

from .training_storage import sha256, content_hash, write_json, write_tensor_file, read_checked
from .collection_contract import Rejection, require
from .transformer_translator import SAM21_MEMORY_SPEC, TransformerStateTranslator, SpatialTransformerConfig

MODEL_REVISION = '746ea3e7d84c366c2d7ac06159e90a1f684bca56'
BENCHMARK_REVISION = 'bcf0a0f6a36c4129d487e5e58e151468d7ca714b'
PACKAGE = Path(__file__).parent


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_lock(approved_by=None):
    from .transformer_translator import PRESETS
    config = SpatialTransformerConfig(**dict(PRESETS['base'].config)).to_dict()
    value = {'schema_version': 'cmmt.lvos_model_lock.v1', 'factory':
             'vos_memory_inspector.transformer_translator:TransformerStateTranslator',
             'preset': 'base', 'config': config, 'model_revision': MODEL_REVISION,
             'approved_by': approved_by, 'source_hashes': {name: sha256(PACKAGE/name) for name in
                         ('transformer_translator.py', 'frozen_tensor_api.py')}}
    return {**value, 'digest': content_hash(value)}


def validate_model_lock(lock, *, require_approval=True):
    require(lock['digest'] == content_hash({k:v for k,v in lock.items() if k!='digest'}), 'MODEL_LOCK_HASH')
    require(lock['model_revision']==MODEL_REVISION, 'MODEL_REVISION')
    require(lock['factory']=='vos_memory_inspector.transformer_translator:TransformerStateTranslator', 'MODEL_FACTORY')
    require(lock['preset']=='base' and lock['config']==model_lock()['config'], 'FROZEN_ARCHITECTURE')
    require(all(sha256(PACKAGE/n)==h for n,h in lock['source_hashes'].items()), 'MODEL_SOURCE_CHANGED')
    if require_approval:
        require(bool(lock.get('approved_by')), 'MODEL_FREEZE_INPUT_MISSING')
    return TransformerStateTranslator(SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC,
        config=SpatialTransformerConfig.from_dict(lock['config']), preset=lock['preset'])


def save_complete(path, payload):
    path = Path(path)
    require(not path.exists(), 'ARTIFACT_EXISTS', str(path))
    entry = write_tensor_file(path, payload)
    write_json(str(path)+'.complete.json', {**entry, 'schema_version': 'cmmt.ready_artifact.v1'})
    return entry


def load_complete(path):
    path = Path(path)
    marker = Path(str(path)+'.complete.json')
    require(marker.is_file(), 'INCOMPLETE_ARTIFACT', str(path))
    entry = json_read(marker)
    require(entry['path']==path.name, 'ARTIFACT_PATH')
    return read_checked(path.parent, entry), entry


class Events:
    def __init__(self, root, identity, kind):
        self.path = Path(root)/'logs/events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity, self.kind = identity, kind
    def emit(self, event, **kw):
        value = {'schema_version': 'cmmt.lvos_event.v1', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                 'run_id': self.identity['run_id'], 'task_id': 'lvos_vertical_slice', 'event': event,
                 'execution_kind': self.kind, 'code_sha': self.identity['code_sha'],
                 'model_config_digest': self.identity['model_config_digest'],
                 'collection_digest': self.identity['collection_digest'], 'epoch': None, 'micro_iteration': None,
                 'optimizer_step': None, 'checkpoint_sha': None, 'eval_id': None, 'metrics': {},
                 'evidence_paths': [], 'duration_seconds': None, **kw}
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n'
        with self.path.open('a', encoding='utf-8') as f:
            f.write(encoded); f.flush(); os.fsync(f.fileno())
        return value


def evidence(paths):
    require(bool(paths), 'PASS_WITHOUT_EVIDENCE')
    return [{'path': str(Path(p).resolve()), 'sha256': sha256(p)} for p in paths]


def verify_evidence(items):
    require(bool(items) and all(Path(e['path']).is_file() and sha256(e['path'])==e['sha256'] for e in items), 'EVIDENCE_CHANGED')


def resources():
    import shutil, platform
    available = {'python': platform.python_version(), 'torch': str(torch.__version__),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cuda': torch.version.cuda,
        'disk_free_bytes': shutil.disk_usage(Path.cwd()).free, 'cpu_ram_bytes': None, 'shm_bytes': None}
    available['cpu_process_rss_bytes']=None
    try:
        available['cpu_ram_bytes'] = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, ValueError):
        pass
    if Path('/dev/shm').exists():
        available['shm_bytes'] = shutil.disk_usage('/dev/shm').free
    if Path('/proc/self/status').exists():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                available['cpu_process_rss_bytes']=int(line.split()[1])*1024
    available['gpu'] = [{'name': torch.cuda.get_device_name(i), 'total_memory': torch.cuda.get_device_properties(i).total_memory}
                        for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return available


def benchmark_module(root, contract):
    import subprocess
    root = Path(root)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip()
    require(revision==contract['benchmark_revision']==BENCHMARK_REVISION, 'BENCHMARK_REVISION')
    require(sha256(root/'best_model.py')==contract['evaluator_sha256'], 'EVALUATOR_CHANGED')
    require(contract.get('approved_by') and contract.get('primary_metric')=='mean_video_retention_three_fractions_percent',
            'METRIC_FREEZE_INPUT_MISSING')
    require(contract['fractions']==[0.25,0.5,0.75] and contract['visible_rule']=='gt_visible_only' and
            contract['undefined_rule']=='null_with_reason' and contract['zero_replay_rule']=='exclude_video_ratio' and
            contract['monitor_metric']=='macro_video_jf_0_1', 'METRIC_CONTRACT')
    spec = importlib.util.spec_from_file_location('cmmt_frozen_best_model',root/'best_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
