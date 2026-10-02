"""명시적인 CPU synthetic vertical slice. SAM weights·실제 cache·GPU를 사용하지 않는다."""
import argparse
from pathlib import Path
import sys
import json
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_lvos_pipeline import fixture_data
from vos_memory_inspector.lvos_snapshot import build_snapshot
from vos_memory_inspector.lvos_contract import model_lock, load_complete
from vos_memory_inspector.lvos_training import LVOSConfig, train_snapshot, load_model_checkpoint
from vos_memory_inspector.training_storage import write_json
from vos_memory_inspector.lvos_cli import draft_export


def main():
    p=argparse.ArgumentParser(description='CPU synthetic one-video overfit + video-disjoint dev smoke')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(); output=args.output.resolve()
    if output.exists():
        p.error('새 output directory를 지정하세요. 기존 smoke를 덮어쓰지 않습니다.')
    output.mkdir(parents=True); started=time.perf_counter(); torch.set_num_threads(2)
    selection,requests,ids,_=fixture_data(output)
    snapshot=build_snapshot(requests,selection,output/'snapshot',trusted=True,stable_seconds=0,pilot_ids=ids,max_shard_mib=2)
    write_json(output/'requests.json',requests)
    lock=model_lock('CPU synthetic smoke only; production approval 아님'); write_json(output/'model-lock.json',lock)
    cfg=LVOSConfig(workers=0,pin_memory=False,microbatch=1,accumulation=3,max_epochs=3)
    overfit=train_snapshot(output/'snapshot',lock,output/'overfit',cfg,device='cpu',mode='overfit')
    dev=train_snapshot(output/'snapshot',lock,output/'dev',cfg,device='cpu')
    model,payload,entry=load_model_checkpoint(output/'dev/checkpoints/epoch-00003.ckpt')
    draft_export([output/'overfit',output/'dev'],[],output/'draft.md')
    summary={'execution_kind':'cpu_synthetic','status':'PASS','actual_sam_weights_used':False,'gpu_verification':False,
        'duration_seconds':time.perf_counter()-started,'models_parameters':model.parameter_count(),
        'overfit_fit_loss':[r['fit']['loss'] for r in overfit],'development_state_loss':[r['dev']['loss'] for r in dev],
        'checkpoint_sha256':entry['sha256'],'snapshot_statistics':snapshot['statistics'],
        'output_bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        'limitations':'synthetic state connectivity only; actual VOS score and GPU speed are BLOCKED'}
    write_json(output/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
