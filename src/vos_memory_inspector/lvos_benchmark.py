"""Pinned baseline Full Replay -> frozen CMMT dev protocol bridge.

Attribution: vos-memory-benchmark, RohSeongmin, baseline/no_handoff.py at dcd3353.
그 파일의 full_replay만 호출한다. baseline split/switch/all-method CLI는 사용하지 않는다.
"""
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
import importlib.util
import subprocess
import sys
import time
import torch
from .collection_contract import require, jpeg_map, READ_FIELDS
from .training_storage import sha256, source_sha256, content_hash, write_json, ExclusiveWriter
from .training_data import read_manifest
from .training_runner import code_provenance
from .lvos_contract import json_read, benchmark_module, BASELINE_REVISION, BENCHMARK_REVISION
from .lvos_budget import Deadline, BudgetStop, check
from .lvos_snapshot import verify_snapshot
from .lvos_evaluation import case_assignments, score_predictions, read_result, validate_rows, target_config_path, validate_protocol_snapshot
from .lvos_metrics import canonical_retention, raw_macro


def baseline_module(root):
    root=Path(root)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==BASELINE_REVISION,'BASELINE_REVISION')
    require(subprocess.run(['git','diff','--exit-code','HEAD'],cwd=root,capture_output=True).returncode==0,'DIRTY_BASELINE_REFERENCE')
    path=root/'baseline/no_handoff.py'
    pin=json_read(Path(__file__).resolve().parents[2]/'configs/lvos_benchmark_reference_lock.json')
    require(source_sha256(path)==pin['sources']['full_replay'],'BASELINE_SOURCE_PIN')
    spec=importlib.util.spec_from_file_location('cmmt_pinned_no_handoff',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module,{'revision':BASELINE_REVISION,'source_sha256':source_sha256(path),
                   'source_hash_policy':'utf8_crlf_to_lf_v1','author':'RohSeongmin','entrypoint':'baseline.no_handoff.full_replay'}


def build_target(snapshot,protocol,device,deadline):
    runtime=protocol['runtime']; sam_root=Path(runtime['sam2_repo']).resolve()
    from .upstream import verify_sam2_checkout
    check(deadline,'target_source_verification')
    require(verify_sam2_checkout(sam_root)==snapshot['models']['target']['upstream_commit'],'TARGET_UPSTREAM')
    require(subprocess.run(['git','diff','--exit-code','HEAD','--','sam2'],cwd=sam_root,capture_output=True).returncode==0,'DIRTY_TARGET_SAM_SOURCE')
    for key,path in (('checkpoint_sha256',runtime['target_checkpoint']),('config_sha256',target_config_path(sam_root,runtime['target_config']))):
        check(deadline,'target_'+key); require(sha256(path)==snapshot['models']['target'][key],'TARGET_HASH')
    check(deadline,'target_model_load')
    sys.path.insert(0,str(sam_root))
    from sam2.build_sam import build_sam2_video_predictor
    from .sam2_lazy_loader import install_sam2_lazy_loader
    from .training_collection import freeze_model
    predictor=freeze_model(build_sam2_video_predictor(runtime['target_config'],runtime['target_checkpoint'],device=device))
    install_sam2_lazy_loader(cache_size=8)
    check(deadline,'target_model_load_complete')
    return predictor


@contextmanager
def encoder_trace(predictor):
    calls=[]; frame=[None]; original_get=predictor._get_image_feature; original_forward=predictor.forward_image
    def get(state,frame_idx,batch_size):
        frame[0]=int(frame_idx)
        try: return original_get(state,frame_idx,batch_size)
        finally: frame[0]=None
    def forward(image):
        require(frame[0] is not None,'ENCODER_FRAME_UNKNOWN'); calls.append(frame[0]); return original_forward(image)
    predictor._get_image_feature=get; predictor.forward_image=forward
    try: yield calls
    finally: predictor._get_image_feature=original_get; predictor.forward_image=original_forward


class ReplaySession:
    def __init__(self,predictor,row,deadline,policy):
        from .sam2_state import init_sam2_inference_state_without_warmup
        self.predictor,self.row,self.deadline,self.policy=predictor,row,deadline,policy
        check(deadline,'replay_video_load')
        self.state=init_sam2_inference_state_without_warmup(predictor,video_path=row['video_dir'],
                         offload_video_to_cpu=True,offload_state_to_cpu=True)
        check(deadline,'replay_video_load_complete')
    def add_prompt(self,frame,mask):
        require(frame==self.row['prompt']['runtime_index'],'REPLAY_PROMPT_FRAME')
        self.predictor.add_new_mask(self.state,frame_idx=frame,obj_id=self.row['object_id'],mask=torch.as_tensor(mask,dtype=torch.bool))
        check(self.deadline,'replay_prompt_complete')
    def track(self,first,last):
        for frame,ids,masks in self.predictor.propagate_in_video(self.state,start_frame_idx=first,max_frame_num_to_track=last-first,reverse=False):
            check(self.deadline,'full_replay_frame')
            require(list(ids)==[self.row['object_id']] and masks.shape[0]==1,'REPLAY_REGISTRY')
            yield int(frame),masks[0,0].detach().cpu().numpy()>0
            if self.policy['name']=='active_window_v1':
                limit=max(self.policy['num_maskmem']-1,self.policy['max_obj_ptrs_in_encoder']-1)
                for store in self.state.get('output_dict_per_obj',{}).values():
                    recent=store['non_cond_frame_outputs']
                    for key in sorted(recent)[:max(0,len(recent)-limit)]: del recent[key]


def replay_group(predictor,rows,baseline,policy,deadline=None):
    """같은 video/object/prompt의 세 fraction을 한번 rollout. Future GT는 scorer만 읽는다."""
    from .runner import load_binary_prompt
    deadline=deadline or Deadline()
    first=rows[0]; prompt=first['prompt']
    require(all(all(row[k]==first[k] for k in ('video_id','video_dir','object_id','prompt','frame_map_sha256')) for row in rows),
            'REPLAY_GROUP_IDENTITY')
    require(sha256(prompt['path'])==prompt['sha256'],'REPLAY_PROMPT_CHANGED')
    mask=load_binary_prompt(prompt['path'],first['object_id'])
    end=max(row['expected_runtime_indices'][-1] for row in rows)
    scored={s['runtime_index'] for row in rows for s in row['scored_frames']}
    observed=[]; predictions={}
    with encoder_trace(predictor) as calls:
        session=ReplaySession(predictor,first,deadline,policy)
        require(not calls,'REPLAY_INIT_UNEXPECTED_ENCODER')
        start=baseline.full_replay(session,prompt['runtime_index'],mask)
        require(start==prompt['runtime_index'],'REPLAY_START')
        for frame,pred in session.track(start,end):
            require(frame not in observed,'REPLAY_DUPLICATE_FRAME'); observed.append(frame)
            if frame in scored: predictions[frame]=pred
    require(observed==list(range(start,end+1)),'REPLAY_ROLLOUT_COVERAGE')
    require(all(start<=frame<=end for frame in calls),'REPLAY_ENCODER_COVERAGE')
    return predictions,{'encoder_frame_ids':calls,'rollout_frames':observed,'initial_prompt_only':True,
                        'prompt_gt_inputs':[prompt['path']],'replayed_once_per_video_object':True}


def adapt_baseline_row(raw,fixed,proof):
    """명시된 생성 proof 없는 과거 row를 현재 계약으로 승격하지 않는다."""
    required={'protocol_digest','target_identity_digest','target_policy_digest','code_sha','baseline_revision','scored_frame_digest','prompt_sha'}
    require(required<=proof.keys(),'BASELINE_ROW_PROVENANCE_MISSING')
    require(proof['baseline_revision']==BASELINE_REVISION and proof['scored_frame_digest']==content_hash(fixed['scored_frames']) and
            proof['prompt_sha']==fixed['prompt']['sha256'],'BASELINE_ROW_PROVENANCE')
    require(raw['video']==fixed['video_id'] and raw['object']==fixed['object_id'] and raw['baseline']=='full_replay' and
            raw['switch_frame']==fixed['switch_runtime_index'] and raw['switch_name']==str(round(fixed['fraction']*100)), 'BASELINE_ROW_IDENTITY')
    return {'case_id':fixed['case_id'],'video':fixed['video_id'],'object_id':fixed['object_id'],'fraction':fixed['fraction'],
            'j':raw['j'],'f':raw['f'],'jf':raw['jf'],'provenance':proof}


def full_replay(root,protocol_path,metric_contract,benchmark_root,baseline_root,output,*,shard_index=0,shard_count=1,
                device='cuda:0',approved=False,max_wall_seconds=None,deadline=None):
    deadline=deadline or Deadline(max_wall_seconds)
    require(approved and max_wall_seconds and torch.device(device).type=='cuda' and torch.cuda.is_available() and
            torch.cuda.device_count()==1 and __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),'GPU_EXECUTION_APPROVAL')
    output=Path(output)
    with ExclusiveWriter(output):
        try:
            snapshot=verify_snapshot(root,deadline=deadline); protocol=read_manifest(protocol_path); contract=json_read(metric_contract)
            require(not snapshot['synthetic'] and protocol['schema_version']=='cmmt.lvos_protocol.v2' and
                    protocol['scope']=='full_development' and protocol['collection_digest']==snapshot['content_sha256'] and
                    protocol['role']=='development' and protocol['source_split']=='train','REPLAY_FULL_FROZEN_DEV_REQUIRED')
            validate_protocol_snapshot(snapshot,protocol)
            metric=benchmark_module(benchmark_root,contract); baseline,reference=baseline_module(baseline_root)
            assignment=case_assignments(protocol,shard_count)
            require(0<=shard_index<shard_count and assignment[shard_index],'EVAL_SHARD_INDEX')
            target=snapshot['models']['target']; read_policy=snapshot['cases'][0]['generating']['effective_model_policy']['target']
            identity={'collection_digest':snapshot['content_sha256'],'protocol_digest':protocol['content_sha256'],
                'target_identity_digest':content_hash(target),'target_policy_digest':content_hash(read_policy),
                'metric_contract_digest':content_hash(contract),'metric_revision':BENCHMARK_REVISION,
                'baseline_reference':reference,'code_sha':code_provenance()['package_source_sha256'],
                'method':'full_replay','shard_index':shard_index,'shard_count':shard_count,'expected_case_ids':assignment[shard_index]}
            if (output/'result.json').exists():
                existing=read_result(output/'result.json'); validate_rows(existing,protocol)
                require(all(existing.get(k)==v for k,v in identity.items()),'REPLAY_REUSE_IDENTITY')
                return existing
            require(not (output/'cases').exists(),'PARTIAL_REPLAY_REQUIRES_NEW_ATTEMPT')
            predictor=build_target(snapshot,protocol,device,deadline)
            require({k:getattr(predictor,k) for k in READ_FIELDS}==read_policy,'TARGET_READ_POLICY')
            groups=defaultdict(list)
            for row in protocol['cases']:
                if row['case_id'] in assignment[shard_index]: groups[(row['video_id'],row['object_id'])].append(row)
            rows=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for group in groups.values():
                    check(deadline,'replay_group_start')
                    _,mapping=jpeg_map(Path(group[0]['video_dir']),deadline=deadline)
                    require(content_hash(mapping)==group[0]['frame_map_sha256'],'EVAL_RGB_CHANGED')
                    predictions,trace=replay_group(predictor,group,baseline,snapshot['memory_policy'],deadline)
                    for fixed in group:
                        chosen={s['runtime_index']:predictions[s['runtime_index']] for s in fixed['scored_frames']}
                        scores=score_predictions(chosen,fixed,metric,deadline)
                        proof={**identity,'scored_frame_digest':content_hash(fixed['scored_frames']),
                               'baseline_revision':BASELINE_REVISION,'prompt_sha':fixed['prompt']['sha256']}
                        raw={'video':fixed['video_id'],'object':fixed['object_id'],'switch_name':str(round(fixed['fraction']*100)),
                             'switch_frame':fixed['switch_runtime_index'],'baseline':'full_replay',**scores}
                        row={**adapt_baseline_row(raw,fixed,proof),**scores,'trace':{**trace,
                             'continuation_frames':fixed['expected_runtime_indices']},'baseline_row':raw}
                        rows.append(row); write_json(output/f"cases/{content_hash(fixed['case_id'])}.json",row)
            check(deadline,'replay_finalization')
            result={**identity,'schema_version':'cmmt.lvos_evaluation.v1','status':'PASS','execution_kind':'gpu_real_checkpoint',
                'scope':protocol['scope'],'protocol_path':str(Path(protocol_path).resolve()),'epoch':0,
                'model_config_digest':content_hash(target),'checkpoint_sha':target['checkpoint_sha256'],
                'checkpoint':protocol['runtime']['target_checkpoint'],'rows':rows,
                'duration_seconds':deadline.elapsed(),'raw_metrics':raw_macro(rows)}
            validate_rows(result,protocol)
            write_json(output/'result.json',result); write_json(output/'READY.json',{'sha256':sha256(output/'result.json')})
            return result
        except BudgetStop as exc:
            write_json(output/'STOPPED.json',{'status':'STOPPED','reason':'budget_stop','incomplete_stage':exc.stage})
            raise


def comparison_table(paths,replay_path,protocol_path,metric_contract,benchmark_root,output):
    protocol=read_manifest(protocol_path); contract=json_read(metric_contract); metric=benchmark_module(benchmark_root,contract)
    replay=read_result(replay_path); validate_rows(replay,protocol)
    require(replay.get('merged') and replay['method']=='full_replay','COMPARISON_REPLAY')
    rows=[]
    for path in paths:
        result=read_result(path); validate_rows(result,protocol)
        require(result.get('merged') and set(result['expected_case_ids'])==set(protocol['expected_case_ids']) and
                result['scope']=='full_development' and all(result[k]==replay[k] for k in
                ('collection_digest','protocol_digest','target_identity_digest','target_policy_digest','metric_contract_digest','execution_kind')),
                'COMPARISON_IDENTITY')
        score=canonical_retention(metric,result['rows'],replay['rows'])
        rows.append({'method':result['method'],'epoch':result['epoch'],'checkpoint_sha':result['checkpoint_sha'],
                     'canonical_retention_percent':score['score'],'raw_j_f_jf':score['raw_method'],'details':score,
                     'artifact':{'path':str(Path(path).resolve()),'sha256':sha256(path)}})
    report={'primary':'mean_video_retention_three_fractions_percent','claim':'comparison on complete frozen development only',
            'legacy_pooled_retention':'different metric; not used','rows':rows,'protocol_digest':protocol['content_sha256']}
    require(not Path(output).exists(),'COMPARISON_EXISTS'); write_json(output,report); return report
