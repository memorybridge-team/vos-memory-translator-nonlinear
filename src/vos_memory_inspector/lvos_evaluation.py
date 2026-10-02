"""동결된 development protocol, target-only no-replay 평가, 단일 selection owner."""
from pathlib import Path
from dataclasses import dataclass, field, asdict, replace
from collections import defaultdict
import math
import time
import sys
import numpy as np
from PIL import Image
import torch

from .collection_contract import require, jpeg_map, READ_FIELDS
from .training_data import read_manifest, save_manifest
from .training_storage import content_hash, sha256, write_json, ExclusiveWriter
from .lvos_contract import json_read, load_complete, save_complete, benchmark_module
from .lvos_snapshot import verify_snapshot
from .lvos_training import load_model_checkpoint


def freeze_protocol(root, source_manifest, runtime, output, *, scope='monitor', videos=None,
                    min_visible_frames=1):
    """GT는 protocol coverage 검사에만 사용. Monitor 영상은 score 전에 명시한다."""
    snapshot = verify_snapshot(root)
    source = read_manifest(source_manifest)
    require(source['content_sha256']==snapshot['selection']['source_manifest_content_sha256'], 'PROTOCOL_SOURCE')
    require(scope in {'monitor','full_development'} and min_visible_frames>=1, 'PROTOCOL_SCOPE')
    raw_by_id = {c['case_id']:c for c in source['cases']}
    dev = [c for c in snapshot['cases'] if c['split']=='development']
    if scope=='full_development':
        require(snapshot['state']=='ready', 'FULL_DEV_REQUIRES_FULL_SNAPSHOT')
        require(videos is None, 'FULL_DEV_CANNOT_SUBSET')
    else:
        require(videos and len(videos)==len(set(videos)) and len(videos)==runtime['monitor_video_count'], 'MONITOR_VIDEOS')
        require(set(videos)<={c['case']['video_id'] for c in dev}, 'MONITOR_FOREIGN_VIDEO')
        dev = [c for c in dev if c['case']['video_id'] in videos]
    require(runtime.get('approved_by') and runtime['ground_truth_format']=='palette_object_id_png', 'RUNTIME_INPUTS')
    settings = {v['video_id']:v for v in runtime['videos']}
    require(len(settings)==len(runtime['videos']), 'DUPLICATE_RUNTIME_VIDEO')
    rows, maps = [], {}
    policy = source['selection_policy']
    require(policy['regular_quantiles']==[0.25,0.5,0.75], 'FRACTION_POLICY')
    for item in dev:
        case, gen = item['case'], item['generating']; raw = raw_by_id[case['case_id']]
        video = case['video_id']; setting = settings[video]
        if video not in maps:
            _, actual = jpeg_map(Path(setting['video_dir']))
            require(actual==gen['frame_map'], 'PROTOCOL_RGB_CHANGED', video)
            maps[video]=actual
        frame_map=maps[video]
        ids=[r['official_id'] for r in frame_map if raw['first_prompt_frame']<=r['official_id']<=raw['future_end_frame']]
        eligible=list(range(policy['min_prefix_frames']-1,len(ids)-policy['min_future_frames']))
        require(bool(eligible), 'FRACTION_NOT_RECOVERABLE')
        fractions=[q for q in policy['regular_quantiles'] if ids[eligible[round(q*(len(eligible)-1))]]==raw['switch_frame']]
        require(len(fractions)==1, 'AMBIGUOUS_FRACTION', case['case_id'])
        fraction=fractions[0]
        if scope=='monitor' and fraction!=0.5:
            continue
        end=max(r['runtime_index'] for r in frame_map if r['official_id']<=raw['future_end_frame'])
        if scope=='monitor':
            end=min(end,case['switch_frame']+64)
        expected=list(range(case['switch_frame']+1,end+1))
        annotations={int(p.stem):p for p in Path(setting['annotation_dir']).glob('*.png') if p.stem.isascii() and p.stem.isdigit()}
        scored=[]
        for index in expected:
            official=frame_map[index]['official_id']
            if official not in annotations:
                continue
            path=annotations[official]
            mask=np.asarray(Image.open(path))
            require(mask.ndim==2, 'GT_PALETTE_FORMAT')
            scored.append({'runtime_index':index,'official_id':official,'path':str(path.resolve()),
                'sha256':sha256(path),'visible':bool((mask==case['object_ids'][0]).any())})
        require(scope!='monitor' or sum(s['visible'] for s in scored)>=min_visible_frames, 'INSUFFICIENT_SCORED_FRAMES', case['case_id'])
        rows.append({'case_id':case['case_id'],'video_id':video,'fraction':fraction,'object_id':case['object_ids'][0],
            'switch_runtime_index':case['switch_frame'],'video_dir':str(Path(setting['video_dir']).resolve()),
            'expected_runtime_indices':expected,'scored_frames':scored,'frame_map_sha256':case['frame_map_sha256']})
    require(rows and (scope!='monitor' or set(videos)=={r['video_id'] for r in rows}), 'EMPTY_PROTOCOL_VIDEO')
    path=Path(output); require(not path.exists(),'PROTOCOL_EXISTS')
    save_manifest(path,{'schema_version':'cmmt.lvos_protocol.v1','scope':scope,'source_split':'train',
        'role':'development','collection_digest':snapshot['content_sha256'],'synthetic':snapshot['synthetic'],
        'models':snapshot['models'],'memory_policy':snapshot['memory_policy'],'runtime':runtime,'cases':rows,
        'frame_policy':'contiguous_runtime_rollout; score only frozen sparse PNG; GT visible only',
        'expected_case_ids':[r['case_id'] for r in rows],'min_visible_frames':min_visible_frames})
    return read_manifest(path)


def case_assignments(protocol, count):
    require(count>0, 'EVAL_SHARD_COUNT')
    videos=sorted({c['video_id'] for c in protocol['cases']})
    assignments=[[c['case_id'] for c in protocol['cases'] if c['video_id'] in videos[i::count]] for i in range(count)]
    flat=sum(assignments,[])
    require(len(flat)==len(set(flat)) and set(flat)==set(protocol['expected_case_ids']), 'EVAL_SHARD_COVERAGE')
    return assignments


def target_config_path(sam_root, config):
    from .training_collection import CONFIGS
    require(config==CONFIGS['target'],'PINNED_TARGET_CONFIG')
    return Path(sam_root)/'sam2'/config


def reconstruct_source(root, snapshot, index):
    """검증된 tensor view만 읽어 원래 K·padding·registry를 재조립한다."""
    from .state_schema import CanonicalState
    item=snapshot['cases'][index]; case=item['case']
    valid=torch.tensor(item['original_validity'],dtype=torch.bool)
    spatial=torch.zeros((*valid.shape,64,64,64),dtype=torch.bfloat16)
    pointer=torch.zeros((*valid.shape,256),dtype=torch.float32)
    frames=torch.tensor(item['original_frames'],dtype=torch.int64); slots=torch.tensor(item['original_slots'],dtype=torch.int64)
    cond=torch.tensor(item['original_conditioning'],dtype=torch.bool); seen=set()
    for entry in snapshot['shards']:
        if entry['split']!=item['split'] or entry['case_index']!=index:
            continue
        payload=None
        # A shard is loaded once for this case, never once per record.
        payload=torch.load(Path(root)/entry['path'],map_location='cpu',weights_only=True)
        for n,ref in enumerate(payload['refs']):
            if ref[0]!=index:
                continue
            key=tuple(ref[1:]); require(key not in seen,'SOURCE_DUPLICATE_RECORD'); seen.add(key)
            spatial[key]=payload['source_spatial'][n]; pointer[key]=payload['source_pointer'][n]
            frames[key]=payload['frame'][n]; slots[key]=payload['slot'][n]; cond[key]=payload['conditioning'][n]
    require(seen=={tuple(x) for x in torch.nonzero(valid).tolist()}, 'SOURCE_RECORD_COVERAGE')
    def floats(value):
        return [floats(v) for v in value] if isinstance(value,list) else float(value)
    return CanonicalState(spatial,pointer,torch.tensor(floats(item['presence_diagnostic']),dtype=torch.float32),frames,slots,cond,valid,
                          tuple(case['object_ids']),case['switch_frame']).validate()


def no_replay_case(predictor, source, model, row, *, deadline=None):
    from .sam2_state import init_sam2_inference_state_without_warmup, inject_sam2_canonical_state
    calls=[]; active_frame=[None]; original_get=predictor._get_image_feature; original_forward=predictor.forward_image
    def get(state,frame_idx,batch_size):
        active_frame[0]=int(frame_idx)
        try:
            return original_get(state,frame_idx,batch_size)
        finally:
            active_frame[0]=None
    def forward(image):
        require(active_frame[0] is not None,'ENCODER_FRAME_UNKNOWN')
        calls.append(active_frame[0]); return original_forward(image)
    predictor._get_image_feature=get; predictor.forward_image=forward
    try:
        started=time.perf_counter()
        state=init_sam2_inference_state_without_warmup(predictor,video_path=row['video_dir'],
                                                     offload_video_to_cpu=True,offload_state_to_cpu=True)
        moved=replace(source,**{n:getattr(source,n).to(predictor.device) for n in
            ('spatial_memory','object_pointer','presence_logits','frame_indices','slot_order','is_conditioning','validity')})
        with torch.autocast(torch.device(predictor.device).type,enabled=False):
            translated=model(moved) if model is not None else moved
        require(translated.spatial_memory.dtype==torch.bfloat16 and translated.object_pointer.dtype==torch.float32, 'HANDOFF_DTYPE')
        for name in ('validity','frame_indices','slot_order','is_conditioning'):
            require(torch.equal(getattr(source,name),getattr(translated,name).cpu()), 'HANDOFF_METADATA')
        info=inject_sam2_canonical_state(translated,predictor=predictor,inference_state=state)
        require(not calls, 'PAST_ENCODER_DURING_HANDOFF')
        handoff_seconds=time.perf_counter()-started
        expected=row['expected_runtime_indices']; predictions={}; observed=[]
        for frame,ids,masks in predictor.propagate_in_video(state,start_frame_idx=expected[0],
                                  max_frame_num_to_track=len(expected)-1,reverse=False):
            require(deadline is None or time.perf_counter()<deadline,'EVAL_TIME_LIMIT')
            require(list(ids)==list(source.object_ids) and masks.shape[0]==1,'ROLLOUT_REGISTRY')
            require(int(frame) not in observed,'ROLLOUT_DUPLICATE_FRAME'); observed.append(int(frame))
            if int(frame) in {s['runtime_index'] for s in row['scored_frames']}:
                predictions[int(frame)]=masks[0,0].detach().cpu().numpy()>0
        require(observed==expected and observed[0]==source.switch_frame+1,'ROLLOUT_COVERAGE')
        require(not [f for f in calls if f<=source.switch_frame],'PAST_ENCODER_CALL')
        return predictions,{'encoder_frame_ids':calls,'past_encoder_call_count':0,'continuation_frames':observed,
            'handoff_seconds':handoff_seconds,'registry':list(source.object_ids),'injection':info,
            'dtype':{'spatial':'bfloat16','pointer':'float32'},'metadata_preserved':True}
    finally:
        predictor._get_image_feature=original_get; predictor.forward_image=original_forward


def score_predictions(predictions,row,metric):
    require(set(predictions)=={s['runtime_index'] for s in row['scored_frames']}, 'SCORED_FRAME_COVERAGE')
    preds=[]; gts=[]
    for item in row['scored_frames']:
        require(sha256(item['path'])==item['sha256'],'GT_CHANGED')
        gt=np.asarray(Image.open(item['path']))==row['object_id']; pred=predictions[item['runtime_index']]
        require(pred.shape==gt.shape,'PRED_GT_SHAPE'); require(bool(gt.any())==item['visible'],'GT_VISIBILITY_CHANGED')
        preds.append(pred); gts.append(gt)
    visible=[(p,g) for p,g in zip(preds,gts) if g.any()]
    if not visible:
        return {'j':None,'f':None,'jf':None,'reason':'NO_VISIBLE_GT','visible_frames':0,'absent_frames':len(gts)}
    j=float(sum(metric.j_score(p,g) for p,g in visible)/len(visible))
    f=float(sum(metric.f_score(p,g) for p,g in visible)/len(visible))
    jf=metric.post_switch_jf(preds,gts)
    require(math.isfinite(jf) and abs(jf-(j+f)/2)<1e-12,'METRIC_FINITE')
    return {'j':j,'f':f,'jf':jf,'reason':None,'visible_frames':len(visible),'absent_frames':len(gts)-len(visible)}


def evaluate(root, protocol_path, checkpoint, metric_contract, benchmark_root, output, *, shard_index=0,
             shard_count=1, method='learned', device='cuda:0', approved=False,max_wall_seconds=None):
    require(approved and torch.device(device).type=='cuda' and torch.cuda.is_available() and
            torch.cuda.device_count()==1 and __import__('os').environ.get('CUDA_VISIBLE_DEVICES'), 'GPU_EXECUTION_APPROVAL')
    require(max_wall_seconds and max_wall_seconds>0,'EVAL_TIME_LIMIT_REQUIRED')
    deadline=time.perf_counter()+max_wall_seconds
    snapshot=verify_snapshot(root); protocol=read_manifest(protocol_path); contract=json_read(metric_contract)
    require(not snapshot['synthetic'] and protocol['collection_digest']==snapshot['content_sha256'],'REAL_EVAL_SNAPSHOT')
    metric=benchmark_module(benchmark_root,contract)
    require(method in {'learned','direct_copy'},'BASELINE_OWNER_REQUIRED')
    model,payload,entry=load_model_checkpoint(checkpoint,device)
    require(payload['identity']['collection_digest']==snapshot['content_sha256'],'EVAL_CHECKPOINT_DATA')
    if method=='direct_copy':
        require(payload['epoch']==0,'DIRECT_COPY_EPOCH_ZERO'); model=None
    assignments=case_assignments(protocol,shard_count)
    require(0<=shard_index<shard_count and assignments[shard_index],'EVAL_SHARD_INDEX')
    runtime=protocol['runtime']; sam_root=Path(runtime['sam2_repo']).resolve()
    from .upstream import verify_sam2_checkout
    require(verify_sam2_checkout(sam_root)==snapshot['models']['target']['upstream_commit'],'TARGET_UPSTREAM')
    import subprocess
    require(subprocess.run(['git','diff','--exit-code','HEAD','--','sam2'],cwd=sam_root,capture_output=True).returncode==0,'DIRTY_TARGET_SAM_SOURCE')
    for key,path in (('checkpoint_sha256',runtime['target_checkpoint']),('config_sha256',target_config_path(sam_root,runtime['target_config']))):
        require(sha256(path)==snapshot['models']['target'][key],'TARGET_HASH')
    sys.path.insert(0,str(sam_root))
    from sam2.build_sam import build_sam2_video_predictor
    from .sam2_lazy_loader import install_sam2_lazy_loader
    from .training_collection import freeze_model
    predictor=freeze_model(build_sam2_video_predictor(runtime['target_config'],runtime['target_checkpoint'],device=device))
    install_sam2_lazy_loader(cache_size=8)
    output=Path(output); by_id={c['case']['case_id']:i for i,c in enumerate(snapshot['cases'])}; rows=[]
    with ExclusiveWriter(output),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        require(not (output/'result.json').exists(),'EVAL_OUTPUT_EXISTS')
        for row in protocol['cases']:
            if row['case_id'] not in assignments[shard_index]:
                continue
            require(time.perf_counter()<deadline,'EVAL_TIME_LIMIT')
            item=snapshot['cases'][by_id[row['case_id']]]
            require({k:getattr(predictor,k) for k in READ_FIELDS}==item['generating']['effective_model_policy']['target'],'TARGET_READ_POLICY')
            _,actual=jpeg_map(Path(row['video_dir']))
            require(content_hash(actual)==row['frame_map_sha256'],'EVAL_RGB_CHANGED')
            source=reconstruct_source(root,snapshot,by_id[row['case_id']]); started=time.perf_counter()
            predictions,trace=no_replay_case(predictor,source,model,row,deadline=deadline)
            scores=score_predictions(predictions,row,metric)
            rows.append({'case_id':row['case_id'],'video':row['video_id'],'fraction':row['fraction'],
                         **scores,'trace':trace,'duration_seconds':time.perf_counter()-started})
            write_json(output/f"cases/{content_hash(row['case_id'])}.json",rows[-1])
            print(f"평가 완료: {row['case_id']}, J&F={scores['jf']}",flush=True)
        result={'schema_version':'cmmt.lvos_evaluation.v1','status':'PASS','execution_kind':'gpu_real_checkpoint',
            'collection_digest':snapshot['content_sha256'],'model_config_digest':payload['identity']['model_config_digest'],
            'protocol_digest':protocol['content_sha256'],'scope':protocol['scope'],'metric_contract_digest':content_hash(contract),
            'protocol_path':str(Path(protocol_path).resolve()),
            'checkpoint_sha':entry['sha256'],'checkpoint':str(Path(checkpoint).resolve()),'epoch':payload['epoch'],
            'method':method,'shard_index':shard_index,'shard_count':shard_count,
            'expected_case_ids':assignments[shard_index],'rows':rows}
        write_json(output/'result.json',result); write_json(output/'READY.json',{'sha256':sha256(output/'result.json')})
        return result


def read_result(path):
    path=Path(path)
    require(json_read(path.parent/'READY.json')['sha256']==sha256(path),'EVALUATION_NOT_READY')
    value=json_read(path); require(value['status']=='PASS','EVALUATION_FAILED'); return value


def validate_rows(result,protocol):
    expected={c['case_id']:c for c in protocol['cases']}
    rows=result['rows']; ids=[r['case_id'] for r in rows]
    require(len(ids)==len(set(ids)) and set(ids)==set(result['expected_case_ids']),'EVAL_RESULT_COVERAGE')
    require(set(ids)<=expected.keys(),'EVAL_FOREIGN_CASE')
    for row in rows:
        fixed=expected[row['case_id']]
        require(row['video']==fixed['video_id'] and row['fraction']==fixed['fraction'],'EVAL_CASE_METADATA')
        require(row['visible_frames']==sum(s['visible'] for s in fixed['scored_frames']) and
                row['absent_frames']==sum(not s['visible'] for s in fixed['scored_frames']), 'EVAL_SCORED_COUNTS')
        require((row['jf'] is None and row['visible_frames']==0 and row.get('reason')=='NO_VISIBLE_GT') or
                (row['jf'] is not None and row['visible_frames']>0 and math.isfinite(row['jf']) and 0<=row['jf']<=1),'EVAL_UNDEFINED')
        if row['jf'] is not None:
            require(all(math.isfinite(row[k]) and 0<=row[k]<=1 for k in ('j','f')) and abs(row['jf']-(row['j']+row['f'])/2)<1e-9,'EVAL_COMPONENTS')


def merge_results(paths,protocol_path,metric_contract,benchmark_root,output, *, replay=None):
    protocol=read_manifest(protocol_path); contract=json_read(metric_contract); metric=benchmark_module(benchmark_root,contract)
    results=[read_result(p) for p in paths]; require(results,'EVAL_EMPTY_MERGE')
    first=results[0]
    keys=('collection_digest','model_config_digest','checkpoint_sha','protocol_digest','metric_contract_digest','method','epoch','execution_kind','shard_count')
    assignments=case_assignments(protocol,first['shard_count'])
    require(len(results)==first['shard_count'] and {r['shard_index'] for r in results}==set(range(first['shard_count'])),'EVAL_MISSING_SHARD')
    for r in results:
        require(all(r[k]==first[k] for k in keys),'EVAL_MIXED_IDENTITY')
        require(r['protocol_digest']==protocol['content_sha256'] and r['collection_digest']==protocol['collection_digest'] and
                r['metric_contract_digest']==content_hash(contract),'EVAL_DIGEST')
        require(r['expected_case_ids']==assignments[r['shard_index']],'EVAL_ASSIGNMENT'); validate_rows(r,protocol)
    rows=sum([r['rows'] for r in results],[]); require({r['case_id'] for r in rows}==set(protocol['expected_case_ids']),'EVAL_FULL_COVERAGE')
    grouped=defaultdict(list)
    for row in rows:
        if row['jf'] is not None:
            grouped[row['video']].append(row['jf'])
    merged={**first,'rows':sorted(rows,key=lambda r:r['case_id']),'expected_case_ids':protocol['expected_case_ids'],
        'merged':True,'shard_index':None,'monitor_score':sum(sum(v)/len(v) for v in grouped.values())/len(grouped) if grouped else None,
        'undefined_case_count':sum(r['jf'] is None for r in rows),
        'primary_score':None,'primary_reason':'FULL_REPLAY_RESULTS_REQUIRED','source_artifacts':[{'path':str(Path(p).resolve()),'sha256':sha256(p)} for p in paths]}
    if replay:
        baseline=read_result(replay); validate_rows(baseline,protocol)
        require(baseline.get('merged') and baseline['method']=='full_replay' and baseline['scope']==protocol['scope'] and
                set(baseline['expected_case_ids'])==set(protocol['expected_case_ids']) and
                all(baseline[k]==merged[k] for k in ('collection_digest','protocol_digest','metric_contract_digest','execution_kind')),'REPLAY_CONTRACT')
        if protocol['scope']=='full_development':
            # Official implementation performs object -> video -> ratio -> fraction mean.
            method_rows=[r for r in merged['rows'] if r['jf'] is not None]
            replay_rows=[r for r in baseline['rows'] if r['jf'] is not None]
            try:
                merged['retention']=metric.selection_score(method_rows,replay_rows)
                merged['primary_score']=merged['retention']['score']; merged['primary_reason']=None
            except ZeroDivisionError:
                merged['primary_reason']='NO_DEFINED_NONZERO_REPLAY_VIDEO_FOR_FRACTION'
            merged['replay_artifact']={'path':str(Path(replay).resolve()),'sha256':sha256(replay)}
    output=Path(output)
    with ExclusiveWriter(output):
        require(not (output/'result.json').exists(),'EVAL_OUTPUT_EXISTS')
        write_json(output/'result.json',merged); write_json(output/'READY.json',{'sha256':sha256(output/'result.json')})
    return merged


@dataclass
class EarlyStopping:
    min_epochs:int=6
    patience:int=5
    min_delta:float=0.001
    stopping_reference:float|None=None
    raw_max:float|None=None
    bad_events:int=0
    consumed:list=field(default_factory=list)
    should_stop:bool=False
    def accept(self,epoch,score):
        require(epoch not in self.consumed and (not self.consumed or epoch>self.consumed[-1]),'ASYNC_EVENT_ORDER')
        require(math.isfinite(score) and 0<=score<=1,'MONITOR_SCORE_RANGE')
        self.consumed.append(epoch)
        self.raw_max=score if self.raw_max is None else max(self.raw_max,score)
        if self.stopping_reference is None or score>self.stopping_reference+self.min_delta:
            self.stopping_reference=score; self.bad_events=0
        elif epoch>=self.min_epochs:
            self.bad_events+=1
        self.should_stop=epoch>=self.min_epochs and self.bad_events>=self.patience
    def state(self):
        return asdict(self)
    def restore(self,value):
        require(all(value[k]==getattr(self,k) for k in ('min_epochs','patience','min_delta')),'EARLY_CONFIG_CHANGED')
        for key,value in value.items():
            setattr(self,key,value)


def consume_monitor_results(output,early,identity,events):
    output=Path(output)
    requests=sorted((output/'evaluation/requests').glob('epoch-*.json'))
    processed=0
    for path in requests:
        request=json_read(path); epoch=request['epoch']
        if epoch==0 or epoch in early.consumed:
            continue
        result_path=output/f'evaluation/results/epoch-{epoch:05d}/result.json'
        if not result_path.exists():
            break
        try:
            result=read_result(result_path); protocol=read_manifest(request['protocol'])
            validate_rows(result,protocol)
            require(result.get('merged') and result['scope']=='monitor' and result['method']=='learned' and
                set(result['expected_case_ids'])==set(protocol['expected_case_ids']) and result['epoch']==epoch and
                result['checkpoint_sha']==request['checkpoint_sha'] and
                result['collection_digest']==identity['collection_digest'] and result['model_config_digest']==identity['model_config_digest'] and
                result['protocol_digest']==protocol['content_sha256'] and
                result['metric_contract_digest']==identity['metric_contract_digest'],'MONITOR_RESULT_IDENTITY')
            early.accept(epoch,result['monitor_score']); processed+=1
            events.emit('monitor_accepted',epoch=epoch,eval_id=result_path.parent.name,metrics={'score':result['monitor_score']},evidence_paths=[str(result_path)])
            prior=json_read(output/'best_monitor.ckpt.json') if (output/'best_monitor.ckpt.json').exists() else None
            if prior is None or result['monitor_score']>prior['score']:
                write_json(output/'best_monitor.ckpt.json',{'path':request['checkpoint'],'sha256':request['checkpoint_sha'],'epoch':epoch,'score':result['monitor_score']})
        except (ValueError,KeyError,OSError) as exc:
            events.emit('monitor_rejected',epoch=epoch,status='FAIL',reason_code=getattr(exc,'code','INVALID_MONITOR_RESULT'),evidence_paths=[str(result_path)])
            break  # 수정 대기. 누락 event를 건너뛰어 patience를 진행하지 않는다.
    return sum(json_read(p)['epoch']!=0 and json_read(p)['epoch'] not in early.consumed for p in requests)


def shortlist(paths,output):
    rows=[read_result(p) for p in paths]
    require(rows and all(r.get('merged') and r['scope']=='monitor' and r['method']=='learned' for r in rows),'SHORTLIST_SCOPE')
    first=rows[0]
    for r in rows:
        validate_rows(r,read_manifest(r['protocol_path']))
        require(set(r['expected_case_ids'])==set(read_manifest(r['protocol_path'])['expected_case_ids']),'SHORTLIST_COVERAGE')
    require(all(all(r[k]==first[k] for k in ('collection_digest','model_config_digest','protocol_digest','metric_contract_digest')) for r in rows),'SHORTLIST_IDENTITY')
    epochs=[r['epoch'] for r in rows]; require(len(epochs)==len(set(epochs)),'SHORTLIST_DUPLICATE_EPOCH')
    require(all(r['monitor_score'] is not None and math.isfinite(r['monitor_score']) for r in rows),'SHORTLIST_SCORE')
    trained=[r for r in rows if r['epoch']>0]; trained.sort(key=lambda r:(-r['monitor_score'],r['epoch']))
    require(trained,'SHORTLIST_NO_TRAINED')
    result={'rule':'top_3_distinct_trained_epochs_by_frozen_monitor_score; exhaustive monitor shortlist only',
            'identity':{k:first[k] for k in ('collection_digest','model_config_digest','metric_contract_digest')},
            'candidates':[{k:r[k] for k in ('epoch','checkpoint','checkpoint_sha','monitor_score')} for r in trained[:3]]}
    write_json(output,result); return result


def select_best(paths,protocol_path,metric_contract,output, *, shortlist_path):
    protocol=read_manifest(protocol_path); contract=json_read(metric_contract)
    require(protocol['scope']=='full_development' and not protocol['synthetic'],'BEST_REQUIRES_REAL_FULL_DEV')
    require(contract.get('approved_by') and contract.get('export_approved_by') and
            contract.get('export_loader')=='vos_memory_inspector.transformer_translator:TransformerStateTranslator.from_payload', 'EXPORT_ADAPTER_CONTRACT_MISSING')
    shortlist_value=json_read(shortlist_path); allowed=shortlist_value['candidates']; require(allowed,'SHORTLIST_REQUIRED')
    results=[read_result(p) for p in paths]; require(len(results)==len(allowed),'FULL_DEV_SHORTLIST_INCOMPLETE')
    require({(r['epoch'],r['checkpoint_sha']) for r in results}=={(r['epoch'],r['checkpoint_sha']) for r in allowed},'SHORTLIST_CHANGED')
    first=results[0]
    for r in results:
        validate_rows(r,protocol)
        require(r.get('merged') and r['scope']=='full_development' and r['method']=='learned' and r['epoch']>0 and
            set(r['expected_case_ids'])==set(protocol['expected_case_ids']) and r['protocol_digest']==protocol['content_sha256'] and
            r['collection_digest']==protocol['collection_digest'] and r['metric_contract_digest']==content_hash(contract) and
            r['execution_kind']=='gpu_real_checkpoint' and r['primary_score'] is not None and math.isfinite(r['primary_score']), 'BEST_RESULT_CONTRACT')
        require(r['model_config_digest']==first['model_config_digest'],'BEST_MODEL_CONFIG_CHANGED')
        require(all(r[k]==shortlist_value['identity'][k] for k in ('collection_digest','model_config_digest','metric_contract_digest')),'BEST_SHORTLIST_IDENTITY')
        payload,entry=load_complete(r['checkpoint'])
        require(entry['sha256']==r['checkpoint_sha'] and payload['epoch']==r['epoch'] and
            payload['identity']['collection_digest']==r['collection_digest'] and
            payload['identity']['model_config_digest']==r['model_config_digest'],'BEST_CHECKPOINT_IDENTITY')
    best=best_candidate(results)
    output=Path(output)
    with ExclusiveWriter(output):
        require(not (output/'best_model.pth').exists(),'BEST_ALREADY_PUBLISHED')
        payload,_=load_complete(best['checkpoint'])
        from .transformer_translator import TransformerStateTranslator
        TransformerStateTranslator.from_payload(payload['export'])  # Exact model-team export schema, no rename-only adapter.
        from .lvos_contract import validate_model_lock
        validate_model_lock(payload['model_lock'])
        selection={'rule':'complete frozen full-development shortlist only; tie 1e-6 earlier epoch',
            'candidates':[{k:r[k] for k in ('epoch','primary_score','checkpoint_sha','checkpoint','monitor_score')} for r in results],
            'winner_epoch':best['epoch'],'evidence':[{'path':str(Path(p).resolve()),'sha256':sha256(p)} for p in paths],
            'limitations':'trained-candidate winner; improvement over epoch-0/direct copy requires their actual comparable results'}
        write_json(output/'selection.json',selection)
        save_complete(output/'best_model.pth',payload['export'])
        return selection


def best_candidate(results):
    require(results and all(r['epoch']>0 and r['primary_score'] is not None and math.isfinite(r['primary_score']) for r in results),
            'NO_VALID_TRAINED_CANDIDATE')
    maximum=max(r['primary_score'] for r in results)
    return min((r for r in results if maximum-r['primary_score']<=1e-6),key=lambda r:r['epoch'])
