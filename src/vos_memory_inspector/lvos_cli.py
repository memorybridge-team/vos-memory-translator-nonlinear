"""LVOS v1 operations. --help/계획/CPU 검사에는 GPU 실행이 없다."""
import argparse
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import nullcontext
import xml.etree.ElementTree as ET
import torch
from .collection_contract import Rejection, require
from .training_storage import write_json, sha256, source_sha256, content_hash
from .training_data import read_manifest
from .lvos_contract import json_read, model_lock, evidence, BENCHMARK_REVISION, safety_suite_hash
from .lvos_budget import Deadline, BudgetStop


def parser():
    p=argparse.ArgumentParser(description='LVOS official-train fit/dev 고정 모델 파이프라인; GPU 실행은 명시 승인·상한 필요')
    s=p.add_subparsers(dest='command',required=True)
    item=s.add_parser('freeze-model',help='모델팀 base config proposal; 승인자는 실제 승인 후 입력')
    item.add_argument('--output',required=True,type=Path); item.add_argument('--approved-by')
    item=s.add_parser('metric-template',help='실제 benchmark 함수를 pin한 미승인 계약 template')
    item.add_argument('--benchmark-root',required=True,type=Path); item.add_argument('--output',required=True,type=Path)
    item=s.add_parser('audit',help='명시한 완료 legacy cache 검사 + 원본 밖 immutable training view')
    for n in ('requests','selection','output'):
        item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--trusted-team-legacy',action='store_true'); item.add_argument('--stable-seconds',type=float,default=60)
    item.add_argument('--max-shard-mib',type=int,default=256); item.add_argument('--pilot-case-id',action='append')
    item=s.add_parser('verify',help='CPU SHA·completion·원본 snapshot 무결성 검사')
    item.add_argument('--snapshot',required=True,type=Path)
    item=s.add_parser('discover',help='CPU read-only 완료 cache -> audit requests; 원본 수정 없음')
    item.add_argument('--cache-root',required=True,action='append',type=Path)
    item.add_argument('--evidence-root',action='append',type=Path,default=[])
    for n in ('selection','output'): item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--rgb-root',type=Path); item.add_argument('--run-evidence',type=Path)
    item.add_argument('--trusted-team-legacy',action='store_true')
    item.add_argument('--stable-seconds',type=float,default=60); item.add_argument('--max-wall-seconds',type=float)
    item.add_argument('--require-complete',action='store_true')
    item=s.add_parser('cpu-test',help='실제 CPU pytest + JUnit/log/checksum evidence report')
    item.add_argument('--output',required=True,type=Path)
    item.add_argument('--basetemp',type=Path,help='새 전용 pytest temp 경로; Windows는 짧은 절대 경로 필수')
    item=s.add_parser('gates',help='G0–G6 evidence summary; 누락 실측은 BLOCKED')
    item.add_argument('--output',required=True,type=Path); item.add_argument('--snapshot',type=Path)
    item.add_argument('--model-lock',type=Path); item.add_argument('--metric-contract',type=Path)
    item.add_argument('--handoff-result',type=Path); item.add_argument('--cpu-test-report',type=Path)
    item.add_argument('--direct-copy-result',type=Path)
    item.add_argument('--device',default='cpu'); item.add_argument('--execute-approved',action='store_true')
    item.add_argument('--max-wall-seconds',type=float)
    item.add_argument('--require-ready',action='store_true',help='BLOCKED도 exit 3; FAIL은 항상 exit 2')
    item=s.add_parser('freeze-protocol',help='score 보기 전 sparse GT coverage를 검사하여 dev protocol 동결')
    for n in ('snapshot','source-manifest','runtime','output'):
        item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--scope',choices=('monitor','full_development'),default='monitor')
    item.add_argument('--video-id',action='append'); item.add_argument('--min-visible-frames',type=int,default=1)
    for cmd in ('initialize','overfit','profile','train'):
        item=s.add_parser(cmd,help='독립 whole-record FP32 '+cmd+'; 기존 collector와 별도 출력')
        for n in ('snapshot','model-lock','config','output'):
            item.add_argument('--'+n,required=True,type=Path)
        item.add_argument('--device',default='cpu' if cmd=='initialize' else 'cuda:0'); item.add_argument('--execute-approved',action='store_true')
        item.add_argument('--max-wall-seconds',type=float); item.add_argument('--max-micro-iterations',type=int)
        item.add_argument('--resume',type=Path); item.add_argument('--gates',type=Path)
        item.add_argument('--monitor-protocol',type=Path); item.add_argument('--metric-contract',type=Path)
    item=s.add_parser('evaluate',help='실제 Base+ target-only no-replay eval; baseline full-replay는 benchmark 담당')
    for n in ('snapshot','protocol','checkpoint','metric-contract','benchmark-root','output'):
        item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--method',choices=('learned','direct_copy'),default='learned')
    item.add_argument('--shard-index',type=int,default=0); item.add_argument('--shard-count',type=int,default=1)
    item.add_argument('--device',default='cuda:0'); item.add_argument('--execute-approved',action='store_true')
    item.add_argument('--max-wall-seconds',required=True,type=float)
    item=s.add_parser('full-replay',help='Frozen full-dev denominator only; 다른 baseline/Moment 실행 없음')
    for n in ('snapshot','protocol','metric-contract','benchmark-root','baseline-root','output'):
        item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--shard-index',type=int,default=0); item.add_argument('--shard-count',type=int,default=1)
    item.add_argument('--device',default='cuda:0'); item.add_argument('--execute-approved',action='store_true')
    item.add_argument('--max-wall-seconds',required=True,type=float)
    item=s.add_parser('compare',help='Complete full-dev canonical retention와 raw J/F/JF 표')
    for n in ('replay','protocol','metric-contract','benchmark-root','output'):
        item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--result',required=True,action='append',type=Path)
    item=s.add_parser('merge-eval',help='CPU immutable shard merge; worker mean을 평균하지 않음')
    for n in ('protocol','metric-contract','benchmark-root','output'):
        item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--result',required=True,action='append',type=Path); item.add_argument('--replay',type=Path)
    item=s.add_parser('shortlist',help='동일 frozen monitor 상위 trained epoch 3개')
    item.add_argument('--result',required=True,action='append',type=Path); item.add_argument('--output',required=True,type=Path)
    item=s.add_parser('select',help='완전한 full-dev shortlist + 승인된 adapter만 best export')
    for n in ('protocol','metric-contract','shortlist','output'):
        item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--result',required=True,action='append',type=Path)
    item.add_argument('--benchmark-root',required=True,type=Path)
    item=s.add_parser('eval-worker',help='Default dry-run; 명시 실행 worker, immutable requests, disjoint video shards')
    for n in ('run','snapshot','benchmark-root'): item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--shard-index',type=int,default=0); item.add_argument('--shard-count',type=int,default=1)
    item.add_argument('--execute',action='store_true'); item.add_argument('--execute-approved',action='store_true')
    item.add_argument('--drain',action='store_true'); item.add_argument('--merge-shards',action='store_true')
    item.add_argument('--device',default='cuda:0'); item.add_argument('--max-wall-seconds',type=float)
    item=s.add_parser('eval-merge',help='CPU verified exactly-once merge; pending shard는 중단')
    for n in ('run','request','benchmark-root'): item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--shard-count',type=int,default=1)
    item=s.add_parser('eval-status',help='학습 완료와 평가 완료를 구분한 read-only status')
    item.add_argument('--run',required=True,type=Path); item.add_argument('--output',required=True,type=Path)
    item=s.add_parser('bundle',help='strict best export + sanitized index + optional resume를 로컬 ZIP으로 저장')
    for n in ('release','run','output'): item.add_argument('--'+n,required=True,type=Path)
    item.add_argument('--include-resume',action='store_true'); item.add_argument('--allow-synthetic',action='store_true')
    item=s.add_parser('launch-plan',help='독립 A/B/C/D command 계획만 생성; 실행 기능 없음')
    item.add_argument('--jobs',required=True,type=Path); item.add_argument('--output',required=True,type=Path)
    item=s.add_parser('draft-export',help='실제 run/result만 Korean Markdown 표로 내보냄')
    item.add_argument('--run',action='append',type=Path,default=[]); item.add_argument('--result',action='append',type=Path,default=[])
    item.add_argument('--output',required=True,type=Path)
    for name in ('train','overfit','profile','initialize','gates','evaluate','full-replay','eval-worker'):
        item=s.choices[name]
        item.add_argument('--lease-root',type=Path); item.add_argument('--host-id')
        item.add_argument('--physical-gpu-uuid'); item.add_argument('--quoted-hourly-rate',type=float)
        item.add_argument('--total-budget',type=float); item.add_argument('--budget-currency',choices=('USD','KRW'))
    item=s.choices['train']
    item.add_argument('--evaluation-mode',choices=('queue','sequential'),default='queue')
    item.add_argument('--benchmark-root',type=Path)
    return p


def approved_gpu(args):
    if torch.device(args.device).type=='cuda':
        require(args.execute_approved and args.max_wall_seconds and args.max_wall_seconds>0,'GPU_EXECUTION_APPROVAL_AND_TIME_LIMIT')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') and torch.cuda.is_available() and torch.cuda.device_count()==1,'ONE_VISIBLE_GPU')
        require(args.lease_root and args.host_id and args.physical_gpu_uuid and
                os.environ['CUDA_VISIBLE_DEVICES']==args.physical_gpu_uuid and args.physical_gpu_uuid.startswith(('GPU-','MIG-')),
                'EXPLICIT_FREE_GPU_UUID_HOST_AND_LEASE_REQUIRED')
        require(args.quoted_hourly_rate is not None and args.quoted_hourly_rate>=0 and args.total_budget is not None and
                args.total_budget>=0 and args.budget_currency,'GPU_RATE_TOTAL_BUDGET_REQUIRED')
        require(args.max_wall_seconds/3600*args.quoted_hourly_rate<=args.total_budget,'GPU_BUDGET_EXCEEDED')


def gpu_lease(args):
    approved_gpu(args)
    if torch.device(args.device).type!='cuda': return nullcontext(None)
    from .lvos_jobs import GPULease
    return GPULease(args.lease_root,args.host_id,args.physical_gpu_uuid,str(getattr(args,'output',getattr(args,'run',args.command))))


def cpu_test(output,basetemp=None):
    output=Path(output).resolve(); require(not (output/'report.json').exists(),'TEST_OUTPUT_EXISTS'); output.mkdir(parents=True,exist_ok=True)
    basetemp=Path(basetemp).resolve() if basetemp else output/'tmp'
    require(not basetemp.exists(),'TEST_TEMP_EXISTS','새 전용 --basetemp 경로를 사용하십시오. 기존 파일을 삭제하지 않습니다.')
    require(os.name!='nt' or len(str(basetemp))<=40,'WINDOWS_SHORT_TEST_TEMP_REQUIRED',
            'cpu-test --basetemp에 길이 40자 이하의 새 전용 절대 경로를 지정하십시오.')
    repo=Path(__file__).resolve().parents[2]
    from .training_runner import code_provenance
    source_sha=code_provenance()['package_source_sha256']; tests_sha=safety_suite_hash()
    command=[sys.executable,'-m','pytest','tests/test_lvos_pipeline.py','tests/test_lvos_integration.py','-q','--junitxml',str(output/'junit.xml'),
             '-p','no:cacheprovider','--basetemp',str(basetemp)]
    started=time.perf_counter()
    with (output/'pytest.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    xml=ET.parse(output/'junit.xml').getroot(); suites=[xml] if xml.tag=='testsuite' else list(xml)
    report={'suite':'lvos_pipeline','execution_kind':'cpu_synthetic','exit_code':result.returncode,
        'tested_source_sha256':source_sha,'tested_suite_sha256':tests_sha,
        'tests':sum(int(x.attrib['tests']) for x in suites),'command':command,'duration_seconds':time.perf_counter()-started,
        'evidence':evidence([output/'pytest.log',output/'junit.xml']), 'status':'PASS' if result.returncode==0 else 'FAIL'}
    if source_sha!=code_provenance()['package_source_sha256'] or tests_sha!=safety_suite_hash():
        report.update(exit_code=2,status='FAIL',reason_code='SOURCE_CHANGED_DURING_TESTS')
    write_json(output/'report.json',report); print((output/'pytest.log').read_text(encoding='utf-8')); return report


def launch_plan(jobs,output):
    values=jobs['jobs']; keys=[j['job_id'] for j in values]
    require(len(keys)==len(set(keys)) and set(keys)<={'A','B','C','D'},'DUPLICATE_JOB')
    devices=set(); roots=set(); commands=[]
    for job in values:
        key=(job['host'],job['physical_gpu_uuid']); root=job['output'].rstrip('/')
        require(key not in devices and root not in roots,'DUPLICATE_DEVICE_OR_NAMESPACE')
        devices.add(key); roots.add(root)
        argv=job['argv']
        require(argv[:3]==[job['python'],'-m','vos_memory_inspector.lvos_cli'] and argv[3] in {'train','evaluate','full-replay'},'LAUNCH_COMMAND')
        require('--output' in argv and argv[argv.index('--output')+1]==job['output'] and
                '--device' in argv and argv[argv.index('--device')+1]=='cuda:0','LAUNCH_LOCAL_DEVICE')
        parser().parse_args(argv[3:])
        commands.append({**job,'status':'OPERATOR_APPROVAL_REQUIRED','command':
            'CUDA_VISIBLE_DEVICES='+shlex.quote(job['physical_gpu_uuid'])+' '+shlex.join(argv)})
    result={'schema_version':'cmmt.lvos_launch_plan.v1','dry_run':True,'execution_performed':False,
            'jobs':commands,'note':'Independent processes; one visible GPU -> cuda:0. No torchrun/DDP.'}
    write_json(output,result); return result


def draft_export(runs,results,output):
    from .lvos_evaluation import read_result
    lines=['# LVOS 실행 근거 요약','', '이 표는 실제 artifact만 읽은 초안입니다. 누락 결과는 대기이며 VOS 개선으로 해석하지 않습니다.','',
        '## 학습','', '|Run|모델 config digest|완료 epoch|fit loss|dev state loss|상태|','|---|---|---:|---:|---:|---|']
    for root in runs:
        cfg=json_read(root/'config.json'); history=json_read(root/'metrics/train.json') if (root/'metrics/train.json').exists() else []
        last=history[-1] if history else None
        current=json_read(root/'STATUS.json')['status'] if (root/'STATUS.json').exists() else '미확인'
        lines.append(f"|{root.name}|{cfg['identity']['model_config_digest']}|{len(history)}|{last['fit']['loss'] if last else '대기'}|{last['dev']['loss'] if last else '대기'}|{current}|")
        if history:
            lines+=['',f'### {root.name} 학습 곡선 (관측값)','', '|epoch|fit loss|dev loss|', '|---:|---:|---:|']
            lines += [f"|{r['epoch']}|{r['fit']['loss']}|{r['dev']['loss']}|" for r in history]
    if not runs:
        lines+=['|대기|미확인|대기|대기|대기|실제 학습 artifact 없음|']
    lines+=['','## 후속 평가','', '|방법|epoch|scope|실행 종류|case/expected|monitor J&F|primary score|checkpoint SHA|', '|---|---:|---|---|---|---:|---:|---|']
    for path in results:
        r=read_result(path)
        lines.append(f"|{r['method']}|{r['epoch']}|{r['scope']}|{r['execution_kind']}|{len(r['rows'])}/{len(r['expected_case_ids'])}|{r.get('monitor_score','대기')}|{r.get('primary_score','대기')}|{r['checkpoint_sha']}|")
    if not results:
        lines+=['|대기|대기|대기|대기|대기|대기|대기|실제 GPU 결과 없음|']
    lines+=['','단일 객체 독립 cache는 joint multi-object equivalence를 입증하지 않습니다. State loss 감소와 VOS 개선은 별도 근거가 필요합니다.','']
    from .training_storage import atomic_write
    atomic_write(output,lambda f:f.write('\n'.join(lines).encode('utf-8')))


def main(argv=None):
    args=parser().parse_args(argv)
    command_started=time.perf_counter()
    try:
        if args.command=='freeze-model':
            require(not args.output.exists(),'MODEL_LOCK_EXISTS'); result=model_lock(args.approved_by); write_json(args.output,result)
        elif args.command=='metric-template':
            require(not args.output.exists(),'METRIC_CONTRACT_EXISTS')
            result={'schema_version':'cmmt.lvos_metric_contract.v2','benchmark_revision':BENCHMARK_REVISION,
                'source_hash_policy':'utf8_crlf_to_lf_v1','evaluator_sha256':source_sha256(args.benchmark_root/'best_model.py'),'primary_metric':'mean_video_retention_three_fractions_percent',
                'fractions':[0.25,0.5,0.75],'visible_rule':'gt_visible_only','undefined_rule':'null_with_reason',
                'zero_replay_rule':'exclude_video_ratio','monitor_metric':'monitor_jf_proxy','monitor_scale':[0,1],'approved_by':None,
                'export_loader':'vos_memory_inspector.transformer_translator:TransformerStateTranslator.from_payload',
                'export_approved_by':None,'baseline_owner':'pinned baseline/no_handoff.py via canonical full-replay bridge'}
            write_json(args.output,result)
        elif args.command=='audit':
            from .lvos_snapshot import build_snapshot
            result=build_snapshot(json_read(args.requests),args.selection,args.output,trusted=args.trusted_team_legacy,
                stable_seconds=args.stable_seconds,max_shard_mib=args.max_shard_mib,pilot_ids=args.pilot_case_id)
        elif args.command=='verify':
            from .lvos_snapshot import verify_snapshot
            result=verify_snapshot(args.snapshot)
        elif args.command=='discover':
            from .lvos_discovery import discover
            result=discover(args.cache_root,args.selection,args.output,evidence_roots=args.evidence_root,rgb_root=args.rgb_root,
                trusted=args.trusted_team_legacy,stable_seconds=args.stable_seconds,run_evidence=args.run_evidence,max_wall_seconds=args.max_wall_seconds)
            print(json.dumps({'command':'discover','status':result['status'],'expected_cases':result['expected_cases'],
                              'completed_pairs':result['completed_pairs'],'valid_memory_records':result['valid_memory_records']},ensure_ascii=False))
            return 4 if result['status']=='STOPPED' else 3 if args.require_complete and result['status']!='READY' else 0
        elif args.command=='cpu-test':
            result=cpu_test(args.output,args.basetemp); require(result['exit_code']==0,'CPU_TEST_FAILED')
        elif args.command=='gates':
            from .lvos_gates import run_gates
            with gpu_lease(args):
                result=run_gates(args.output,root=args.snapshot,lock=json_read(args.model_lock) if args.model_lock else None,
                    device=args.device,metric_contract=args.metric_contract,handoff_result=args.handoff_result,cpu_test_report=args.cpu_test_report,
                    max_wall_seconds=args.max_wall_seconds,direct_copy_result=args.direct_copy_result,
                    deadline=Deadline(args.max_wall_seconds,started=command_started))
            statuses={g['status'] for g in result['gates']}
            status='FAIL' if 'FAIL' in statuses else 'BLOCKED' if 'BLOCKED' in statuses else 'PASS'
            print(json.dumps({'command':'gates','output':str(args.output),'status':status,
                              'require_ready':args.require_ready,'unattended_training_ready':result['unattended_training_ready']},ensure_ascii=False))
            return 2 if status=='FAIL' else 3 if args.require_ready and not result['unattended_training_ready'] else 0
        elif args.command=='freeze-protocol':
            from .lvos_evaluation import freeze_protocol
            result=freeze_protocol(args.snapshot,args.source_manifest,json_read(args.runtime),args.output,
                scope=args.scope,videos=args.video_id,min_visible_frames=args.min_visible_frames)
        elif args.command in {'initialize','train','overfit','profile'}:
            from .lvos_training import LVOSConfig, train_snapshot
            cfg=LVOSConfig(**json_read(args.config))
            if args.max_wall_seconds is not None:
                cfg.max_wall_seconds=args.max_wall_seconds
            if args.max_micro_iterations is not None:
                cfg.max_micro_iterations=args.max_micro_iterations
            if args.quoted_hourly_rate is not None: cfg.quoted_hourly_rate=args.quoted_hourly_rate
            budget=Deadline(cfg.max_wall_seconds,started=command_started)
            callback=None
            with gpu_lease(args) as lease:
                if getattr(args,'evaluation_mode',None)=='sequential':
                    require(args.benchmark_root and args.monitor_protocol,'SEQUENTIAL_EVALUATOR_INPUTS')
                    from .lvos_jobs import run_worker
                    def callback(run,deadline):
                        value=run_worker(run,args.snapshot,args.benchmark_root,dry_run=False,approved=args.execute_approved,
                            device=args.device,max_wall_seconds=cfg.max_wall_seconds,deadline=deadline,merge=True,lease=lease)
                        require(value['status']=='DONE','SEQUENTIAL_EVALUATION_INCOMPLETE')
                result=train_snapshot(args.snapshot,json_read(args.model_lock),args.output,cfg,device=args.device,mode=args.command,
                    resume=args.resume,gates=args.gates,monitor_protocol=args.monitor_protocol,metric_contract=args.metric_contract,
                    deadline=budget,evaluation_callback=callback)
            run_status=json_read(args.output/'STATUS.json')
            if run_status.get('reason')=='budget_stop':
                print(json.dumps({'command':args.command,'status':'STOPPED','reason':'budget_stop','output':str(args.output)},ensure_ascii=False))
                return 4
        elif args.command=='evaluate':
            from .lvos_evaluation import evaluate
            with gpu_lease(args):
                result=evaluate(args.snapshot,args.protocol,args.checkpoint,args.metric_contract,args.benchmark_root,args.output,
                    shard_index=args.shard_index,shard_count=args.shard_count,method=args.method,device=args.device,approved=args.execute_approved,
                    max_wall_seconds=args.max_wall_seconds,deadline=Deadline(args.max_wall_seconds,started=command_started))
        elif args.command=='merge-eval':
            from .lvos_evaluation import merge_results
            result=merge_results(args.result,args.protocol,args.metric_contract,args.benchmark_root,args.output,replay=args.replay)
        elif args.command=='full-replay':
            from .lvos_benchmark import full_replay
            with gpu_lease(args):
                result=full_replay(args.snapshot,args.protocol,args.metric_contract,args.benchmark_root,args.baseline_root,args.output,
                    shard_index=args.shard_index,shard_count=args.shard_count,device=args.device,approved=args.execute_approved,
                    max_wall_seconds=args.max_wall_seconds,deadline=Deadline(args.max_wall_seconds,started=command_started))
        elif args.command=='compare':
            from .lvos_benchmark import comparison_table
            result=comparison_table(args.result,args.replay,args.protocol,args.metric_contract,args.benchmark_root,args.output)
        elif args.command=='shortlist':
            from .lvos_evaluation import shortlist
            result=shortlist(args.result,args.output)
        elif args.command=='select':
            from .lvos_evaluation import select_best
            result=select_best(args.result,args.protocol,args.metric_contract,args.output,shortlist_path=args.shortlist,benchmark_root=args.benchmark_root)
        elif args.command=='eval-worker':
            from .lvos_jobs import run_worker
            context=gpu_lease(args) if args.execute else nullcontext(None)
            with context as lease:
                result=run_worker(args.run,args.snapshot,args.benchmark_root,index=args.shard_index,count=args.shard_count,
                    dry_run=not args.execute,drain=args.drain,approved=args.execute_approved,device=args.device,
                    max_wall_seconds=args.max_wall_seconds,deadline=Deadline(args.max_wall_seconds,started=command_started),
                    merge=args.merge_shards,lease=lease)
            print(json.dumps(result,ensure_ascii=False))
            return 4 if result['status']=='STOPPED' else 2 if result['status']=='FAIL' else 0
        elif args.command=='eval-merge':
            from .lvos_jobs import merge_request
            result=merge_request(args.run,args.request,args.benchmark_root,count=args.shard_count)
        elif args.command=='eval-status':
            from .lvos_jobs import pending_requests
            pending=pending_requests(args.run)
            result={'training':json_read(args.run/'STATUS.json'),'evaluation_status':'PENDING' if pending else 'COMPLETED',
                    'pending_requests':pending,'promotion_status':'requires complete full-dev shortlist select'}
            require(not args.output.exists(),'STATUS_OUTPUT_EXISTS'); write_json(args.output,result)
        elif args.command=='bundle':
            from .lvos_release import bundle_release
            result=bundle_release(args.release,args.run,args.output,include_resume=args.include_resume,allow_synthetic=args.allow_synthetic)
        elif args.command=='launch-plan':
            result=launch_plan(json_read(args.jobs),args.output)
        elif args.command=='draft-export':
            draft_export(args.run,args.result,args.output); result={'output':str(args.output)}
        print(json.dumps({'command':args.command,'output':str(getattr(args,'output','')),'status':'DONE'},ensure_ascii=False))
        return 0
    except BudgetStop as exc:
        if getattr(args,'output',None):
            write_json(args.output/'STOPPED.json',{'status':'STOPPED','reason':'budget_stop','incomplete_stage':exc.stage})
        print(json.dumps({'command':args.command,'status':'STOPPED','reason':'budget_stop','incomplete_stage':exc.stage},ensure_ascii=False))
        return 4
    except (ValueError,OSError,KeyError,RuntimeError,TypeError) as exc:
        detail={'command':args.command,'status':'FAIL','reason_code':getattr(exc,'code','INPUT_OR_EXECUTION_ERROR'),'detail':str(exc)}
        print(json.dumps(detail,ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
