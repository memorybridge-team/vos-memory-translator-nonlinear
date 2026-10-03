"""작은 reproducibility bundle. Explicit files only; cache/predictions/secrets scan 없음."""
from pathlib import Path
import zipfile
import shutil
from .collection_contract import require
from .lvos_contract import json_read, load_complete, MODEL_REVISION, BENCHMARK_REVISION, BASELINE_REVISION
from .training_storage import sha256, write_json, ExclusiveWriter
from .transformer_translator import TransformerStateTranslator


def bundle_release(release,run,output,*,include_resume=False,allow_synthetic=False):
    release,run,output=[Path(p).resolve() for p in (release,run,output)]
    require(not output.is_relative_to(release) and not output.is_relative_to(run),'BUNDLE_INSIDE_SOURCE')
    payload,entry=load_complete(release/'best_model.pth')
    TransformerStateTranslator.from_payload(payload)
    selection=json_read(release/'selection.json'); config=json_read(run/'config.json')
    synthetic=selection.get('execution_kind')=='cpu_synthetic'
    require(not synthetic or allow_synthetic,'SYNTHETIC_BUNDLE_EXPLICIT_ONLY')
    require(selection.get('selection_claim')=='best among evaluated shortlist; not global best epoch','BUNDLE_SELECTION_CLAIM')
    files={'best_model.pth':release/'best_model.pth','best_model.pth.complete.json':Path(str(release/'best_model.pth')+'.complete.json'),
           'selection.json':release/'selection.json'}
    if include_resume:
        last=json_read(run/'last.ckpt.json'); path=(run/last['path']).resolve()
        require(path.is_relative_to(run/'checkpoints'),'BUNDLE_RESUME_PATH')
        _,proof=load_complete(path); require(proof['sha256']==last['sha256'],'BUNDLE_RESUME_CHANGED')
        files['resume.ckpt']=path
    with ExclusiveWriter(output):
        require(not (output/'artifact-index.json').exists() and not (output/'bundle.zip').exists(),'BUNDLE_EXISTS')
        index=[]
        for name,path in files.items():
            require(path.stat().st_size<=32*1024**2,'BUNDLE_FILE_TOO_LARGE')
            digest=sha256(path); dest=output/'files'/name
            dest.parent.mkdir(exist_ok=True); shutil.copyfile(path,dest)
            require(sha256(path)==sha256(dest)==digest,'BUNDLE_SOURCE_CHANGED')
            index.append({'logical_path':'files/'+name,'bytes':dest.stat().st_size,'sha256':digest})
        if include_resume:
            last_entry=next(e for e in index if e['logical_path']=='files/resume.ckpt')
            write_json(output/'files/resume.ckpt.complete.json',{'path':'resume.ckpt','bytes':last_entry['bytes'],
                'sha256':last_entry['sha256'],'schema_version':'cmmt.ready_artifact.v1'})
            p=output/'files/resume.ckpt.complete.json'
            index.append({'logical_path':'files/'+p.name,'bytes':p.stat().st_size,'sha256':sha256(p)})
        record={'schema_version':'cmmt.lvos_artifact_index.v1','execution_kind':'cpu_synthetic' if synthetic else 'gpu_real_checkpoint',
            'model_revision':MODEL_REVISION,'metric_revision':BENCHMARK_REVISION,'baseline_revision':BASELINE_REVISION,
            'run_source_sha256':config['identity']['code_sha'],'model_config_digest':config['identity']['model_config_digest'],
            'training_revision':config['identity'].get('code_revision','unavailable'),
            'collection_digest':config['identity']['collection_digest'],
            'protocol_digest':selection.get('protocol_digest'),
            'monitor_protocol_digest':config['identity'].get('monitor_protocol_digest'),
            'metric_contract_digest':config['identity'].get('metric_contract_digest'),
            'selection_claim':selection['selection_claim'],'coverage':selection.get('coverage','selection.json evidence'),
            'checkpoint_sha256':entry['sha256'],'files':index,'sanitized':True}
        write_json(output/'artifact-index.json',record)
        with zipfile.ZipFile(output/'bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output/'artifact-index.json','artifact-index.json')
            for value in index: archive.write(output/value['logical_path'],value['logical_path'])
        proof={'bytes':(output/'bundle.zip').stat().st_size,'sha256':sha256(output/'bundle.zip'),'index_sha256':sha256(output/'artifact-index.json')}
        write_json(output/'checksums.json',proof); return record
