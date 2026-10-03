"""고정 best_model.selection_score를 final/table 공통 reducer로 사용."""
from collections import defaultdict
import math
from .collection_contract import require


PRIMARY = 'mean_video_retention_three_fractions_percent'
MONITOR = 'monitor_jf_proxy'


def raw_macro(rows):
    result={}
    for key in ('j','f','jf'):
        groups=defaultdict(list)
        for row in rows:
            if row.get(key) is not None: groups[row['video']].append(row[key])
        result[key]=sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None
    return result


def canonical_retention(metric, method_rows, replay_rows):
    method={r['case_id']:r for r in method_rows}; replay={r['case_id']:r for r in replay_rows}
    require(len(method)==len(method_rows) and len(replay)==len(replay_rows) and method.keys()==replay.keys(),
            'RETENTION_CASE_COVERAGE')
    exclusions=[]; m=[]; p=[]
    for key,row in method.items():
        baseline=replay[key]
        require(all(row.get(k)==baseline.get(k) for k in ('video','fraction','object_id')), 'RETENTION_CASE_IDENTITY')
        require((row['jf'] is None)==(baseline['jf'] is None),'RETENTION_UNDEFINED_MISMATCH')
        if row['jf'] is None:
            exclusions.append({'case_id':key,'reason':'NO_VISIBLE_GT'}); continue
        require(math.isfinite(row['jf']) and math.isfinite(baseline['jf']),'RETENTION_NONFINITE')
        m.append(row); p.append(baseline)
    fractions=[.25,.5,.75]
    require({r['fraction'] for r in method_rows}==set(fractions), 'RETENTION_FRACTION_COVERAGE')
    eligible={}
    for fraction in fractions:
        videos=metric.video_scores(p,fraction)
        eligible[str(fraction)]=sorted(v for v,x in videos.items() if x>0)
        exclusions.extend({'video':v,'fraction':fraction,'reason':'ZERO_REPLAY'} for v,x in videos.items() if x<=0)
    if any(not values for values in eligible.values()):
        return {'primary_metric':PRIMARY,'score':None,'reason':'NO_DEFINED_NONZERO_REPLAY_VIDEO_FOR_FRACTION',
                'eligible_videos':eligible,'exclusions':exclusions,'raw_method':raw_macro(method_rows),'raw_replay':raw_macro(replay_rows)}
    value=metric.selection_score(m,p)  # canonical object -> video ratio -> fraction order; no clipping.
    require(all(math.isfinite(x) for x in value.values()),'RETENTION_NONFINITE')
    return {**value,'primary_metric':PRIMARY,'reason':None,'eligible_videos':eligible,'exclusions':exclusions,
            'raw_method':raw_macro(method_rows),'raw_replay':raw_macro(replay_rows)}
