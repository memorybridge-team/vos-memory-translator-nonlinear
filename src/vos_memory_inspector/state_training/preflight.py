"""Readiness means full indexed data verified, never a successful sample run."""
import math
from pathlib import Path

from .data import load_pair, resident_bytes

REFERENCE_INDEX_SHA256 = '3ae8b00b5cfdf7df624a6c1b476d95ceee7ce259f796338b02975c201556ae9c'


def inspect(index, config, *, sample_cases=None):
    config.validate()
    missing, available, failures = [], [], []
    for entry in index.entries:
        if not Path(entry['path']).is_file() or not Path(entry['path'] + '.sha256').is_file():
            missing.append(dict(case_id=entry['case']['case_id'], path=entry['path']))
        else:
            available.append(entry)
    # Spread samples across datasets/roles rather than selecting only the first prefix.
    groups = {}
    for entry in available:
        groups.setdefault((entry['case']['dataset'], entry['case']['training_role'],
                           entry['case'].get('origin_paired_split')), []).append(entry)
    interleaved = []
    for offset in range(max((len(g) for g in groups.values()), default=0)):
        interleaved.extend(g[offset] for g in groups.values() if offset < len(g))
    selected = interleaved if sample_cases is None else interleaved[:sample_cases]
    sums = {'spatial': [0., 0], 'pointer': [0., 0]}
    verified, records, peak_case = 0, 0, 0
    spec = None
    for entry in selected:
        try:
            value, _ = load_pair(entry, expected_spec=spec)
            s, t = value['source_canonical'], value['target_canonical']
            spec = s.spec
            if index.official and (spec.feature_channels, spec.height, spec.width, spec.pointer_dim) != (64, 64, 64, 256):
                raise ValueError('official SAM2 memory tensor spec mismatch')
            verified += 1
            records += entry['records']
            peak_case = max(peak_case, resident_bytes(value))
            if entry['case']['training_role'] == 'train':
                for name, tensor in (('spatial', t.spatial_memory), ('pointer', t.object_pointer)):
                    x = tensor[t.validity].double()
                    sums[name][0] += float(x.square().sum())
                    sums[name][1] += x.numel()
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as exc:
            failures.append(dict(case_id=entry['case']['case_id'], path=entry['path'], error=str(exc)))
    complete = not missing and not failures and verified == len(index.entries)
    scales = {k: math.sqrt(s / n) for k, (s, n) in sums.items() if n}
    normalization = dict(scope='train_only' if complete else 'sample_only', scales=scales, sums=sums)
    if complete and (len(scales) != 2 or any(v <= 0 or not math.isfinite(v) for v in scales.values())):
        failures.append(dict(error='train RMS is zero or invalid'))
        complete = False
    if complete and index.normalization:
        prior = index.normalization
        if prior.get('scope') != 'train_only' or any(
            not math.isclose(scales[k], prior['scales'][k], rel_tol=1e-6, abs_tol=1e-10) for k in scales):
            failures.append(dict(error='recomputed train RMS differs from official index'))
            complete = False
    per_record = 2 * 64 ** 3 * 2 + 2 * 256 * 4
    if spec is not None:
        per_record = 2 * spec.feature_channels * spec.height * spec.width * 2 + 2 * spec.pointer_dim * 4
    workers = max(config.workers, 1)
    queued = max(1, config.workers * config.prefetch_factor) * config.microbatch * per_record
    return dict(schema_version='cmmt.state_training_preflight.v1', data_identity=index.identity,
                official=index.official, full_data_ready=complete and index.official,
                sample_data_ready=complete and not index.official,
                checked_cases=verified, checked_records=records, expected_cases=len(index.entries),
                missing=missing, invalid=failures, normalization=normalization,
                source_spec=spec.to_dict() if spec is not None else None,
                memory=dict(worker_cache_ceiling_bytes=workers * config.cache_bytes,
                            queued_tensor_bytes=queued, observed_peak_case_bytes=peak_case,
                            note='per process; excludes model, activations, optimizer and Python overhead'),
                historical_comparison_data_match=complete and index.official and
                    index.identity['index_sha256'] == REFERENCE_INDEX_SHA256,
                selective_handoff_verified=False, cuda_training_executed=False)


def require_ready(report, *, allow_sample=False):
    if report['invalid'] or report['missing'] or not (
        report['full_data_ready'] or (allow_sample and report['sample_data_ready'])):
        raise ValueError('data is not ready: run full preflight; sample success does not authorize official training')
