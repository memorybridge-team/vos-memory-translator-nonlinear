"""Readiness means full indexed data verified, never a successful sample run."""
from dataclasses import dataclass
import math
from pathlib import Path

from .data import load_pair, resident_bytes, file_stamp, sidecar_hash
from .storage import digest

REPORT_SCHEMA = 'cmmt.state_training_preflight.v2'
RECEIPT_FIELDS = ('schema_version', 'data_identity', 'official', 'full_data_ready',
                  'sample_data_ready', 'checked_cases', 'checked_records', 'expected_cases',
                  'missing', 'invalid', 'normalization', 'source_spec', 'files')

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
    files = []
    spec = None
    for entry in selected:
        try:
            value, stamp = load_pair({k: v for k, v in entry.items() if k != 'verified_stamp'}, expected_spec=spec)
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
            files.append(dict(case_id=entry['case']['case_id'], sha256=entry['sha256'],
                              stamp=list(stamp)))
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
    report = dict(schema_version=REPORT_SCHEMA, data_identity=index.identity, files=files,
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
    report['receipt_digest'] = digest({key: report[key] for key in RECEIPT_FIELDS})
    return report


def require_ready(report, *, allow_sample=False):
    if report['invalid'] or report['missing'] or not (
        report['full_data_ready'] or (allow_sample and report['sample_data_ready'])):
        raise ValueError('data is not ready: run full preflight; sample success does not authorize official training')


@dataclass(frozen=True)
class ValidatedPreflight:
    """In-process evidence of metadata checking before process-group creation."""
    index: object
    report: dict
    roles: tuple


def normalization_scales(normalization):
    if not isinstance(normalization, dict) or normalization.get('scope') != 'train_only':
        raise ValueError('verified train_only normalization required')
    scales = normalization.get('scales', {})
    if set(scales) != {'spatial', 'pointer'} or any(
        not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
        for v in scales.values()
    ):
        raise ValueError('positive finite spatial/pointer normalization scales required')
    return scales


def bind_stamps(index, report):
    receipts = {item['case_id']: tuple(item['stamp']) for item in report['files']}
    for entry in index.entries:
        if entry['case']['case_id'] in receipts:
            entry['verified_stamp'] = receipts[entry['case']['case_id']]


def validate_readiness(index, report, *, roles=('train', 'validation'), allow_sample=False):
    """Check only metadata and tiny sidecars. Never hash or load pair tensors.

    Full hashing/RMS computation belongs to the standalone preflight command.
    Stamps bind this report to the inspected files; the loader still verifies
    SHA256 and tensor contracts whenever a case is loaded during training.
    """
    if isinstance(report, ValidatedPreflight):
        if report.index is not index or not set(roles).issubset(report.roles):
            raise ValueError('validated preflight index/roles mismatch')
        return report
    if not isinstance(report, dict) or report.get('schema_version') != REPORT_SCHEMA:
        raise ValueError('preflight v2 report required; rerun standalone preflight')
    if any(key not in report for key in RECEIPT_FIELDS) or report.get('receipt_digest') != digest(
        {key: report[key] for key in RECEIPT_FIELDS}
    ):
        raise ValueError('preflight receipt digest mismatch')
    require_ready(report, allow_sample=allow_sample)
    if report['data_identity'] != index.identity or report['official'] != index.official:
        raise ValueError('preflight index identity mismatch')
    if report['expected_cases'] != len(index.entries) or report['checked_cases'] != len(index.entries):
        raise ValueError('preflight case coverage mismatch')
    if report['checked_records'] != sum(e['records'] for e in index.entries):
        raise ValueError('preflight record coverage mismatch')
    if len(report['files']) != len(index.entries):
        raise ValueError('preflight file coverage mismatch')
    files = {item['case_id']: item for item in report['files']}
    if len(files) != len(index.entries) or set(files) != {e['case']['case_id'] for e in index.entries}:
        raise ValueError('preflight file identity mismatch')
    spec = report['source_spec']
    from ..state_schema import StateSpec
    parsed_spec = StateSpec(**spec)
    if index.official and (parsed_spec.feature_channels, parsed_spec.height, parsed_spec.width,
                           parsed_spec.pointer_dim) != (64, 64, 64, 256):
        raise ValueError('official preflight tensor spec mismatch')
    scales = normalization_scales(report['normalization'])
    train_records = sum(e['records'] for e in index.role('train'))
    expected_elements = dict(spatial=train_records * parsed_spec.feature_channels * parsed_spec.height * parsed_spec.width,
                             pointer=train_records * parsed_spec.pointer_dim)
    for key, count in expected_elements.items():
        total, observed_count = report['normalization']['sums'][key]
        if observed_count != count or count < 1 or not math.isfinite(total) or total <= 0:
            raise ValueError('preflight normalization observation mismatch')
        if not math.isclose(scales[key] ** 2, total / count, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('preflight normalization sums/scales mismatch')
    if index.normalization:
        declared = normalization_scales(index.normalization)
        if any(not math.isclose(scales[key], declared[key], rel_tol=1e-6, abs_tol=1e-10) for key in scales):
            raise ValueError('preflight normalization differs from index')
    for entry in index.entries:
        receipt = files[entry['case']['case_id']]
        stamp = receipt['stamp']
        if receipt['sha256'] != entry['sha256'] or len(stamp) != 2 or any(
            not isinstance(value, int) or value < 0 for value in stamp
        ):
            raise ValueError('preflight file hash/stamp mismatch')
        if entry['case']['training_role'] not in roles:
            continue
        path = entry['path']
        if file_stamp(path) != tuple(stamp) or Path(path + '.writer.lock').exists():
            raise ValueError(f'file changed after preflight: {path}')
        if sidecar_hash(path + '.sha256') != entry['sha256']:
            raise ValueError(f'checksum sidecar changed after preflight: {path}')
    bind_stamps(index, report)
    return ValidatedPreflight(index, report, tuple(roles))
