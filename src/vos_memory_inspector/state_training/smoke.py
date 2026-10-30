"""Bounded CPU checks on available real pairs without inventing a validation split."""
from pathlib import Path
import torch

from .data import load_pair
from .preflight import inspect
from .ridge import collect_statistics, load_ridge, SCHEMA
from .statistics import solve
from .storage import save_json, save_torch
from .trainer import construct_model


def cpu_smoke(index, config, name, output, *, options=None, ridge_path=None, sample_cases=8, steps=2):
    if sample_cases < 1 or steps < 1:
        raise ValueError('positive sample cases/steps required')
    output = Path(output)
    if any(output.resolve().is_relative_to(Path(e['path']).parent) for e in index.entries):
        raise ValueError('CPU smoke output must be outside original cache roots')
    if output.exists() and any(output.iterdir()):
        raise ValueError('CPU smoke output must be empty')
    config.workers = 0
    report = inspect(index, config, sample_cases=sample_cases)
    if report['invalid']:
        raise ValueError(f"CPU sample verification failed: {report['invalid']}")
    groups = {}
    for e in index.role('train'):
        if Path(e['path']).is_file() and Path(e['path'] + '.sha256').is_file():
            groups.setdefault((e['case']['dataset'], e['case'].get('origin_paired_split')), []).append(e)
    selected = []
    for offset in range(max((len(g) for g in groups.values()), default=0)):
        selected.extend(g[offset] for g in groups.values() if offset < len(g))
        if len(selected) >= sample_cases:
            break
    selected = selected[:sample_cases]
    if not selected:
        raise ValueError('no available train samples')
    pair, _ = load_pair(selected[0])
    spec = pair['source_canonical'].spec
    torch.manual_seed(config.seed)
    model = construct_model(name, spec, options)
    if getattr(model, 'skip_mode', 'identity') == 'affine':
        if ridge_path:
            ridge = load_ridge(ridge_path)
            if ridge['provenance']['data_identity'] != index.identity:
                raise ValueError('ridge data identity mismatch')
        else:
            # This initializes a smoke model only. It is not validation-selected
            # and its identity deliberately cannot match official training.
            _, stats = collect_statistics(selected, config)
            components = {}
            for c in ('spatial', 'pointer'):
                a = solve(stats[c], 1e-4)
                components[c] = dict(weight=a.weight.float(), bias=a.bias.float(), diagnostics=a.diagnostics)
            init_path = output / 'sample_initialization.pth'
            save_torch(init_path, dict(schema_version=SCHEMA, source_spec=spec.to_dict(), target_spec=spec.to_dict(),
                       **components, provenance=dict(data_identity={'sample_only': True, 'parent_index': index.identity,
                       'cases': [e['case']['case_id'] for e in selected]}, selection_role='none_cpu_smoke',
                       refit_after_selection=False)))
            ridge = load_ridge(init_path)
        model.initialize_from_ridge(ridge)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    results = []
    for e in selected:
        value, _ = load_pair(e, expected_spec=spec)
        s, t = value['source_canonical'], value['target_canonical']
        first, last = None, None
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            a, p = model.translate_tensors(s.spatial_memory, s.object_pointer, s.validity)
            loss = (a[s.validity] - t.spatial_memory[s.validity].float()).square().mean()
            loss = loss + (p[s.validity] - t.object_pointer[s.validity]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite CPU smoke loss')
            loss.backward()
            if any(g.grad is not None and not torch.isfinite(g.grad).all() for g in model.parameters()):
                raise ValueError('nonfinite CPU smoke gradient')
            first = float(loss.detach()) if first is None else first
            last = float(loss.detach())
            optimizer.step()
        results.append(dict(case_id=e['case']['case_id'], runtime_switch=s.switch_frame,
                            indexed_frame=e['case']['switch_frame'], records=e['records'],
                            initial_raw_loss=first, final_raw_loss=last, finite_forward_backward=True))
    result = dict(schema_version='cmmt.state_training_cpu_smoke.v1', model=name, options=options or {},
                  samples=results, preflight=report, cuda_training_executed=False,
                  official_training_completed=False, validation_or_benchmark_score=False,
                  official_ridge_initialization_used=ridge_path is not None,
                  selective_handoff_verified=False)
    save_json(output / 'cpu_smoke_report.json', result)
    return result
