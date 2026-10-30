"""Preflight, ridge fit, CPU smoke and explicit full-data training commands."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .data import load_index
from .determinism import configure_determinism, execution_profile
from .policy import TrainConfig
from .preflight import inspect, require_ready, validate_readiness
from .ridge import fit_ridge, load_ridge, StreamingRidgeTranslator
from .storage import save_json, digest
from .trainer import train, construct_model
from .smoke import cpu_smoke


def _json(path):
    return json.loads(Path(path).read_text())


def _options(args):
    raw = _json(args.config) if args.config else {}
    if set(raw) - {'training', 'model', 'model_options'}:
        raise ValueError('unknown experiment config section')
    config = TrainConfig(**raw.get('training', {}))
    config.validate()
    return config, args.model or raw.get('model', 'transformer'), raw.get('model_options', {})


def parser():
    p = argparse.ArgumentParser(prog='cmmt-state-training')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'ridge-fit', 'train', 'cpu-smoke', 'evaluate'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--index', type=Path, required=True)
        cmd.add_argument('--root-map', type=Path, help='JSON object mapping original path prefixes to local roots')
        cmd.add_argument('--config', type=Path)
        cmd.add_argument('--output', type=Path, required=True)
        cmd.add_argument('--model', choices=('transformer', 'residual_mlp', 'mlp_context', 'ridge') if name == 'evaluate' else ('transformer', 'residual_mlp', 'mlp_context'))
        cmd.add_argument('--ridge-init', type=Path)
        if name in ('preflight', 'cpu-smoke', 'ridge-fit', 'evaluate'):
            cmd.add_argument('--sample-index', action='store_true', help='allow explicitly labelled CPU-only fixture indexes')
        if name in ('preflight', 'cpu-smoke'):
            cmd.add_argument('--sample-cases', type=int, help='inspect only N available cases; cannot make full_data_ready true')
        if name == 'evaluate':
            cmd.add_argument('--model-artifact', type=Path, required=True)
        if name == 'preflight':
            cmd.add_argument('--checkpoint', type=Path, help='optional epoch checkpoint to inspect for resume')
            cmd.add_argument('--world-size', type=int, default=1, help='planned training process count')
        if name in ('train', 'ridge-fit', 'evaluate'):
            cmd.add_argument('--preflight-report', type=Path, required=name == 'train')
        if name == 'train':
            cmd.add_argument('--distributed-timeout-seconds', type=float, default=180.)
            cmd.add_argument('--device', required=True, choices=('cpu', 'cuda'))
            cmd.add_argument('--resume', type=Path)
        if name == 'cpu-smoke':
            cmd.add_argument('--steps', type=int, default=2)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config, name, options = _options(args)
    if args.command == 'preflight' and args.sample_cases is not None and args.sample_cases < 1:
        raise ValueError('--sample-cases must be positive')
    index = load_index(args.index, _json(args.root_map) if args.root_map else None,
                       allow_sample=getattr(args, 'sample_index', False))
    # Every command writes only derived artifacts outside original cache folders.
    if any(args.output.resolve().is_relative_to(Path(e['path']).parent) for e in index.entries):
        raise ValueError('output must be outside cache roots')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if args.command != 'train' and world != 1:
        raise ValueError('use one process for preflight, ridge-fit and CPU smoke')
    if args.command == 'preflight':
        config.validate(args.world_size)
        report = inspect(index, config, sample_cases=args.sample_cases)
        report['planned_world_size'] = args.world_size
        if args.ridge_init:
            ridge = load_ridge(args.ridge_init)
            report['ridge_initialization_ready'] = (options.get('skip_mode', 'identity') == 'affine' and
                ridge['provenance']['data_identity'] == index.identity and
                ridge['source_spec'] == report['source_spec'] and ridge['target_spec'] == report['source_spec'])
        else:
            report['ridge_initialization_ready'] = options.get('skip_mode', 'identity') != 'affine'
        report['model_configuration'] = dict(name=name, options=options)
        if report['source_spec']:
            from ..state_schema import StateSpec
            model = construct_model(name, StateSpec(**report['source_spec']), options)
            if args.ridge_init and report['ridge_initialization_ready']:
                model.initialize_from_ridge(ridge)
            report['model_parameters'] = model.parameter_count()
            report['memory']['model_parameters_fp32_bytes'] = model.parameter_count() * 4
            report['memory']['optimizer_and_gradient_estimate_bytes'] = model.parameter_count() * 12
        report['checkpoint_ready'] = args.checkpoint is None
        if args.checkpoint:
            try:
                checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
                identity = checkpoint['identity']
                from .trainer import restore_model
                restored = restore_model(name, checkpoint['model'])
                report['checkpoint_ready'] = (
                    checkpoint.get('schema_version') == 'cmmt.record_training_checkpoint.v1' and
                    identity['data'] == index.identity and identity['config'] == config.logical() and
                    identity['model'] == name and identity['options'] == options and
                    identity.get('determinism') == execution_profile(
                        'cuda' if identity.get('determinism', {}).get('cublas_workspace_config') else 'cpu') and
                    identity['world_size'] == args.world_size and
                    identity['ridge_sha256'] == (ridge['provenance']['artifact_sha256'] if args.ridge_init else None) and
                    restored.source_spec.to_dict() == report['source_spec'] and
                    checkpoint['signature'] == digest(identity)
                )
                report['checkpoint_epoch'] = checkpoint['epoch']
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                report['checkpoint_ready'] = False
                report['checkpoint_error'] = str(exc)
        report['launch_ready'] = report['full_data_ready'] and report['ridge_initialization_ready'] and report['checkpoint_ready']
        save_json(args.output, report)
        print(json.dumps({k: report[k] for k in ('full_data_ready', 'checked_cases', 'expected_cases', 'ridge_initialization_ready')}))
        return 0 if report['launch_ready'] else 2
    if args.command == 'evaluate':
        from .evaluation import evaluate_artifact
        evaluate_artifact(index, config, name, args.model_artifact, args.output, allow_sample=args.sample_index,
                          readiness=_json(args.preflight_report) if args.preflight_report else None)
        return 0
    if args.command == 'ridge-fit':
        report = (validate_readiness(index, _json(args.preflight_report), allow_sample=args.sample_index).report
                  if args.preflight_report else inspect(index, config))
        require_ready(report, allow_sample=args.sample_index)
        if args.output.exists():
            raise ValueError('ridge output exists; choose a new file')
        ridge = fit_ridge(index, config, args.output)
        print(json.dumps(dict(artifact=str(args.output), sha256=ridge['provenance']['artifact_sha256'])))
        return 0
    if args.command == 'cpu-smoke':
        result = cpu_smoke(index, config, name, args.output, options=options,
                           ridge_path=args.ridge_init, sample_cases=8 if args.sample_cases is None else args.sample_cases, steps=args.steps)
        print(json.dumps(dict(checked_cases=len(result['samples']), cuda_training_executed=False)))
        return 0
    # Expensive standalone preflight must finish before torchrun. Each process
    # checks its local file metadata before creating any process group.
    if not math.isfinite(args.distributed_timeout_seconds) or args.distributed_timeout_seconds <= 0:
        raise ValueError('distributed timeout must be positive and finite')
    readiness = validate_readiness(index, _json(args.preflight_report))
    configure_determinism(args.device)
    initialized_here = False
    try:
        if args.device == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA unavailable')
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            torch.cuda.set_device(local_rank)
            device = f'cuda:{local_rank}'
        else:
            device = 'cpu'
        if world > 1:
            from datetime import timedelta
            dist.init_process_group(backend='nccl' if args.device == 'cuda' else 'gloo',
                                    timeout=timedelta(seconds=args.distributed_timeout_seconds))
            initialized_here = True
        train(index, config, name, args.output, options=options, ridge_path=args.ridge_init,
              device=device, resume=args.resume, readiness=readiness)
    finally:
        if initialized_here:
            dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
