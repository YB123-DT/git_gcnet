"""Strict original Flat / frozen Nested replay using protected GCNet masks.

This entry currently shares the approved MOSEI seed66 runtime, not a universal
cross-dataset evaluator. It never starts stage-two training. A single parent
checkpoint is replayed across all eight rates; its mean must not be confused
with the per-rate selected-checkpoint training summary.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path


def validate_parent_config(expected, recorded):
    """State keys cannot detect raw Gap / residual Gap and other flag changes."""
    left = json.loads(json.dumps(expected, sort_keys=True))
    right = json.loads(json.dumps(recorded, sort_keys=True))
    changed = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key)
                     or (key in left) != (key in right))
    if changed:
        raise ValueError('Parent config differs from selected unified variant: ' + ', '.join(changed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('A0', 'A1', 'A2', 'A3'), default='A3')
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--parent-config', type=Path, required=True,
                        help='Complete native TrainConfig JSON from this parent run')
    parser.add_argument('--nested-checkpoint', type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-batches', type=int, help='Diagnostic subset only, not full-test scores')
    args = parser.parse_args(argv)
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Replay output must be new/empty; never overwrite predictions')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    from . import protocol, runtime
    from .stage2 import FrozenNestedModel, restore_parent, restore_block

    bundle = runtime.setup(args.variant, args.output, args.gpu_uuid)
    recorded = json.loads(args.parent_config.read_text())
    recorded_cfg = bundle['trainer'].TrainConfig(**recorded)
    validate_parent_config(asdict(bundle['cfg']), asdict(recorded_cfg))
    parent = bundle['model']
    parent_identity = restore_parent(parent, args.parent_checkpoint,
                                     expected_sha256=args.parent_sha256)
    parent_identity['config_path'] = str(args.parent_config.resolve())
    parent_identity['config_sha256'] = runtime.sha(args.parent_config)
    saved = torch.load(args.parent_checkpoint, map_location='cpu', weights_only=False)
    if 'config' in saved:
        validate_parent_config(asdict(recorded_cfg),
            asdict(bundle['trainer'].TrainConfig(**saved['config'])))
    del saved
    nested_identity = None
    model = parent
    if args.nested_checkpoint is not None:
        with runtime.isolated_rng(66):
            model = FrozenNestedModel(parent).to('cuda:0')
        nested_identity = restore_block(model, args.nested_checkpoint, parent_identity['sha256'])
    metrics = {}
    for rate in protocol.RATES:
        metrics[rate], arrays = runtime.evaluate(bundle, model, rate, args.max_batches)
        np.savez_compressed(args.output / f'rate_{rate.replace(".", "p")}.npz', **arrays)
    report = dict(status='complete', kind='SINGLE_CHECKPOINT_EXACT_MASK_REPLAY',
        label='INTERNAL TEST-ORACLE CHECKPOINT; paired frozen replay, not independent generalization',
        variant=args.variant, parent=parent_identity, nested=nested_identity,
        config=asdict(bundle['cfg']), environment=bundle['environment'], gpu=bundle['gpu'],
        identity=bundle['identity'], rates=metrics, aggregate=protocol.aggregate(metrics),
        partial_test=args.max_batches is not None, max_batches=args.max_batches)
    runtime.save_json(args.output / 'REPLAY.json', report)
    print(json.dumps(report['aggregate'], indent=2))


if __name__ == '__main__':
    main()
