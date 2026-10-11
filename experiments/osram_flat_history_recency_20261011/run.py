"""Seed66 screen for strict-past modality history metadata in original Flat."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from experiments.osram_complete150_20261011 import run as continuation
from experiments.osram_complete150_20261011 import recover_run as recovery
from experiments.osram_complete150_20261011.run import now, read, sha, write


LABEL = 'INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle'
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66')
DATA = Path('/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json')


def specs():
    return [dict(id='history_recency_seed66', dataset='CMUMOSI',
                 model='history_recency', seed=66, fold=1)]


def schedule():
    return ((100, .001), (150, .0001))


def candidate_config(reference):
    expected = dict(
        dataset='CMUMOSI', seed=66, epochs=100, learning_rate=.001,
        lr_schedule='constant', backbone_type='osram', training_objective='emotion-only',
        checkpoint_selection='test-oracle-per-rate', train_rate_mode='cyclic',
        osram_readout_fusion='flat', osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
        osram_bidirectional=False, disable_unused_aux_modules=True,
    )
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError(f'Unexpected cfg84 {key}')
    forbidden = (
        'paired_history_views', 'osram_relation_block', 'osram_base_first_gap',
        'osram_post_grn', 'classification_completion', 'completion_write_to_memory',
        'osram_decision_correction', 'osram_gap_increment_filter',
        'osram_local_evidence_gate', 'osram_hierarchical_evidence_gate',
        'osram_memory_only_adapter', 'osram_ced_block', 'osram_history_recency',
    )
    if any(reference.get(key, False) for key in forbidden):
        raise ValueError('Reference contains a competing method')
    if reference.get('osram_meaningful_block', 'none') != 'none':
        raise ValueError('Reference contains a meaningful block')
    if reference.get('osram_readout_candidate', 'none') != 'none':
        raise ValueError('Reference contains a readout candidate')
    if reference.get('osram_counterfactual_readout', 'none') != 'none':
        raise ValueError('Reference contains a counterfactual readout')
    return dict(reference, osram_history_recency=True)


def _source_hashes(source):
    names = ['config.py']
    for directory in (
        'gcnet_missing_m3', 'gcnet_modality_jepa', 'gcnet',
        'gcnet_jepa_replacement', 'gcnet_plci_jepa',
        'experiments/osram_complete150_20261011',
        'experiments/osram_flat_history_recency_20261011',
    ):
        names.extend(str(path.relative_to(source)) for path in (source / directory).rglob('*.py'))
    return {name: sha(source / name) for name in sorted(set(names))}


def prepare(root, commit):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    if (root / 'MANIFEST.json').exists():
        raise ValueError('Existing experiment, do not overwrite')
    source = Path(__file__).resolve().parents[2]
    base = asdict(TrainConfig(**read(REFERENCE / 'config.json')))
    data = read(DATA)
    verified = {}
    for name, digest in data['files'].items():
        if sha(name) != digest:
            raise ValueError(f'Data hash mismatch: {name}')
        stat = Path(name).stat()
        verified[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest)
    config = asdict(TrainConfig(**candidate_config(base)))
    delta = {key: dict(baseline=base[key], candidate=value)
             for key, value in config.items() if value != base[key]}
    expected_delta = {'osram_history_recency': dict(baseline=False, candidate=True)}
    if delta != expected_delta:
        raise ValueError(f'Unexpected config delta: {delta}')
    task = dict(specs()[0], reference=str(REFERENCE), original=None,
        source=str(source), config_path=str(REFERENCE / 'config.json'),
        config_sha256=sha(REFERENCE / 'config.json'), config=config,
        configuration_delta=delta, mode='fresh100-then-full-state50')
    manifest = dict(label=LABEL, runner_commit=commit,
        runner_sha256=sha(continuation.__file__), tasks=[task],
        data={'CMUMOSI': data}, data_verified=verified,
        sources={str(source): dict(commit=commit, hashes=_source_hashes(source))},
        prepared_at=now(), environment=dict(
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            host=platform.node()))
    write(root / 'MANIFEST.json', manifest)
    compatibility = root / 'source_4eed138/run.py'
    compatibility.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(continuation.__file__, compatibility)


def smoke(root, gpu_uuid):
    import torch
    from types import SimpleNamespace
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_missing_m3.mc_fwc import forward_view, _task_loss
    from gcnet_missing_m3.train_mc_fwc import make_loaders, build_model
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal the verified host UUID')
    cfg = trainer.TrainConfig(**candidate_config(read(REFERENCE / 'config.json')))
    trainer.set_random_seed(cfg.seed)
    torch.set_num_threads(1)
    data = read(DATA)
    train, _, _, dimensions, _ = make_loaders(SimpleNamespace(
        dataset_root=data['dataset_root']), cfg)
    trainer.set_random_seed(trainer.SeedBundle(cfg.seed).derive('missing_m3_model_init:fold:5'))
    model = build_model(cfg, dimensions).train()
    groups, _ = trainer._optimizer_parameter_groups(model, cfg)
    optimizer = torch.optim.Adam(groups, weight_decay=cfg.weight_decay)
    batch = trainer._move_batch(next(iter(train)), torch.device(cfg.device))
    view = trainer._prepare_view(batch, trainer._schedules(cfg, 'train')[.5], 0, dimensions)
    rows = []
    started = time.perf_counter()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = forward_view(model, view)
        logits = output[0] if isinstance(output, tuple) else output
        loss = _task_loss(logits, view, cfg.dataset)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite smoke loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        metadata_grad = model.osram.emotion_adapter[1].weight.grad[:, -6:]
        optimizer.step()
        rows.append(dict(
            step=step + 1, loss=float(loss), preclip_norm=float(norm),
            metadata_column_gradient_l1=float(metadata_grad.abs().sum()),
            history_recency=model.osram.last_diagnostics.get('history_recency')))
    write(root / 'SMOKE.json', dict(
        label='ENGINEERING ONLY', steps=rows,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.perf_counter() - started,
        valid_utterances=int(view['umask'].sum())))


def summarize(root):
    baseline_metrics = read(REFERENCE / 'metrics.json')
    run = root / 'runs/history_recency_seed66'
    provenance = read(run / 'PROVENANCE.json')
    if provenance.get('status') != 'complete' or not provenance.get('outputs_verified'):
        raise ValueError('history recency run is not complete')
    metrics = read(run / 'stage150/metrics.json')
    if metrics['mask_sha256'] != baseline_metrics['mask_sha256']:
        raise ValueError('evaluation masks differ from Flat')
    scores = {key: 100 * value['weighted_f1'] for key, value in metrics['test'].items()}
    baseline = {key: 100 * value['weighted_f1']
                for key, value in baseline_metrics['test'].items()}
    result = dict(
        label=LABEL, status='complete', seed=66, epochs=150,
        schedule='100@0.001+50@0.0001',
        history_recency=dict(
            per_rate_wf1=scores, mean8=sum(scores.values()) / 8,
            high=sum(scores[str(index / 10)] for index in (5, 6, 7)) / 3,
            selected_epoch_by_rate=metrics.get('selected_epoch_by_rate'),
            parameters=read(root / 'SMOKE.json')['parameters']),
        flat_reference=dict(
            per_rate_wf1=baseline, mean8=sum(baseline.values()) / 8,
            high=sum(baseline[str(index / 10)] for index in (5, 6, 7)) / 3))
    write(root / 'SUMMARY.json', result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    selected = sum((args.prepare, args.train, args.recover, args.smoke, args.summarize))
    if selected != 1:
        parser.error('Choose exactly one operation')
    if args.prepare:
        if not args.commit:
            parser.error('--commit is required for prepare')
        prepare(args.root, args.commit)
    elif args.train:
        continuation.train(args.root, 0, args.gpu_uuid)
    elif args.recover:
        recovery.train(args.root, 0, args.gpu_uuid)
    elif args.smoke:
        smoke(args.root, args.gpu_uuid)
    else:
        summarize(args.root)


if __name__ == '__main__':
    main()

