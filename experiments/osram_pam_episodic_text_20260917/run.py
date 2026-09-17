"""Train PAM-E on CMU-MOSI with the locked causal OSRAM protocol."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_causal_nojepa_20260910 import run as base

ROOT = Path('/data2/yb/remote_experiments/osram_pam_episodic_text_20260917')
SEEDS = (66, 67, 68, 69, 70)
# GPU 4 is known to be unreliable on the current biggpu host.
_DEFAULT_GPUS = (1, 2, 3, 5, 7)


def configuration(seed):
    from gcnet_missing_m3.train_gcnet import TrainConfig

    if seed not in SEEDS:
        raise ValueError(f'Unsupported PAM-E seed: {seed}')
    inherited, source, _ = base.configuration(seed)
    values = asdict(inherited)
    values.update(
        training_objective='pam-episodic-text',
        completion_path='pam-episodic-text',
        teacher_mode='ema',
        teacher_checkpoint=None,
        target_space='all-modalities',
        text_subspace_checkpoint=None,
        pam_key_dim=64,
        pam_loss_weight=0.05,
        checkpoint_selection='test-oracle-per-rate',
        evaluate_test=True,
    )
    cfg = TrainConfig(**values)
    if not (
        cfg.dataset == 'CMUMOSI'
        and cfg.seed == seed
        and cfg.backbone_type == 'osram'
        and cfg.osram_bidirectional is False
        and cfg.osram_write_step == .6
        and cfg.osram_readout_fusion == 'flat'
        and cfg.fusion_type == 'mean'
        and cfg.train_rate_mode == 'cyclic'
        and cfg.epochs == 100
    ):
        raise ValueError('Inherited causal OSRAM protocol is not locked')
    return cfg, source, inherited


def _source_files():
    return (
        'gcnet_missing_m3/osram.py',
        'gcnet_missing_m3/model.py',
        'gcnet_missing_m3/pam_episodic.py',
        'gcnet_missing_m3/b2.py',
        'gcnet_missing_m3/train_gcnet.py',
        'experiments/osram_pam_episodic_text_20260917/run.py',
    )


def train(seed):
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment
    from experiments.osram_causal_readout_20260910.run import FEATURES, sha, write_json
    from experiments.osram_causal_readout_20260910.summarize import mask_hashes

    cfg, source, inherited = configuration(seed)
    output = ROOT / 'mosi' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=False)
    source_sha = {name: sha(REPO / name) for name in _source_files()}
    provenance = {
        'status': 'training',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'reference': str(source),
        'configuration_delta': {
            key: [asdict(inherited).get(key), asdict(cfg).get(key)]
            for key in asdict(cfg)
            if asdict(inherited).get(key) != asdict(cfg).get(key)
        },
        'from_scratch': True,
        'source_checkpoint_loaded_into_model': False,
        'selection_protocol': 'per-rate-test-oracle',
        'label': 'INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT',
        'feature_root': str(FEATURES),
        'reference_sha256': {name: sha(source / name) for name in ('config.json', 'metrics.json')},
        'source_sha256': source_sha,
    }
    write_json(output / 'PROVENANCE.json', provenance)
    write_json(output / 'config.json', asdict(cfg))
    try:
        torch.set_num_threads(6)
        roots = [str(FEATURES / name) for name in (
            'wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / 'metrics.json').read_text())
        if metrics['selection_protocol'] != 'per-rate-test-oracle':
            raise ValueError('PAM-E did not use per-rate Test-oracle selection')
        reference_metrics = json.loads((source / 'metrics.json').read_text())
        if mask_hashes(metrics) != mask_hashes(reference_metrics):
            raise ValueError('PAM-E test masks differ from the inherited protocol')
    except BaseException as error:
        provenance.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'PROVENANCE.json', provenance)
        raise
    provenance.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'PROVENANCE.json', provenance)
    print(f'COMPLETE PAM-E MOSI seed={seed}', flush=True)


def _gpus():
    raw = os.environ.get('PAM_E_GPUS', '')
    if raw.strip():
        gpus = tuple(int(value.strip()) for value in raw.split(',') if value.strip())
    else:
        gpus = _DEFAULT_GPUS
    if len(gpus) != len(SEEDS):
        raise ValueError('PAM_E_GPUS must contain one GPU index per seed')
    return gpus


def launch():
    import subprocess

    gpus = _gpus()
    for seed in SEEDS:
        configuration(seed)
        if (ROOT / 'mosi' / f'seed_{seed}').exists():
            raise FileExistsError(f'PAM-E seed {seed} output already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'QUEUE.json'
    if manifest.exists():
        raise FileExistsError(f'PAM-E queue already exists: {manifest}')
    state = {'status': 'starting', 'started_utc': datetime.now(timezone.utc).isoformat(), 'tasks': []}
    manifest.write_text(json.dumps(state, indent=2) + '\n')
    children = []
    for seed, gpu in zip(SEEDS, gpus):
        log_path = ROOT / f'seed{seed}.log'
        with log_path.open('x') as log:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='6',
                       MKL_NUM_THREADS='6', PYTHONPATH=str(REPO))
            child = subprocess.Popen([sys.executable, '-u', __file__, '--seed', str(seed)],
                                     cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'seed': seed, 'gpu': gpu, 'pid': child.pid, 'log': str(log_path), 'status': 'running'}
        state['tasks'].append(row)
        children.append((child, row))
        manifest.write_text(json.dumps(state, indent=2) + '\n')
        print(f'START PAM-E seed={seed} GPU={gpu} PID={child.pid}', flush=True)
    state['status'] = 'running'
    manifest.write_text(json.dumps(state, indent=2) + '\n')
    for child, row in children:
        row['exit_code'] = child.wait()
        row['status'] = 'complete' if row['exit_code'] == 0 else 'failed'
        manifest.write_text(json.dumps(state, indent=2) + '\n')
    state.update(status='complete' if all(row['exit_code'] == 0 for _, row in children) else 'failed',
                 completed_utc=datetime.now(timezone.utc).isoformat())
    manifest.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch', action='store_true')
    group.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.launch:
        launch()
    else:
        train(args.seed)
