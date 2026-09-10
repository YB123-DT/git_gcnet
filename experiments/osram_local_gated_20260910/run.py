"""Approved five-seed local-gated run; inherit Flat and select each rate independently."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_causal_readout_20260910.run import (
    FULL, FEATURES, SEEDS, sha, training_settings, write_json,
)
from experiments.osram_causal_readout_20260910.summarize import extract_best, mask_hashes

ROOT = Path('/data2/yb/remote_experiments/osram_local_gated_20260910')


def configuration(seed):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    source = FULL / f'seed_{seed}'
    old = json.loads((source / 'config.json').read_text())
    # Reuse the locked-reference checks, not the historical ablation setting.
    training_settings(old, 'local-only', seed)
    extract_best(json.loads((source / 'history.json').read_text()))
    mask_hashes(json.loads((source / 'metrics.json').read_text()))
    cfg = TrainConfig(**dict(old, osram_readout_fusion='local-gated',
                             checkpoint_selection='test-oracle-per-rate'))
    before, after = asdict(TrainConfig(**old)), asdict(cfg)
    delta = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
    assert delta == {'osram_readout_fusion': ['flat', 'local-gated'],
                     'checkpoint_selection': ['test-oracle', 'test-oracle-per-rate']}, delta
    return cfg, source, delta


def train(seed):
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment
    cfg, source, delta = configuration(seed)
    output = ROOT / 'mosi' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(status='training', started_utc=datetime.now(timezone.utc).isoformat(),
                      reference=str(source), configuration_delta=delta, from_scratch=True,
                      source_checkpoint_loaded_into_model=False,
                      selection_protocol='per-rate-test-oracle',
                      label='INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT',
                      reference_sha256={n: sha(source / n) for n in
                                        ('config.json', 'history.json', 'metrics.json')},
                      source_sha256={n: sha(REPO / n) for n in
                                     ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py',
                                      'gcnet_missing_m3/train_gcnet.py',
                                      'experiments/osram_local_gated_20260910/run.py')})
    write_json(output / 'PROVENANCE.json', provenance)
    try:
        torch.set_num_threads(6)
        print(f'TRAIN seed={seed} local-gated cyclic eta=.6 epochs=100 per-rate-test-oracle', flush=True)
        roots = [str(FEATURES / n) for n in
                 ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        run_experiment(cfg, *roots, output_dir=str(output))
        extract_best(json.loads((output / 'history.json').read_text()))
        metrics = json.loads((output / 'metrics.json').read_text())
        assert metrics['selection_protocol'] == 'per-rate-test-oracle'
        assert mask_hashes(metrics) == mask_hashes(json.loads((source / 'metrics.json').read_text()))
    except BaseException as error:
        provenance.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'PROVENANCE.json', provenance)
        raise
    provenance.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'PROVENANCE.json', provenance)
    print(f'COMPLETE seed={seed}', flush=True)


def launch():
    # Validate every inherited config before creating any GPU process.
    for seed in SEEDS:
        configuration(seed)
        if (ROOT / 'mosi' / f'seed_{seed}').exists():
            raise FileExistsError(f'seed {seed} already has output; refusing duplicate launch')
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'QUEUE.json'
    state = dict(status='starting', started_utc=datetime.now(timezone.utc).isoformat(), tasks=[])
    with manifest.open('x') as handle:
        json.dump(state, handle)
    children = []
    for seed, gpu in zip(SEEDS, (2, 2, 2, 3, 3)):
        log_path = ROOT / f'seed{seed}.log'
        with log_path.open('x') as log:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='6',
                       MKL_NUM_THREADS='6', PYTHONPATH=str(REPO))
            child = subprocess.Popen([sys.executable, '-u', __file__, '--seed', str(seed)],
                                     cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(seed=seed, gpu=gpu, pid=child.pid, log=str(log_path), status='running')
        children.append((child, row))
        state['tasks'].append(row)
        write_json(manifest, state)
        print(f'START seed={seed} GPU={gpu} PID={child.pid}', flush=True)
    state['status'] = 'running'
    write_json(manifest, state)
    for child, row in children:
        row['exit_code'] = child.wait()
        row['status'] = 'complete' if row['exit_code'] == 0 else 'failed'
        write_json(manifest, state)
    state.update(status='complete' if all(r['exit_code'] == 0 for _, r in children) else 'failed',
                 completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch', action='store_true')
    group.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    launch() if args.launch else train(args.seed)
