"""Approved MOSI Future-State run, inherited protocol, per-rate Test oracle."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_causal_nojepa_20260910 import run as base

ROOT = Path('/data2/yb/remote_experiments/osram_future_state_20260914')


def configuration(seed):
    cfg, source, _ = base.configuration(seed)
    cfg = replace(cfg, training_objective='future-state')
    assert cfg.train_rate_mode == 'cyclic' and cfg.epochs == 100
    assert cfg.checkpoint_selection == 'test-oracle-per-rate'
    return cfg, source, {
        'training_objective': ['joint', 'future-state'],
        'checkpoint_selection': ['test-oracle', 'test-oracle-per-rate'],
    }


def train(seed):
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, source, delta = configuration(seed)
    output = ROOT / 'mosi' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(
        status='training', started_utc=datetime.now(timezone.utc).isoformat(),
        reference=str(source), configuration_delta=delta, from_scratch=True,
        source_checkpoint_loaded_into_model=False,
        selection_protocol='per-rate-test-oracle',
        label='INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT',
        implementation_commit='3f88cb3',
        reference_sha256={n: base.runner.sha(source / n) for n in
                          ('config.json', 'history.json', 'metrics.json')},
        source_sha256={n: base.runner.sha(REPO / n) for n in (
            'gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py',
            'gcnet_missing_m3/train_gcnet.py', 'gcnet_missing_m3/future_state.py',
            'experiments/osram_future_state_20260914/run.py')})
    base.runner.write_json(output / 'PROVENANCE.json', provenance)
    try:
        torch.set_num_threads(6)
        print(f'TRAIN seed={seed} Future-State causal eta=.6 cyclic 100epochs per-rate-test-oracle', flush=True)
        roots = [str(base.runner.FEATURES / n) for n in
                 ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        run_experiment(cfg, *roots, output_dir=str(output))
        best = base.runner.extract_best(json.loads((output / 'history.json').read_text()))
        metrics = json.loads((output / 'metrics.json').read_text())
        assert metrics['selection_protocol'] == 'per-rate-test-oracle'
        assert all(metrics['selected_epoch_by_rate'][r] == best[r]['epoch'] for r in best)
        assert base.runner.mask_hashes(metrics) == base.runner.mask_hashes(
            json.loads((source / 'metrics.json').read_text()))
    except BaseException as error:
        provenance.update(status='failed', error=f'{type(error).__name__}: {error}')
        base.runner.write_json(output / 'PROVENANCE.json', provenance)
        raise
    provenance.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    base.runner.write_json(output / 'PROVENANCE.json', provenance)
    print(f'COMPLETE seed={seed}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--launch', action='store_true')
    group.add_argument('--seed', type=int, choices=base.runner.SEEDS)
    args = parser.parse_args()
    if args.check:
        for seed in base.runner.SEEDS:
            cfg, _, delta = configuration(seed)
            print(seed, cfg.training_objective, cfg.checkpoint_selection, delta)
    elif args.launch:
        base.runner.ROOT = ROOT
        base.runner.configuration = configuration
        base.runner.__file__ = __file__
        base.runner.launch(gpus=(6, 6, 6, 6, 6))
    else:
        train(args.seed)
