"""Causal eta=.6 Flat: remove all JEPA supervision, retain task objective."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_local_cross_attn_20260910 import run as runner

ROOT = Path('/data2/yb/remote_experiments/osram_causal_nojepa_20260910')


def configuration(seed):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    source = runner.FULL / f'seed_{seed}'
    old = json.loads((source / 'config.json').read_text())
    runner.training_settings(old, 'local-only', seed)
    runner.extract_best(json.loads((source / 'history.json').read_text()))
    runner.mask_hashes(json.loads((source / 'metrics.json').read_text()))
    cfg = TrainConfig(**dict(old, training_objective='emotion-only',
                             checkpoint_selection='test-oracle-per-rate'))
    before, after = asdict(TrainConfig(**old)), asdict(cfg)
    delta = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
    assert delta == {'training_objective': ['joint', 'emotion-only'],
                     'checkpoint_selection': ['test-oracle', 'test-oracle-per-rate']}, delta
    assert cfg.osram_readout_fusion == 'flat' and cfg.osram_write_step == .6
    assert not cfg.osram_bidirectional
    return cfg, source, delta


def train(seed):
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment
    cfg, source, delta = configuration(seed)
    output = ROOT / 'mosi' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(status='training', started_utc=datetime.now(timezone.utc).isoformat(),
        reference=str(source), configuration_delta=delta, from_scratch=True,
        selection_protocol='per-rate-test-oracle',
        label='INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT',
        reference_sha256={n: runner.sha(source / n) for n in ('config.json','history.json','metrics.json')},
        source_sha256={n: runner.sha(REPO / n) for n in (
            'gcnet_missing_m3/osram.py','gcnet_missing_m3/model.py',
            'gcnet_missing_m3/train_gcnet.py','experiments/osram_causal_nojepa_20260910/run.py')})
    if cfg.training_objective == 'complete-state':
        provenance['source_sha256'].update({n: runner.sha(REPO / n) for n in (
            'gcnet_missing_m3/complete_state.py', 'experiments/osram_complete_state_20260910/run.py')})
    runner.write_json(output / 'PROVENANCE.json', provenance)
    try:
        torch.set_num_threads(6)
        print(f'TRAIN seed={seed} Flat {cfg.training_objective} cyclic eta=.6 epochs=100', flush=True)
        roots = [str(runner.FEATURES / n) for n in
                 ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        run_experiment(cfg, *roots, output_dir=str(output))
        history = json.loads((output / 'history.json').read_text())
        runner.extract_best(history)
        metrics = json.loads((output / 'metrics.json').read_text())
        assert metrics['selection_protocol'] == 'per-rate-test-oracle'
        assert runner.mask_hashes(metrics) == runner.mask_hashes(json.loads((source / 'metrics.json').read_text()))
    except BaseException as error:
        provenance.update(status='failed', error=f'{type(error).__name__}: {error}')
        runner.write_json(output / 'PROVENANCE.json', provenance)
        raise
    provenance.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    runner.write_json(output / 'PROVENANCE.json', provenance)
    print(f'COMPLETE seed={seed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--launch', action='store_true')
    g.add_argument('--seed', type=int, choices=runner.SEEDS)
    args = p.parse_args()
    if args.launch:
        runner.ROOT, runner.configuration, runner.__file__ = ROOT, configuration, __file__
        runner.launch()
    else:
        train(args.seed)
