"""One fresh MOSI seed/variant per process; never resume or overwrite output."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REMOTE = Path('/data2/yb/remote_experiments')
ROOT = REMOTE / 'osram_causal_readout_20260910'
FULL = REMOTE / 'osram_write_step_train_20260909/mosi'
FEATURES = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features')
SEEDS = (66, 67, 68, 69, 70)
VARIANTS = ('local-only', 'local-base', 'local-gap')
RATES = tuple(str(r / 10) for r in range(8))


def training_settings(old, variant, seed):
    required = dict(dataset='CMUMOSI', seed=seed, backbone_type='osram', fusion_type='mean',
                    osram_bidirectional=False, osram_forward_slot_reuse=False,
                    osram_write_step=.6, osram_num_heads=8, osram_key_dim=32,
                    osram_value_dim=32, osram_output_dim=700, osram_ablation='full',
                    osram_predictor_mode='structured', osram_query_availability=True,
                    train_rate_mode='cyclic', epochs=100, checkpoint_selection='test-oracle',
                    training_objective='joint', evaluate_test=True)
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError('Only the approved five seeds and three variants are allowed')
    mismatch = {k: old.get(k) for k, v in required.items() if old.get(k) != v}
    if (mismatch or old.get('initial_backbone_checkpoint') is not None
            or old.get('completion_path', 'none') != 'none'
            or old.get('b2_base_checkpoint') is not None
            or old.get('b2_pretrain_checkpoint') is not None
            or old.get('osram_emotion_ablation', 'full') != 'full'):
        raise ValueError(f'Reference is not locked from-scratch causal Full: {mismatch}')
    return dict(old, osram_emotion_ablation=variant)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def worker(variant, seed):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    from experiments.osram_causal_readout_20260910.summarize import extract_best, mask_hashes

    source = FULL / f'seed_{seed}'
    output = ROOT / 'mosi' / variant / f'seed_{seed}'
    old = json.loads((source / 'config.json').read_text())
    settings = training_settings(old, variant, seed)
    extract_best(json.loads((source / 'history.json').read_text()))
    mask_hashes(json.loads((source / 'metrics.json').read_text()))
    config = TrainConfig(**settings)
    before, after = asdict(TrainConfig(**old)), asdict(config)
    delta = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
    if delta != {'osram_emotion_ablation': ['full', variant]}:
        raise ValueError(f'Unexpected config delta: {delta}')
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    provenance = dict(status='training', started_utc=datetime.now(timezone.utc).isoformat(),
                      reference=str(source), from_scratch=True,
                      source_checkpoint_loaded_into_model=False, configuration_delta=delta,
                      feature_root=str(FEATURES), reporting_protocol='per-rate-test-oracle-earliest-tie',
                      checkpoint_protocol='8-rate-mean-test-oracle',
                      reference_sha256={name: sha(source / name) for name in
                                        ('config.json', 'history.json', 'metrics.json')},
                      source_sha256={name: sha(repo / name) for name in
                                     ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py',
                                      'gcnet_missing_m3/train_gcnet.py',
                                      'experiments/osram_causal_readout_20260910/run.py')})
    write_json(output / 'config.json', after)
    write_json(output / 'PROVENANCE.json', provenance)
    try:
        torch.set_num_threads(6)
        roots = [str(FEATURES / name) for name in
                 ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        run_experiment(config, *roots, output_dir=str(output))
        extract_best(json.loads((output / 'history.json').read_text()))
        if mask_hashes(json.loads((output / 'metrics.json').read_text())) != mask_hashes(
                json.loads((source / 'metrics.json').read_text())):
            raise ValueError('Test mask hashes differ from inherited Full')
    except BaseException as error:
        provenance.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'PROVENANCE.json', provenance)
        raise
    provenance.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'PROVENANCE.json', provenance)
    print(f'COMPLETE mosi {variant} seed={seed}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--seed', required=True, type=int, choices=SEEDS)
    args = parser.parse_args()
    worker(args.variant, args.seed)


if __name__ == '__main__':
    main()
