"""Single eta=.6 training lane; IEMOCAP4 then MOSI, each five seeds.

Run --worker DATASET --seed N for one from-scratch task, or no worker to
launch the locked queue. Never resume or overwrite an existing task directory.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REMOTE = Path('/data2/yb/remote_experiments')
ROOT = REMOTE / 'osram_write_step_train_20260909'
SOURCES = {
    'iemocap4': REMOTE / 'osram_forward_only_iemocap_20260908/iemocap4',
    'mosi': REMOTE / 'osram_forward_only_mosi_20260908',
}
FEATURES = {
    'iemocap4': Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP/features'),
    'mosi': Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features'),
}
SEEDS = (66, 67, 68, 69, 70)


def training_settings(old):
    if (old.get('initial_backbone_checkpoint') is not None
            or old.get('osram_bidirectional') is not False
            or old.get('osram_forward_slot_reuse', False)
            or old.get('osram_write_step', 1.) != 1.
            or old.get('backbone_type') != 'osram'
            or old.get('fusion_type') != 'mean'
            or old.get('train_rate_mode') != 'cyclic'
            or old.get('epochs') != 100
            or old.get('checkpoint_selection') != 'test-oracle'):
        raise ValueError('Reference is not the locked from-scratch cyclic causal OSRAM')
    return dict(old, osram_write_step=.6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def worker(dataset, seed):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    source = SOURCES[dataset] / f'seed_{seed}'
    output = ROOT / dataset / f'seed_{seed}'
    if output.exists():
        raise FileExistsError(output)
    old = json.loads((source / 'config.json').read_text())
    history = json.loads((source / 'history.json').read_text())
    checkpoint = torch.load(source / 'best.pt', map_location='cpu', weights_only=False)
    if old != checkpoint['config'] or len(history) != 100 or old['seed'] != seed:
        raise ValueError('Reference checkpoint/config/budget mismatch')
    if checkpoint.get('selection_protocol') != '8-rate-mean-test-oracle':
        raise ValueError('Reference selection mismatch')
    cfg = TrainConfig(**training_settings(old))
    before, after = asdict(TrainConfig(**old)), asdict(cfg)
    delta = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
    if delta != {'osram_write_step': [1., .6]}:
        raise ValueError(f'Unexpected configuration changes: {delta}')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'PROVENANCE.json', {
        'status': 'training', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'reference': str(source), 'reference_checkpoint_sha256': sha(source / 'best.pt'),
        'reference_config_sha256': sha(source / 'config.json'),
        'reference_epoch': checkpoint['epoch'], 'configuration_delta': delta,
        'from_scratch': True, 'source_checkpoint_loaded_into_model': False,
        'feature_root': str(FEATURES[dataset]),
        'source_sha256': {str(p): sha(p) for p in map(Path, [
            'gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py',
            'gcnet_missing_m3/train_gcnet.py', 'gcnet_missing_m3/loss.py',
            'gcnet_missing_m3/mixed_rate.py', 'gcnet_modality_jepa/mask_schedule.py'])},
    })
    del checkpoint
    torch.set_num_threads(6)
    roots = [str(FEATURES[dataset] / name) for name in
             ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    run_experiment(cfg, *roots, output_dir=str(output))
    # Separate CPU processes avoid retaining training graphs during diagnostics.
    for label, override in [('C_train06_test06', None), ('D_train06_test1', '1.0')]:
        cmd = [sys.executable, '-m', 'gcnet_missing_m3.evaluate_write_intervention',
               '--checkpoint', str(output / 'best.pt'), '--feature-root', str(FEATURES[dataset]),
               '--output-dir', str(output / label), '--modes', 'reference', '--device', 'cpu']
        if override is not None:
            cmd += ['--evaluation-write-step', override]
        with (output / f'{label}.log').open('w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    provenance = json.loads((output / 'PROVENANCE.json').read_text())
    provenance.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'PROVENANCE.json', provenance)
    print(f'COMPLETE {dataset} seed={seed} train.6 and C/D evaluation', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=tuple(SOURCES))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.worker:
        if args.seed is None:
            parser.error('--worker requires --seed')
        return worker(args.worker, args.seed)
    ROOT.mkdir(parents=True, exist_ok=True)
    status_path = ROOT / 'QUEUE.json'
    if status_path.exists():
        raise FileExistsError('Queue already exists; inspect it instead of launching duplicates')
    status = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(), 'tasks': []}
    write_json(status_path, status)
    for dataset in SOURCES:
        children = []
        for seed, gpu in zip(SEEDS, (0, 0, 0, 1, 1)):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='6', MKL_NUM_THREADS='6')
            log_path = ROOT / f'{dataset}_seed{seed}.log'
            log = log_path.open('w')
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                        '--worker', dataset, '--seed', str(seed)],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            row = dict(dataset=dataset, seed=seed, gpu=gpu, pid=process.pid, log=str(log_path), status='running')
            status['tasks'].append(row)
            children.append((process, log, row))
            write_json(status_path, status)
        failed = False
        for process, log, row in children:
            row.update(exit_code=process.wait())
            log.close()
            row['status'] = 'complete' if row['exit_code'] == 0 else 'failed'
            failed |= row['exit_code'] != 0
            write_json(status_path, status)
        if failed:
            status['status'] = 'failed'
            write_json(status_path, status)
            raise RuntimeError('Dataset has failed tasks; next dataset not launched')
    status.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)


if __name__ == '__main__':
    main()
