"""One authorized seed66 run; unchanged cfg84 protocol, Gap-only evidence gates."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_local_evidence_gate_20260930 import run as prior

UUID = 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--reference-root', type=Path, default=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'))
    parser.add_argument('--wait-core20', type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = args.output_root / 'seed_66'
        if output.exists():
            raise FileExistsError(output)
        reference = args.reference_root / 'seed_66'
        config = prior.configuration_dict(66, args.reference_root)
        config.update(osram_local_evidence_gate_gap_only=True, osram_local_evidence_gate_reg_type='l2')
        if prior.read(reference / 'PROVENANCE.json')['status'] != 'complete':
            raise ValueError('Reference not complete')
        if prior.read(reference / 'metrics.json')['selection_protocol'] != 'per-rate-test-oracle':
            raise ValueError('Reference protocol mismatch')
        record = dict(label=prior.LABEL, seed=66, server='biggpu', gpu=6, gpu_uuid=UUID,
                      config=config, code_commit=args.code_commit,
                      source_sha256={p: prior.sha(REPO / p) for p in ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py', 'gcnet_missing_m3/train_gcnet.py', 'experiments/osram_gap_only_evidence_gate_20261005/run.py')},
                      reference_config_sha256=prior.sha(reference / 'config.json'),
                      reference_metrics_sha256=prior.sha(reference / 'metrics.json'),
                      from_scratch=True, pid=os.getpid(), status='queued',
                      created_utc=datetime.now(timezone.utc).isoformat())
        prior.write(args.output_root / 'STATUS.json', record)
        while args.wait_core20:
            state = prior.read(args.wait_core20)
            if not any(r['status'] in ('running', 'pending') for r in state['runs']):
                break
            time.sleep(20)
        while True:
            card = subprocess.check_output(['nvidia-smi', '-i', '6', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
            if card[0].strip() != UUID:
                raise RuntimeError('GPU6 UUID mismatch')
            if int(card[1]) >= 4000:
                break
            time.sleep(20)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
            raise ValueError('Requires explicit GPU6 UUID visibility')
        import torch
        from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
        torch.set_num_threads(2)
        output.mkdir()
        record.update(status='training', started_utc=datetime.now(timezone.utc).isoformat())
        prior.write(args.output_root / 'STATUS.json', record)
        prior.write(output / 'PROVENANCE.json', record)
        try:
            run_experiment(TrainConfig(**config), *prior.feature_roots(args), output_dir=str(output))
            prior.verify_outputs(output, reference)
            comparison = {name: prior.rate_scores(prior.read(path / 'metrics.json')) for name, path in
                          (('flat', reference), ('gap_only_gate', output))}
            prior.write(args.output_root / 'SUMMARY.json', dict(label=prior.LABEL, seed=66, **comparison))
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            prior.write(output / 'PROVENANCE.json', record)
            prior.write(args.output_root / 'STATUS.json', record)
            raise
        record.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), mask_validation='canonical_row_multiset')
        prior.write(output / 'PROVENANCE.json', record)
        prior.write(args.output_root / 'STATUS.json', record)


if __name__ == '__main__':
    main()
