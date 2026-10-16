"""Two bounded replications of original B3 4/3; immutable seed66 source reused."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TRAINING_SHA = '737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c'
REMOTE = Path('/home/qinjinghui/remote_experiments')
SOURCE_ROOT = REMOTE / 'osram_meld_b3_code_737fdb1'
SOURCE_SUMMARY = REMOTE / 'osram_meld_b3_structure_hpo_20261016/optuna/runs/b3_struct_004/SUMMARY.json'
OUTPUT = REMOTE / 'osram_meld_b3_default43_3seed_20261016'
GPU_UUID = 'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'


def prepare(source, source_path):
    expected = dict(dataset='MELD', seed=66, epochs=50, batch_size=32,
        learning_rate=.001, weight_decay=1e-5, dropout=.5,
        osram_mbt_bottlenecks=4, osram_mbt_rounds=3,
        osram_meaningful_block='mbt_role_bottleneck',
        osram_output_dim=1600, osram_adapter_hidden_dim=0,
        osram_disable_local_skip=False, missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='accuracy',
        training_objective='emotion-only', preserve_selection_artifacts=True)
    if (source.get('status') != 'complete' or source.get('method') != 'B3'
            or source.get('provenance', {}).get('git_sha') != TRAINING_SHA
            or any(source['config'].get(k) != v for k, v in expected.items())):
        raise ValueError('Require the original default4/3 completed seed66, unchanged')
    return dict(label='INTERNAL TEST-ORACLE THREE-SEED REPLICATION',
        training_git_sha=TRAINING_SHA, new_training_budget=2,
        configurations=[dict(id='B3_default43', source_summary=str(source_path), summary=source)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--controller-sha', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    # Only the frozen original modules are imported: controller code does not
    # supply or patch a model, optimizer, sampler, or worker implementation.
    sys.path.insert(0, str(SOURCE_ROOT))
    from experiments.osram_meld_block2_auto_20261014 import replicate
    from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
    from experiments.osram_meld_b6_structure_hpo_20261015.run import StructureScheduler

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE_ROOT, text=True).strip() != TRAINING_SHA
            or subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=SOURCE_ROOT, text=True).strip()):
        raise RuntimeError('Original numerical source must remain immutable')
    source = json.loads(SOURCE_SUMMARY.read_text())
    manifest = prepare(source, SOURCE_SUMMARY)
    for name, expected in source['provenance']['code_hashes'].items():
        if digest(SOURCE_ROOT / name) != expected:
            raise RuntimeError('Source hash mismatch: ' + name)
    from experiments.osram_meld_block2_auto_20261014.worker import DATASET, DATASET_SHA
    if digest(DATASET) != DATASET_SHA or shutil.disk_usage(REMOTE).free < 25 * 2**30:
        raise RuntimeError('Feature hash or disk preflight failed')
    for i in range(8):
        if not (SOURCE_SUMMARY.parent / f'best_miss_0p{i}.pt').is_file():
            raise RuntimeError('Reused seed66 checkpoint missing')
    smoke_path = REMOTE / 'osram_meld_b3_structure_hpo_20261016/smoke/SUMMARY.json'
    smoke = json.loads(smoke_path.read_text())
    if (smoke.get('status') != 'complete' or not smoke.get('smoke_only')
            or smoke['provenance']['git_sha'] != TRAINING_SHA
            or smoke['config']['osram_mbt_bottlenecks'] != 8
            or smoke['config']['osram_mbt_rounds'] != 4):
        raise RuntimeError('Original larger-structure smoke invalid')
    peak = smoke['peak_memory_mib']
    if not 0 < peak <= 4000:
        raise RuntimeError('Unexpected original smoke allocation')
    manifest.update(controller_git_sha=args.controller_sha, controller_sha256=digest(__file__),
        source_summary_sha256=digest(SOURCE_SUMMARY), dataset_sha256=DATASET_SHA,
        reused_smoke=dict(path=str(smoke_path), sha256=digest(smoke_path), peak_memory_mib=peak),
        gpu=3, gpu_uuid=GPU_UUID, max_new_processes=2, max_total_gpu_processes=4,
        paired_replay='not requested; exact selected-epoch Test-best predictions only')
    jobs = replicate.make_jobs(manifest['configurations'], args.output)

    class ReplicationScheduler(StructureScheduler):
        def command(self, job, config_path, uuid):
            return super().command(job, config_path, uuid) + [
                '--testbest-only', '--replication-seed', str(job['config']['seed'])]

        def failure(self, row, reason, returncode=None):
            super().failure(row, reason, returncode)
            # Do not retry an incomplete RNG recovery silently.
            if row['status'] == 'pending':
                row.update(status='failed', resume_blocked='Inspect full CUDA RNG recovery before retry')

    scheduler = ReplicationScheduler(args.output / 'queue', gpus=(3,), per_gpu=2,
        code_root=SOURCE_ROOT, smoke_costs={'B3': dict(peak_memory_mib=peak, disk_bytes=3 * 2**30)})
    if scheduler.gpu_snapshot()[3]['uuid'] != GPU_UUID:
        raise RuntimeError('Physical GPU3 UUID changed')
    print(json.dumps(dict(preflight='passed', jobs=[j['id'] for j in jobs],
                          source_git_sha=TRAINING_SHA, reused_smoke_peak_mib=peak)), flush=True)
    if args.check_only:
        return
    with (args.output / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'MANIFEST.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError('Do not mix replication sources/configs')
        write_json(path, manifest)
        runner = dict(status='running', pid=os.getpid(), started_unix=time.time(),
                      controller_git_sha=args.controller_sha, training_git_sha=TRAINING_SHA)
        write_json(args.output / 'RUNNER.json', runner)
        try:
            records = scheduler.run_jobs(jobs)
            result = replicate.summarize(manifest['configurations'], records, args.output)
            # Fix only the reused report's legacy title/footer, never training.
            report = args.output / 'RESULT.md'
            text = report.read_text().replace('MELD top Test-best ACC configurations', 'MELD default B3 4/3')
            text = text.replace('Original seed66 source is 7abb855.', 'Original seed66 and new workers use immutable737fdb1.')
            report.write_text(text)
            runner['status'] = result['status']
        except BaseException as error:
            runner.update(status='failed', error=repr(error))
            raise
        finally:
            runner['finished_unix'] = time.time()
            write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
