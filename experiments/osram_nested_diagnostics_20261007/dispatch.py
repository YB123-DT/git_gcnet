"""Finite, inference-only old-Nested diagnostic queue. Never trains or uses GPU4."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

REMOTE = Path('/data2/yb/remote_experiments')
OUTPUT = REMOTE / 'osram_nested_diagnostics_20261007'
GPUS = (6,)
MAX_CONCURRENT = 2
LABEL = 'INTERNAL DIAGNOSTIC ONLY; existing per-rate Test-oracle; no training'


def tasks():
    rows = []
    for seed in (66, 67, 68):
        for dataset, folds in [('CMUMOSI', (1,)), ('IEMOCAPFour', range(1, 6)),
                               ('IEMOCAPSix', range(1, 6)), ('CMUMOSEI', (1,))]:
            for fold in folds:
                if dataset == 'CMUMOSI':
                    if seed == 66:
                        source = REMOTE / 'osram_new40_gpu0123_20261004/attempt2/runs/nested_gnn_rooted_evidence/seed_66'
                    else:
                        source = REMOTE / f'osram_readout_top3_3seed_20261005/runs/nested_gnn_rooted_evidence/seed_{seed}'
                    data = '/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'
                else:
                    source = REMOTE / f'osram_nested_cross_dataset_20261006/{dataset}/seed_{seed}/fold_{fold}'
                    data = ('/data2/yb/paper/GCNet_repro_cmumosei_10seed_20260819/dataset' if dataset == 'CMUMOSEI'
                            else '/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
                rows.append(dict(dataset=dataset, seed=seed, fold=fold, source=str(source),
                                 dataset_root=data, key=f'{dataset}_seed{seed}_fold{fold}'))
    # Initial samples cover each dataset, then finish the remaining fixed queue.
    initial = [next(r for r in rows if r['dataset'] == d) for d in
               ('CMUMOSI', 'IEMOCAPFour', 'IEMOCAPSix', 'CMUMOSEI')]
    return initial + [r for r in rows if r not in initial]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_sources():
    target = OUTPUT / 'SOURCES.json'
    if target.exists():
        raise FileExistsError('Do not overwrite prior source verification')
    records = []
    for task in tasks():
        root = Path(task['source'])
        cfg = json.loads((root / 'config.json').read_text())
        for key in ('dataset', 'seed', 'fold'):
            if cfg[key] != task[key]:
                raise ValueError(f'Source identity mismatch {root}: {key}')
        if (cfg['osram_meaningful_block'] != 'nested_gnn_rooted_evidence'
                or cfg['training_objective'] != 'emotion-only'
                or cfg['osram_bidirectional'] or cfg['osram_readout_fusion'] != 'flat'):
            raise ValueError(f'Not the pinned original Nested model: {root}')
        hashes = {name: sha(root / name) for name in ('config.json', 'metrics.json')}
        for i in range(8):
            for name in (f'best_miss_0p{i}.pt', f'predictions_miss_0p{i}.npz'):
                hashes[name] = sha(root / name)
        folder = 'IEMOCAP' if task['dataset'].startswith('IEMOCAP') else task['dataset']
        features = Path(task['dataset_root']) / folder / 'features'
        for feature in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT'):
            if not (features / feature).is_dir():
                raise FileNotFoundError(features / feature)
        records.append(dict(task, hashes=hashes))
        print('Verified', task['key'], flush=True)
    write(target, dict(label=LABEL, created_utc=now(), tasks=records))


def gpu_info():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in output.splitlines():
        index, uuid, free = [x.strip() for x in line.split(',')]
        if int(index) in GPUS:
            rows[int(index)] = dict(uuid=uuid, free=int(free))
    return rows


def dispatch():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'DISPATCH.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = OUTPUT / 'DISPATCH.json'
    if status_path.exists():
        raise FileExistsError('Existing queue must be reconciled, never submitted twice')
    manifest = json.loads((OUTPUT / 'SOURCES.json').read_text())
    code_root = Path(__file__).resolve().parents[2]
    state = dict(label=LABEL, status='running', pid=os.getpid(), started_utc=now(),
                 gpu_whitelist=list(GPUS), max_concurrent=MAX_CONCURRENT,
                 code_commit=os.environ.get('NESTED_DIAG_CODE_COMMIT'),
                 code_sha256={str(p.relative_to(code_root)): sha(p) for p in
                    sorted(list((code_root / 'gcnet_missing_m3').glob('*.py'))
                        + list((code_root / 'gcnet_modality_jepa').glob('*.py'))
                        + list((code_root / 'experiments/osram_nested_diagnostics_20261007').glob('*.py'))
                        + [code_root / 'config.py', code_root / 'experiments/osram_gap_increment_audit_20261003/run.py'])},
                 source_manifest_sha256=sha(OUTPUT / 'SOURCES.json'),
                 tasks=[dict(r, status='planned') for r in manifest['tasks']])
    active = []
    write(status_path, state)
    while any(r['status'] in ('planned', 'running') for r in state['tasks']):
        changed = False
        for row, child, stream in list(active):
            exit_code = child.poll()
            if exit_code is None:
                continue
            stream.close()
            result = Path(row['output']) / 'STATUS.json'
            successful = exit_code == 0 and result.exists() and json.loads(result.read_text()).get('status') == 'complete'
            row.update(status='complete' if successful else 'failed', exit_code=exit_code, finished_utc=now())
            active.remove((row, child, stream))
            changed = True
        if changed:
            summary = subprocess.run([sys.executable, '-m', 'experiments.osram_nested_diagnostics_20261007.summarize',
                                      '--root', str(OUTPUT)], cwd=code_root, check=False)
            state['last_summary_exit_code'] = summary.returncode
        resources = gpu_info()
        for row in state['tasks']:
            if row['status'] != 'planned' or len(active) >= MAX_CONCURRENT:
                continue
            required = 18000 if row['dataset'] == 'CMUMOSEI' else 6500
            device = next((i for i in GPUS if resources[i]['free'] >= required), None)
            if device is None:
                continue
            root = Path(row['source'])
            if any(sha(root / name) != digest for name, digest in row['hashes'].items()):
                row.update(status='failed', error='Source checkpoint/artifact changed before dispatch')
                continue
            out = OUTPUT / 'results' / row['key']
            log = OUTPUT / 'logs' / (row['key'] + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', '-m', 'experiments.osram_nested_diagnostics_20261007.evaluate',
                       '--source', row['source'], '--dataset-root', row['dataset_root'],
                       '--output', str(out), '--device', 'cuda']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=resources[device]['uuid'],
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       GCNET_CACHE_ROOT=str(OUTPUT / ('cache_' + row['dataset'])))
            stream = log.open('x')
            child = subprocess.Popen(command, cwd=code_root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            row.update(status='running', pid=child.pid, gpu=device, gpu_uuid=resources[device]['uuid'],
                       command=command, output=str(out), log=str(log), started_utc=now())
            active.append((row, child, stream))
            resources[device]['free'] -= required
            write(status_path, state)
        write(status_path, state)
        if any(r['status'] in ('planned', 'running') for r in state['tasks']):
            time.sleep(15)
    state.update(status='complete' if all(r['status'] == 'complete' for r in state['tasks']) else 'completed_with_failures',
                 finished_utc=now())
    write(status_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-sources', action='store_true')
    args = parser.parse_args()
    verify_sources() if args.verify_sources else dispatch()
