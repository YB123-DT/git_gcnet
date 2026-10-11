"""Delete 23 approved completed round1/round2 checkpoints; preserve evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket

BASE = Path('/data2/yb/remote_experiments')
GROUPS = {
    'osram_meaningful20_20261003_round1': [
        'dgcnn_dynamic_edgeconv', 'tabnet', 'rrn_evidence', 'ed_hnn',
        'graph_multiset_transformer', 'otke', 'node', 'allset_transformer',
        'equilibrium_aggregation', 'capsule_variational_bayes',
        'capsule_dynamic_routing', 'slot_attention'],
    'osram_meaningful20_round2_20261004': [
        'dense_coattention', 'mcan_encoder_decoder', 'fspool_fsunpool_evidence',
        'mac_control_read_write', 'conditional_rational_quadratic_spline_coupling',
        'learned_robust_pca_evidence_decomposition', 'dual_attention_symbolic_relations',
        'compositional_search_retrieval', 'rat_spn_evidence_circuit',
        'spdnet_bimap_reeig_logeig', 'tokenlearner_fuser_v11'],
}
ROOTS = [BASE / batch / 'runs' / name for batch, names in GROUPS.items() for name in names]
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    with path.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Fail closed if any live Python process references these experiment batches.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            parts = (process / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if parts and b'python' in Path(parts[0].decode(errors='replace')).name.encode():
            command = b' '.join(parts).decode(errors='replace')
            assert not any(str(r) in command for r in ROOTS), command
    files, evidence, runs = [], [], []
    for root in ROOTS:
        assert root.is_dir() and not root.is_symlink()
        children = sorted(root.glob('seed_66'))
        assert children
        for run in children:
            assert not run.is_symlink()
            provenance = json.loads((run / 'PROVENANCE.json').read_text())
            assert provenance['status'] == 'complete' and provenance['outputs_verified']
            assert (run / 'metrics.json').is_file() and (run / 'config.json').is_file()
            assert len(list(run.glob('predictions_miss_*.npz'))) == 8
            metrics = json.loads((run / 'metrics.json').read_text())
            score = sum(metrics['test'][str(i / 10)]['weighted_f1'] for i in range(8)) * 100 / 8
            assert score < 80.3, (str(run), score)
            runs.append(str(run))
            for path in sorted(run.rglob('*')):
                assert not path.is_symlink(), str(path)
                if not path.is_file():
                    continue
                assert path.resolve().is_relative_to(root.resolve())
                stat = path.stat()
                row = dict(path=str(path), size=stat.st_size, sha256=sha(path),
                           inode=stat.st_ino, device=stat.st_dev, links=stat.st_nlink)
                (files if path.suffix in WEIGHTS else evidence).append(row)
    assert len(runs) == 23 and len(files) == 839, (len(runs), len(files))
    report = dict(status='inventoried', host=socket.gethostname(), runs=runs,
                  checkpoints=files, retained_evidence=evidence,
                  requested_scope='23 explicitly approved round1/round2 candidates', script_sha256=sha(Path(__file__)),
                  free_before={p: shutil.disk_usage(p).free for p in ('/data1', '/data2')})
    save(args.output / 'MANIFEST.json', report)
    print('Inventory saved:', len(files), 'checkpoints, bytes:', sum(r['size'] for r in files), flush=True)
    if not args.execute:
        return
    # No recursive directory deletion. Validate exact inode/content immediately
    # before unlink; a partial failure remains traceable to the durable manifest.
    for row in files:
        path = Path(row['path'])
        stat = path.stat()
        assert (stat.st_ino, stat.st_dev, stat.st_size) == (row['inode'], row['device'], row['size'])
        assert sha(path) == row['sha256']
        path.unlink()
    for row in evidence:
        path = Path(row['path'])
        assert path.is_file() and sha(path) == row['sha256'], str(path)
    assert all(not Path(row['path']).exists() for row in files)
    report.update(status='complete', deleted_files=len(files),
                  deleted_logical_bytes=sum(r['size'] for r in files),
                  evidence_verified=len(evidence),
                  free_after={p: shutil.disk_usage(p).free for p in ('/data1', '/data2')})
    save(args.output / 'MANIFEST.json', report)
    summary = {k: v for k, v in report.items() if k not in ('checkpoints', 'retained_evidence')}
    save(args.output / 'SUMMARY.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()


