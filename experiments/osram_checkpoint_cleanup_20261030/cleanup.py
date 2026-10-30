"""Bounded, score-backed obsolete weight cleanup; immutable exact-path manifest."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil

from experiments.osram_signature_neumiss_cleanup_20261026 import cleanup as common

BASE = Path('/data2/yb/remote_experiments')
FAMILIES = {
    'osram_v3_three_seed_20261021',
    'osram_v3_connection_screen_20261022',
    'osram_v3_paired_interfaces_20261022',
    'osram_source_role_direct_cross_20261020',
    'osram_mosi_memory_grid_20261022',
    'osram_nested_cmerc_tuning_20261023',
    'osram_reg_only_hparam_sweep_20260919_100',
}


def preserve_run(family, run):
    """Deliberately conservative exclusions within the user-approved families."""
    if 'smoke' in run.parts or any(p.startswith('smoke_') for p in run.parts):
        return False
    name = run.name
    if name in ('none', 'flat', 'baseline') or 'baseline' in name:
        return True
    if family == 'osram_mosi_memory_grid_20261022':
        return name in ('head_48', 'step_0p4')
    if family == 'osram_source_role_direct_cross_20261020':
        return run.as_posix().endswith('/CMUMOSEI/fold_1/seed_66/direct')
    if family == 'osram_nested_cmerc_tuning_20261023':
        return name in {'nested_dim128', 'nested_dim32', 'nested_gnn_rooted_evidence',
                        'mosei_nested_mlp128', 'mosei_nested_depth2'}
    if family == 'osram_reg_only_hparam_sweep_20260919_100':
        return name.startswith(('cfg84_', 'cfg88_'))
    return False


def has_scores(record):
    if record.get('status') not in (None, 'complete', 'completed'):
        return False
    best = record.get('best_by_rate', {})
    if any(isinstance(v, dict) and v.get('epoch') is not None and
           isinstance(v.get('metrics'), dict) and v['metrics'] for v in best.values()):
        return True
    return bool(record.get('selected_weighted_f1_by_rate') and
                record.get('selected_epoch_by_rate'))


def walk(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(n for n in dirs if not n.startswith(
            ('source_', 'controller_', 'snapshot_', 'code_', '.', '__')) and n not in
            ('source', 'snapshot', 'repo', 'data', 'dataset', 'feature_cache',
             'cache', 'features', 'node_modules', 'experiments')
             and not n.endswith('_source'))
        yield Path(directory), sorted(files)


def fingerprint(path):
    return dict(path=str(path), size=path.stat().st_size, sha256=common.sha(path))


def external_dependencies(runs):
    """Retain exact referenced model paths/runs in non-snapshot project records.

    This is a conservative search, not a claim to statically resolve every Python
    path expression. Hardcoded controls/winners above additionally cover dynamic
    path construction in current experiments.
    """
    protected, records = set(), []
    prefix = str(BASE) + '/'
    pattern = re.compile(re.escape(prefix) + r'''[^\s"'<>),\]\}]+''')
    # Only current/postdating output registries can reference these 2026-10-20+
    # runs. Archived historical source copies are not active runtime registries;
    # the independent current-repository audit supplements this bounded search.
    scan_roots = []
    for family in sorted(BASE.iterdir()):
        date = re.search(r'20[0-9]{6}', family.name)
        if (family.is_dir() and not family.is_symlink() and family.name not in FAMILIES
                and 'cleanup' not in family.name
                and not family.name.startswith(('code_', 'source_', 'controller_', '.'))
                and '_code_' not in family.name
                and (date is None or date.group(0) >= '20261020')):
            scan_roots.append(family)
    for directory, files in (item for root in scan_roots for item in walk(root)):
        if any(directory.is_relative_to(BASE / f) for f in FAMILIES):
            continue
        relative = directory.relative_to(BASE)
        if relative.parts and 'cleanup' in relative.parts[0]:
            continue  # Past deletion inventories are not runtime dependencies.
        for name in files:
            path = directory / name
            if name.startswith(('history', 'predictions', 'diagnostics', 'train_log', 'epoch_')):
                continue
            if path.suffix not in ('.json', '.md', '.py', '.csv') or path.is_symlink():
                continue
            if path.stat().st_size > 16 * 1024**2:
                continue
            try:
                body = path.read_text(errors='replace')
            except OSError:
                continue
            if not any(f in body for f in FAMILIES):
                continue
            touched = set()
            for text in pattern.findall(body):
                ref = Path(text.rstrip('`.:;'))
                for run in runs:
                    if ref == run or (ref.parent == run and common.MODEL.fullmatch(ref.name)):
                        touched.add(str(run))
            if touched:
                protected.update(touched)
                records.append(dict(**fingerprint(path), protected_runs=sorted(touched)))
    return protected, records


def inventory(output):
    candidates, retained_runs, skipped = [], [], []
    roots = {BASE / f for f in FAMILIES}
    snapshot = common.live_paths()
    for root in sorted(roots):
        if not root.is_dir() or root.resolve() != root:
            raise ValueError('Missing or redirected authorized root: ' + str(root))
        for run, names in walk(root):
            weights = [run / n for n in names if common.MODEL.fullmatch(n)]
            if not weights:
                continue
            proof = next((run / n for n in ('SUMMARY.json', 'metrics.json') if n in names), None)
            if preserve_run(root.name, run):
                retained_runs.append(dict(run=str(run), reason='baseline/current/representative'))
                continue
            record = common.read(proof) if proof else {}
            if not has_scores(record):
                skipped.append(dict(run=str(run), reason='incomplete or missing saved scores'))
                continue
            for path in weights:
                # A selected-rate model must have its own saved score, not merely
                # a score for a different rate somewhere in the same directory.
                rate_match = re.search(r'best_miss_0p([0-7])', path.name)
                if rate_match:
                    rate = '0.' + rate_match.group(1)
                    if rate not in record.get('best_by_rate', {}) and rate not in record.get('selected_weighted_f1_by_rate', {}):
                        skipped.append(dict(path=str(path), reason='missing matching rate score'))
                        continue
                common.assert_not_live([dict(run=str(run), path=str(path))], snapshot)
                candidates.append((path, root, run))
    protected, dependencies = external_dependencies({r for _, _, r in candidates})
    print(json.dumps(dict(phase='dependencies_checked', candidates=len(candidates),
        protected_runs=len(protected), records=len(dependencies))), flush=True)
    targets = []
    for path, root, run in candidates:
        if str(run) in protected:
            retained_runs.append(dict(run=str(run), reason='external checkpoint/run reference'))
            continue
        targets.append(common.target_row(path, root, run))
        if len(targets) % 100 == 0:
            print(json.dumps(dict(phase='hash_weights', count=len(targets))), flush=True)
    if not targets:
        raise ValueError('No eligible score-backed obsolete model files')
    target_paths = {r['path'] for r in targets}
    retained = []
    for root in sorted(roots):
        for directory, names in walk(root):
            for name in names:
                path = directory / name
                if str(path) not in target_paths and path.is_file() and not path.is_symlink():
                    retained.append(fingerprint(path))
    common.assert_not_live(targets)
    common.save_new(output, dict(status='planned', created_at=common.now(),
        authorization='2026-10-30 user permits obsolete new-Missing weights cleanup in seven listed families',
        executor_sha256=common.sha(__file__), helper_sha256=common.sha(common.__file__),
        targets=targets, retained=retained, dependency_records=dependencies,
        retained_runs=list({(r['run'], r['reason']): r for r in retained_runs}.values()),
        skipped=skipped, recoverable=False))
    print(json.dumps(dict(status='planned', targets=len(targets), retained=len(retained),
        bytes=sum(r['size'] for r in targets), manifest=str(output))), flush=True)


def check_fingerprints(rows):
    for row in rows:
        path = Path(row['path'])
        if path.is_symlink() or path.stat().st_size != row['size'] or common.sha(path) != row['sha256']:
            raise ValueError('Retained evidence/dependency changed: ' + row['path'])


def execute(manifest, output):
    plan = common.read(manifest)
    if plan['status'] != 'planned' or plan['executor_sha256'] != common.sha(__file__):
        raise ValueError('Require unchanged planned manifest/executor')
    if plan.get('helper_sha256', common.sha(common.__file__)) != common.sha(common.__file__):
        raise ValueError('Safety helper changed')
    roots = {BASE / f for f in FAMILIES}
    rows = plan['targets']
    if not rows or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate targets')
    for row in rows:
        common.validate_target(row, roots)
        if common.sha(row['path']) != row['sha256']:
            raise ValueError('Target contents changed: ' + row['path'])
    check_fingerprints(plan['retained'])
    check_fingerprints(plan['dependency_records'])
    common.assert_not_live(rows)
    output.mkdir(parents=True, exist_ok=False)
    before = shutil.disk_usage(BASE).free
    with (output / 'DELETED.jsonl').open('x') as journal:
        for index, row in enumerate(rows):
            if index % 100 == 0:
                common.assert_not_live(rows[index:])
            common.validate_target(row, roots)
            journal.write(json.dumps(row) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            Path(row['path']).unlink()
    if any(Path(r['path']).exists() for r in rows):
        raise ValueError('Target remains after unlink')
    check_fingerprints(plan['retained'])
    check_fingerprints(plan['dependency_records'])
    groups = {}
    for root in sorted(roots):
        subset = [r for r in rows if r['root'] == str(root)]
        groups[root.name] = dict(count=len(subset), bytes=sum(r['size'] for r in subset))
    receipt = dict(status='complete', completed_at=common.now(), manifest=str(manifest),
        manifest_sha256=common.sha(manifest), executor_sha256=common.sha(__file__),
        deleted_count=len(rows), deleted_bytes=sum(r['size'] for r in rows),
        deleted_allocated_bytes=sum(r['allocated_bytes'] for r in rows),
        retained_count=len(plan['retained']), retained_sha256_unchanged=True,
        dependency_records_unchanged=True, by_family=groups, recoverable=False,
        disk_free_before=before, disk_free_after=shutil.disk_usage(BASE).free)
    common.save_new(output / 'RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inventory').add_argument('--output', type=Path, required=True)
    ex = sub.add_parser('execute')
    ex.add_argument('--manifest', type=Path, required=True)
    ex.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory(args.output)
    else:
        execute(args.manifest, args.output)


if __name__ == '__main__':
    main()
