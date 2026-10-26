"""Exact, audited model-weight removal; no directory or wildcard deletion."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

BASE = Path('/data2/yb/remote_experiments')
NESTED = BASE/'osram_frozen_nested_3seed_20261026/execution_v1'
MODEL = re.compile(r'(?:best_epoch_[0-9]+|best_miss_0p[0-7](?:_epoch_[0-9]+)?|last(?:_training)?)[.]pt')
FAMILIES = {
    'osram_neumiss_fixed_rate_stage2_20261025': 'method',
    'osram_neumiss_stage2_3seed_20261025': 'method',
    'osram_signature_cyclic68_20261025': 'method',
    'osram_signature_cyclic_multi_20261025': 'method',
    'osram_signature_iemocap_cyclic_20261025': 'method',
    'osram_signature_noarea_iemocap68_20261025': 'method',
    'osram_signature_noarea_mosi66_20261025': 'method',
    'osram_signature_noarea_mosi68_20261025': 'method',
    'osram_signature_stage2_extension_20261025': 'method',
    'osram_recent_noarea_serial_20261025': 'noarea',
    'osram_four_block_mosi_20261025': 'mixed',
    'osram_three_block_joint_mosi_20261025': 'mixed',
}
ROOTS = {BASE/name for name in FAMILIES}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def save_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def allowed_method(summary, run, family):
    mode = summary.get('mode', summary.get('identity', {}).get('mode'))
    method = summary.get('identity', {}).get('method', summary.get('method'))
    if family == 'noarea':
        return mode in ('serial', 'parallel')
    return mode in ('signature', 'signature_noarea', 'neumiss') or method in ('signature', 'neumiss')


def stat_fields(path):
    path = Path(path)
    s = path.lstat()
    if (path.resolve() != path or not stat.S_ISREG(s.st_mode) or s.st_nlink != 1
            or s.st_uid != os.getuid()):
        raise ValueError('Require owned regular exact file, no symlink/hardlink: '+str(path))
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev,
        uid=s.st_uid, links=s.st_nlink, allocated_bytes=s.st_blocks*512)


def target_row(path, root, run):
    row = dict(path=str(path), root=str(root), run=str(run), **stat_fields(path))
    row['sha256'] = sha(path)
    validate_target(row, {root})
    return row


def validate_target(row, allowed_roots=None):
    allowed_roots = ROOTS if allowed_roots is None else allowed_roots
    path, root, run = (Path(row[k]) for k in ('path','root','run'))
    if (root not in allowed_roots or root == path or not path.is_relative_to(root)
            or not run.is_relative_to(root) or not path.is_relative_to(run)
            or not MODEL.fullmatch(path.name)):
        raise ValueError('Target outside authorized exact method/run/model scope')
    current = stat_fields(path)
    if any(current[k] != row[k] for k in current):
        raise ValueError('Inventoried target changed: '+str(path))


def live_paths():
    outputs, opened = [], set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            for i, value in enumerate(argv[:-1]):
                if value in ('--output','--output-dir','--root') and argv[i+1].startswith('/'):
                    outputs.append(dict(pid=int(proc.name), path=argv[i+1]))
            for fd in (proc/'fd').iterdir():
                try:
                    linked = os.readlink(fd)
                    if linked.startswith('/'):
                        opened.add(linked)
                except OSError:
                    pass
        except OSError:
            pass
    return outputs, opened


def assert_not_live(rows, snapshot=None):
    outputs, opened = live_paths() if snapshot is None else snapshot
    for row in rows:
        run, path = Path(row['run']), row['path']
        if path in opened or any(run.is_relative_to(Path(p['path'])) or Path(p['path']).is_relative_to(run) for p in outputs):
            raise ValueError('Active run/open checkpoint must be retained: '+path)


def nested_references():
    q = read(NESTED/'QUEUE.json')
    if q.get('expected_train_jobs') != 128 or len(q.get('train', [])) != 128:
        raise ValueError('Unexpected live Nested queue scope')
    protected = {}
    for j in q['train']:
        d, s, f, r = (j[k] for k in ('dataset','seed','fold','rate'))
        if s not in (67,68) or r not in tuple(str(k/10) for k in range(8)):
            raise ValueError('Unexpected Nested seed/rate')
        if d == 'MELD':
            continue
        if d == 'IEMOCAPSix' and f in range(1,6):
            ref = BASE/'osram_signature_iemocap_cyclic_20261025/execution_v1/train'/f'IEMOCAPSix_{s}_fold{f}_{r}'
        elif d == 'CMUMOSI' and s == 68 and f == 1:
            ref = BASE/'osram_signature_cyclic68_20261025/execution_v1/train'/r
        elif d in ('CMUMOSI','CMUMOSEI') and f == 1:
            ref = BASE/'osram_signature_cyclic_multi_20261025/execution_v2/train'/f'{d}_{s}_{r}'
        else:
            raise ValueError('Unexpected Nested dataset/fold')
        a = read(ref/'SUMMARY.json'); b = a['best']; path = (ref/b['checkpoint']).resolve()
        if a['status'] != 'complete' or path.parent != ref.resolve() or not MODEL.fullmatch(path.name):
            raise ValueError('Unsafe/incomplete same-seed Signature reference')
        protected[str(path)] = b['checkpoint_sha256']
    if len(protected) != 112:
        raise ValueError('Require112 exact Signature dependencies')
    return q['source_commit'], protected


def nested_finished(root, expected_source):
    try:
        q, result = read(root/'QUEUE.json'), read(root/'RESULT.json')
        return (q.get('source_commit') == result.get('source_commit') == expected_source
            and q.get('status') == result.get('status') == 'complete'
            and len(q.get('train', [])) == 128 and all(j.get('status') == 'complete' for j in q['train'])
            and result.get('validated_rate_results') == 192 and not result.get('failures'))
    except (OSError, ValueError, KeyError):
        return False


def inventory(output):
    source, protected = nested_references()
    targets, deferred, skipped, runs = [], [], [], set()
    live_snapshot = live_paths()
    for name, family in FAMILIES.items():
        root = BASE/name
        if not root.is_dir() or root.resolve() != root:
            raise ValueError('Missing or redirected approved root: '+str(root))
        for dp, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [n for n in dirs if not n.startswith(('source_', '.', '__'))
                and n not in ('dataset','data','feature_cache','cache','features')]
            run = Path(dp)
            if 'SUMMARY.json' not in names:
                continue
            summary = read(run/'SUMMARY.json')
            if not allowed_method(summary, run, family):
                continue
            for filename in names:
                if not MODEL.fullmatch(filename):
                    continue
                path = run/filename
                try:
                    row = target_row(path, root, run)
                    assert_not_live([row], live_snapshot)
                except (OSError, ValueError) as error:
                    skipped.append(dict(path=str(path), reason=str(error)))
                    continue
                if str(path) in protected:
                    if row['sha256'] != protected[str(path)]:
                        raise ValueError('Protected dependency SHA differs')
                    deferred.append(row)
                else:
                    targets.append(row)
                runs.add(run)
    if len(deferred) != 112 or not targets:
        raise ValueError('Missing deferred dependencies or no immediate weights')
    assert_not_live(targets+deferred)
    paths = {r['path'] for r in targets+deferred}
    evidence = []
    for run in sorted(runs):
        for dp, dirs, names in os.walk(run, followlinks=False):
            dirs[:] = [n for n in dirs if not n.startswith(('source_', '.', '__'))]
            for name in names:
                path = Path(dp)/name
                if str(path) in paths or path.is_symlink() or not path.is_file():
                    continue
                evidence.append(dict(path=str(path), size=path.stat().st_size, sha256=sha(path)))
    plan = dict(status='planned', created_at=now(), source_sha256=sha(Path(__file__)),
        authorization='User explicitly requests Signature/NeuMiss checkpoints deletion; preserve all other methods and evidence',
        immediate=targets, deferred=deferred, skipped=skipped, retained_evidence=evidence,
        nested_queue=str(NESTED), nested_source_commit=source, recoverable=False,
        counts=dict(immediate=len(targets), deferred=len(deferred), skipped=len(skipped), retained_evidence=len(evidence)),
        immediate_bytes=sum(r['size'] for r in targets), deferred_bytes=sum(r['size'] for r in deferred))
    save_new(output, plan)
    print(json.dumps({k:plan[k] for k in ('status','counts','immediate_bytes','deferred_bytes')}), flush=True)


def execute(manifest, phase, output):
    plan = read(manifest)
    if plan['status'] != 'planned' or sha(Path(__file__)) != plan['source_sha256'] or phase not in ('immediate','deferred'):
        raise ValueError('Require original exact reviewed plan/executor')
    if phase == 'deferred' and not nested_finished(Path(plan['nested_queue']), plan['nested_source_commit']):
        raise ValueError('Nested queue/collector not complete; retain dependencies')
    rows = plan[phase]
    if not rows or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate exact target list')
    # All checks precede the first irreversible unlink.
    for row in rows:
        validate_target(row)
        if sha(row['path']) != row['sha256']:
            raise ValueError('Inventoried weight SHA changed')
    for row in plan['retained_evidence']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Retained evidence changed: '+row['path'])
    assert_not_live(rows)
    output.mkdir(parents=True, exist_ok=False)
    before = shutil.disk_usage(BASE).free
    deleted = []
    with (output/'DELETED.jsonl').open('x') as journal:
        for row in rows:
            validate_target(row)
            journal.write(json.dumps(row)+'\n'); journal.flush(); os.fsync(journal.fileno())
            Path(row['path']).unlink()
            deleted.append(row)
    if any(Path(r['path']).exists() for r in rows):
        raise ValueError('A target remains')
    for row in plan['retained_evidence']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Retained evidence changed after removal: '+row['path'])
    if phase == 'immediate':
        for row in plan['deferred']:
            validate_target(row)
            if sha(row['path']) != row['sha256']:
                raise ValueError('Nested dependency changed')
    receipt = dict(status='complete', phase=phase, completed_at=now(), manifest=str(manifest),
        manifest_sha256=sha(manifest), executor_sha256=sha(Path(__file__)),
        deleted_count=len(deleted), deleted_bytes=sum(r['size'] for r in deleted),
        deleted_allocated_bytes=sum(r['allocated_bytes'] for r in deleted),
        by_root=dict(Counter(Path(r['root']).name for r in deleted)),
        retained_evidence_count=len(plan['retained_evidence']), retained_evidence_sha256_unchanged=True,
        nested_dependencies_retained=(phase == 'immediate'), recoverable=False,
        disk_free_before=before, disk_free_after=shutil.disk_usage(BASE).free)
    save_new(output/'RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--phase', choices=('immediate','deferred'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wait-for-nested', action='store_true')
    args = parser.parse_args()
    if args.inventory:
        if any((args.manifest, args.phase, args.output, args.wait_for_nested)):
            parser.error('Inventory is read-only and exclusive')
        inventory(args.inventory)
        return
    if not all((args.manifest, args.phase, args.output)):
        parser.error('Execution requires manifest, phase and unique receipt output')
    if args.wait_for_nested:
        if args.phase != 'deferred':
            parser.error('Wait only applies to112 deferred exact targets')
        plan = read(args.manifest); deadline = time.monotonic()+48*3600
        while not nested_finished(Path(plan['nested_queue']), plan['nested_source_commit']):
            if time.monotonic() > deadline:
                raise TimeoutError('Nested completion not verified after48h; no deferred weights deleted')
            print(json.dumps(dict(status='waiting', at=now(), deleted=0)), flush=True)
            time.sleep(30)
    execute(args.manifest, args.phase, args.output)


if __name__ == '__main__':
    main()
