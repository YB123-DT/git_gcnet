"""Apply the existing shared smoke once per accepted method, then publish READY."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .manifest import now, read, sha, validate_round, verify_snapshot, write
from .preflight import BANNED_UUID, query_gpus, validate_gpu, validate_readiness
from .dispatch import load_policy, allowed_gpu


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'reference', 'snapshot', 'manifest', 'dataset', 'cpu-log'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cpu-command', required=True, help='Exact command that produced the shared passing CPU log')
    p.add_argument('--once', action='store_true', help='Check at most one new method; never wait for resources')
    p.add_argument('--dispatch-policy', type=Path)
    p.add_argument('--group', choices=('all','bundle','other'), default='all')
    p.add_argument('--poll-seconds',type=float,default=30)
    return p


def advance(args):
    root, source = args.root.resolve(), args.snapshot.resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot = verify_snapshot(source)
    manifest = read(args.manifest)
    ids=validate_round(manifest)
    policy=load_policy(args.dispatch_policy,ids) if args.dispatch_policy else None
    if args.poll_seconds<1 or (args.group!='all' and policy is None):
        raise ValueError('Named groups require dispatch policy and polling >=1 second')
    cards=manifest['cards']
    if args.group!='all':
        selected=policy['bundle_ids' if args.group=='bundle' else 'other_ids']
        cards=[card for card in cards if card['id'] in selected]
    suffix='' if args.group=='all' else '_'+args.group
    status_path=root/('ADVANCE'+suffix+'.json')
    manifest_name = str(args.manifest.resolve().relative_to(source))
    if snapshot['source_sha256'].get(manifest_name) != sha(args.manifest):
        raise ValueError('The accepted subset must be included in its immutable source snapshot')
    cpu = args.cpu_log.resolve()
    if not cpu.is_file() or '\nOK\n' not in cpu.read_text():
        raise ValueError('The shared minimal CPU smoke must pass before CUDA readiness')
    pinned = {'manifest': sha(args.manifest), 'dataset': sha(args.dataset), 'cpu': sha(cpu)}
    with (root / ('advance'+suffix+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for card in cards:
            name = card['id']
            ready = root / 'readiness' / name / 'READY.json'
            if ready.exists():
                existing = read(ready)
                original = verify_snapshot(existing['snapshot_root'])
                validate_readiness(existing, candidate=name, design_sha256=card['design_sha256'],
                                   source_sha256=original['source_sha256'])
                continue  # Never replace an old candidate's readiness with a newer snapshot.
            check = root / 'checks' / name
            if check.exists():
                print(f'{name}: prior attempt retained; inspect its specific failure before retry', flush=True)
                continue
            while True:
                measured = [read(path).get('profile', {}).get('peak_mib', 0)
                            for path in (root / 'checks').glob('*/profile.json')
                            if read(path).get('status') == 'passed']
                peak = max(measured, default=0)
                required_free = max(12288, peak + max(2048, .2 * peak)) if peak else 16384
                resources = query_gpus()
                mapping = {index: row['uuid'] for index, row in resources.items()}
                eligible = []
                disk_ok = shutil.disk_usage(root).free / 1024 ** 3 >= 16
                cpu_ok = os.getloadavg()[0] <= max(1, os.cpu_count() or 1)
                for index, row in resources.items():
                    if index == '4' or row['uuid'] == BANNED_UUID: continue
                    validate_gpu(index, row['uuid'], mapping)
                    if policy and not allowed_gpu(policy,name,index,root): continue
                    if (disk_ok and cpu_ok and row['free_mib'] >= required_free
                            and row['utilization'] < 90 and row['temperature'] < 85):
                        eligible.append((row['utilization'], -row['free_mib'], index))
                if eligible: break
                write(status_path, dict(status='waiting_resources', next=name,group=args.group,
                    required_free_mib=required_free, round=2, target_count=20, updated_at=now()))
                if args.once: return
                time.sleep(args.poll_seconds)
            index = min(eligible)[-1]
            for key, path in (('manifest', args.manifest), ('dataset', args.dataset), ('cpu', cpu)):
                if sha(path) != pinned[key]: raise ValueError(f'Frozen smoke input changed: {key}')
            command = [sys.executable, '-u', '-m', 'experiments.osram_meaningful20_round2_20261004.cuda_check',
                '--candidate', name, '--reference', str(args.reference.resolve()),
                '--dataset', str(args.dataset.resolve()), '--output', str(check),
                '--gpu-index', index, '--gpu-uuid', mapping[index]]
            write(status_path, dict(status='checking', candidate=name,group=args.group,gpu_index=index,
                gpu_uuid=mapping[index], snapshot=str(source), command=command, updated_at=now()))
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=mapping[index], PYTHONPATH=str(source),
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            with (root / f'preflight_{name}.log').open('x') as log:
                result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                if args.once: return
                continue
            profile = read(check / 'profile.json')
            if profile['status'] != 'passed': raise ValueError('Successful smoke exit lacks passing evidence')
            verify_snapshot(source)
            for key, path in (('manifest', args.manifest), ('dataset', args.dataset), ('cpu', cpu)):
                if sha(path) != pinned[key]: raise ValueError(f'Frozen smoke input changed: {key}')
            write(ready, dict(status='ready', candidate=name, snapshot_root=str(source),
                design_sha256=card['design_sha256'], source_sha256=snapshot['source_sha256'],
                cpu=dict(command=args.cpu_command, returncode=0, log=str(cpu), sha256=pinned['cpu']),
                cuda=dict(command=command, returncode=0, log=profile['log'], sha256=profile['log_sha256'],
                          gpu_index=index, gpu_uuid=mapping[index]), profile=profile['profile']))
            print(f'{name}: shared smoke passed; ready for policy-controlled training admission', flush=True)
            if args.once: return
        write(status_path, dict(status='accepted_subset_checks_finished',round=2,group=args.group,
            accepted_count=len(cards),target_count=20,updated_at=now()))


if __name__ == '__main__': advance(parser().parse_args())
