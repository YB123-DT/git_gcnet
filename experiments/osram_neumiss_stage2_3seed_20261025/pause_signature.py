"""Checkpoint-preserving pause for exactly the superseded IEMOCAP queue.

No deletion, checkpoint mutation, GPU allocation, or broad process matching.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time

ROOT = Path('/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/execution_v1')
SOURCE = ROOT.parent / 'source_21e46f2'
SHA = '21e46f2ecd092345818659694db8ff372672c35e'
CONTROLLER = 4165163
MODULE = 'experiments.osram_signature_iemocap_cyclic_20261025.scheduler'
WORKER = 'experiments.osram_signature_iemocap_cyclic_20261025.run'
REASON = 'User requested only NeuMiss completion; preserve Signature for explicit future resume.'


def now():
    return datetime.now(timezone.utc).isoformat()


def matches(argv, module, option, value):
    try:
        return (argv[argv.index('-m') + 1] == module
                and argv[argv.index(option) + 1] == value
                and argv[argv.index('--source-commit') + 1] == SHA)
    except (ValueError, IndexError):
        return False


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        if p.stat().st_uid != os.getuid():
            return None
        args = [s.decode() for s in (p / 'cmdline').read_bytes().split(b'\0') if s]
        if (p / 'cwd').resolve() != SOURCE:
            return None
        return args
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def signal_exact(pid, module, option, value, sig):
    args = process(pid)
    if args is None:
        return False
    if not matches(args, module, option, value):
        raise RuntimeError(f'PID identity mismatch, refusing signal: {pid}')
    os.kill(pid, sig)
    return True


def discover_workers():
    found = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        args = process(pid)
        if not args or WORKER not in args:
            continue
        try:
            output = Path(args[args.index('--output') + 1])
        except (ValueError, IndexError):
            continue
        if output.parent == ROOT / 'train' and matches(args, WORKER, '--output', str(output)):
            found[pid] = output
    return found


def validate_state(state):
    required = {'block', 'optimizer', 'identity', 'epoch', 'history', 'best', 'rng'}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError('Checkpoint is not complete training state')
    if (not state['block'] or not {'state', 'param_groups'}.issubset(state['optimizer'])
            or not state['optimizer']['param_groups']):
        raise ValueError('Missing model/optimizer state')
    if (not isinstance(state['epoch'], int) or state['epoch'] < 1
            or len(state['history']) != state['epoch']):
        raise ValueError('Epoch/history mismatch')
    identity = state['identity']
    if identity.get('source_commit') != SHA or not identity.get('parent', {}).get('checkpoint'):
        raise ValueError('Missing original immutable identity/parent')
    if not {'python', 'numpy', 'torch', 'cuda'}.issubset(state['rng']):
        raise ValueError('Missing full RNG state')
    if not isinstance(state['best'], dict) or not {
            'epoch', 'checkpoint', 'predictions', 'checkpoint_sha256', 'prediction_sha256'}.issubset(state['best']):
        raise ValueError('Missing selected artifacts')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoint(output):
    import torch
    path = output / 'last.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_state(state)
    parent = state['identity']['parent']
    if not Path(parent['checkpoint']).is_file() or digest(parent['checkpoint']) != parent['checkpoint_sha256']:
        raise ValueError('Original parent checkpoint is missing or changed')
    best = state['best']
    for key, hash_key in [('checkpoint', 'checkpoint_sha256'), ('predictions', 'prediction_sha256')]:
        artifact = output / best[key]
        if not artifact.is_file() or digest(artifact) != best[hash_key]:
            raise ValueError(f'Selected {key} missing/changed')
    return dict(output=str(output), epoch=state['epoch'], last_checkpoint_sha256=digest(path),
                optimizer=True, rng=True, history_length=len(state['history']),
                parent_checkpoint=parent['checkpoint'], parent_sha256=parent['checkpoint_sha256'],
                best=best)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        name = stream.name
    os.replace(name, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    audit = dict(status='checking', reason=REASON, started_at=now(), queue=str(ROOT),
                 source_commit=SHA, controller_pid=CONTROLLER, workers=[])
    save(args.audit, audit)
    if not signal_exact(CONTROLLER, MODULE, '--root', str(ROOT), signal.SIGSTOP):
        raise RuntimeError('Expected admission controller not running; refuse blind pause')
    audit['controller_admission_stopped_at'] = now()
    save(args.audit, audit)
    print('Admission controller stopped after exact PID/module/root/source/UID/cwd validation', flush=True)
    try:
        workers = discover_workers()
        for pid, output in workers.items():
            deadline = time.monotonic() + 90
            while True:
                try:
                    checkpoint(output)
                    break
                except (FileNotFoundError, EOFError, RuntimeError, ValueError):
                    if process(pid) is None or time.monotonic() >= deadline:
                        raise
                    time.sleep(2)
            if signal_exact(pid, WORKER, '--output', str(output), signal.SIGSTOP):
                # Atomically saved last.pt is stable once worker is stopped.
                record = checkpoint(output)
                signal_exact(pid, WORKER, '--output', str(output), signal.SIGTERM)
                signal_exact(pid, WORKER, '--output', str(output), signal.SIGCONT)
                limit = time.monotonic() + 15
                while process(pid) is not None and time.monotonic() < limit:
                    time.sleep(.1)
                if process(pid) is not None:
                    raise RuntimeError(f'Worker did not exit after SIGTERM: {pid}')
                after = checkpoint(output)
                if after != record:
                    raise RuntimeError(f'Checkpoint changed across pause: {output}')
                record.update(pid=pid, verified_after_exit=True, paused_at=now())
                audit['workers'].append(record)
                save(args.audit, audit)
                print(json.dumps(dict(pid=pid, output=output.name, epoch=record['epoch'], paused=True)), flush=True)
        if discover_workers():
            raise RuntimeError('Owned workers remain after bounded pause')
        signal_exact(CONTROLLER, MODULE, '--root', str(ROOT), signal.SIGTERM)
        signal_exact(CONTROLLER, MODULE, '--root', str(ROOT), signal.SIGCONT)
        deadline = time.monotonic() + 15
        while process(CONTROLLER) is not None and time.monotonic() < deadline:
            time.sleep(.1)
        if process(CONTROLLER) is not None:
            raise RuntimeError('Admission controller did not exit')
        state = json.loads((ROOT / 'QUEUE.json').read_text())
        by_output = {record['output']: record for record in audit['workers']}
        completed = 0
        for job in state['train']:
            result = Path(job.get('output', '/nonexistent')) / 'RESULT.json'
            if result.exists() and json.loads(result.read_text()).get('status') == 'complete':
                job['status'] = 'complete'
                completed += 1
            elif job.get('output') in by_output:
                job.update(status='paused', recovery='full last.pt', paused_at=now(),
                           pause_reason=REASON, paused_epoch=by_output[job['output']]['epoch'])
            elif job['status'] == 'running':
                raise RuntimeError(f'Unaccounted running job: {job["id"]}')
        state.update(status='paused', pause_reason=REASON, paused_at=now(),
                     automatic_resume=False, pause_audit=str(args.audit))
        save(ROOT / 'QUEUE.json', state)
        audit.update(status='paused', finished_at=now(), completed_jobs=completed,
                     paused_workers=len(audit['workers']), owned_workers_remaining=0,
                     controller_remaining=False, deleted_files=0)
        save(args.audit, audit)
        print(json.dumps({k: v for k, v in audit.items() if k != 'workers'}), flush=True)
    except BaseException as error:
        audit.update(status='failed', error=repr(error), failed_at=now(),
                     remaining_owned_workers=list(discover_workers()))
        save(args.audit, audit)
        raise


if __name__ == '__main__':
    main()
