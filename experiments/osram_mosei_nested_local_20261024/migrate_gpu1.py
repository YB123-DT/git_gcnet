"""Audited GPU2 -> GPU1 relocation of exactly the four approved MOSEI runs.

Run this file OUTSIDE the immutable training snapshot. Numerical worker source
is unchanged. Three full epoch checkpoints are copied and resumed; the fourth
has no epoch checkpoint and gets one fresh, separately recorded OOM retry.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

GPU1 = 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'


def plan_migration(state, output):
    from experiments.osram_mosei_nested_local_20261024.run import VARIANTS
    jobs = state['jobs']
    if len(jobs) != 4 or [(j['variant'], j['method'], j['nested_learning_rate'])
                           for j in jobs] != list(VARIANTS):
        raise ValueError('Only the original four variants may migrate')
    result = []
    for i, before in enumerate(jobs):
        if before['status'] != ('running' if i < 3 else 'failed'):
            raise ValueError('Expected three live runs and the single startup failure')
        job = copy.deepcopy(before)
        job.pop('error', None)
        job.pop('exit_code', None)
        job.pop('finished_at', None)
        job.pop('pid', None)
        job.pop('started_at', None)
        path = Path(output) / 'runs/CMUMOSEI/fold_1/seed_66' / job['variant']
        job.update(previous_output=before['output'], previous_pid=before.get('pid'),
                   output=str(path), log=str(path)+'.log', gpu_uuid=GPU1,
                   status='pending', resume=i < 3)
        command = job['command']
        command[command.index('--gpu-uuid')+1] = GPU1
        command[command.index('--output')+1] = str(path)
        if i < 3:
            command.append('--resume')
        result.append(job)
    return result


def validate_process(pid, command, source, *, proc_root=Path('/proc')):
    process = Path(proc_root) / str(pid)
    actual = (process/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if actual != [os.fsencode(x) for x in command] or (process/'cwd').resolve() != Path(source).resolve():
        raise RuntimeError('PID ownership/source differs; refusing signal: '+str(pid))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.source = args.source.resolve()
    sys.path.insert(0, str(args.source))
    from experiments.osram_nested_cmerc_tuning_20261023 import run as common
    from experiments.osram_mosei_nested_combinations_20261024.run import can_admit
    from experiments.osram_iemocap6_nested128_3seed_20261024.run import own_gpu_memory_mib
    from experiments.osram_mosei_nested_local_20261024.report import write_report
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    import torch
    torch.set_num_threads(1)
    original = args.original / 'QUEUE.json'
    old = json.loads(original.read_text())
    manifest = common.verify_source_snapshot(args.source, old['source_commit'])
    device = common.base.devices([GPU1])[GPU1]
    if device['index'] != 1 or device['total_mib']-device['used_mib'] < 4*6144+2048:
        raise RuntimeError('Require healthy physical GPU1 with four-job reserve before relocation')
    if common.available_gib() < 64 or shutil.disk_usage(args.output.parent).free/2**30 < 30:
        raise RuntimeError('Insufficient RAM or disk for independent recovery copies')
    jobs = plan_migration(old, args.output)
    failed = Path(old['jobs'][3]['output'])
    failure = json.loads((failed/'SUMMARY.json').read_text())
    if 'out of memory' not in failure.get('error', '').lower() or (failed/'last.pt').exists():
        raise ValueError('Fresh retry allowed only for documented pre-epoch CUDA OOM')
    controller = old['pid']
    cmd = [os.fsdecode(x) for x in (Path('/proc')/str(controller)/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
    if cmd[cmd.index('-m')+1] != 'experiments.osram_mosei_nested_local_20261024.run' or Path(cmd[cmd.index('--output')+1]).resolve() != args.original.resolve():
        raise RuntimeError('Original controller ownership mismatch')
    validate_process(controller, cmd, args.source)
    for job in old['jobs'][:3]:
        validate_process(job['pid'], job['command'], args.source)
    args.output.mkdir(exist_ok=False)
    common.write(args.output/'ORIGINAL_QUEUE.json', old)
    paused, terminated = [], False
    proofs = {}
    try:
        # Freeze only these owned processes while checking an epoch-consistent
        # disk snapshot. If interrupted during persist_epoch, unfreeze and retry.
        for attempt in range(4):
            os.kill(controller, signal.SIGSTOP); paused.append(controller)
            for job in old['jobs'][:3]:
                validate_process(job['pid'], job['command'], args.source)
                os.kill(job['pid'], signal.SIGSTOP); paused.append(job['pid'])
            try:
                proofs.clear()
                for job in old['jobs'][:3]:
                    path = Path(job['output'])
                    state = torch.load(path/'last.pt', map_location='cpu', weights_only=False)
                    summary = json.loads((path/'SUMMARY.json').read_text())
                    if state['epoch'] != summary['epoch'] or len(state['rng']['cuda']) != 1:
                        raise ValueError('Epoch commit in flight; wait for consistent checkpoint')
                    if not state['optimizer']['state'] or len(state['history']) != state['epoch']:
                        raise ValueError('Incomplete Adam/history recovery')
                    validate_selected(path, state['best_by_rate'])
                    proofs[job['variant']] = dict(epoch=state['epoch'],
                        last_sha256=hashlib.sha256((path/'last.pt').read_bytes()).hexdigest(),
                        optimizer_groups=len(state['optimizer']['param_groups']), cuda_rng_states=1)
                    del state
                break
            except ValueError:
                for pid in reversed(paused):
                    os.kill(pid, signal.SIGCONT)
                paused.clear()
                if attempt == 3:
                    raise
                time.sleep(5)
        for job in jobs[:3]:
            shutil.copytree(job['previous_output'], job['output'], ignore=shutil.ignore_patterns('RUN.lock', '*.tmp'))
            source = Path(job['output'])/'last.pt'
            if hashlib.sha256(source.read_bytes()).hexdigest() != proofs[job['variant']]['last_sha256']:
                raise ValueError('Copied recovery checkpoint differs')
        common.write(args.output/'RECOVERY.json', dict(status='copies_verified',
            source_manifest=manifest, original_queue=str(original), target_gpu=GPU1,
            checkpoints=proofs, failed_retry=dict(original=str(failed), reason=failure['error']),
            note='Discard unfinished epoch only; restore model, Adam, epoch, history, CPU/CUDA RNG. Same V100 model; no bitwise continuation claim across devices.',
            started_at=time.time()))
        # All copies are valid before terminating any original process. Never
        # signal foreign GPU users or use a name-based kill command.
        for pid, command in [(controller, cmd)]+[(j['pid'], j['command']) for j in old['jobs'][:3]]:
            validate_process(pid, command, args.source)
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        paused.clear(); terminated = True
        for unused in range(100):
            live = []
            for pid in [controller]+[j['pid'] for j in old['jobs'][:3]]:
                stat = Path('/proc')/str(pid)/'stat'
                if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] != 'Z':
                    live.append(pid)
            if not live:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Original owned processes did not exit; no duplicate launch')
        old.update(status='migrated_to_gpu1', migration_output=str(args.output), migrated_at=time.time())
        common.write(original, old)
    finally:
        if not terminated:
            for pid in reversed(paused):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
    state = dict(status='pending', pid=os.getpid(), source_commit=old['source_commit'],
        gpu_uuid=GPU1, maximum_parallel=4, maximum_jobs=4, jobs=jobs,
        started_at=time.time(), label='INTERNAL TEST-ORACLE SCREEN',
        source_manifest_sha256=manifest['manifest_sha256'])
    common.base.ROOT = args.source
    children = {}
    def record():
        state['updated_at'] = time.time()
        common.write(args.output/'QUEUE.json', state)
    record()
    while True:
        for i, proc in list(children.items()):
            if common.poll(proc, jobs[i]):
                del children[i]
        pending = [i for i,j in enumerate(jobs) if j['status']=='pending']
        if any(j['status']=='failed' for j in jobs):
            state['status'] = 'failure_draining_active'
            if not children:
                state['status']='failed'; record(); return
        elif not children and not pending:
            write_report(args.output, state)
            state.update(status='complete', finished_at=time.time()); record(); return
        else:
            dev = common.base.devices([GPU1])[GPU1]
            resources = dict(total_mib=dev['total_mib'], used_mib=dev['used_mib'],
                own_used_mib=own_gpu_memory_mib(GPU1,[p.pid for p in children.values()]),
                active_count=len(children), ram_gib=common.available_gib(),
                disk_gib=shutil.disk_usage(args.output).free/2**30)
            state['resources']=resources
            if pending and can_admit(**resources):
                common.verify_source_snapshot(args.source, old['source_commit'])
                i=pending[0]; children[i]=common.base.launch(jobs[i])
                state['status']='running'; record(); time.sleep(4); continue
            state['status']='running' if children else 'waiting_for_resources'
        record(); time.sleep(15)


if __name__ == '__main__':
    main()
