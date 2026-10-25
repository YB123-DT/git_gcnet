"""Audited one-time extra admission; never signal a training worker.

User authorized remaining MOSEI rates immediately, preserving every live run.
Only hand off the queue controller so it adopts the extra workers instead of
later launching duplicates. The immutable training source/config is unchanged.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

CONTROLLER = 'experiments.osram_frozen_nested_cross_20261025.scheduler'


def extra_targets(jobs):
    return [j for j in jobs if j['dataset'] == 'CMUMOSEI' and j['seed'] == 66
        and j['fold'] == 1 and j['rate'] in ('0.5','0.6','0.7') and j['status'] == 'pending']


def validate_controller(state,command,root):
    if (state['status'] != 'train' or state['expected_train_jobs'] != 56
        or CONTROLLER not in command or '--root' not in command
        or command[command.index('--root')+1] != str(root)
        or '--source-commit' not in command
        or command[command.index('--source-commit')+1] != state['source_commit']):
        raise ValueError('Not the exact approved controller/root/source')


def resume_controller(root,state,command,environment,audit,lock,lock_acquired):
    if not lock_acquired:
        raise RuntimeError('No scheduler lock ownership; cannot write queue or start replacement')
    status = Path('/proc')/str(audit['old_controller_pid'])/'status'
    if status.exists() and not any(line.startswith('State:') and '\tZ' in line
            for line in status.read_text().splitlines()):
        raise RuntimeError('Old controller has not exited; no queue overwrite or replacement')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import save,now
    save(root/'QUEUE.json',state)
    fcntl.flock(lock,fcntl.LOCK_UN)
    restart = command if '--resume' in command else command+['--resume']
    with (root.parent/'controller_v1.log').open('a') as log:
        child=subprocess.Popen(restart,cwd=state['cwd'],env=environment,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    audit.update(new_controller_pid=child.pid,controller_command=restart,finished_at=now())
    return child.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args();root = args.root.resolve()
    if root.name != 'execution_v1' or root.parent.name != 'osram_frozen_nested_cross_20261025':
        raise ValueError('Only the existing cross-dataset queue is in scope')
    from experiments.osram_frozen_nested_cross_20261025 import scheduler as queue
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import resources,save,now
    from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits
    with (root/'extra_admission.lock').open('a') as admission_lock:
        fcntl.flock(admission_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = json.loads((root/'QUEUE.json').read_text());pid = state['pid']
        proc = Path('/proc')/str(pid)
        command = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        validate_controller(state,command,root)
        if proc.stat().st_uid != os.getuid() or (proc/'cwd').resolve() != Path(state['cwd']).resolve():
            raise ValueError('Controller UID/cwd mismatch')
        if not extra_targets(state['train']):
            print('All remaining MOSEI rates already admitted; no process touched');return
        environment = dict(x.decode().split('=',1) for x in (proc/'environ').read_bytes().split(b'\0') if x)
        audit = dict(started_at=now(),operation='extra pending MOSEI only; preserve all live workers',
            source_commit=state['source_commit'],old_controller_pid=pid,launched=[],
            control_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            automatic_per_gpu_limit=4,temporary_peak_per_gpu_limit=5,expected_train_jobs=56)
        stopped,killed,new_controller,lock_acquired = False,False,None,False
        scheduler_lock = (root/'scheduler.lock').open('a')
        try:
            # Freeze only the scheduler briefly while taking its last consistent
            # persisted queue; detached training workers continue uninterrupted.
            os.kill(pid,signal.SIGSTOP);stopped=True
            deadline=time.monotonic()+5
            while not any(line.startswith('State:') and '\tT' in line
                    for line in (proc/'status').read_text().splitlines()):
                if time.monotonic()>deadline:raise RuntimeError('Controller did not stop for consistent queue handoff')
                time.sleep(.01)
            state = json.loads((root/'QUEUE.json').read_text())
            validate_controller(state,command,root)
            known = {j.get('pid') for phase in ('smoke','train') for j in state.get(phase,[])}
            children = [int(p) for p in (proc/'task'/str(pid)/'children').read_text().split()]
            for child in children:
                cmdfile = Path('/proc')/str(child)/'cmdline'
                if child not in known and cmdfile.exists() and cmdfile.read_bytes():
                    raise RuntimeError('Unpersisted child detected; restore controller and retry safely')
            live = [j for j in state['train'] if j['status'] == 'running' and queue.process_matches(j)]
            if pid in {j['pid'] for j in live}:raise ValueError('Controller cannot be a training PID')
            audit['workers_before'] = [{k:j[k] for k in ('id','pid','command','gpu_uuid')} for j in live]
            targets = extra_targets(state['train'])
            uuids = list(state['gpu_map'].values())
            if set(map(int,state['gpu_map'])) != {0,1,2}:raise ValueError('Require original healthy GPU0/1/2 map')
            snapshot = resources(root,uuids)
            peers = list(live);assignments=[]
            for job in targets:
                for uuid in sorted(uuids,key=lambda u:sum(j.get('gpu_uuid')==u for j in peers)):
                    card_peers = [j for j in peers if j.get('gpu_uuid')==uuid]
                    if fits(snapshot,uuid,job,card_peers,state['admission_measurements'],5):
                        assignments.append((job,uuid));peers.append(dict(job,gpu_uuid=uuid));break
                else:raise RuntimeError('Extra MOSEI does not fit measured capacity; existing workers unchanged')
            save(root/f'QUEUE.before_extra_{pid}.json',state)
            # Terminate exactly the controller, never its process group/children.
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False;killed=True
            deadline=time.monotonic()+10
            while True:
                try:
                    fcntl.flock(scheduler_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    lock_acquired=True;break
                except BlockingIOError:
                    if time.monotonic()>deadline:raise RuntimeError('Controller lock did not release')
                    time.sleep(.1)
            launch_args=argparse.Namespace(root=root,source_commit=state['source_commit'],resume=False)
            for job,uuid in assignments:
                queue.launch(launch_args,job,uuid,'train')
                audit['launched'].append({k:job[k] for k in ('id','pid','command','gpu_uuid','output')})
                save(root/'QUEUE.json',state)
            state['train'].sort(key=lambda j:j['dataset']!='CMUMOSEI')
            state.setdefault('extra_admissions',[]).append(audit)
            save(root/'QUEUE.json',state)
        finally:
            if stopped:os.kill(pid,signal.SIGCONT)
            if killed:
                new_controller=resume_controller(root,state,command,environment,audit,scheduler_lock,lock_acquired)
                audit['workers_unchanged_or_naturally_finished'] = all(queue.process_matches(j)
                    or json.loads((Path(j['output'])/'SUMMARY.json').read_text()).get('status')=='complete' for j in live)
                save(root/f'EXTRA_MOSEI_{pid}.json',audit)
        print(json.dumps(dict(controller=new_controller,extra=audit['launched'],
            existing_workers_preserved=audit.get('workers_unchanged_or_naturally_finished'))))


if __name__=='__main__':main()
