"""Cancel only IEMOCAPFour Flat; adopt other live jobs without restarting."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def process(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        return dict(state=stat[0], start=stat[19], command=command)
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); root = a.root
    source = root / 'source_4eed138/run.py'
    spec = importlib.util.spec_from_file_location('old_runner', source)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    lock = (root / 'SCOPE47.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'CANCEL_FOUR_FLAT.json').exists(), 'Already handed off; reconcile first'
    state = old.read(root / 'QUEUE.json')
    coordinator = state['pid']; info = process(coordinator)
    assert info and str(source) in info['command'] and '--dispatch' in info['command']
    os.kill(coordinator, signal.SIGSTOP)
    time.sleep(.2)
    assert process(coordinator)['state'] in ('T', 't')
    state = old.read(root / 'QUEUE.json')
    old.write(root / 'QUEUE_BEFORE_CANCEL_FOUR_FLAT.json', state)
    rows = state['tasks']
    excluded = [r for r in rows if r['id'].startswith('IEMOCAPFour_flat_')]
    assert len(excluded) == 15 and len(rows) == 62
    targets, survivors = [], []
    for row in rows:
        if row['status'] == 'running':
            proc = process(row['pid'])
            assert proc and str(source) in proc['command']
            assert ' '.join(row['command']) == proc['command'].strip()
            row['start_ticks'] = proc['start']
            (targets if row in excluded else survivors).append(row['pid'])
    record = dict(status='cancelling', excluded_ids=[r['id'] for r in excluded],
                  stopped_child_pids=targets, untouched_child_pids=survivors,
                  old_coordinator_pid=coordinator, new_coordinator_pid=os.getpid(),
                  requested_scope=47, started_at=old.now(), controller_sha256=old.sha(Path(__file__)))
    old.write(root / 'CANCEL_FOUR_FLAT.json', record)
    for row in excluded:
        if row['status'] == 'running':
            os.kill(row['pid'], signal.SIGTERM)
            for _ in range(100):
                proc = process(row['pid'])
                if proc is None or proc['state'] == 'Z': break
                time.sleep(.1)
            proc = process(row['pid'])
            assert proc is None or proc['state'] == 'Z', 'Target did not terminate; inspect, do not kill others'
            output = root / 'runs' / row['id']
            prov = old.read(output / 'PROVENANCE.json')
            prov.update(status='cancelled_by_user', cancelled_at=old.now(),
                        reason='IEMOCAPFour Flat removed from requested scope')
            old.write(output / 'PROVENANCE.json', prov)
        row.update(status='cancelled_by_user', cancelled_at=old.now())
    # Stop only the obsolete scheduler, never its process group/live children.
    os.kill(coordinator, signal.SIGTERM)
    os.kill(coordinator, signal.SIGCONT)
    for pid in survivors:
        assert process(pid) is not None, pid
    state.update(pid=os.getpid(), status='running', active_scope=47,
                 excluded_scope='IEMOCAPFour Flat all15', controller=str(Path(__file__)))
    old.write(root / 'QUEUE.json', state)
    record.update(status='handoff_complete', completed_at=old.now())
    old.write(root / 'CANCEL_FOUR_FLAT.json', record)
    children = {}; halted = False
    while True:
        for i,row in enumerate(rows):
            if row['status'] != 'running': continue
            proc = process(row['pid'])
            if proc and proc['start'] == row['start_ticks'] and proc['state'] != 'Z': continue
            if i in children: children[i].poll()
            prov = root / 'runs' / row['id'] / 'PROVENANCE.json'
            ok = prov.exists() and old.read(prov).get('status') == 'complete' and old.read(prov).get('outputs_verified')
            row['status'] = 'complete' if ok else 'failed'
            halted |= not ok
        for gpu,(uuid,free) in old.gpu_info().items():
            if halted: break
            active = [r for r in rows if r['status']=='running' and r.get('gpu')==gpu]
            count = sum(r['status']=='running' for r in rows)
            if len(active)>=2 or free-len(active)*old.RESERVE<old.RESERVE: continue
            if shutil.disk_usage(root).free < (30+count*5)*2**30: continue
            i = next((i for i,r in enumerate(rows) if r['status']=='pending'), None)
            if i is None: break
            row = rows[i]
            assert not row['id'].startswith('IEMOCAPFour_flat_')
            log = root / 'logs' / (row['id']+'.log')
            command=[sys.executable,'-u',str(source),'--root',str(root),'--train',str(i),'--gpu-uuid',uuid]
            with log.open('x') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            children[i]=child
            proc=process(child.pid); assert proc
            row.update(status='running',pid=child.pid,start_ticks=proc['start'],gpu=gpu,uuid=uuid,
                       command=command,log=str(log))
            old.write(root/'QUEUE.json',state)
            time.sleep(4)
        state['status']='failed_draining' if halted else 'running'
        old.write(root/'QUEUE.json',state)
        if not any(r['status']=='running' for r in rows) and (halted or not any(r['status']=='pending' for r in rows)):
            state.update(status='failed' if halted else 'complete',finished_at=old.now())
            old.write(root/'QUEUE.json',state)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
