"""Drain GPU7; admit at most five jobs each on GPUs1/2/3 without restarts."""
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


def proc(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return s[0],s[19]
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root
    source=root/'source_4eed138/run.py'
    spec=importlib.util.spec_from_file_location('runner',source)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    lock=(root/'GPU123_FIVE.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'GPU123_FIVE.json').exists()
    state=old.read(root/'QUEUE.json');previous=state['pid']
    command=Path(f'/proc/{previous}/cmdline').read_bytes().replace(b'\\0', b' ').decode()
    assert str(root/'schedule_gpu123_three.py') in command
    os.kill(previous,signal.SIGSTOP)
    time.sleep(.2)
    assert proc(previous)[0] in ('T','t')
    state=old.read(root/'QUEUE.json')
    old.write(root/'QUEUE_BEFORE_GPU123_FIVE.json',state)
    rows=state['tasks'];running=[]
    for row in rows:
        if row['status']=='running':
            assert proc(row['pid']) and proc(row['pid'])[1]==row['start_ticks']
            running.append(dict(id=row['id'],pid=row['pid'],gpu=row['gpu']))
    assert sum(r['status']=='cancelled_by_user' for r in rows)==15
    record=dict(status='adopted',preserved_running=running,
                previous_scheduler_paused=previous,coordinator_pid=os.getpid(),
                new_gpu_whitelist=[1,2,3],max_per_gpu=5,drain_only=[7],
                controller_sha256=old.sha(Path(__file__)),started_at=old.now())
    old.write(root/'GPU123_FIVE.json',record)
    state.update(pid=os.getpid(),status='running',controller=str(Path(__file__)),
                 gpu_whitelist=[1,2,3],max_per_gpu=5,drain_only=[7])
    old.write(root/'QUEUE.json',state)
    children={};halted=False
    while True:
        for i,row in enumerate(rows):
            if row['status']!='running':continue
            info=proc(row['pid'])
            if info and info[1]==row['start_ticks'] and info[0]!='Z':continue
            if i in children:children[i].poll()
            path=root/'runs'/row['id']/'PROVENANCE.json'
            ok=path.exists() and old.read(path).get('status')=='complete' and old.read(path).get('outputs_verified')
            row['status']='complete' if ok else 'failed';halted|=not ok
        for gpu,(uuid,free) in old.gpu_info().items():
            if gpu not in (1,2,3):continue
            if halted:break
            active=[r for r in rows if r['status']=='running' and r.get('gpu')==gpu]
            count=sum(r['status']=='running' for r in rows)
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                         '--format=csv,noheader,nounits'],text=True)
            used={}
            for line in raw.splitlines():
                fields=[x.strip() for x in line.split(',')]
                if len(fields)==2 and fields[0].isdigit() and fields[1].isdigit():
                    used[int(fields[0])]=int(fields[1])
            # Reserve peak headroom only, not both peak and already allocated VRAM.
            adjusted=free-sum(max(0,6000-used.get(r['pid'],0)) for r in active)
            if len(active)>=5 or adjusted<6000+1500:continue
            if shutil.disk_usage(root).free<(30+count*5)*2**30:continue
            i=next((i for i,r in enumerate(rows) if r['status']=='pending'),None)
            if i is None:break
            row=rows[i];assert not row['id'].startswith('IEMOCAPFour_flat_')
            recovery=row.get('resume_full_state',False)
            script=root/'recover_run.py' if recovery else source
            log=root/'logs'/(row['id']+('_recovery' if recovery else '')+'.log')
            cmd=[sys.executable,'-u',str(script),'--root',str(root),'--train',str(i),'--gpu-uuid',uuid]
            with log.open('x') as stream:
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            children[i]=child
            row.update(status='running',pid=child.pid,start_ticks=proc(child.pid)[1],gpu=gpu,uuid=uuid,command=cmd,log=str(log))
            old.write(root/'QUEUE.json',state);time.sleep(4)
        state['status']='failed_draining' if halted else 'running';old.write(root/'QUEUE.json',state)
        if not any(r['status']=='running' for r in rows) and (halted or not any(r['status']=='pending' for r in rows)):
            state.update(status='failed' if halted else 'complete',finished_at=old.now());old.write(root/'QUEUE.json',state);return
        time.sleep(15)


if __name__=='__main__':main()


