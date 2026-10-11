"""Recover six interrupted children; preserve47-task scope and isolate sessions."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
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
    lock=(root/'RECOVERY_SCOPE47.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'RECOVERY_SCOPE47.json').exists()
    state=old.read(root/'QUEUE.json');previous=state['pid']
    assert proc(previous)[0] in ('T','t'), 'Previous scheduler must remain paused'
    old.write(root/'QUEUE_BEFORE_RECOVERY_SCOPE47.json',state)
    rows=state['tasks'];recover=[]
    for i,row in enumerate(rows):
        if row['status']=='failed':
            assert row['pid'] in old.read(root/'CANCEL_FOUR_FLAT.json')['untouched_child_pids']
            assert proc(row['pid']) is None or proc(row['pid'])[0]=='Z'
            assert list((root/'runs'/row['id']).glob('stage*/last_training.pt'))
            row.update(status='pending',resume_full_state=True,interrupted_pid=row['pid'])
            recover.append(row['id'])
        elif row['status']=='running':
            assert proc(row['pid']) and proc(row['pid'])[1]==row['start_ticks']
    assert len(recover)==6 and sum(r['status']=='cancelled_by_user' for r in rows)==15
    record=dict(status='recovering',recover_ids=recover,previous_scheduler_paused=previous,
                coordinator_pid=os.getpid(),start_new_session=True,started_at=old.now())
    old.write(root/'RECOVERY_SCOPE47.json',record)
    state.update(pid=os.getpid(),status='running',controller=str(Path(__file__)))
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
            if halted:break
            active=[r for r in rows if r['status']=='running' and r.get('gpu')==gpu]
            count=sum(r['status']=='running' for r in rows)
            if len(active)>=2 or free-len(active)*old.RESERVE<old.RESERVE:continue
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
