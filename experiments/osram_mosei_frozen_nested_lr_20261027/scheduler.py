"""Two LR settings, sixteen own-rate jobs; healthy GPU0/1/2, max4/card."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .protocol import EPOCHS,LRS,NUMERIC,PARAMETERS,jobs_for,validate_scope
from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory,now,resources,save

GPUS={0:'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
      1:'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049',
      2:'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
MODULE='experiments.osram_mosei_frozen_nested_lr_20261027.run'


def validate_state(state,source):
    if state.get('source_commit')!=source:raise ValueError('Immutable source changed')
    for phase in ('smoke','train'):
        expected={j['id']:j for j in jobs_for(phase)}
        actual=state.get(phase,[])
        if len(actual)!=len(expected) or len({j['id'] for j in actual})!=len(expected):
            raise ValueError('Incomplete or expanded finite queue')
        for j in actual:
            if j['id'] not in expected or any(j[k]!=expected[j['id']][k] for k in ('config','rate')):
                raise ValueError('Changed approved LR/rate scope')
            validate_scope(j['config'],j['rate'])


def matches(job):
    try:
        p=Path('/proc')/str(job['pid'])
        argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        return p.stat().st_uid==os.getuid() and argv==job['command'] and (p/'cwd').resolve()==NUMERIC
    except (KeyError,OSError,UnicodeError):return False


def finish(job,code,phase,source):
    job.update(returncode=code,finished_at=now())
    try:
        summary=json.loads((Path(job['output'])/'SUMMARY.json').read_text())
        expected='smoke_passed' if phase=='smoke' else 'complete'
        identity=summary['identity']
        if (code!=0 or summary['status']!=expected or summary['epoch']!=(2 if phase=='smoke' else EPOCHS)
            or summary['total_parameters']!=PARAMETERS or not all(summary['checks'].values())
            or identity['source_commit']!=source or identity['config']!=job['config']
            or identity['evaluation_rate']!=job['rate'] or identity['learning_rate']!=LRS[job['config']]):
            raise ValueError('Worker incomplete/mismatched; inspect log')
        job.update(status=expected,summary=str(Path(job['output'])/'SUMMARY.json'),
            peak_memory_mib=summary['peak_memory_mib'],peak_rss_gib=summary['peak_rss_gib'])
    except Exception as error:job.update(status='failed',error=repr(error))


def launch(args,job,phase,uuid):
    output=args.root/phase/job['id']
    resume=output.exists()
    if resume and (not args.resume or not (output/'last.pt').exists()):
        raise ValueError('Existing run requires explicit full-state recovery')
    output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m',MODULE,'--config',job['config'],'--rate',job['rate'],
        '--output',str(output),'--gpu-uuid',uuid,'--source-commit',args.source_commit]
    if phase=='smoke':command+=['--smoke']
    if resume:command+=['--resume']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (output/'worker.log').open('a') as log:
        process=subprocess.Popen(command,cwd=NUMERIC,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job.update(status='running',pid=process.pid,command=command,cwd=str(NUMERIC),gpu_uuid=uuid,
        source_commit=args.source_commit,output=str(output),log=str(output/'worker.log'),started_at=now(),resume=resume)
    return process


def run_phase(args,state,phase):
    children={}
    jobs=state[phase]
    if args.resume:
        for job in jobs:
            if matches(job):job['status']='running';continue
            if job.get('output') and (Path(job['output'])/'SUMMARY.json').exists():
                finish(job,0,phase,args.source_commit)
                if job['status']!='failed':continue
            if job.get('output') and not (Path(job['output'])/'last.pt').exists():
                raise ValueError('Partial run lacks recoverable block/Adam/RNG state')
            job['status']='pending'
    while any(j['status'] in ('pending','running') for j in jobs):
        for job in jobs:
            if job['status']!='running':continue
            child=children.get(job['id'])
            code=child.poll() if child is not None else (None if matches(job) else 0)
            if code is not None:
                finish(job,code,phase,args.source_commit);children.pop(job['id'],None)
        if any(j['status']=='failed' for j in jobs):
            for job in jobs:
                if job['status']=='pending':job.update(status='blocked',error='Worker failure; other workers preserved')
        else:
            inventory=gpu_inventory()
            if any(inventory[u]['host_index']!=i for i,u in GPUS.items()):
                raise RuntimeError('Healthy physical GPU/UUID mapping changed')
            snap=resources(args.root,list(GPUS.values()));state['latest_resources']=snap
            pending=next((j for j in jobs if j['status']=='pending'),None)
            if pending:
                # Completed same-architecture full runs peaked near1GiB/4GiB RAM.
                # Conservative floors also reserve imports/cache work before visible allocation.
                measured=[j for j in state['smoke']+state['train'] if 'peak_memory_mib' in j]
                peak=max([2048.]+[j['peak_memory_mib'] for j in measured])
                ram=max([8.]+[j['peak_rss_gib'] for j in measured])
                for uuid in sorted(GPUS.values(),key=lambda u:sum(j['status']=='running' and j.get('gpu_uuid')==u for j in jobs)):
                    peers=[j for j in jobs if j['status']=='running' and j.get('gpu_uuid')==uuid]
                    limit=1 if phase=='smoke' else args.per_gpu
                    if (len(peers)<limit and snap['gpus'][uuid]['free_mib']>=4096+1.25*peak*(len(peers)+1)
                        and snap['ram_available_gib']>=24+ram*(len(peers)+1) and snap['disk_free_gib']>=24):
                        children[pending['id']]=launch(args,pending,phase,uuid);break
        state.update(status=phase,updated_at=now())
        save(args.root/'QUEUE.json',state)
        if any(j['status'] in ('pending','running') for j in jobs):time.sleep(3)
    if any(j['status']!=('smoke_passed' if phase=='smoke' else 'complete') for j in jobs):
        raise RuntimeError('Failed/blocked '+phase+' jobs; no automatic protocol changes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--per-gpu',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.per_gpu<=4:raise ValueError('At most4/card and12 global')
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.root/'QUEUE.json'
    if path.exists() and not args.resume:raise FileExistsError('Existing queue; explicit resume required')
    state=(json.loads(path.read_text()) if path.exists() else dict(source_commit=args.source_commit,
        smoke=jobs_for('smoke'),train=jobs_for('train'),gpu_map=GPUS,started_at=now(),
        pid=os.getpid(),command=sys.argv,expected_train_jobs=16))
    validate_state(state,args.source_commit)
    try:
        run_phase(args,state,'smoke');run_phase(args,state,'train')
        from .summarize import summarize
        summarize(args.root,args.source_commit,state['train'])
        state.update(status='complete',finished_at=now())
    except BaseException as error:
        state.update(status='failed',error=repr(error),updated_at=now(),
            preserved_live_pids=[j['pid'] for p in ('smoke','train') for j in state[p] if matches(j)])
        raise
    finally:save(path,state)


if __name__=='__main__':main()
