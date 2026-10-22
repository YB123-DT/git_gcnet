"""Persistent two-job queue: real-batch smokes then measured concurrent admission."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_source_relation_attention_20261021.run import gpu_free
from .worker import MODES
from .report import summarize


def worker_command(args,mode,phase):
    if mode not in MODES or phase not in ('smoke','train'):
        raise ValueError('Only two approved MOSEI capacity arms')
    return [sys.executable,'-u','-m','experiments.osram_mosei_v3_capacity_20261022.worker',
        '--mode',mode,'--phase',phase,'--seed','66','--gpu-uuid',args.gpu_uuid,
        '--source-commit',args.source_commit,'--output',str(args.output/('smoke' if phase=='smoke' else 'runs')/mode)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args()
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    if args.output.exists():
        raise FileExistsError('Fresh finite queue required; no silent repeats')
    if shutil.disk_usage(args.output.parent).free<20*2**30:
        raise RuntimeError('Need20GiB disk reserve')
    index,free=gpu_free(args.gpu_uuid)
    if available_gib()<64 or free<10000:
        raise RuntimeError('Insufficient64GiB host RAM or10GiB smoke GPU reserve')
    args.output.mkdir()
    lock=(args.output/'QUEUE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='smoke',source_commit=args.source_commit,pid=os.getpid(),
        gpu_uuid=args.gpu_uuid,physical_gpu=index,max_parallel=2,jobs=[],smokes=[])
    def save():
        temporary=args.output/'QUEUE.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(args.output/'QUEUE.json')
    save();tick=time.monotonic();processes=[]
    try:
        for mode in MODES:
            _,free=gpu_free(args.gpu_uuid)
            if available_gib()<64 or free<10000:
                raise RuntimeError('Resource availability changed before smoke')
            log=args.output/('smoke_'+mode+'.log')
            with log.open('w') as stream:
                done=subprocess.run(worker_command(args,mode,'smoke'),stdout=stream,stderr=subprocess.STDOUT)
            p=args.output/'smoke'/mode/'SUMMARY.json'
            s=json.loads(p.read_text()) if p.exists() else {}
            state['smokes'].append(dict(mode=mode,status=s.get('status'),returncode=done.returncode,
                peak_memory_mib=s.get('peak_memory_mib'),seconds=s.get('seconds'),log=str(log)))
            save()
            if done.returncode or s.get('status')!='complete':
                raise RuntimeError('Real-batch smoke failed: '+mode)
        _,free=gpu_free(args.gpu_uuid)
        required=2000+1.6*sum(r['peak_memory_mib'] for r in state['smokes'])
        if free<required or available_gib()<64:
            raise RuntimeError('Measured peaks cannot safely admit both arms; no training launched')
        state.update(status='running',free_before_train_mib=free,estimated_required_mib=required);save()
        for mode in MODES:
            # Recheck resources against this job's measured peak, accounting
            # for its peer even if that process has not allocated memory yet.
            _,now_free=gpu_free(args.gpu_uuid)
            if now_free<required or available_gib()<64:
                raise RuntimeError('Resource admission changed; do not kill already running peer')
            log=args.output/('train_'+mode+'.log')
            with log.open('w') as stream:
                proc=subprocess.Popen(worker_command(args,mode,'train'),stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',
                             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            processes.append(proc)
            state['jobs'].append(dict(mode=mode,pid=proc.pid,status='running',command=worker_command(args,mode,'train'),
                output=str(args.output/'runs'/mode),log=str(log)));save()
        while True:
            active=False
            for proc,job in zip(processes,state['jobs']):
                code=proc.poll()
                p=Path(job['output'])/'SUMMARY.json';s=json.loads(p.read_text()) if p.exists() else {}
                job.update(epoch=s.get('epoch'),worker_status=s.get('status'))
                if code is None:
                    active=True
                else:
                    job.update(returncode=code,status='complete' if code==0 and s.get('status')=='complete' else 'failed')
            state['seconds']=time.monotonic()-tick;save();summarize(args.output)
            if not active:
                break
            time.sleep(15)
        state['status']='complete' if all(r['status']=='complete' for r in state['jobs']) else 'failed'
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['seconds']=time.monotonic()-tick;save()


if __name__=='__main__':
    main()
