"""Persistent two-shard launcher; no main-model training and no duplicate starts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.osram_frozen_memory_audit_20261009.run import GPUS, LABEL, REPO, now, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'LAUNCH.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'LAUNCH.json').exists():
        raise FileExistsError('Inspect existing runs; never relaunch silently')
    if not (REPO/'SNAPSHOT.json').is_file():
        raise ValueError('A sealed source snapshot is required')
    if shutil.disk_usage(root).free < 12*2**30:
        raise ValueError('Need 12GiB available disk')
    for gpu,uuid in GPUS.items():
        text=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        actual,free=[s.strip() for s in text.split(',')]
        if actual!=uuid or float(free)<6000:
            raise ValueError(f'GPU{gpu} health/capacity check failed')
    state=dict(label=LABEL,status='running',pid=os.getpid(),started_utc=now(),source=str(REPO),runs=[])
    write(root/'LAUNCH.json',state)
    children=[]
    for name,gpu,rates in (('low',2,[0,.1,.2,.3]),('high',3,[.4,.5,.6,.7])):
        output=root/'runs'/name
        command=[sys.executable,'-u','-m','experiments.osram_frozen_memory_audit_20261009.run',
                 '--output',str(output),'--gpu',str(gpu),'--rates',*[str(r) for r in rates]]
        log=(root/f'{name}.log').open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[gpu],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['runs'].append(dict(name=name,gpu=gpu,gpu_uuid=GPUS[gpu],pid=child.pid,
                                 output=str(output),log=str(root/f'{name}.log'),command=command))
        children.append((child,log))
        write(root/'LAUNCH.json',state)
    codes=[]
    for (child,log),record in zip(children,state['runs']):
        code=child.wait()
        log.close()
        codes.append(code)
        record['exit_code']=code
        write(root/'LAUNCH.json',state)
    command=[sys.executable,'-m','experiments.osram_frozen_memory_audit_20261009.analyze',
             '--roots',*[str(root/'runs'/name) for name in ('low','high')],'--output',str(root/'summary')]
    with (root/'analysis.log').open('x') as log:
        report=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    summary_path=root/'summary/SUMMARY.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    verified=summary.get('verification_passed') is True and summary.get('partial') is False
    state.update(status='complete' if codes==[0,0] and report.returncode==0 and verified else 'failed',
                 analysis_exit_code=report.returncode,finished_utc=now())
    write(root/'LAUNCH.json',state)
    if state['status']!='complete':
        raise RuntimeError('Diagnostic shard/report failed; inspect preserved logs')


if __name__=='__main__':
    main()
