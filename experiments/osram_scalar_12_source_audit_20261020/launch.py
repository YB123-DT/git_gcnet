"""Two finite frozen scalar audits sharing one healthy GPU; never train/retry."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import devices
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib,write

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/data2/yb/reproduction_workspace/envs/s0/bin/python'


def build_jobs(output,source_commit,gpu_uuid):
    return [dict(dataset=d,output=str(Path(output)/d),status='pending',
        command=[PYTHON,'-u','-m','experiments.osram_scalar_12_source_audit_20261020.run',
            '--dataset',d,'--gpu-uuid',gpu_uuid,'--output',str(Path(output)/d),'--source-commit',source_commit])
        for d in ('CMUMOSI','CMUMOSEI')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    lock=(args.output/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    card=devices([args.gpu_uuid])[args.gpu_uuid]
    if (shutil.disk_usage(args.output).free<12*2**30 or available_gib()<32
            or card['total_mib']-card['used_mib']<12000):
        raise RuntimeError('Require12GiB disk,32GiB RAM,12GB free healthy GPU; no peers touched')
    jobs=build_jobs(args.output,args.source_commit,args.gpu_uuid)
    plan=dict(status='running',no_training=True,source_commit=args.source_commit,gpu_uuid=args.gpu_uuid,
        gpu=card,pid=os.getpid(),jobs=jobs,max_processes=2,no_expansion=True,started_at=time.time())
    write(args.output/'QUEUE.json',plan)
    processes=[]
    for job in jobs:
        log=(args.output/(job['dataset']+'.log')).open('x')
        process=subprocess.Popen(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            start_new_session=True)
        job.update(status='running',pid=process.pid,log=log.name)
        processes.append((job,process,log));write(args.output/'QUEUE.json',plan)
    while processes:
        pending=[]
        for job,process,log in processes:
            code=process.poll()
            if code is None: pending.append((job,process,log));continue
            log.close();f=Path(job['output'])/'SUMMARY.json'
            s=json.loads(f.read_text()) if f.exists() else {}
            success=code==0 and s.get('status')=='complete' and s.get('backbone_unchanged')
            job.update(status='complete' if success else 'failed',returncode=code,error=s.get('error'),finished_at=time.time())
        processes=pending;write(args.output/'QUEUE.json',plan)
        if processes: time.sleep(5)
    plan.update(status='complete' if all(j['status']=='complete' for j in jobs) else 'failed',finished_at=time.time())
    write(args.output/'QUEUE.json',plan)
    if plan['status']!='complete': raise RuntimeError('Audit failed; no automatic retry or new training')


if __name__=='__main__': main()
