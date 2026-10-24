"""Seven new jobs on physical GPU0/1; account for the already running Flat256."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .worker import MODES

GPU0='GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'
GPU1='GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'
EXTERNAL=Path('/data2/yb/remote_experiments/osram_mosei_flat256_20261024/execution_v1/runs/flat256')
ROOT=Path(__file__).resolve().parents[2]


def plan_jobs(output,source_commit):
    order=('f128','f384','f768','f256_n64','f256_n128','f512_n128','f512_n256')
    jobs=[]
    for i,mode in enumerate(order):
        uuid=GPU0 if i<4 else GPU1
        path=Path(output)/'runs'/mode
        cmd=[sys.executable,'-u','-m','experiments.osram_mosei_flat_nested_capacity_20261024.worker',
            '--mode',mode,'--phase','train','--seed','66','--gpu-uuid',uuid,
            '--source-commit',source_commit,'--output',str(path)]
        jobs.append(dict(mode=mode,output=str(path),command=cmd,log=str(path)+'.log',
            gpu_uuid=uuid,reserve_mib=8192 if mode=='f512_n256' else 6144,status='pending'))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib,write
    from experiments.osram_mosi_v3_capacity_20261022.run import resident_memory
    from experiments.osram_source_relation_attention_20261021.run import gpu_free
    from .report import summarize
    if args.output.exists() and not args.resume:raise FileExistsError('Independent output required')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'QUEUE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT.parent/'SOURCE_MANIFEST.json').read_text())
    def verify():
        for relative,digest in manifest['source_hashes'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Immutable source changed: '+relative)
    verify()
    for uuid,index in ((GPU0,0),(GPU1,1)):
        if gpu_free(uuid)[0]!=index:raise RuntimeError('Physical GPU UUID mapping changed')
    jobs=plan_jobs(args.output,args.source_commit)
    if args.resume:
        state=json.loads((args.output/'QUEUE.json').read_text())
        if state['source_commit']!=args.source_commit:raise ValueError('Source identity differs')
        jobs=state['jobs']
        for j in jobs:
            if j['status']=='complete':continue
            if j['status']=='failed':raise ValueError('No silent failed-job retry')
            if j['status']=='running':
                if Path('/proc/'+str(j['pid'])).exists():raise RuntimeError('Worker alive; refuse duplicate')
                s=json.loads((Path(j['output'])/'SUMMARY.json').read_text())
                if s.get('status')=='complete':j['status']='complete';continue
                if not (Path(j['output'])/'last.pt').exists():raise ValueError('No full recovery checkpoint')
                j['status']='pending'
                if '--resume' not in j['command']:j['command'].append('--resume')
        state.update(pid=os.getpid(),status='resuming')
    else:
        state=dict(status='pending',pid=os.getpid(),source_commit=args.source_commit,jobs=jobs,
            maximum_new_jobs=7,total_including_existing=8,maximum_per_gpu=4,
            existing_flat256=str(EXTERNAL),started_at=time.time(),label='INTERNAL TEST-ORACLE SCREEN')
    children={}
    def record():
        state['updated_at']=time.time();write(args.output/'QUEUE.json',state)
    record()
    while True:
        for i,proc in list(children.items()):
            code=proc.poll()
            if code is None:continue
            j=jobs[i];f=Path(j['output'])/'SUMMARY.json'
            s=json.loads(f.read_text()) if f.exists() else {}
            j.update(status='complete' if code==0 and s.get('status')=='complete' and s.get('epoch')==60 else 'failed',
                returncode=code,finished_at=time.time());del children[i]
        e=json.loads((EXTERNAL/'SUMMARY.json').read_text())
        state['existing_flat256_status']=e['status'];state['existing_flat256_epoch']=e.get('epoch')
        external_live=e['status'] not in ('failed','complete')
        if external_live and not Path('/proc/'+str(e['pid'])).exists():
            raise RuntimeError('Existing Flat256 exited without completed output; no duplicate restart')
        failed=any(j['status']=='failed' for j in jobs) or e['status']=='failed'
        if failed:
            state['status']='failure_draining_active'
            if not children:state['status']='failed';record();return
        elif not children and all(j['status']=='complete' for j in jobs) and not external_live:
            summarize(args.output,state,final=True)
            state.update(status='complete',finished_at=time.time());record();return
        else:
            resident=resident_memory()
            for uuid in (GPU0,GPU1):
                live=[j for j in jobs if j['status']=='running' and j['gpu_uuid']==uuid]
                if uuid==GPU1 and external_live:
                    live=live+[dict(pid=e['pid'],reserve_mib=6144)]
                for i,j in enumerate(jobs):
                    if j['status']!='pending' or j['gpu_uuid']!=uuid or len(live)>=4:continue
                    required=2048+j['reserve_mib']+sum(max(0,x['reserve_mib']-resident.get(x['pid'],0)) for x in live)
                    if available_gib()<64 or shutil.disk_usage(args.output).free<20*2**30 or gpu_free(uuid)[1]<required:continue
                    verify();Path(j['output']).parent.mkdir(parents=True,exist_ok=True)
                    with Path(j['log']).open('a' if '--resume' in j['command'] else 'x') as log:
                        proc=subprocess.Popen(j['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,start_new_session=True,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                    j.update(pid=proc.pid,status='running',started_at=time.time())
                    children[i]=proc;live.append(j);record();time.sleep(4)
            state['status']='running' if children or external_live else 'waiting_for_resources'
        record();summarize(args.output,state,final=False);time.sleep(15)


if __name__=='__main__':main()
