"""Four approved direct-input runs; two concurrent admissions on healthy GPU7."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_core20_20261005.run import LABEL, now, write
from experiments.osram_nested_sweep_20261005.dispatch import scores

GPU = 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'
METHODS = ('m03_gatv2_direct', 'm05_pna_direct', 'cwn_cellular_direct', 'perceiver_io_direct')


def available(running):
    raw=subprocess.check_output(['nvidia-smi','--id=7',
        '--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    index,identifier,free=[s.strip() for s in raw.split(',')]
    if index!='7' or identifier!=GPU:
        raise ValueError('Explicit healthy GPU7 whitelist mismatch; GPU4 forbidden')
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                 '--format=csv,noheader,nounits'],text=True)
    used={int(line.split(',')[0]):float(line.split(',')[1]) for line in raw.splitlines()
          if line.strip() and line.split(',')[1].strip().isdigit()}
    growth=sum(max(0,5000-used.get(child.pid,0)) for child,_,_ in running)
    return float(free), float(free)>=growth+5000+2048


def outcome(child,row):
    path=Path(row['output'])/'PROVENANCE.json'
    prov=json.loads(path.read_text()) if path.exists() else {}
    row.update(status='complete' if child.returncode==0 and prov.get('outputs_verified') else 'failed',
               exit_code=child.returncode,error=prov.get('error'),finished_utc=now())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--data-manifest',type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    lock=(root/'DISPATCH.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'DISPATCH.json').exists():
        raise FileExistsError('Inspect prior processes and full recovery checkpoints first')
    source=Path(__file__).resolve().parents[2]
    state=dict(label=LABEL,status='running',pid=os.getpid(),source=str(source),
               gpu=7,gpu_uuid=GPU,seed=66,concurrency=2,started_utc=now(),
               runs=[dict(method=m,status='pending') for m in METHODS])
    write(root/'DISPATCH.json',state)
    data=json.loads(args.data_manifest.read_text())
    running=[]
    while running or any(r['status']=='pending' for r in state['runs']):
        for child,stream,row in list(running):
            if child.poll() is not None:
                stream.close(); outcome(child,row); running.remove((child,stream,row))
                write(root/'DISPATCH.json',state)
        pending=next((r for r in state['runs'] if r['status']=='pending'),None)
        if pending is not None and len(running)<2:
            free,admitted=available(running)
            state['free_mib_last_check']=free
            if admitted and shutil.disk_usage(root).free/2**30>=12:
                output=root/'runs'/pending['method']/'seed_66'
                output.mkdir(parents=True,exist_ok=False)
                command=[sys.executable,'-u','-m','experiments.osram_core20_20261005.run',
                    '--method',pending['method'],'--seed','66','--reference',str(args.reference),
                    '--data-manifest',str(args.data_manifest),'--output',str(output),'--gpu-uuid',GPU]
                log=output/'train.log'; stream=log.open('x')
                child=subprocess.Popen(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,GCNET_DATASET_ROOT=data['dataset_root'],
                             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                pending.update(status='running',pid=child.pid,command=command,output=str(output),
                               log=str(log),free_mib_at_launch=free,started_utc=now())
                running.append((child,stream,pending))
            write(root/'DISPATCH.json',state)
        if running or any(r['status']=='pending' for r in state['runs']):
            time.sleep(15)
    state.update(status='complete' if all(r['status']=='complete' for r in state['runs']) else 'failed',
                 finished_utc=now())
    write(root/'DISPATCH.json',state)
    rows=[dict(method=r['method'],status=r['status'],
          **(scores(Path(r['output'])) if r['status']=='complete' else {'error':r.get('error')}))
          for r in state['runs']]
    write(root/'SUMMARY.json',dict(label=LABEL,runs=rows,reference=scores(args.reference.parent)))


if __name__=='__main__':
    main()
