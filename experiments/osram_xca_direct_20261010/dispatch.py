"""Two approved direct evidence transforms, original protocol, concurrent GPU7."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_core20_20261005.run import LABEL, now, write
from experiments.osram_nested_sweep_20261005.dispatch import scores

GPU = 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'
METHODS = ('m28_xcit_xca_direct','neural_production_direct')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--data-manifest',type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'DISPATCH.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'DISPATCH.json').exists():
        raise FileExistsError('Inspect prior runs before any resubmission')
    source=Path(__file__).resolve().parents[2]
    state=dict(label=LABEL,status='pending',pid=os.getpid(),source=str(source),
               gpu=7,gpu_uuid=GPU,seed=66,runs=[],started_utc=now())
    while True:
        raw=subprocess.check_output(['nvidia-smi','--id=7',
            '--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        index,identifier,free=[s.strip() for s in raw.split(',')]
        if index!='7' or identifier!=GPU:
            raise ValueError('Healthy physical GPU7 UUID mismatch; GPU4 forbidden')
        state['free_mib']=float(free)
        write(root/'DISPATCH.json',state)
        if float(free)>=12500 and shutil.disk_usage(root).free/2**30>=26:
            break
        time.sleep(15)
    children=[]
    try:
        for method in METHODS:
            output=root/'runs'/method/'seed_66'
            output.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,'-u','-m','experiments.osram_core20_20261005.run',
                '--method',method,'--seed','66','--reference',str(args.reference),
                '--data-manifest',str(args.data_manifest),'--output',str(output),'--gpu-uuid',GPU]
            stream=(output/'train.log').open('x')
            child=subprocess.Popen(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='2',
                         MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            row=dict(method=method,status='running',pid=child.pid,command=command,
                     output=str(output),log=str(output/'train.log'),started_utc=now())
            state['runs'].append(row)
            children.append((child,stream,row))
            state['status']='running'
            write(root/'DISPATCH.json',state)
        while any(child.poll() is None for child,_,_ in children):
            for child,stream,row in children:
                code=child.poll()
                if code is not None and row['status']=='running':
                    stream.close()
                    path=Path(row['output'])/'PROVENANCE.json'
                    import json
                    prov=json.loads(path.read_text()) if path.exists() else {}
                    row.update(status='complete' if code==0 and prov.get('outputs_verified') else 'failed',
                               exit_code=code,error=prov.get('error'),finished_utc=now())
                    write(root/'DISPATCH.json',state)
            time.sleep(15)
        for child,stream,row in children:
            stream.close()
            import json
            path=Path(row['output'])/'PROVENANCE.json'
            prov=json.loads(path.read_text()) if path.exists() else {}
            row.update(status='complete' if child.returncode==0 and prov.get('outputs_verified') else 'failed',
                       exit_code=child.returncode,error=prov.get('error'),finished_utc=now())
        state['status']='complete' if all(r['status']=='complete' for r in state['runs']) else 'failed'
        state['finished_utc']=now()
        write(root/'DISPATCH.json',state)
        table=[dict(method=r['method'],status=r['status'],
                    **(scores(Path(r['output'])) if r['status']=='complete' else {})) for r in state['runs']]
        write(root/'SUMMARY.json',dict(label=LABEL,runs=table,reference=scores(args.reference.parent)))
    finally:
        for _,stream,_ in children:
            stream.close()


if __name__=='__main__':
    main()
