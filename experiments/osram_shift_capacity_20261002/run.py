"""Four concurrent scalar-filter capacity variants; original single-view protocol."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.osram_paired_history_rho025_20261002.sweep import (
    REFERENCE,DATASET,gpu_check,common,read,write,sha)
VARIANTS={'D2-W128':(2,128),'D2-W256':(2,256),'D3-W128':(3,128),'D3-W256':(3,256)}


def configuration(name):
    common.configuration_dict(66,REFERENCE)
    cfg=read(REFERENCE/'seed_66/config.json')
    depth,width=VARIANTS[name]
    return dict(cfg,osram_readout_fusion='memory-shift-residual',
                osram_shift_filter_depth=depth,osram_shift_filter_width=width)


def train(args):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig,run_experiment
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu
    gpu_check(args.gpu)
    cfg=configuration(args.variant)
    output=args.root/args.variant
    output.mkdir(exist_ok=False)
    record=dict(status='training',config=cfg,code_commit=args.commit,gpu=args.gpu,
                started_at=datetime.now(timezone.utc).isoformat(),label=common.LABEL,
                source_sha256={p:sha(REPO/p) for p in ('gcnet_missing_m3/train_gcnet.py',
                    'gcnet_missing_m3/model.py','gcnet_missing_m3/osram.py')})
    write(output/'PROVENANCE.json',record)
    try:
        torch.set_num_threads(2)
        roots=[str(DATASET/'CMUMOSI/features'/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        run_experiment(TrainConfig(**cfg),*roots,output_dir=str(output))
        common.verify_outputs(output,REFERENCE/'seed_66')
        record['status']='complete'
    except BaseException as e:
        record.update(status='failed',error=repr(e));raise
    finally:
        record['updated_at']=datetime.now(timezone.utc).isoformat()
        write(output/'PROVENANCE.json',record)


def coordinate(args):
    args.root.mkdir(parents=True,exist_ok=False)
    write(args.root/'status.json',dict(status='running',commit=args.commit))
    children=[];records=[]
    try:
        for name in VARIANTS:
            assert gpu_check(args.gpu)>3500,'Insufficient free memory; do not alter batch or kill other users'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2',GCNET_DATASET_ROOT=str(DATASET),PYTHONPATH=str(REPO))
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(args.root),
                     '--variant',name,'--gpu',args.gpu,'--commit',args.commit]
            with (args.root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);records.append(dict(variant=name,pid=child.pid,gpu=args.gpu,status='running'))
            write(args.root/'children.json',records)
            time.sleep(15)
        while True:
            for child,r in zip(children,records):
                result=child.poll();r.update(returncode=result,status='running' if result is None else ('complete' if result==0 else 'failed'))
            write(args.root/'children.json',records)
            if all(c.poll() is not None for c in children): break
            time.sleep(30)
        assert all(c.returncode==0 for c in children),'Run failed; inspect logs'
    except BaseException as e:
        write(args.root/'status.json',dict(status='failed',error=repr(e)));raise
    write(args.root/'status.json',dict(status='complete',commit=args.commit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--gpu',choices=('5','6'),default='5')
    p.add_argument('--commit',required=True);a=p.parse_args()
    if a.variant:train(a)
    else:coordinate(a)
