"""Confirm three prespecified rho values with seeds66/67/68, reusing seed66."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.osram_paired_history_rho025_20261002.sweep import (
    REFERENCE,DATASET,UUIDS,gpu_check,common,read,write,sha)

RHOS=(.05,.10,.25)
OLD=Path('/data2/yb/remote_experiments/osram_paired_rho_sweep_20261002/runs')
OLD_IDS={.05:'C01',.10:'C02',.25:'C05'}


def configuration(rho,seed):
    common.configuration_dict(seed,REFERENCE)  # Validation only; discard its query-adapter variant.
    original=read(REFERENCE/f'seed_{seed}/config.json')
    return dict(original,paired_history_views=True,history_drop_prob=.2,
                history_contrast_weight=0.,history_contrast_temperature=.1,
                history_task_view2_weight=rho)


def run_name(rho,seed):
    return f'rho_{rho:.2f}_seed_{seed}'


def train(args):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig,run_experiment
    gpu='5' if args.seed==67 else '6'
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu
    gpu_check(gpu)
    output=args.root/run_name(args.rho,args.seed)
    output.mkdir(exist_ok=False)
    cfg=configuration(args.rho,args.seed)
    record=dict(status='training',seed=args.seed,rho=args.rho,gpu=gpu,code_commit=args.commit,
                config=cfg,started_at=datetime.now(timezone.utc).isoformat(),
                source_sha256={f:sha(REPO/f) for f in ('gcnet_missing_m3/train_gcnet.py',
                    'gcnet_missing_m3/paired_views.py','gcnet_missing_m3/osram.py','gcnet_missing_m3/model.py')})
    write(output/'PROVENANCE.json',record)
    try:
        torch.set_num_threads(2)
        roots=[str(DATASET/'CMUMOSI/features'/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        run_experiment(TrainConfig(**cfg),*roots,output_dir=str(output))
        common.verify_outputs(output,REFERENCE/f'seed_{args.seed}')
        record['status']='complete'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        record['updated_at']=datetime.now(timezone.utc).isoformat()
        write(output/'PROVENANCE.json',record)


def summarize(root):
    rows=[]
    for rho in RHOS:
        for seed in (66,67,68):
            folder=OLD/OLD_IDS[rho] if seed==66 else root/run_name(rho,seed)
            assert read(folder/'PROVENANCE.json')['status']=='complete'
            assert len(read(folder/'history.json'))==100
            metrics=read(folder/'metrics.json')
            scores=[100*metrics['test'][f'{i/10:.1f}']['weighted_f1'] for i in range(8)]
            base=read(REFERENCE/f'seed_{seed}/metrics.json')
            flat=[100*base['test'][f'{i/10:.1f}']['weighted_f1'] for i in range(8)]
            rows.append(dict(rho=rho,seed=seed,per_rate=scores,mean=statistics.mean(scores),
                high=statistics.mean(scores[5:]),flat_mean=statistics.mean(flat),flat_high=statistics.mean(flat[5:])))
    write(root/'SUMMARY.json',dict(label=common.LABEL,seed66_reused=True,rows=rows))


def coordinate(args):
    children=[];records=[]
    for gpu in UUIDS: assert gpu_check(gpu)>25000
    for rho in RHOS:
        for seed in (67,68):
            gpu='5' if seed==67 else '6'
            assert gpu_check(gpu)>5000
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',GCNET_DATASET_ROOT=str(DATASET),PYTHONPATH=str(REPO))
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(args.root),
                '--seed',str(seed),'--rho',str(rho),'--commit',args.commit]
            with (args.root/(run_name(rho,seed)+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);records.append(dict(seed=seed,rho=rho,gpu=gpu,pid=child.pid,status='running'))
            write(args.root/'children.json',records)
            time.sleep(4)
    while any(c.poll() is None for c in children):
        for child,record in zip(children,records):
            result=child.poll()
            record['status']='running' if result is None else ('complete' if result==0 else 'failed')
            record['returncode']=result
        write(args.root/'children.json',records)
        time.sleep(30)
    if any(c.returncode!=0 for c in children): raise RuntimeError('training failed; inspect logs, no blind restart')
    summarize(args.root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--seed',type=int,choices=(67,68))
    p.add_argument('--rho',type=float,choices=RHOS);a=p.parse_args()
    if a.seed is not None: train(a)
    else:
        a.root.mkdir(parents=True,exist_ok=False)
        write(a.root/'status.json',dict(status='running',code_commit=a.commit))
        try:
            coordinate(a)
        except BaseException as e:
            write(a.root/'status.json',dict(status='failed',error=repr(e)));raise
        write(a.root/'status.json',dict(status='complete',code_commit=a.commit))
