"""Ten authorized rho runs, five concurrent processes on each of host GPUs5/6."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.osram_paired_history_views_20261001.run import common, read, write, sha

RHOS=(.05,.10,.15,.20,.25,.30,.35,.40,.45,.60)
UUIDS={'5':'GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62','6':'GPU-e4cafb17-818e-216a-b94a-7440063a9153'}
REFERENCE=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full')
DATASET=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')


def config(index):
    base=read(Path(__file__).with_name('config.json'))
    assert base['history_contrast_weight']==0 and base['epochs']==100 and base['seed']==66
    return dict(base,history_task_view2_weight=RHOS[index])


def gpu_check(gpu):
    values=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.free',
                                    '--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert values[0].strip()==UUIDS[gpu]
    return int(values[1])


def train(args):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig,run_experiment
    gpu='5' if args.index<5 else '6'
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu
    gpu_check(gpu)
    out=args.root/f'C{args.index+1:02d}'
    out.mkdir(exist_ok=False)
    cfg=config(args.index)
    record=dict(status='training',gpu=gpu,gpu_uuid=UUIDS[gpu],config=cfg,
        started_at=datetime.now(timezone.utc).isoformat(),code_commit=args.commit,
        source_sha256={p:sha(REPO/p) for p in ['gcnet_missing_m3/train_gcnet.py',
            'gcnet_missing_m3/paired_views.py','gcnet_missing_m3/model.py','gcnet_missing_m3/osram.py']},
        label=common.LABEL,from_scratch=True)
    write(out/'PROVENANCE.json',record)
    try:
        torch.set_num_threads(2)
        roots=[str(DATASET/'CMUMOSI/features'/name) for name in
               ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        run_experiment(TrainConfig(**cfg),*roots,output_dir=str(out))
        common.verify_outputs(out,REFERENCE/'seed_66')
        record['status']='complete'
    except BaseException as error:
        record.update(status='failed',error=repr(error))
        raise
    finally:
        record['updated_at']=datetime.now(timezone.utc).isoformat()
        write(out/'PROVENANCE.json',record)


def launch(args):
    args.root.mkdir(parents=True,exist_ok=False)
    for gpu in UUIDS:
        assert gpu_check(gpu)>28000,'Require empty cards for five-way concurrency'
    records=[]
    # Interleave cards. No wait for a training process to finish before launching another.
    for index in (0,5,1,6,2,7,3,8,4,9):
        gpu='5' if index<5 else '6'
        assert gpu_check(gpu)>4500,'Insufficient remaining memory; do not change batch size'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                 OPENBLAS_NUM_THREADS='2',GCNET_DATASET_ROOT=str(DATASET),PYTHONPATH=str(REPO))
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(args.root),
                 '--index',str(index),'--commit',args.commit]
        with (args.root/f'C{index+1:02d}.log').open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(run=f'C{index+1:02d}',rho=RHOS[index],gpu=gpu,pid=child.pid,command=command))
        write(args.root/'launch.json',dict(code_commit=args.commit,children=records,
              started_at=datetime.now(timezone.utc).isoformat(),requested_concurrency=10))
        print(records[-1],flush=True)
        time.sleep(4)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--index',type=int,choices=range(10));p.add_argument('--commit',required=True)
    a=p.parse_args()
    if a.index is None: launch(a)
    else: train(a)
