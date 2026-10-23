"""One seed66 completion, then automatically publish three-seed Flat records."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

OLD = Path('/data2/yb/remote_experiments/osram_mosei_gcnet_flat_lr1e4_20261021/execution_v1/runs/CMUMOSEI')


def train_command(output, source_commit, gpu_uuid, resume=False):
    command = [sys.executable, '-u', '-m', 'experiments.osram_mosei_gcnet_flat_lr1e4_20261021.worker',
        '--phase', 'train', '--mode', 'flat', '--seed', '66', '--source-commit', source_commit,
        '--gpu-uuid', gpu_uuid, '--output', str(output)]
    return command + (['--resume'] if resume else [])


def publish(output, new_run):
    # Read selected predictions only; no extra model inference or mask sampling.
    import numpy as np
    from experiments.osram_tasr_20261018.runner_helpers import sha, write_json
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    from experiments.osram_value_covariance_20261023.report import binary_metrics
    rows, seeds = [], {}
    for seed in (66,67,68):
        directory = new_run if seed == 66 else OLD/f'seed_{seed}'/'flat'
        s = json.loads((directory/'SUMMARY.json').read_text())
        if (s['status'] != 'complete' or s['epoch'] != 60 or s['seed'] != seed
                or s['mode'] != 'flat' or s['config']['learning_rate'] != 1e-4
                or s['config']['missing_mask_protocol'] != 'gcnet-random-mask-v1'):
            raise ValueError('Require completed matching original Flat trajectory')
        validate_selected(directory, s['best_by_rate'])
        for rate, best in s['best_by_rate'].items():
            with np.load(directory/best['predictions'], allow_pickle=False) as a:
                y, p = a['labels'].reshape(-1), a['prediction'].reshape(-1)
                if len(y) != 4659 or not np.isfinite(p).all():
                    raise ValueError('Missing/invalid selected predictions')
                wf = binary_metrics(y,p)['W_F1']
                if abs(wf-best['metrics']['router']['WF1']) > 1e-8:
                    raise ValueError('Selected metric/prediction mismatch')
            row = dict(seed=seed, rate=rate, epoch=best['epoch'], **best['metrics']['router'])
            rows.append(row)
        seeds[str(seed)] = dict(path=str(directory), summary_sha256=sha(directory/'SUMMARY.json'),
            source_commit=s['source_commit'], reused=seed != 66, aggregate=s['aggregate']['router'])
    aggregate = {scope:{metric:dict(mean=statistics.mean(s['aggregate'][scope][metric] for s in seeds.values()),
                                    sample_sd=statistics.stdev(s['aggregate'][scope][metric] for s in seeds.values()))
                        for metric in ('ACC','WF1','MSE')} for scope in ('mean','high')}
    write_json(output/'RESULT.json',dict(status='complete',label='INTERNAL TEST-ORACLE SCREEN',seeds=seeds,aggregate=aggregate))
    with (output/'PER_RATE.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# MOSEI original Flat constant1e-4 three-seed completion','',
        'INTERNAL TEST-ORACLE SCREEN. Only seed66 is new;67/68 reused read-only.',
        '60epochs,GCNet-random-mask-v1,original Flat;each rate keeps its original TEST-best W-F1 checkpoint.',
        'Masks/seed-specific initialization follow the existing low-LR protocol. No new inference or reselection.',
        '', '| Seed | Eight-rate ACC | W-F1 | High ACC | High W-F1 |','|---:|---:|---:|---:|---:|']
    for seed, s in seeds.items():
        a,b=s['aggregate']['mean'],s['aggregate']['high']
        lines.append(f"| {seed} | {a['ACC']:.4f} | {a['WF1']:.4f} | {b['ACC']:.4f} | {b['WF1']:.4f} |")
    lines += ['', 'Three-seed sample mean/SD and source identities are in RESULT.json; selected metrics/epochs in PER_RATE.csv.',
        'Eight-rate mean can combine eight checkpoints and is not single-model deployment performance.',
        'Exploratory test selection, not independent generalization evidence. No automatic new models/HPO/seeds.','']
    (output/'RESULT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    from experiments.osram_tasr_20261018.runner_helpers import write_json
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_source_relation_attention_20261021.run import gpu_free
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    gpu=guard_gpu(args.gpu_uuid)
    if args.output.exists() and not args.resume:raise FileExistsError('Use independent new output')
    if available_gib()<64 or shutil.disk_usage(args.output.parent).free<12*2**30 or gpu_free(args.gpu_uuid)[1]<6144:
        raise RuntimeError('Insufficient headroom; no tasks were stopped or settings changed')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record=args.output/'QUEUE.json';run=args.output/'runs'/'CMUMOSEI'/'seed_66'/'flat'
    command=train_command(run,args.source_commit,args.gpu_uuid,args.resume)
    state=dict(status='preflight',seed=66,dataset='CMUMOSEI',mode='flat',learning_rate=1e-4,
        epochs=60,source_commit=args.source_commit,gpu=gpu,command=command,pid=os.getpid(),started_at=time.time())
    write_json(record,state)
    try:
        with (args.output/'train_seed66.log').open('a' if args.resume else 'w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            state.update(status='running',worker_pid=process.pid,log=str(args.output/'train_seed66.log'));write_json(record,state)
            code=process.wait()
        if code:raise RuntimeError('Worker failed with exit '+str(code)+'; see retained log')
        publish(args.output,run)
        state.update(status='complete',completed_at=time.time())
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:write_json(record,state)


if __name__=='__main__':main()
