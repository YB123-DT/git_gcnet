"""Exactly two current-missing Flat runs, resource checked and concurrent."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import (
    BAD_UUID,make_job,devices,launch,poll,recover_jobs,validate_summary)
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib,write


def plan_jobs(output,sha,gpus):
    if len(gpus)!=1 or not gpus[0].startswith('GPU-') or gpus[0]==BAD_UUID:raise ValueError('One healthy GPU required')
    return [make_job('experiments.osram_mosei_gcnet_flat_lr1e4_20261021.worker',
        Path(output)/'runs'/'CMUMOSEI'/f'seed_{seed}'/'flat',sha,gpus[0],'flat',seed,'CMUMOSEI') for seed in (67,68)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();jobs=plan_jobs(args.output,args.source_commit,[args.gpu_uuid])
    if args.output.exists() and not args.resume:raise FileExistsError('Fresh queue required')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'QUEUE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'QUEUE.json'
    if args.resume:
        state=json.loads(path.read_text())
        if state['source_commit']!=args.source_commit or state['gpu_uuid']!=args.gpu_uuid:raise ValueError('Resume identity differs')
        if state['status']=='complete':raise ValueError('Completed queue must not repeat')
        recover_jobs(state)
    else:state=dict(status='preflight',source_commit=args.source_commit,gpu_uuid=args.gpu_uuid,
        pid=os.getpid(),jobs=jobs,started_at=time.time(),maximum_parallel=2,
        verification='Reuse identical original Flat numerical path and completed GPU checks; no new model math')
    children={}
    try:
        if shutil.disk_usage(args.output).free<12*2**30:raise RuntimeError('Require12GiB disk; do not delete old checkpoints')
        while any(j['status'] in ('pending','running') for j in state['jobs']):
            for index,process in list(children.items()):
                if poll(process,state['jobs'][index]):children.pop(index)
            failed=any(j['status']=='failed' for j in state['jobs'])
            for index,j in enumerate(state['jobs']):
                if failed or j['status']!='pending':continue
                gpu=devices([args.gpu_uuid])[args.gpu_uuid]
                # Prior same-data direct-source peak was1.8GiB; allocate a
                # conservative4GiB per original Flat plus2GiB card margin.
                if gpu['total_mib']-gpu['used_mib']<4096*2+2048 or available_gib()<64:continue
                children[index]=launch(j);write(path,state);time.sleep(2)
            state.update(status='draining_after_failure' if failed else 'training',updated_at=time.time());write(path,state)
            if failed and not children:raise RuntimeError('Child failure; no automatic retry')
            time.sleep(5)
        if any(j['status']!='complete' for j in state['jobs']):raise RuntimeError('Incomplete jobs')
        summaries=[validate_summary(j) for j in state['jobs']]
        result=dict(status='complete',label='INTERNAL TEST-ORACLE SCREEN',
            per_seed={str(s['seed']):dict(aggregate=s['aggregate']['router'],evaluation=s['evaluation']) for s in summaries})
        write(args.output/'RESULT.json',result)
        lines=['# MOSEI Flat67/68 constant1e-4,GCNet missing','',result['label'],'',
            'Not a matched learning-rate control for the previously trained1e-3 direct-source model.','',
            '| Seed | Rate | Selected epoch | ACC | W-F1 |','|---:|---|---:|---:|---:|']
        for s in summaries:
            for r,v in s['evaluation'].items():lines.append(f"| {s['seed']} | {r} | {v['selected_epoch']} | {v['router']['ACC']:.4f} | {v['router']['WF1']:.4f} |")
            for scope in ('mean','high'):
                v=s['aggregate']['router'][scope];lines.append(f"| {s['seed']} | {scope} | — | {v['ACC']:.4f} | {v['WF1']:.4f} |")
        (args.output/'RESULT.md').write_text('\n'.join(lines)+'\n')
        state.update(status='complete',completed_at=time.time());write(path,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),surviving_children=[p.pid for p in children.values()]);write(path,state);raise


if __name__=='__main__':main()
