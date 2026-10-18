"""Eight finite MELD Flat jobs: fixed LR1e-5, four WD values, seeds67/68.

The existing verified optimizer-screen worker is reused in isolated processes.
Only orchestration globals are adapted; numerical trainer/model/masks are not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.osram_meld_optimizer_screen_20261018 import run as screen
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler,write_json

SEEDS=(67,68)
WEIGHT_DECAYS=(0.,1e-5,1e-4,5e-4)
CONTROL_ROOT=Path('/home/qinjinghui/remote_experiments/osram_meld_small_lr_3seed_20261018/runs_v1')
GPU_UUIDS={0:'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
    2:'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
_ORIGINAL_SOURCE_CONFIG=screen.source_config
_ORIGINAL_CHECK_SNAPSHOT=screen.check_snapshot


def check_seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only seeds67/68 are authorized for this paired follow-up')


def control_config(seed):
    check_seed(seed)
    return dict(_ORIGINAL_SOURCE_CONFIG(),seed=seed,learning_rate=1e-5,weight_decay=2e-4)


def make_config(seed,weight_decay):
    check_seed(seed)
    if type(weight_decay) not in (int,float) or weight_decay not in WEIGHT_DECAYS:
        raise ValueError('Only four predeclared weight-decay values are authorized')
    return dict(control_config(seed),weight_decay=weight_decay)


def job_specs():
    return [dict(id=f'S{seed}_WD{index}',seed=seed,weight_decay=wd)
        for seed in SEEDS for index,wd in enumerate(WEIGHT_DECAYS)]


def validate_control(path,seed):
    summary=json.loads((path/'SUMMARY.json').read_text())
    trace=json.loads((path/'MASK_TRACE.json').read_text())
    initial=json.loads((path/'INITIAL_STATE.json').read_text())
    identity=summary.get('identity',{})
    if (summary.get('status')!='complete' or summary.get('config')!=control_config(seed)
            or summary.get('parameters')!=13290795
            or identity.get('numerical_commit')!=screen.original.HISTORICAL
            or identity.get('dataset_sha256')!=screen.original.DATA_SHA
            or initial!=dict(sha256=summary.get('initial_state_sha256'),parameters=13290795)):
        raise ValueError('Require completed same-seed LR1e-5 WD2e-4 original-Flat control')
    epochs={str(i) for i in range(1,51)};rates={str(i/10) for i in range(8)}
    if set(trace.get('train',{}))!=epochs or set(trace.get('selection',{}))!=epochs:
        raise ValueError('Require all50 actual train/selection mask traces')
    for epoch in epochs:
        if set(trace['selection'][epoch])!=rates:raise ValueError('Missing selected-rate mask evidence')
        for row in [trace['train'][epoch],*trace['selection'][epoch].values()]:
            if not all(k in row for k in ('sha256','N','batches')):
                raise ValueError('Incomplete actual mask-stream observation')
    return summary


def check_source(driver_commit):
    original=_ORIGINAL_CHECK_SNAPSHOT()
    manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or screen.original.sha(path)!=digest:
            raise ValueError('Fixed-LR follow-up source changed: '+name)
    if not driver_commit or driver_commit=='unrecorded':raise ValueError('Immutable driver commit required')
    return dict(original,git_sha=driver_commit,fixed_lr_wd_manifest=manifest)


def worker(args):
    """Adapt the audited worker only inside this fresh child interpreter."""
    check_seed(args.seed)
    spec=next((s for s in job_specs() if s['id']==args.job and s['seed']==args.seed),None)
    if spec is None:raise ValueError('Job does not match the approved seed/WD grid')
    source=dict(_ORIGINAL_SOURCE_CONFIG(),seed=args.seed)
    screen.source_config=lambda:dict(source)
    screen.JOB_SPECS={s['id']:(1e-5,s['weight_decay']) for s in job_specs() if s['seed']==args.seed}
    screen.GPU_UUIDS=dict(GPU_UUIDS)
    screen.validate_control=lambda path:validate_control(path,args.seed)
    screen.check_snapshot=lambda:check_source(args.driver_commit)
    args.control=args.controls_root/f'seed_{args.seed}'
    screen.worker(args)


class FixedLRScheduler(Scheduler):
    def __init__(self,args):
        self.args=args
        super().__init__(args.output/'queue',gpus=(0,2),per_gpu=4,code_root=ROOT,
            smoke_costs={'Flat':dict(peak_memory_mib=4096,disk_bytes=2*2**30)})

    def command(self,job,config_path,gpu_uuid):
        return [self.python,'-u',str(Path(__file__)),'--worker','--job',job['id'],
            '--seed',str(job['config']['seed']),'--config',str(config_path),
            '--output',job['output'],'--model-source',str(self.args.model_source),
            '--controls-root',str(self.args.controls_root),'--gpu-uuid',gpu_uuid,
            '--driver-commit',self.args.driver_commit]


def aggregate_pair(rows,weight_decay):
    if len(rows)!=2 or {r['config']['seed'] for r in rows}!={67,68}:
        raise ValueError('Two complete paired seeds67/68 required; never label as three seeds')
    for row in rows:
        if row['status']!='complete' or row['config']!=dict(control_config(row['config']['seed']),weight_decay=weight_decay):
            raise ValueError('Incomplete or mismatched paired result')
    result=dict(N_seeds=2,seeds=[67,68],weight_decay=weight_decay)
    for scope in ('mean','high'):
        result[scope]={metric:statistics.mean(r['testbest'][scope][metric] for r in rows)
            for metric in ('ACC','UA','WF1')}
        result[scope+'_sample_std']={metric:statistics.stdev(r['testbest'][scope][metric] for r in rows)
            for metric in ('ACC','UA','WF1')}
    return result


def render(result):
    lines=['# MELD fixed learning-rate weight-decay follow-up','',
        'INTERNAL TEST-ORACLE SCREEN. LR1e-5, original Flat1600 + Local Skip.',
        'Only weight_decay changes; GCNet random-mask-v1, cmerc-repro,50epochs,batch32.',
        'The aggregate below contains TWO paired seeds67/68, not three. WD2e-4 controls are reused.',
        '', '| Seed | weight_decay | ACC | UA | W-F1 | high ACC | high UA |',
        '|---|---:|---:|---:|---:|---:|---:|']
    rows=list(result['controls'].values())+[r for r in result['results'].values() if r.get('status')=='complete']
    for row in sorted(rows,key=lambda r:(r['config']['weight_decay'],r['config']['seed'])):
        cfg=row['config'];score=row['testbest']
        lines.append(f'| {cfg["seed"]} | {cfg["weight_decay"]:g} | '+' | '.join(
            f'{score[s][m]:.3f}' for s,m in [('mean','ACC'),('mean','UA'),('mean','WF1'),('high','ACC'),('high','UA')])+' |')
    lines+=['','| weight_decay | seeds67/68 mean ACC | mean UA | mean W-F1 |','|---:|---:|---:|---:|']
    for key,row in result['aggregate'].items():
        lines.append(f'| {float(key):g} | '+' | '.join(f'{row["mean"][m]:.3f}' for m in ('ACC','UA','WF1'))+' |')
    lines+=['','Same-seed fresh initialization and all50 actual train/test mask streams are checked against controls.',
        'UA/W-F1 correspond to the same ACC-selected forward. Selected epochs can differ.','']
    return '\n'.join(lines)


def controller(args):
    source=check_source(args.driver_commit)
    controls={str(seed):validate_control(args.controls_root/f'seed_{seed}',seed) for seed in SEEDS}
    if args.output.exists() and not args.resume:raise FileExistsError('Existing output requires explicit same-source --resume')
    if screen.original.sha(screen.original.DATASET)!=screen.original.DATA_SHA or shutil.disk_usage(args.output.parent).free<20*2**30:
        raise ValueError('Dataset hash or20GiB free-space check failed')
    scheduler=FixedLRScheduler(args)
    snapshot=scheduler.gpu_snapshot()
    if any(snapshot[index]['uuid']!=uuid for index,uuid in GPU_UUIDS.items()):
        raise ValueError('Healthy physical GPU0/2 UUID mapping changed')
    jobs=[dict(id=s['id'],method='Flat',config=make_config(s['seed'],s['weight_decay']),
        output=str(args.output/s['id'])) for s in job_specs()]
    record=dict(status='running',pid=__import__('os').getpid(),started_unix=time.time(),jobs=8,
        seeds=[67,68],learning_rate=1e-5,gpus=GPU_UUIDS,per_gpu=4,
        driver_commit=args.driver_commit,source=source)
    write_json(args.output/'RUNNER.json',record)
    try:
        completed=scheduler.run_jobs(jobs)
        results={row['id']:row.get('summary',{}) for row in completed}
        aggregates={'0.0002':aggregate_pair(list(controls.values()),2e-4)}
        for wd in WEIGHT_DECAYS:
            paired=[r for r in results.values() if r.get('status')=='complete' and r['config']['weight_decay']==wd]
            if len(paired)==2:aggregates[str(wd)]=aggregate_pair(paired,wd)
        result=dict(status='complete' if all(row['status']=='complete' for row in completed) else 'failed',
            controls=controls,results=results,aggregate=aggregates,
            failures={row['id']:row.get('failures') for row in completed if row['status']!='complete'})
        write_json(args.output/'RESULTS.json',result)
        (args.output/'RESULT.md').write_text(render(result));record['status']=result['status']
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        record['finished_unix']=time.time();write_json(args.output/'RUNNER.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--driver-commit',required=True)
    p.add_argument('--model-source',type=Path,default=screen.MODEL_SOURCE)
    p.add_argument('--controls-root',type=Path,default=CONTROL_ROOT)
    p.add_argument('--resume',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--job',choices=[s['id'] for s in job_specs()]);p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--config',type=Path);p.add_argument('--gpu-uuid')
    args=p.parse_args()
    if args.worker and any(getattr(args,n) is None for n in ('job','seed','config','gpu_uuid')):
        p.error('Worker requires job/seed/config/gpu-uuid')
    (worker if args.worker else controller)(args)


if __name__=='__main__':main()
