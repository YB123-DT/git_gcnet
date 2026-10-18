"""Reuse completed O6 seed66 and train ONLY seeds67/68 with its fixed config."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.osram_meld_optimizer_screen_20261018 import run as screen
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler,write_json

SEED66=Path('/home/qinjinghui/remote_experiments/osram_meld_optimizer_screen_20261018/runs_v1/O6')
GPU=1
UUID='GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'


def make_config(seed):
    if type(seed) is not int or seed not in (67,68):
        raise ValueError('Only missing seeds67/68 are authorized; seed66 is reused')
    return dict(screen.make_config(screen.source_config(),'O6'),seed=seed)


def aggregate_seeds(rows):
    if set(rows)!={'66','67','68'}:raise ValueError('Three complete seeds required')
    base=screen.make_config(screen.source_config(),'O6')
    for key,row in rows.items():
        if row['status']!='complete' or row['config']!=dict(base,seed=int(key)):
            raise ValueError('Incomplete or protocol-mismatched seed')
    return {name:{metric:function([row['testbest'][scope][metric] for row in rows.values()])
            for metric in ('ACC','UA','WF1')}
            for name,scope,function in [('mean','mean',statistics.mean),
                ('std','mean',statistics.stdev),('high_mean','high',statistics.mean),
                ('high_std','high',statistics.stdev)]}


def check_source():
    manifest=screen.check_snapshot()  # immutable shared helpers, same as seed66
    expected=json.loads(Path(__file__).with_name('SOURCE_MANIFEST.json').read_text())
    for name,digest in expected['files'].items():
        if screen.original.sha(ROOT/name)!=digest:raise ValueError('Extension source changed: '+name)
    return manifest,expected


def worker(args):
    _,extension=check_source()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,
        GCNET_MELD_PICKLE=str(screen.original.DATASET),
        GCNET_DATASET_ROOT=str(screen.MODEL_SOURCE/'dataset'),
        GCNET_CACHE_ROOT=str(args.output/'feature_cache'))
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment,repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    import torch
    gpu=guard_gpu(args.gpu_uuid)
    if gpu['host_index']!=GPU or gpu['uuid']!=UUID:raise ValueError('Only healthy GPU1 authorized')
    environment=require_replication_environment(torch);repair_uuid_rng(torch)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    config=json.loads(args.config.read_text())
    if config!=make_config(args.seed) or screen.original.sha(screen.original.DATASET)!=screen.original.DATA_SHA:
        raise ValueError('Wrong fixed optimizer config or features')
    trainer=screen.original.load_historical(screen.MODEL_SOURCE)
    from gcnet_missing_m3.training_resume import TrainingState
    from experiments.osram_meld_write_tuning_20261017.capture import TrainingCapture
    from experiments.osram_meld_fixed_rates_20261018.run import selected_result
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(numerical_commit=screen.original.HISTORICAL,extension=extension,
            config_sha256=screen.original.sha(args.config),dataset_sha256=screen.original.DATA_SHA)
        path=args.output/'SUMMARY.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if previous['identity']!=identity:raise ValueError('Source or config changed on resume')
            if previous['status']=='complete':return
        summary=dict(status='running',seed=args.seed,config=config,identity=identity,
            protocol='INTERNAL TEST-ORACLE SCREEN; per-rate ACC; same-forward UA/WF1',
            environment=environment,gpu=gpu,pid=os.getpid(),started_unix=time.time(),from_scratch=True,
            seed66_reused=str(SEED66),cross_seed_masks_equal=False)
        write_json(path,summary)
        capture=TrainingCapture(trainer,args.output)
        # W0 here means independent initialization: do NOT compare seed67/68's
        # initialization/masks to seed66 or load seed66 weights. This reuses only
        # the artifact/recovery observer, not W0's optimizer or numerical model.
        Parent=screen.original.state_class(TrainingState,capture,args.output,'W0',args.output)
        class SeedState(Parent):
            def commit_epoch(self,**kwargs):
                actual=screen.constant_optimizer_check(
                    screen.audit_optimizer_groups(self.model,self.optimizer.param_groups),config)
                if capture.modelparameters!=13290795:raise ValueError('Original Flat changed')
                super().commit_epoch(**kwargs)
                p=args.output/'EPOCH_STATUS.json';row=json.loads(p.read_text())
                row.update(seed=args.seed,actual_optimizer=actual)
                write_json(p,row)
        tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
        try:
            with capture:
                trainer.run_experiment(trainer.TrainConfig(**config),'MELD-pickle','MELD-pickle',
                    'MELD-pickle',args.output,training_state=SeedState(args.output,identity=identity))
                capture.write_trace()
            if len(json.loads((args.output/'history.json').read_text()))!=50:
                raise ValueError('Incomplete training')
            selected={str(i/10):selected_result(args.output,i/10) for i in range(8)}
            summary.update(status='complete',testbest=screen.original.aggregate(selected),
                parameters=capture.modelparameters,initial_state_sha256=capture.initial_state_sha256,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        except BaseException as error:
            summary.update(status='failed',error=repr(error));raise
        finally:
            summary.update(seconds=time.monotonic()-tick,finished_unix=time.time())
            write_json(path,summary)


class SeedScheduler(Scheduler):
    def __init__(self,output):
        super().__init__(output/'queue',gpus=(GPU,),per_gpu=2,code_root=ROOT,
            smoke_costs={'Flat':dict(peak_memory_mib=4096,disk_bytes=2*2**30)})
    def command(self,job,config_path,gpu_uuid):
        return [self.python,'-u',str(Path(__file__)),'--worker','--seed',str(job['config']['seed']),
            '--config',str(config_path),'--output',job['output'],'--gpu-uuid',gpu_uuid]


def controller(args):
    check_source()
    reused=json.loads((SEED66/'SUMMARY.json').read_text())
    if reused['status']!='complete' or reused['config']!=screen.make_config(screen.source_config(),'O6'):
        raise ValueError('Completed unchanged O6 seed66 required')
    if args.output.exists() and not args.resume:raise FileExistsError('Use explicit same-source --resume')
    if shutil.disk_usage(args.output.parent).free<10*2**30:raise ValueError('Need10GiB output space')
    scheduler=SeedScheduler(args.output)
    if scheduler.gpu_snapshot()[GPU]['uuid']!=UUID:raise ValueError('Physical GPU1 identity changed')
    jobs=[dict(id=f'S{seed}',method='Flat',config=make_config(seed),
        output=str(args.output/f'seed_{seed}')) for seed in (67,68)]
    record=dict(status='running',pid=os.getpid(),started_unix=time.time(),new_seeds=[67,68],
        reused_seed66=str(SEED66),gpu=GPU,gpu_uuid=UUID)
    write_json(args.output/'RUNNER.json',record)
    try:
        completed=scheduler.run_jobs(jobs)
        rows={'66':reused}
        rows.update({str(row['config']['seed']):row['summary'] for row in completed if row['status']=='complete'})
        result=dict(status='complete' if len(rows)==3 else 'failed',seeds=rows,
            failures={row['id']:row.get('failures') for row in completed if row['status']!='complete'})
        if len(rows)==3:result['aggregate']=aggregate_seeds(rows)
        write_json(args.output/'RESULTS.json',result)
        lines=['# MELD fixed small-LR three-seed confirmation','',
            'INTERNAL TEST-ORACLE SCREEN; ACC-selected corresponding UA/W-F1.',
            'Seed66 reused; seeds67/68 from scratch. No additional HPO.','',
            '| Seed | ACC | UA | W-F1 |','|---|---:|---:|---:|']
        for seed,row in rows.items():
            lines.append('| '+seed+' | '+' | '.join(f'{row["testbest"]["mean"][m]:.3f}' for m in ('ACC','UA','WF1'))+' |')
        if 'aggregate' in result:
            a=result['aggregate'];lines.append('| mean ± sample std | '+' | '.join(
                f'{a["mean"][m]:.3f} ± {a["std"][m]:.3f}' for m in ('ACC','UA','WF1'))+' |')
        (args.output/'RESULT.md').write_text('\n'.join(lines)+'\n')
        record['status']=result['status']
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        record['finished_unix']=time.time();write_json(args.output/'RUNNER.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--seed',type=int,choices=(67,68))
    p.add_argument('--config',type=Path);p.add_argument('--gpu-uuid')
    args=p.parse_args()
    if args.worker and any(getattr(args,n) is None for n in ('seed','config','gpu_uuid')):
        p.error('Worker requires seed/config/gpu-uuid')
    (worker if args.worker else controller)(args)


if __name__=='__main__':main()
