"""Eight finite from-scratch MELD Flat LR/weight-decay runs, no Probe fitting."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.osram_meld_write_tuning_20261017 import run as original
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler,write_json

JOB_SPECS = {'O0':(2e-4,2e-4),'O1':(2e-4,1e-3),
             'O2':(1e-4,2e-4),'O3':(1e-4,1e-3),
             'O4':(3e-5,2e-4),'O5':(3e-5,1e-3),
             'O6':(1e-5,2e-4),'O7':(1e-5,1e-3)}
GPU_UUIDS = {1:'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049',
             2:'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
MODEL_SOURCE = Path('/data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138')
CONTROL = Path('/data2/yb/remote_experiments/osram_meld_write_tuning_20261017/runs_v1/W0')


def source_config():
    return json.loads(original.REFERENCE.read_text())['source_config']


def make_config(base,job):
    if job not in JOB_SPECS:
        raise ValueError('Only eight finite optimizer configurations are authorized')
    if base != source_config():
        raise ValueError('Original Flat protocol changed outside LR/weight_decay')
    lr,decay = JOB_SPECS[job]
    return dict(deepcopy(base),learning_rate=lr,weight_decay=decay)


def check_snapshot():
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or original.sha(path)!=digest:
            raise ValueError('Experiment source changed: '+name)
    return manifest


def constant_optimizer_check(groups,config):
    if not groups or config['lr_schedule']!='constant':
        raise ValueError('Require nonempty original constant-LR optimizer groups')
    multipliers = {'backbone':config['backbone_lr_multiplier'],
        'projector':config['projector_lr_multiplier'],'classifier':config['classifier_lr_multiplier'],
        'predictor':1.,'other':1.}
    result = dict(rates=[],weight_decay=[])
    for group in groups:
        name = group.get('name')
        if name not in multipliers:
            raise ValueError('Unknown optimizer parameter-group identity')
        lr,decay = float(group['lr']),float(group['weight_decay'])
        if (not math.isfinite(lr) or not math.isfinite(decay)
                or not math.isclose(lr,config['learning_rate']*multipliers[name],abs_tol=1e-15,rel_tol=1e-12)
                or not math.isclose(decay,config['weight_decay'],abs_tol=1e-15,rel_tol=1e-12)):
            raise ValueError('Actual optimizer LR or weight_decay differs from this job')
        result['rates'].append(lr)
        result['weight_decay'].append(decay)
    return result


def audit_optimizer_groups(model,groups):
    """The b4 trainer has unnamed groups; label COPIES by actual ownership.

    Never add names to the optimizer itself or change its order/parameters.
    This mirrors its existing prefix partition solely for read-only auditing.
    """
    ownership={}
    for name,parameter in model.named_parameters():
        if name.startswith('osram.'):
            block='backbone'
        elif name.startswith('observed_set.'):
            block='projector'
        elif name.startswith(('smax_fc.','conditioned_readout.','affine_readout.','decision_head.')):
            block='classifier'
        elif name.startswith(('missing_predictor.','source_only_predictor.')):
            block='predictor'
        else:
            block='other'
        ownership[id(parameter)]=block
    result=[]
    for group in groups:
        blocks={ownership.get(id(p)) for p in group['params']}
        if len(blocks)!=1 or None in blocks:
            raise ValueError('Unknown or mixed optimizer parameter ownership')
        name=next(iter(blocks))
        if group.get('name',name)!=name:
            raise ValueError('Optimizer group name differs from its parameter ownership')
        result.append(dict(group,name=name))
    return result


def assert_epoch_masks(trace,reference,epoch):
    key=str(epoch)
    try:
        for phase in ('train','selection'):
            left,right = trace[phase][key],reference[phase][key]
            if phase=='selection':
                rates={str(i/10) for i in range(8)}
                if set(left)!=rates or set(right)!=rates:
                    raise ValueError('Eight actual selection masks required')
                rows=[(left[r],right[r]) for r in sorted(rates)]
            else:
                rows=[(left,right)]
            if any(a.get(field)!=b.get(field) or field not in a or field not in b
                   for a,b in rows for field in ('sha256','batches','N')):
                raise ValueError(f'Actual {phase} mask stream differs from W0 at epoch{epoch}')
    except (KeyError,TypeError) as error:
        raise ValueError('Missing actual epoch mask evidence') from error


def validate_control(path):
    summary=json.loads((path/'SUMMARY.json').read_text())
    trace=json.loads((path/'MASK_TRACE.json').read_text())
    if (summary['status']!='complete' or summary['config']!=source_config()
            or summary['parameters']!=13290795
            or summary['provenance']['numerical_commit']!=original.HISTORICAL
            or summary['provenance']['data_sha256']!=original.DATA_SHA
            or set(trace['train'])!={str(i) for i in range(1,51)}
            or set(trace['selection'])!={str(i) for i in range(1,51)}):
        raise ValueError('Need completed, unchanged original constant-LR W0 control')
    return summary


def state_factory(base,capture,output,job,control,config):
    Parent=original.state_class(base,capture,output,job,control)
    reference=json.loads((control/'MASK_TRACE.json').read_text())
    class ObservedState(Parent):
        def bind(self,model,optimizer,**kwargs):
            super().bind(model,optimizer,**kwargs)
            if capture.modelparameters!=13290795:
                raise ValueError('Original Flat parameter count changed')
            self.lr_history=[]
            return self

        def restore(self):
            restored=super().restore()
            if restored is not None:
                self.lr_history=restored['auxiliary_state']['optimizer_screen']
                if len(self.lr_history)!=restored['next_epoch']:
                    raise ValueError('Actual optimizer history recovery mismatch')
            return restored

        def commit_epoch(self,**kwargs):
            epoch=kwargs['next_epoch']
            actual=constant_optimizer_check(audit_optimizer_groups(self.model,self.optimizer.param_groups),config)
            assert_epoch_masks(capture.trace,reference,epoch)
            if len(self.lr_history)!=epoch-1:
                raise ValueError('Optimizer observation skipped an epoch')
            self.lr_history.append(dict(epoch=epoch,**actual))
            auxiliary=dict(kwargs.get('auxiliary_state') or {})
            auxiliary['optimizer_screen']=self.lr_history
            kwargs['auxiliary_state']=auxiliary
            super().commit_epoch(**kwargs)
            progress=json.loads((output/'EPOCH_STATUS.json').read_text())
            progress.update(actual_optimizer=actual,paired_masks_equal_through_epoch=epoch,
                            initialization_equal=True,job=job)
            write_json(output/'EPOCH_STATUS.json',progress)
            write_json(output/'OPTIMIZER_HISTORY.json',self.lr_history)
    return ObservedState


def worker(args):
    import fcntl
    manifest=check_snapshot()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,GCNET_MELD_PICKLE=str(original.DATASET),
        GCNET_DATASET_ROOT=str(args.model_source/'dataset'),GCNET_CACHE_ROOT=str(args.output/'feature_cache'))
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment,repair_uuid_rng
    import torch
    gpu=guard_gpu(args.gpu_uuid)
    if GPU_UUIDS.get(gpu['host_index'])!=gpu['uuid']:
        raise ValueError('Only healthy physical GPU1/2 are authorized for this queue')
    environment=require_replication_environment(torch)
    repair_uuid_rng(torch)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    validate_control(args.control)
    config=json.loads(args.config.read_text())
    if config!=make_config(source_config(),args.job) or original.sha(original.DATASET)!=original.DATA_SHA:
        raise ValueError('Effective config or accepted ECERC features changed')
    trainer=original.load_historical(args.model_source)
    from gcnet_missing_m3.training_resume import TrainingState
    from experiments.osram_meld_write_tuning_20261017.capture import TrainingCapture
    from experiments.osram_meld_fixed_rates_20261018.run import selected_result
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(experiment_git_sha=manifest['git_sha'],numerical_commit=original.HISTORICAL,
            config_sha256=original.sha(args.config),dataset_sha256=original.DATA_SHA,
            source_manifest_sha256=original.sha(ROOT/'SOURCE_MANIFEST.json'),
            control_summary_sha256=original.sha(args.control/'SUMMARY.json'),
            control_mask_trace_sha256=original.sha(args.control/'MASK_TRACE.json'))
        path=args.output/'SUMMARY.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if previous['identity']!=identity:
                raise ValueError('Refuse different source/config/data on resume')
            if previous['status']=='complete':
                return
        summary=dict(status='running',job=args.job,config=config,identity=identity,
            protocol='INTERNAL TEST-ORACLE SCREEN; per-rate ACC with same-forward UA/WF1',
            environment=environment,gpu=gpu,pid=os.getpid(),started_unix=time.time(),
            from_scratch=True,control=str(args.control))
        write_json(path,summary)
        capture=TrainingCapture(trainer,args.output)
        state=state_factory(TrainingState,capture,args.output,args.job,args.control,config)(args.output,identity=identity)
        tick=time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            with capture:
                trainer.run_experiment(trainer.TrainConfig(**config),'MELD-pickle','MELD-pickle',
                                       'MELD-pickle',args.output,training_state=state)
                capture.write_trace()
            progress=json.loads((args.output/'EPOCH_STATUS.json').read_text())
            if progress['epoch']!=50 or progress['paired_masks_equal_through_epoch']!=50:
                raise ValueError('Incomplete training/mask verification')
            selected={str(i/10):selected_result(args.output,i/10) for i in range(8)}
            summary.update(status='complete',testbest=original.aggregate(selected),
                parameters=capture.modelparameters,initial_state_sha256=capture.initial_state_sha256,
                paired_masks_equal_through_epoch=50,initialization_equal=True,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        except BaseException as error:
            summary.update(status='failed',error=repr(error))
            raise
        finally:
            summary.update(seconds=time.monotonic()-tick,finished_unix=time.time())
            write_json(path,summary)


class OptimizerScheduler(Scheduler):
    def __init__(self,root,*,model_source,control):
        self.model_source,self.control=Path(model_source),Path(control)
        super().__init__(root,gpus=(1,2),per_gpu=4,code_root=ROOT,
            smoke_costs={'Flat':dict(peak_memory_mib=4096,disk_bytes=2*2**30)})

    def command(self,job,config_path,gpu_uuid):
        return [self.python,'-u',str(Path(__file__)),'--worker','--job',job['id'],
            '--config',str(config_path),'--output',job['output'],'--model-source',str(self.model_source),
            '--control',str(self.control),'--gpu-uuid',gpu_uuid]


def render(result):
    lines=['# MELD Flat optimizer screen','','INTERNAL TEST-ORACLE SCREEN; seed66 only.',
        '', 'Only LR and Adam weight_decay differ. Original Flat1600 + Local Skip; no added Block.',
        'Original GCNet random-mask-v1, ECERC,50epochs,batch32,constant LR,cmerc-repro.',
        'ACC-best corresponding UA/W-F1 use saved selected-epoch predictions. No UA-best mixing.',
        '', '| Job | LR | weight_decay | ACC | UA | W-F1 | high ACC | high UA |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    control=result['control']['testbest']
    lines.append('| W0 reused | 1e-3 | 1e-5 | '+' | '.join(f'{control[s][m]:.3f}'
        for s,m in (('mean','ACC'),('mean','UA'),('mean','WF1'),('high','ACC'),('high','UA')))+' |')
    for job,row in result['results'].items():
        if row.get('status')!='complete':
            continue
        cfg,scores=row['config'],row['testbest']
        lines.append(f'| {job} | {cfg["learning_rate"]:g} | {cfg["weight_decay"]:g} | '+' | '.join(
            f'{scores[s][m]:.3f}' for s,m in (('mean','ACC'),('mean','UA'),('mean','WF1'),('high','ACC'),('high','UA')))+' |')
    lines+=['','Actual initialization and all50 epochs of train/test mask streams are checked against W0.',
        'ACC-selected epochs can differ: matching streams does not imply matching selected masks.',
        'This is one-seed exploratory Test-oracle HPO, not independent paper performance.',
        'No automatic seed expansion. Small LR may require more epochs;50epoch budget is unchanged.',
        'Per-rate metrics, epochs, class recall/support, checkpoint/mask hashes and failures: RESULTS.json.','']
    return '\n'.join(lines)


def controller(args):
    manifest=check_snapshot()
    control=validate_control(args.control)
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing output requires explicit same-source --resume')
    if (original.sha(original.DATASET)!=original.DATA_SHA
            or shutil.disk_usage(args.output.parent).free<20*2**30):
        raise ValueError('Data hash or20GiB disk admission failed')
    args.output.mkdir(parents=True,exist_ok=True)
    scheduler=OptimizerScheduler(args.output/'queue',model_source=args.model_source,control=args.control)
    snapshot=scheduler.gpu_snapshot()
    if any(snapshot[g]['uuid']!=uuid for g,uuid in GPU_UUIDS.items()):
        raise ValueError('Physical GPU1/2 UUID mapping changed')
    jobs=[dict(id=job,method='Flat',config=make_config(source_config(),job),output=str(args.output/job))
          for job in JOB_SPECS]
    runner=dict(status='running',pid=os.getpid(),started_unix=time.time(),jobs=8,epochs=50,seed=66,
        gpus=GPU_UUIDS,per_gpu=4,experiment_git_sha=manifest['git_sha'],
        numerical_commit=original.HISTORICAL,control=str(args.control),later_search=False)
    write_json(args.output/'RUNNER.json',runner)
    try:
        completed=scheduler.run_jobs(jobs)
        result=dict(status='complete' if all(r['status']=='complete' for r in completed) else 'failed',
            control=control,results={r['id']:r.get('summary',{}) for r in completed},
            failures={r['id']:r['failures'] for r in completed if r['status']!='complete'})
        write_json(args.output/'RESULTS.json',result)
        (args.output/'RESULT.md').write_text(render(result))
        runner['status']=result['status']
    except BaseException as error:
        runner.update(status='failed',error=repr(error))
        raise
    finally:
        runner['finished_unix']=time.time()
        write_json(args.output/'RUNNER.json',runner)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-source',type=Path,default=MODEL_SOURCE)
    parser.add_argument('--control',type=Path,default=CONTROL)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--job',choices=JOB_SPECS)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.worker and any(getattr(args,k) is None for k in ('job','config','gpu_uuid')):
        parser.error('Worker requires job/config/gpu-uuid')
    (worker if args.worker else controller)(args)


if __name__=='__main__':
    main()
