"""Finish cosine seeds67/68, then test cosine+step.3 at seeds66/67/68."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.osram_meld_cosine_20261017 import run as cosine
from experiments.osram_meld_write_tuning_20261017 import run as original
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler,write_json

GPU_UUID = cosine.GPU_UUID
JOB_SPECS = {'C67':(67,.6),'C68':(68,.6),'S66':(66,.3),'S67':(67,.3),'S68':(68,.3)}


def source_config():
    return json.loads(original.REFERENCE.read_text())['source_config']


def make_config(base, job):
    if job not in JOB_SPECS:
        raise ValueError('Only five new jobs; completed C66 must be reused')
    seed,step = JOB_SPECS[job]
    return dict(cosine.make_config(base),seed=seed,osram_write_step=step)


def control_for(job, output, reused):
    if job not in JOB_SPECS:
        raise ValueError('Unknown finite job')
    if job.startswith('C'):
        return None
    seed = JOB_SPECS[job][0]
    return Path(reused) if seed == 66 else Path(output)/f'C{seed}'


def validate_reuse(path):
    summary = json.loads((path/'SUMMARY.json').read_text())
    if summary['status'] != 'complete' or summary['config'] != cosine.make_config(source_config()):
        raise ValueError('Require completed original cosine seed66, exactly matching config')
    if summary['paired_masks_equal_through_epoch'] != 50 or summary['parameters'] != 13290795:
        raise ValueError('Completed seed66 failed its original pairing checks')
    if summary['provenance']['numerical_commit'] != original.HISTORICAL or summary['provenance']['data_sha256'] != original.DATA_SHA:
        raise ValueError('Reused numerical source or data version differs')
    for i in range(8):
        if not (path/f'best_miss_0p{i}.pt').is_file() or not (path/'ua_best'/f'best_miss_0p{i}.pt').is_file():
            raise ValueError('Reused seed66 checkpoint missing')
    return summary


def worker(args):
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_reuse(args.reused)
        config = json.loads(args.config.read_text())
        if config != make_config(source_config(),args.job):
            raise ValueError('Effective finite job config changed')
        control = control_for(args.job,args.output.parent,args.reused)
        if control is not None:
            reference = json.loads((control/'SUMMARY.json').read_text())
            expected = dict(config,osram_write_step=.6)
            if reference['status'] != 'complete' or reference['config'] != expected:
                raise ValueError('Step comparison requires completed same-seed cosine control')
        # Reuse the numerical worker, same-forward ACC/UA capture, atomic full
        # recovery and LR observer. Only finite config/control routing differs.
        original.make_config = lambda base,variant:make_config(base,variant)
        original.state_class = cosine.state_factory(config,paired_control=control is not None)
        args.variant, args.control = args.job, control or args.reused
        write_json(args.output/'ENTRY_PROVENANCE.json',dict(job=args.job,config=config,
            source_sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            script_sha256=original.sha(Path(__file__)),
            observer_sha256=original.sha(Path(cosine.__file__)),
            reused_seed66=str(args.reused),paired_control=str(control) if control else None,
            from_scratch=True,authorized_new_jobs=5))
        try:
            original.worker(args)
            path = args.output/'SUMMARY.json'
            summary = json.loads(path.read_text())
            progress = json.loads((args.output/'EPOCH_STATUS.json').read_text())
            if progress['epoch'] != 50 or (control is not None and progress['paired_masks_equal_through_epoch'] != 50):
                raise ValueError('Incomplete training or step-pair verification')
            summary.update(testbest_ua=progress['best_ua'],job=args.job,
                paired_control=str(control) if control else None,
                paired_masks_equal_through_epoch=progress['paired_masks_equal_through_epoch'],
                initialization_equal=progress['initialization_equal'],
                script_sha256=original.sha(Path(__file__)),observer_sha256=original.sha(Path(cosine.__file__)),
                note='ACC-best and UA-best are separate; C67/C68 have no same-seed constant-LR paired control.')
            write_json(path,summary)
        except BaseException as error:
            path = args.output/'SUMMARY.json'
            summary = json.loads(path.read_text()) if path.exists() else {}
            summary.update(status='failed',error=repr(error))
            write_json(path,summary)
            raise


class ReplicationScheduler(Scheduler):
    def __init__(self,root,*,source,reused):
        self.source,self.reused = Path(source),Path(reused)
        super().__init__(root,gpus=(5,),per_gpu=4,code_root=ROOT,
            smoke_costs={'Flat':dict(peak_memory_mib=4096,disk_bytes=3*2**30)})

    def command(self,job,config_path,gpu_uuid):
        return [self.python,'-u',str(Path(__file__)),'--worker','--job',job['id'],
            '--config',str(config_path),'--output',job['output'],
            '--model-source',str(self.source),'--reused',str(self.reused),'--gpu-uuid',gpu_uuid]

    def admit(self,job):
        if job['id'].startswith('S'):
            controls = [self.state['jobs'].get(name,{}) for name in ('C67','C68')]
            if any(row.get('status')=='failed' for row in controls):
                self.failure(job,'Cosine replication failed; second-stage step comparison not launched')
                return None
            if any(row.get('status')!='complete' for row in controls):
                return None
        return super().admit(job)


def summarize(reused,completed):
    import numpy as np
    summaries = {row['id']:row['summary'] for row in completed if row['status']=='complete'}
    groups = {'cosine':{'66':reused},'cosine_step03':{}}
    for job,summary in summaries.items():
        groups['cosine' if job.startswith('C') else 'cosine_step03'][str(JOB_SPECS[job][0])] = summary
    aggregate = {}
    for group,rows in groups.items():
        if set(rows) != {'66','67','68'}:
            aggregate[group] = dict(status='incomplete',completed_seeds=sorted(rows))
            continue
        aggregate[group] = {}
        for selection in ('testbest','testbest_ua'):
            aggregate[group][selection] = {subset:{metric:{
                'mean':float(np.mean([r[selection][subset][metric] for r in rows.values()])),
                'sample_std':float(np.std([r[selection][subset][metric] for r in rows.values()],ddof=1))}
                for metric in ('ACC','UA','WF1','CE')} for subset in ('mean','high')}
    return dict(groups=groups,aggregate=aggregate)


def controller(args):
    if args.output.exists() and not args.resume:
        raise FileExistsError('Never overwrite runs; use --resume only with unchanged source/config')
    if args.resume and not args.output.exists():
        raise FileNotFoundError('No queue to resume')
    reused = validate_reuse(args.reused)
    if original.sha(original.DATASET) != original.DATA_SHA:
        raise ValueError('Accepted MELD ECERC data hash changed')
    if shutil.disk_usage(args.output.parent).free < 18*2**30:
        raise RuntimeError('Need 18GiB admission for five retained ACC/UA-best runs and margins')
    args.output.mkdir(parents=True,exist_ok=True)
    scheduler = ReplicationScheduler(args.output/'queue',source=args.model_source,reused=args.reused)
    snapshot = scheduler.gpu_snapshot()
    if snapshot[5]['uuid'] != GPU_UUID:
        raise ValueError('Healthy physical GPU5 mapping changed')
    jobs = [dict(id=job,method='Flat',config=make_config(source_config(),job),output=str(args.output/job)) for job in JOB_SPECS]
    meta = dict(status='running',pid=os.getpid(),started_unix=time.time(),authorized_new_jobs=5,
        reused_seed66=str(args.reused),gpu=snapshot[5],second_stage='wait for completed C67 and C68',
        source_sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        numerical_sha=original.HISTORICAL,protocol='INTERNAL TEST-ORACLE SCREEN; ACC-best and UA-best separate')
    write_json(args.output/'RUNNER.json',meta)
    try:
        completed = scheduler.run_jobs(jobs)
        ok = all(row['status']=='complete' for row in completed)
        result = summarize(reused,completed)
        result.update(status='complete' if ok else 'failed',failures={r['id']:r['failures'] for r in completed if r['status']!='complete'})
        write_json(args.output/'RESULTS.json',result)
        meta['status'] = result['status']
    except BaseException as error:
        meta.update(status='failed',error=repr(error))
        raise
    finally:
        meta['finished_unix'] = time.time()
        write_json(args.output/'RUNNER.json',meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-source',type=Path,required=True)
    parser.add_argument('--reused',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--job',choices=JOB_SPECS)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    if args.worker and (args.job is None or args.config is None or args.gpu_uuid != GPU_UUID or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID):
        parser.error('Worker requires a finite job/config and UUID-bound healthy physical GPU5')
    (worker if args.worker else controller)(args)


if __name__ == '__main__':
    main()
