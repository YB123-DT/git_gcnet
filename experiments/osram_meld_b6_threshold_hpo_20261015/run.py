"""Finite threshold/init and B6-only LR HPO; fixed best-ACC rank2/14."""
from __future__ import annotations

import argparse
import fcntl
from copy import deepcopy
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
from experiments.osram_meld_b6_structure_hpo_20261015.run import StructureScheduler,run_search,FLAT

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/osram_meld_b6_structure_hpo_20261015/SUMMARY.json'
DEFAULT_OUTPUT=Path('/home/qinjinghui/remote_experiments/osram_meld_b6_threshold_hpo_20261015')
INITIAL_CONFIGS=({'threshold_init':.1,'b6_lr_multiplier':1.},
                 {'threshold_init':.03,'b6_lr_multiplier':1.},
                 {'threshold_init':.3,'b6_lr_multiplier':1.},
                 {'threshold_init':.1,'b6_lr_multiplier':.1},
                 {'threshold_init':.1,'b6_lr_multiplier':10.})
SEARCH_SPACE={'threshold_init':{'low':.01,'high':1.,'log':True},
              'b6_lr_multiplier':{'low':.1,'high':30.,'log':True}}
REPORT_METADATA=dict(title='B6 threshold initialization and independent LR, fixed rank2/14',
    scope='Only threshold_init and B6-only optimizer multiplier; old trial16 main settings unchanged',
    control_note='Trial0 reruns threshold_init0.1/B6 multiplier1 at this source, rank2/14. No old score counted as a new run.')


def make_config(threshold_init,b6_lr_multiplier):
    for name,v in (('threshold_init',threshold_init),('b6_lr_multiplier',b6_lr_multiplier)):
        spec=SEARCH_SPACE[name]
        if (isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v)
                or not spec['low']<=v<=spec['high']):
            raise ValueError('Out-of-range finite threshold HPO setting: '+name)
    source=json.loads(SOURCE.read_text())
    row=next(t for t in source['trials'] if t['number']==3)
    c=deepcopy(row['config'])
    if (source['status']!='complete' or row['state']!='COMPLETE' or c['seed']!=66
            or c['epochs']!=50 or (c['osram_rpca_rank'],c['osram_rpca_steps'])!=(2,14)
            or c['missing_mask_protocol']!='gcnet-random-mask-v1'):
        raise ValueError('Frozen source/config changed')
    c.update(osram_rpca_threshold_init=float(threshold_init),osram_rpca_lr_multiplier=float(b6_lr_multiplier))
    return c


class ThresholdScheduler(StructureScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs,per_gpu_limit=6)
        self.global_per_gpu_limit=6


def make_sampler():
    import optuna
    return optuna.samplers.TPESampler(seed=66,n_startup_trials=5,
                                      constant_liar=True,multivariate=False)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--gpus',default='2')
    p.add_argument('--per-gpu',type=int,default=6)
    p.add_argument('--trials',type=int,default=20)
    a=p.parse_args(argv);a.gpus=[int(v) for v in a.gpus.split(',')]
    if a.gpus!=[2] or a.per_gpu!=6 or a.trials!=20:
        p.error('Authorized scope: physicalGPU2only,per-card6,total20,seed66')
    return a


def execute(a):
    import optuna
    import torch
    import numpy as np
    from experiments.osram_meld_block2_auto_20261014.worker import DATASET,DATASET_SHA,source_hashes
    from gcnet_missing_m3.training_resume import _sha
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Require committed immutable source')
    if _sha(DATASET)!=DATASET_SHA or shutil.disk_usage(a.output.parent).free<20*2**30:
        raise RuntimeError('Dataset or disk preflight failed')
    flat=json.loads(FLAT.read_text())
    if flat['status']!='complete' or flat['method']!='Flat':
        raise ValueError('Common-mask Flat reference missing')
    a.output.mkdir(parents=True,exist_ok=True)
    provenance=dict(source_git_sha=sha,code_hashes=source_hashes(),launcher_sha256=_sha(Path(__file__)),
        controller_sha256=_sha(ROOT/'experiments/osram_meld_b6_structure_hpo_20261015/run.py'),
        frozen_source_sha256=_sha(SOURCE),dataset_sha256=DATASET_SHA,config=make_config(.1,1.),
        requested_gpus=a.gpus,per_gpu_total_limit=6,global_pending_limit=len(a.gpus)*6,
        initial_configs=list(INITIAL_CONFIGS),search_space=SEARCH_SPACE,total_trials=20,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,optuna=optuna.__version__))
    path=a.output/'PREFLIGHT.json'
    if path.exists() and json.loads(path.read_text())!=provenance:
        raise ValueError('Do not mix source/config/protocol versions')
    write_json(path,provenance)
    runner=dict(status='running',pid=os.getpid(),source_git_sha=sha,started_unix=time.time())
    write_json(a.output/'RUNNER.json',runner)
    try:
        smoke=a.output/'smoke';settings=a.output/'smoke_config.json'
        write_json(settings,make_config(1.,30.))
        if not (smoke/'SUMMARY.json').exists():
            probe=ThresholdScheduler(a.output/'queue',gpus=a.gpus,per_gpu=6)
            snapshot=probe.gpu_snapshot()
            gpu=max(a.gpus,key=lambda g:snapshot[g]['free_mib'])
            if snapshot[gpu]['free_mib']<6000:
                raise RuntimeError('No admitted real-batch smoke capacity; other jobs untouched')
            uuid=snapshot[gpu]['uuid']
            cmd=[sys.executable,'-u','-m','experiments.osram_meld_block2_auto_20261014.worker',
                 '--config',str(settings),'--output',str(smoke),'--gpu-uuid',uuid,'--smoke']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with (a.output/'smoke.log').open('ab') as handle:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        s=json.loads((smoke/'SUMMARY.json').read_text())
        if s['status']!='complete' or not s.get('smoke_only') or s['config']!=make_config(1.,30.):
            raise RuntimeError('Smoke incomplete/incompatible')
        groups=s['optimizer_parameter_groups']
        if (not math.isclose(groups['rpca']['learning_rate'],30*make_config(1.,30.)['learning_rate'])
                or groups['rpca']['parameter_count']!=29):
            raise RuntimeError('Smoke did not use independent B6 learning rate')
        scheduler=ThresholdScheduler(a.output/'queue',gpus=a.gpus,per_gpu=6,
            smoke_costs={'B6':{'peak_memory_mib':s['peak_memory_mib'],'disk_bytes':2*2**30}})
        result=run_search(a.output,flat['paired'],scheduler,budget=20,max_pending=len(a.gpus)*6,
            config_factory=make_config,initial_configs=INITIAL_CONFIGS,search_space=SEARCH_SPACE,
            study_name='meld_B6_rank2_steps14_threshold_lr_seed66',sampler=make_sampler(),metadata=REPORT_METADATA)
        runner.update(status=result['status'],finished_unix=time.time())
    except BaseException as error:
        runner.update(status='failed',error=repr(error),finished_unix=time.time());raise
    finally:
        write_json(a.output/'RUNNER.json',runner)


def main(argv=None):
    a=parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.controller.lock').open('a+') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another controller already owns this study; no trials submitted') from error
        execute(a)


if __name__=='__main__':
    main()
