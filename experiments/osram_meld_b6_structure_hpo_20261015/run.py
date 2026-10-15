"""Finite B6 structural TPE; freeze trial16 training hyperparameters."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json
from experiments.osram_meld_block2_auto_20261014.optuna_search import paired_objective

ROOT = Path(__file__).resolve().parents[2]
INITIAL_CONFIGS = ({'rank':1,'steps':14}, {'rank':1,'steps':8},
                   {'rank':1,'steps':24}, {'rank':2,'steps':14}, {'rank':3,'steps':14})
SOURCE = ROOT / 'experiments/osram_meld_block2_topacc_3seed_20261015/MANIFEST.json'
FLAT = Path('/data2/yb/remote_experiments/osram_meld_block2_auto_20261014_v2/screen/Flat/SUMMARY.json')
DEFAULT_OUTPUT = Path('/home/qinjinghui/remote_experiments/osram_meld_b6_structure_hpo_20261015')


def make_config(rank, steps):
    if (type(rank) is not int or rank not in (1,2,3) or type(steps) is not int
            or steps not in range(4,29,2)):
        raise ValueError('B6 HPO requires rank1/2/3 and even steps4..28')
    records = json.loads(SOURCE.read_text())['configurations']
    source = next(c['summary'] for c in records if c['id']=='B6_trial16')
    config = deepcopy(source['config'])
    if (source['status']!='complete' or source['method']!='B6' or config['seed']!=66
            or config['epochs']!=50 or config['missing_mask_protocol']!='gcnet-random-mask-v1'
            or config['osram_meaningful_block']!='learned_robust_pca_evidence_decomposition'):
        raise ValueError('Frozen trial16 provenance/protocol changed')
    config.update(osram_rpca_rank=rank,osram_rpca_steps=steps)
    return config


def available_slots(limit, visible_pids, owned_pids):
    """Count actual processes plus our not-yet-visible launch reservations once."""
    return max(0, limit-len(set(visible_pids)|set(owned_pids)))


def gpu_provenance(gpus):
    return dict(gpu_whitelist=list(gpus),per_gpu_total_limit=4,global_limit=len(gpus)*4)


class StructureScheduler(Scheduler):
    def gpu_snapshot(self):
        snapshot=super().gpu_snapshot()
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'],text=True)
        pids=defaultdict(set)
        for line in raw.splitlines():
            if line.strip():
                uuid,pid=(p.strip() for p in line.split(','))
                pids[uuid].add(int(pid))
        for gpu,v in snapshot.items():
            owned=[r['pid'] for r in self.state['jobs'].values()
                   if r.get('gpu')==gpu and r['status'] in ('running','launching') and r.get('pid')]
            # External/other-queue jobs are observed, never signalled or adopted.
            if not available_slots(4,pids[v['uuid']],owned):
                v['free_mib']=0
        return snapshot


def unique_tpe_sampler():
    import optuna
    class UniqueTPE(optuna.samplers.TPESampler):
        def sample_independent(self, study, trial, param_name, distribution):
            pairs={(t.params.get('rank'),t.params.get('steps')) for t in study.get_trials(deepcopy=False)
                   if t.number!=trial.number and {'rank','steps'}<=set(t.params)}
            if param_name=='rank':
                allowed=[r for r in (1,2,3) if any((r,s) not in pairs for s in range(4,29,2))]
            else:
                allowed=[s for s in range(4,29,2) if (trial.params['rank'],s) not in pairs]
            if not allowed:
                raise RuntimeError('Unique structural search space exhausted')
            for _ in range(32):
                value=super().sample_independent(study,trial,param_name,distribution)
                if value in allowed:
                    return value
            # Conditional uniqueness fallback is explicit, deterministic, bounded.
            study._storage.set_trial_user_attr(trial._trial_id,'unique_fallback_'+param_name,True)
            return random.Random(66+trial.number).choice(allowed)
    return UniqueTPE(seed=66,n_startup_trials=5,constant_liar=True)


def build_report(rows):
    report={'label':'INTERNAL TEST-ORACLE SCREEN; seed66 only',
            'scope':'B6 structural HPO with frozen trial16 training hyperparameters',
            'trials':[], 'best_paired_objective':None,'best_testbest_ACC':None}
    for row in rows:
        out={k:row[k] for k in ('number','state','params','value')}
        out['failure']=row.get('user_attrs',{}).get('failure')
        if row['state']=='COMPLETE':
            s=row['user_attrs']['summary']
            for key in ('testbest','paired','parameters','block_parameters','training_seconds','peakmemorybytes'):
                out[key]=s[key]
            out['config']=s.get('config')
            out['provenance']=s.get('provenance')
        report['trials'].append(out)
    done=[r for r in report['trials'] if r['state']=='COMPLETE']
    if done:
        report['best_paired_objective']=max(done,key=lambda r:r['value'])['number']
        report['best_testbest_ACC']=max(done,key=lambda r:r['testbest']['mean']['ACC'])['number']
    return report


def export(study, output):
    rows=[dict(number=t.number,state=t.state.name,params=t.params,value=t.value,user_attrs=t.user_attrs)
          for t in study.trials]
    write_json(output/'TRIALS.json',rows)
    report=build_report(rows)
    write_json(output/'SUMMARY.json',report)
    cells=[]
    lines=['# B6 structural HPO, fixed trial16 training configuration','',report['label'],'',
           '| Trial | Rank | Steps | State | Test-best ACC | UA | Paired ACC | UA | Objective |',
           '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in report['trials']:
        tb=r.get('testbest',{}).get('mean',{})
        pm=r.get('paired',{}).get('on',{}).get('mean',{})
        vals=[tb.get('ACC'),tb.get('UA'),pm.get('ACC'),pm.get('UA'),r['value']]
        text=['—' if v is None else f'{v:.4f}' for v in vals]
        lines.append(f"| {r['number']} | {r['params'].get('rank','—')} | {r['params'].get('steps','—')} | {r['state']} | {' | '.join(text)} |")
        if r['state']=='COMPLETE':
            for protocol in ('testbest','paired'):
                data=r['testbest'] if protocol=='testbest' else r['paired']['on']
                for rate,v in dict(data['rates'],mean=data['mean'],high=data['high']).items():
                    cells.append(dict(trial=r['number'],rank=r['params']['rank'],steps=r['params']['steps'],
                                      protocol=protocol,rate=rate,**{k:v[k] for k in ('ACC','UA','WF1','CE')}))
    lines+=['',f"Best paired objective trial: {report['best_paired_objective']}",
            f"Highest Test-best ACC trial: {report['best_testbest_ACC']}",
            '', 'All scores are test-selected exploration, not independent test performance.',
            'Trial0 repeats rank1/steps14 at this immutable source. No old score is counted as a new run.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    if cells:
        with (output/'RESULT.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    return report


def run_search(output, flat_paired, scheduler, budget=20, max_pending=12):
    import optuna
    study=optuna.create_study(study_name='meld_B6_trial16_structure_seed66',
        storage='sqlite:///'+str(output/'STRUCTURE.sqlite3'),direction='maximize',
        sampler=unique_tpe_sampler(),load_if_exists=True)
    protocol={'budget':budget,'max_pending':max_pending,'base_config':make_config(1,14),
              'initial':list(INITIAL_CONFIGS),'flat_paired':flat_paired,
              'objective':'0.6*delta_ACC+0.4*delta_UA','warmstart_parallel':5}
    if study.user_attrs.get('protocol',protocol)!=protocol:
        raise ValueError('Persistent study settings changed')
    study.set_user_attr('protocol',protocol)
    if not study.trials:
        for c in INITIAL_CONFIGS:
            study.enqueue_trial(c)
    def job(t):
        name=f'b6_struct_{t.number:03d}'
        return dict(id=name,method='B6',config=make_config(**t.params),
                    output=str(output/'optuna'/'runs'/name))
    while True:
        trials=study.get_trials(deepcopy=False)
        issued=[t for t in trials if t.state!=optuna.trial.TrialState.WAITING]
        running=[]
        for t in issued:
            if t.state==optuna.trial.TrialState.RUNNING:
                # Recover an ask->suggest crash before any job was submitted;
                # suggestions already persisted remain exactly unchanged.
                live=optuna.Trial(study,t._trial_id)
                live.suggest_categorical('rank',[1,2,3])
                live.suggest_int('steps',4,28,step=2)
                running.append(live)
        # First ask all5fixed configurations concurrently; thereafter replenish
        # only after recording completed results. Constant liar accounts for pending jobs.
        width=5 if len(issued)<5 else max_pending
        for _ in range(min(width-len(running),budget-len(issued))):
            t=study.ask()
            t.suggest_categorical('rank',[1,2,3])
            t.suggest_int('steps',4,28,step=2)
            running.append(t)
        if not running:
            break
        records=scheduler.run_jobs([job(t) for t in running],
                                  return_on_first_completion=len(issued)>=5)
        for t,r in zip(running,records):
            if r['status'] not in ('complete','failed'):
                continue
            study._storage.set_trial_user_attr(t._trial_id,'queue_record',
                {k:v for k,v in r.items() if k!='summary'})
            try:
                if r['status']!='complete':
                    raise ValueError('Training/replay failed; see queue failure/log')
                value=paired_objective(r['summary'],flat_paired)
                study._storage.set_trial_user_attr(t._trial_id,'summary',r['summary'])
            except (KeyError,ValueError,TypeError) as error:
                study._storage.set_trial_user_attr(t._trial_id,'failure',str(error))
                study.tell(t.number,state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(t.number,value)
        export(study,output)
    report=export(study,output)
    report['status']='complete' if all(r['state']=='COMPLETE' for r in report['trials']) and len(report['trials'])==budget else 'completed_with_failures'
    write_json(output/'SUMMARY.json',report)
    return report


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--gpus',default='0,2,3')
    p.add_argument('--per-gpu',type=int,default=4)
    p.add_argument('--trials',type=int,default=20)
    a=p.parse_args(argv)
    a.gpus=tuple(int(v) for v in a.gpus.split(','))
    if (not a.gpus or len(set(a.gpus))!=len(a.gpus) or not set(a.gpus)<={0,2,3}
            or a.per_gpu!=4 or a.trials!=20):
        p.error('Authorized scope: GPU0/2/3, per-gpu4, twenty trials')
    return a


def main(argv=None):
    a=parse_args(argv)
    import optuna
    import torch
    import numpy as np
    from experiments.osram_meld_block2_auto_20261014.worker import DATASET, DATASET_SHA, source_hashes
    from gcnet_missing_m3.training_resume import _sha
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Require committed immutable training source')
    if _sha(DATASET)!=DATASET_SHA or shutil.disk_usage(a.output.parent).free<20*2**30:
        raise RuntimeError('Dataset SHA or disk preflight failed')
    flat=json.loads(FLAT.read_text())
    if flat['status']!='complete' or flat['method']!='Flat':
        raise ValueError('Accepted common-mask Flat result missing')
    a.output.mkdir(parents=True,exist_ok=True)
    provenance=dict(source_git_sha=sha,code_hashes=source_hashes(),
        launcher_sha256=_sha(Path(__file__)),frozen_source_sha256=_sha(SOURCE),
        dataset_sha256=DATASET_SHA,config=make_config(1,14),
        **gpu_provenance(a.gpus),
        initial_configs=list(INITIAL_CONFIGS),trials=20,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,optuna=optuna.__version__))
    previous=a.output/'PREFLIGHT.json'
    if previous.exists() and json.loads(previous.read_text())!=provenance:
        raise ValueError('Do not mix source/config versions')
    write_json(previous,provenance)
    runner=dict(status='running',pid=os.getpid(),source_git_sha=sha,started_unix=time.time())
    write_json(a.output/'RUNNER.json',runner)
    try:
        smoke=a.output/'smoke'
        smoke_config=a.output/'smoke_config.json'
        write_json(smoke_config,make_config(3,28))
        if not (smoke/'SUMMARY.json').exists():
            probe=StructureScheduler(a.output/'queue',gpus=a.gpus,per_gpu=a.per_gpu)
            snapshot=probe.gpu_snapshot()
            gpu=max(a.gpus,key=lambda g:snapshot[g]['free_mib'])
            if snapshot[gpu]['free_mib']<6000:
                raise RuntimeError('Insufficient space for worst-size real-batch smoke; external tasks untouched')
            uuid=snapshot[gpu]['uuid']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            cmd=[sys.executable,'-u','-m','experiments.osram_meld_block2_auto_20261014.worker',
                 '--config',str(smoke_config),'--output',str(smoke),'--gpu-uuid',uuid,'--smoke']
            with (a.output/'smoke.log').open('ab') as stream:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        s=json.loads((smoke/'SUMMARY.json').read_text())
        if s['status']!='complete' or s['config']!=make_config(3,28) or not s.get('smoke_only'):
            raise RuntimeError('Worst-size smoke incomplete/incompatible')
        scheduler=StructureScheduler(a.output/'queue',gpus=a.gpus,per_gpu=a.per_gpu,
            smoke_costs={'B6':{'peak_memory_mib':s['peak_memory_mib'],'disk_bytes':2*2**30}})
        report=run_search(a.output,flat['paired'],scheduler,budget=a.trials,max_pending=len(a.gpus)*4)
        runner.update(status=report['status'],finished_unix=time.time())
    except BaseException as error:
        runner.update(status='failed',error=repr(error),finished_unix=time.time())
        raise
    finally:
        write_json(a.output/'RUNNER.json',runner)


if __name__=='__main__':
    main()
