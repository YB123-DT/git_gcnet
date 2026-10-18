"""Finite six-job queue: MELD seed66 + IEMOCAP6 seed66 fivefold, no HPO."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.osram_paired_meld_iemocap6_20261018.spec import build_jobs,aggregate_jobs,PATTERNS,RATES,METRICS
from experiments.osram_paired_meld_iemocap6_20261018.worker import GPUS,HELPERS
from experiments.osram_persistent_patterns_20261018.run import write,check_source

GATE_IDS=('MELD_seed66_fold1','IEMOCAPSix_seed66_fold5')


def integration_ready(output):
    for name in GATE_IDS:
        path=Path(output)/name/'FIRST_EPOCH_INTEGRATION.json'
        if not path.is_file() or json.loads(path.read_text()).get('status')!='complete':
            return False
    return True


def baseline_scores():
    meld=json.loads((ROOT/'experiments/osram_meld_wd_seed66_20261018/BASELINE.json').read_text())
    iemo=json.loads((ROOT/'experiments/osram_iemocap_flat_3seed_cmerc_20261016/SUMMARY.json').read_text())
    fixed=json.loads((ROOT/'experiments/osram_persistent_patterns_20261018/SUMMARY.json').read_text())
    return dict(MELD=dict(random=next(r['testbest'] for r in meld['runs'] if r['seed']==66),
                         persistent=fixed['datasets']['MELD']['aggregation']['per_seed']['66']),
        IEMOCAPSix=dict(random=iemo['datasets']['IEMOCAPSix']['seed_fivefold']['66'],
                       persistent=fixed['datasets']['IEMOCAPSix']['aggregation']['per_seed']['66']))


def render(result,baseline):
    lines=['# Paired task-only transfer: MELD and IEMOCAP6','',
        'INTERNAL TEST-ORACLE SCREEN. Seed66 only; IEMOCAP6 means require all five folds.',
        'Status: '+result['status'], '',
        'Existing rho=.5, history_drop_prob=.2, no contrastive loss. Original no-JEPA Flat/CHM unchanged.',
        'MELD lr1e-5/wd1e-5/50epochs; IEMOCAP6 lr1e-3/wd1e-5/100epochs; batch32; gcnet-random-mask-v1.',
        'Task CE supervises ALL valid utterances in BOTH views; same-current/different-history anchors do not restrict task loss.',
        'Inference uses one view; no training-time InfoNCE/projector call at inference.',
        'View2 is only a nonempty subset of View1, not persistent-pattern training.', '',
        '| Dataset | Random rate | Flat ACC | Paired ACC | Flat UA | Paired UA | Flat WF1 | Paired WF1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,row in result['datasets'].items():
        for rate in (*RATES,'mean','high'):
            original=baseline[name]['random']['rates'][rate] if rate in RATES else baseline[name]['random'][rate]
            new=row['random']['rates'][rate] if rate in RATES else row['random'][rate]
            lines.append('| '+name+' | '+rate+' | '+' | '.join(f'{x[m]:.3f}' for m in ('ACC','UA','WF1') for x in (original,new))+' |')
    lines+=['','Random per-rate BEST scores retain the original resampled Test-oracle selection protocol; do not call them a common-mask test.',
        '', '| Dataset | Persistent observed set | Flat ACC | Paired ACC | Flat UA | Paired UA | Flat WF1 | Paired WF1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,row in result['datasets'].items():
        for pattern in (*PATTERNS,'macro_six','macro_seven'):
            selected=PATTERNS[:-1] if pattern=='macro_six' else PATTERNS
            old=baseline[name]['persistent'][pattern] if pattern in PATTERNS else {
                m:sum(baseline[name]['persistent'][p][m] for p in selected)/len(selected) for m in METRICS}
            new=row['persistent']['patterns'][pattern] if pattern in PATTERNS else row['persistent'][pattern]
            lines.append('| '+name+' | '+pattern+' | '+' | '.join(f'{x[m]:.3f}' for m in ('ACC','UA','WF1') for x in (old,new))+' |')
    lines+=['', 'Persistent support is fixed across each whole conversation. Singles use .7 BEST, pairs .3, full .0.',
        'These are not seven independently trained models. No performance claim before all relevant folds complete.',
        'Checkpoints, exact selected masks/predictions, full optimizer/RNG state and persistent predictions are retained remotely.',
        'The inherited inert training projector is initialized in fork_rng, receives no task-only gradient, and is excluded from backbone initialization comparisons.',
        'Existing tests passed in both original numerical checkouts; first committed training epochs validate real CE/backward/optimizer/saving and memory admission.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();manifest=check_source()
    if args.output.exists() and not args.resume:raise FileExistsError('New output required, or explicit same-source --resume')
    if shutil.disk_usage(args.output.parent).free<20*2**30:raise RuntimeError('Need20GiB output/cache headroom')
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=build_jobs(args.output)
    # Begin with one real epoch per dataset; use the longest-fold condition as
    # integration/peak bound rather than launching an extra disposable smoke.
    order={name:i for i,name in enumerate((*GATE_IDS,'IEMOCAPSix_seed66_fold1',
        'IEMOCAPSix_seed66_fold2','IEMOCAPSix_seed66_fold3','IEMOCAPSix_seed66_fold4'))}
    jobs.sort(key=lambda j:order[j['id']])
    plan=dict(source_commit=args.commit,manifest=manifest,jobs=jobs,gpus=GPUS,max_per_gpu=3,
              seed=66,planned_training_jobs=6,extra_hpo=False,baseline=baseline_scores())
    path=args.output/'MANIFEST.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(plan)):
        raise ValueError('Cannot mix experiment sources/configs')
    write(path,plan)
    sys.path.append(str(HELPERS))
    from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler

    class PairedScheduler(Scheduler):
        def command(self,job,config_path,uuid):
            return [self.python,'-u',str(Path(__file__).with_name('worker.py')),'--job',job['id'],
                '--output-root',str(args.output),'--config',str(config_path),'--gpu-uuid',uuid,'--commit',args.commit]

        def admit(self,job):
            if job['id'] not in GATE_IDS and not integration_ready(args.output):
                if any(self.state['jobs'].get(g,{}).get('status')=='failed' for g in GATE_IDS):
                    self.failure(job,'Real first-epoch integration failed; dependent jobs not launched')
                return None
            peaks=[]
            for gate in GATE_IDS:
                path=args.output/gate/'FIRST_EPOCH_INTEGRATION.json'
                if path.exists():peaks.append(json.loads(path.read_text())['peak_memory_mib'])
            if peaks:self.smoke_costs['paired_task_only_rho05']['peak_memory_mib']=max(6144,max(peaks)*1.4)
            return super().admit(job)

    scheduler=PairedScheduler(args.output/'queue',gpus=(0,3),per_gpu=3,code_root=ROOT,
        smoke_costs={'paired_task_only_rho05':dict(peak_memory_mib=6144,disk_bytes=2*2**30)})
    snapshot=scheduler.gpu_snapshot()
    if any(snapshot[g]['uuid']!=uuid for g,uuid in GPUS.items()):raise RuntimeError('Healthy physical GPU0/3 UUID changed')
    records=scheduler.run_jobs(jobs)
    summaries={r['id']:r['summary'] for r in records if r['status']=='complete'}
    result=aggregate_jobs(jobs,summaries)
    if any(r['status']=='failed' for r in records):result['status']='failed'
    result.update(source_commit=args.commit,baseline=plan['baseline'],
                  jobs=[{k:v for k,v in r.items() if k!='summary'} for r in records])
    write(args.output/'RESULTS.json',result)
    (args.output/'RESULT.md').write_text(render(result,plan['baseline']))
    if result['status']!='complete':raise RuntimeError('Finite paired transfer incomplete; inspect individual failure records')


if __name__=='__main__':main()
