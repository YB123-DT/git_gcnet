"""Finite seed66 transfer of the existing rho=.5 paired task-only experiment."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = ('A','T','V','AT','AV','TV','ATV')
METRICS = ('ACC','UA','WF1','CE')
RATES = tuple(str(i/10) for i in range(8))


def paired_config(original):
    return dict(original, paired_history_views=True, history_drop_prob=.2,
                history_task_view2_weight=.5, history_contrast_weight=0.)


def build_jobs(output):
    meld = json.loads((ROOT/'experiments/osram_meld_wd_seed66_20261018/BASELINE.json').read_text())
    old = json.loads((ROOT/'experiments/osram_persistent_patterns_20261018/SUMMARY.json').read_text())
    sources = [('MELD',7,1,dict(meld['config_without_seed'],seed=66),
                next(r['output'] for r in meld['runs'] if r['seed']==66))]
    iemo = old['datasets']['IEMOCAPSix']
    for fold in range(1,6):
        reference = next(s for s in iemo['sources'] if s['seed']==66 and s['fold']==fold)
        sources.append(('IEMOCAPSix',6,fold,dict(iemo['common_training_config'],seed=66,fold=fold),reference['output']))
    jobs = []
    for dataset,classes,fold,base,reference in sources:
        expected = dict(dataset=dataset,seed=66,fold=fold,backbone_type='osram',
            training_objective='emotion-only',missing_mask_protocol='gcnet-random-mask-v1',
            osram_readout_fusion='flat',osram_output_dim=1600,osram_meaningful_block='none',
            osram_write_step=.6,osram_bidirectional=False,paired_history_views=False,
            batch_size=32,weight_decay=1e-5,train_rate_mode='cyclic',
            checkpoint_selection='test-oracle-per-rate',
            learning_rate=1e-5 if dataset=='MELD' else .001,
            epochs=50 if dataset=='MELD' else 100)
        if any(base.get(k)!=v for k,v in expected.items()) or base.get('osram_disable_local_skip',False):
            raise ValueError('Accepted Flat protocol changed: '+dataset)
        name=f'{dataset}_seed66_fold{fold}'
        jobs.append(dict(id=name,dataset=dataset,classes=classes,seed=66,fold=fold,
            config=paired_config(base),baseline_config=base,baseline_output=reference,
            output=str(Path(output)/name),method='paired_task_only_rho05'))
    return jobs


def _mean(rows):
    return {m:statistics.mean(r[m] for r in rows) for m in METRICS}


def aggregate_jobs(jobs,summaries):
    complete = {j['id']:summaries[j['id']] for j in jobs
                if j['id'] in summaries and summaries[j['id']].get('status')=='complete'}
    result=dict(status='complete' if len(complete)==6 else 'running',
                completed_count=len(complete),planned_count=6,datasets={})
    for dataset in ('MELD','IEMOCAPSix'):
        needed=[j for j in jobs if j['dataset']==dataset]
        if any(j['id'] not in complete for j in needed):
            continue
        rows=[complete[j['id']] for j in needed]
        random={rate:_mean([r['testbest']['rates'][rate] for r in rows]) for rate in RATES}
        fixed={p:_mean([r['persistent']['patterns'][p]['metrics'] for r in rows]) for p in PATTERNS}
        result['datasets'][dataset]=dict(folds={str(r['fold']):r for r in rows},
            random=dict(rates=random,mean=_mean(list(random.values())),
                        high=_mean([random[str(i/10)] for i in (5,6,7)])),
            persistent=dict(patterns=fixed,macro_six=_mean([fixed[p] for p in PATTERNS[:-1]]),
                            macro_seven=_mean(list(fixed.values()))))
    return result
