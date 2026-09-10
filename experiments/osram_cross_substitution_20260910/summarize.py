"""Paired frozen-checkpoint substitution report; never selects an epoch."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st

MODES = ('normal', 'no_gap', 'no_base', 'base_to_gap', 'gap_to_base')


def summarize(root):
    reports = [json.loads((root/f'seed{s}.json').read_text()) for s in range(66,71)]
    assert all(len(r['rows']) == 8*4*5 and all(r['mask_matches'].values()) for r in reports)
    raw = [dict(seed=r['seed'], rate=row['rate'], group=row['group'], mode=row['mode'],
                count=row['count'], nonzero_count=row['nonzero_count'],
                weighted_f1=None if row['metrics'] is None else row['metrics']['weighted_f1']*100)
           for r in reports for row in r['rows']]
    with (root/'per_seed_rate.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0])); w.writeheader(); w.writerows(raw)
    summary = []
    for group in ('one_missing', 'AT', 'AV', 'TV'):
        for rate in [i/10 for i in range(8)]:
            subset = [x for x in raw if x['group']==group and x['rate']==rate]
            for mode in MODES:
                values = [x['weighted_f1'] for x in subset if x['mode']==mode and x['weighted_f1'] is not None]
                summary.append(dict(group=group,rate=rate,mode=mode,n_seeds=len(values),
                    mean=None if not values else st.mean(values), sd=None if len(values)<2 else st.stdev(values)))
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines=['# Frozen Flat Base/Gap cross-substitution', '',
           '**INTERNAL DIAGNOSTIC ONLY.** MOSI seeds 66–70; causal eta=.6; existing `best.pt`.',
           'No training, no new epoch selection. Existing checkpoints were selected by eight-rate-mean Test oracle;',
           'they are NOT the per-rate maxima reported in the preceding readout experiment.', '',
           'One original model forward per batch. Replay only emotion_adapter, local_skip, emotion_norm and smax_fc.',
           'Normal replay logits are exactly equal; every test mask hash matches the reference.',
           'Only exactly-one-missing valid utterances are patched. W-F1 excludes labels equal to zero.',
           'Donor slots remain intact: Base→Gap retains Base; Gap→Base retains the active Gap.',
           'Raw magnitude and slot positions are preserved; no normalization/learned alignment is added.', '',
           '| Seed | Existing checkpoint epoch |', '|---|---:|']
    lines += [f"| {r['seed']} | {r['checkpoint_epoch']} |" for r in reports]
    for group in ('one_missing','AT','AV','TV'):
        lines += ['',f'## {group}: five-seed W-F1 mean ± sample SD (%)','',
            '| Rate | Nonzero samples per seed | Normal | No Gap | No Base | Base→Gap | Gap→Base |',
            '|---|---|---|---|---|---|---|']
        for rate in [i/10 for i in range(8)]:
            cells=[]
            for mode in MODES:
                item=next(s for s in summary if s['group']==group and s['rate']==rate and s['mode']==mode)
                cells.append('N/A' if item['mean'] is None else f"{item['mean']:.3f} ± {item['sd']:.3f} (n={item['n_seeds']})" if item['sd'] is not None else f"{item['mean']:.3f} (n=1)")
            counts=[x['nonzero_count'] for x in raw if x['group']==group and x['rate']==rate and x['mode']=='normal']
            lines.append('| '+str(rate)+' | '+','.join(map(str,counts))+' | '+' | '.join(cells)+' |')
    lines += ['', '## Recovery contrasts', '',
        'Descriptive equal-rate averages within each seed, then equal-seed averages. Only rates with eligible samples in all five seeds are included.',
        'Not an eight-rate benchmark score. Recovery is substitution minus the corresponding deletion, not proof of semantic identity.', '',
        '| Contrast | Mean delta (pp) | SD across seeds | Positive seeds |', '|---|---:|---:|---:|']
    eligible_rates=[i/10 for i in range(8) if all(next(x for x in raw if x['seed']==s and x['group']=='one_missing' and x['rate']==i/10 and x['mode']=='normal')['nonzero_count']>0 for s in range(66,71))]
    for a,b in [('no_gap','normal'),('no_base','normal'),('base_to_gap','no_gap'),('gap_to_base','no_base'),('base_to_gap','normal'),('gap_to_base','normal')]:
        delta=[]
        for seed in range(66,71):
            def value(mode,rate):
                return next(x['weighted_f1'] for x in raw if x['seed']==seed and x['group']=='one_missing' and x['rate']==rate and x['mode']==mode)
            delta.append(st.mean(value(a,r)-value(b,r) for r in eligible_rates))
        lines.append(f'| {a} − {b} | {st.mean(delta):+.3f} | {st.stdev(delta):.3f} | {sum(d>0 for d in delta)}/5 |')
    lines += ['',f'Included rates: {eligible_rates}.', '',
        'Failure to substitute can reflect slot-specific decoding, scaling or off-distribution combinations; it does not establish unrelated semantic content.',
        'Successful substitution supports functional replaceability in this frozen classifier, not geometric equality.',
        'No CKA/cosine analysis, model modification or extra training was performed.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[-14:]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);summarize(p.parse_args().root)
