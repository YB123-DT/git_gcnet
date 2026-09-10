"""Summarize completed per-rate Test-oracle runs; do not choose a mean checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.osram_causal_readout_20260910.run import FULL, SEEDS, RATES
from experiments.osram_causal_readout_20260910.summarize import extract_best, mask_hashes


def save(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(root):
    rows=[];data={}
    for seed in SEEDS:
        for mode,p in [('flat',FULL/f'seed_{seed}'),('cross',root/'mosi'/f'seed_{seed}')]:
            best=extract_best(json.loads((p/'history.json').read_text()))
            metrics=json.loads((p/'metrics.json').read_text())
            if mode=='cross':
                assert json.loads((p/'PROVENANCE.json').read_text())['status']=='complete'
                assert metrics['selection_protocol']=='per-rate-test-oracle'
                assert mask_hashes(metrics)==mask_hashes(json.loads((FULL/f'seed_{seed}'/'metrics.json').read_text()))
                for rate in RATES:
                    assert metrics['selected_epoch_by_rate'][rate]==best[rate]['epoch']
            data[(seed,mode)]=best
            rows.extend(dict(seed=seed,mode=mode,rate=r,selected_epoch=best[r]['epoch'],weighted_f1=100*best[r]['weighted_f1']) for r in RATES)
    save(root/'per_seed_rate.csv',rows)
    summary=[]
    for name,rates in [(r,[r]) for r in RATES]+[('all8',RATES),('high',('0.5','0.6','0.7'))]:
        f=[st.mean(data[(s,'flat')][r]['weighted_f1']*100 for r in rates) for s in SEEDS]
        c=[st.mean(data[(s,'cross')][r]['weighted_f1']*100 for r in rates) for s in SEEDS]
        delta=[b-a for a,b in zip(f,c)]
        summary.append(dict(rate=name,flat_mean=st.mean(f),flat_sd=st.stdev(f),cross_mean=st.mean(c),cross_sd=st.stdev(c),delta=st.mean(delta),positive_seeds=sum(d>0 for d in delta)))
    save(root/'summary.csv',summary)
    lines=['# MOSI Local Cross-Attention result','',
        '**INTERNAL DIAGNOSTIC ONLY: each seed × rate independently selects highest Test W-F1, earliest tie.**',
        'Five seeds66–70, 100epochs, cyclic, causal OSRAM write step .6. Flat inherited from full histories.',
        'Eight-rate/high-rate averages below are descriptive, never used to choose a checkpoint.', '',
        '| Rate | Flat mean ± SD | Cross-attn mean ± SD | Delta pp | Positive seeds |','|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['rate']} | {r['flat_mean']:.3f} ± {r['flat_sd']:.3f} | {r['cross_mean']:.3f} ± {r['cross_sd']:.3f} | {r['delta']:+.3f} | {r['positive_seeds']}/5 |")
    lines+=['','All test masks matched. Per-seed/rate selected epochs are in per_seed_rate.csv.',
        'This is not a parameter-matched isolated compression experiment. See DESIGN.md.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
