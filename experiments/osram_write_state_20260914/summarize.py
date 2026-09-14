"""Completed WSC vs inherited controls; independent per-rate Test-oracle."""
import csv
import json
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.osram_causal_readout_20260910.run import FULL,SEEDS,RATES
from experiments.osram_causal_readout_20260910.summarize import extract_best,mask_hashes

ROOT=Path('/data2/yb/remote_experiments/osram_write_state_20260914')
PATHS={'Joint':FULL,'No-JEPA':ROOT.parent/'osram_causal_nojepa_20260910/mosi',
       'State-JEPA':ROOT.parent/'osram_complete_state_20260910/mosi','WSC':ROOT/'mosi'}


def save(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    data={};rows=[]
    for seed in SEEDS:
        ref=json.loads((FULL/f'seed_{seed}'/'metrics.json').read_text())
        for mode,root in PATHS.items():
            p=root/f'seed_{seed}';h=json.loads((p/'history.json').read_text())
            best=extract_best(h);m=json.loads((p/'metrics.json').read_text())
            assert mask_hashes(m)==mask_hashes(ref)
            if mode!='Joint':
                assert json.loads((p/'PROVENANCE.json').read_text())['status']=='complete'
                assert m['selection_protocol']=='per-rate-test-oracle'
                assert all(m['selected_epoch_by_rate'][r]==best[r]['epoch'] for r in RATES)
            data[seed,mode]=best
            for r in RATES:rows.append(dict(seed=seed,mode=mode,rate=r,weighted_f1=100*best[r]['weighted_f1'],selected_epoch=best[r]['epoch']))
    save(ROOT/'per_seed_rate.csv',rows);summary=[]
    for label,rates in [(r,[r]) for r in RATES]+[('all8',RATES),('high',('0.5','0.6','0.7'))]:
        v={m:[st.mean(data[s,m][r]['weighted_f1']*100 for r in rates) for s in SEEDS] for m in PATHS}
        row=dict(rate=label)
        for m,a in v.items():row.update({m+'_mean':st.mean(a),m+'_sd':st.stdev(a)})
        for m in ('Joint','No-JEPA','State-JEPA'):
            d=[a-b for a,b in zip(v['WSC'],v[m])]
            row.update({'delta_vs_'+m:st.mean(d),'positive_vs_'+m:sum(x>0 for x in d)})
        summary.append(row)
    save(ROOT/'summary.csv',summary)
    lines=['# WSC-OSRAM MOSI result','',
        '**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**',
        'Five seeds66–70,100epochs,cyclic,causal eta=.6 Flat. Each seed × rate independently selects highest Test W-F1, earliest tie.',
        'Means are descriptive only, never checkpoint-selection criteria. All test masks matched. Controls inherited.', '',
        '| Rate | Joint | No-JEPA | State-JEPA | WSC mean±SD | WSC−Joint pp | Positive vs Joint |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        cells=[r['rate']]+[f"{r[m+'_mean']:.3f}" for m in ('Joint','No-JEPA','State-JEPA')]
        cells += [f"{r['WSC_mean']:.3f} ± {r['WSC_sd']:.3f}",f"{r['delta_vs_Joint']:+.3f}",f"{r['positive_vs_Joint']}/5"]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','No target-quality or memory-retention mechanism is inferred from these scores alone.',
            'No automatic tuning or additional training launched. Detailed seed/epoch selections: per_seed_rate.csv.']
    result='\n'.join(lines)+'\n';(ROOT/'RESULT.md').write_text(result);print(result)


if __name__=='__main__':main()
