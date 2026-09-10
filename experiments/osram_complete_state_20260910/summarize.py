"""Three completed objectives, independently selected per seed and test rate."""
import csv
import json
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.osram_causal_readout_20260910.run import FULL, SEEDS, RATES
from experiments.osram_causal_readout_20260910.summarize import extract_best, mask_hashes

ROOT=Path('/data2/yb/remote_experiments/osram_complete_state_20260910')
NOJEPA=Path('/data2/yb/remote_experiments/osram_causal_nojepa_20260910/mosi')


def save(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    data={};rows=[];losses=[]
    for seed in SEEDS:
        ref=json.loads((FULL/f'seed_{seed}'/'metrics.json').read_text())
        for name,root in [('joint',FULL),('emotion_only',NOJEPA),('state',ROOT/'mosi')]:
            p=root/f'seed_{seed}'
            cfg=json.loads((p/'config.json').read_text())
            assert cfg['osram_write_step']==.6 and not cfg['osram_bidirectional']
            assert cfg.get('osram_readout_fusion','flat')=='flat'
            assert cfg['train_rate_mode']=='cyclic'
            assert cfg['training_objective']=={'joint':'joint','emotion_only':'emotion-only','state':'complete-state'}[name]
            h=json.loads((p/'history.json').read_text());best=extract_best(h)
            m=json.loads((p/'metrics.json').read_text())
            assert mask_hashes(m)==mask_hashes(ref)
            if name!='joint':
                assert json.loads((p/'PROVENANCE.json').read_text())['status']=='complete'
                assert m['selection_protocol']=='per-rate-test-oracle'
                assert all(m['selected_epoch_by_rate'][r]==best[r]['epoch'] for r in RATES)
            if name=='emotion_only':assert all(v['train']['jepa_loss']==0 for v in h)
            if name=='state':
                assert all(v['train']['state_loss']==v['train']['jepa_loss'] for v in h)
                losses.append(dict(seed=seed,first_state_loss=h[0]['train']['state_loss'],last_state_loss=h[-1]['train']['state_loss']))
            data[seed,name]=best
            for r in RATES:rows.append(dict(seed=seed,mode=name,rate=r,selected_epoch=best[r]['epoch'],weighted_f1=100*best[r]['weighted_f1']))
    save(ROOT/'per_seed_rate.csv',rows);save(ROOT/'state_loss_endpoints.csv',losses)
    summary=[]
    for label,rates in [(r,[r]) for r in RATES]+[('all8',RATES),('high',('0.5','0.6','0.7'))]:
        values={name:[st.mean(data[s,name][r]['weighted_f1']*100 for r in rates) for s in SEEDS]
                for name in ('joint','emotion_only','state')}
        row=dict(rate=label)
        for name,v in values.items():row.update({name+'_mean':st.mean(v),name+'_sd':st.stdev(v)})
        for refname in ('joint','emotion_only'):
            d=[a-b for a,b in zip(values['state'],values[refname])]
            row.update({'delta_vs_'+refname:st.mean(d),'positive_vs_'+refname:sum(v>0 for v in d)})
        summary.append(row)
    save(ROOT/'summary.csv',summary)
    lines=['# Complete-View Local-State JEPA: MOSI results','',
        '**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**',
        'Seeds66–70;100epochs;causal Flat eta=.6;cyclic. Each seed × rate independently selects maximum Test W-F1 (earliest tie).',
        'All averages are descriptive, never used for checkpoint selection. All test mask hashes matched.', '',
        '| rate | Joint mean±SD | Emotion-only mean±SD | State mean±SD | State−Joint pp | State−Emotion pp | positive vs Joint / Emotion |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v in summary:
        cells=[v['rate']]+[f"{v[n+'_mean']:.3f} ± {v[n+'_sd']:.3f}" for n in ('joint','emotion_only','state')]
        cells += [f"{v['delta_vs_joint']:+.3f}",f"{v['delta_vs_emotion_only']:+.3f}",f"{v['positive_vs_joint']}/5 ; {v['positive_vs_emotion_only']}/5"]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','Full histories and selected epochs are archived. Emotion-only JEPA loss was zero in all epochs.',
            'State loss decline alone is not evidence of sample-specific prediction or absence of collapse; no representation audit is claimed.']
    text='\n'.join(lines)+'\n';(ROOT/'RESULT.md').write_text(text);print(text)


if __name__=='__main__':main()
