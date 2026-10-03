"""Selected D3-W256 three-seed results, existing per-rate BEST only."""
import json
from pathlib import Path
import statistics as st

root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
reference=read(root.parent/'osram_paired_history_rho025_20261002/three_seed_results/SUMMARY.json')
flat={r['seed']:r for r in reference['rows'] if r['rho']==.05}
rows=[]
for seed in (66,67,68):
    p=root/'results/D3-W256' if seed==66 else root/'three_seed_results'/f'D3-W256_seed_{seed}'
    assert read(p/'PROVENANCE.json')['status']=='complete'
    assert len(read(p/'history.json'))==100
    cfg=read(p/'config.json')
    assert cfg['seed']==seed and cfg['osram_shift_filter_depth']==3 and cfg['osram_shift_filter_width']==256
    metrics=read(p/'metrics.json');assert metrics['selection_protocol']=='per-rate-test-oracle'
    values=[100*metrics['test'][f'{i/10:.1f}']['weighted_f1'] for i in range(8)]
    rows.append(dict(seed=seed,per_rate=values,mean=st.mean(values),high=st.mean(values[5:]),
                     flat_mean=flat[seed]['flat_mean'],flat_high=flat[seed]['flat_high']))
report=['# D3-W256 three-seed confirmation — complete','',
        'Seed66 reused;67/68 completed100epochs and each retained8best checkpoints.',
        'INTERNAL per-rate Test-oracle diagnostics, not formal validation-selected paper results.',
        'W-F1%, differences percentage points. No last-epoch substitution.','',
        '|Seed|Flat eight-rate|D3-W256 eight-rate|Delta|Flat high|D3-W256 high|Delta|',
        '|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    report.append(f"|{r['seed']}|{r['flat_mean']:.3f}|{r['mean']:.3f}|{r['mean']-r['flat_mean']:+.3f}|{r['flat_high']:.3f}|{r['high']:.3f}|{r['high']-r['flat_high']:+.3f}|")
means={k:st.mean(r[k] for r in rows) for k in ('flat_mean','mean','flat_high','high')}
report.append('|Mean|'+'|'.join(f'{v:.3f}' for v in (means['flat_mean'],means['mean'],means['mean']-means['flat_mean'],means['flat_high'],means['high'],means['high']-means['flat_high']))+'|')
report+=['','|Model|Eight-rate mean ± sample SD|High missing mean ± sample SD|','|---|---:|---:|']
for name,m,h in [('Flat','flat_mean','flat_high'),('D3-W256','mean','high')]:
    report.append(f"|{name}|{means[m]:.3f} ± {st.stdev(r[m] for r in rows):.3f}|{means[h]:.3f} ± {st.stdev(r[h] for r in rows):.3f}|")
report+=['','Both eight-rate and high-missing scores are lower than Flat for all three seeds.',
         'Seed66 near-parity does not persist as a robust result. Selected after four seed66 configurations;',
         'no significance claim, new tuning, or new training in this report.','']
(root/'THREE_SEED_RESULT.md').write_text('\n'.join(report))
(root/'three_seed_results/SUMMARY.json').write_text(json.dumps(dict(rows=rows,means=means),indent=2)+'\n')
print('\n'.join(report))
