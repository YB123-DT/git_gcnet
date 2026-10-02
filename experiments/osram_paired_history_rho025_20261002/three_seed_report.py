"""Validate and report all nine selected rho/seed outcomes."""
import json
from pathlib import Path
import statistics as st

root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
source=root/'three_seed_results'
assert read(source/'status.json')['status']=='complete'
rows=read(source/'SUMMARY.json')['rows']
assert len(rows)==9
masks={}
for r in rows:
    folder=(root/'results'/{.05:'C01',.1:'C02',.25:'C05'}[r['rho']] if r['seed']==66
            else source/f"rho_{r['rho']:.2f}_seed_{r['seed']}")
    assert read(folder/'PROVENANCE.json')['status']=='complete'
    history=read(folder/'history.json'); assert len(history)==100
    metrics=read(folder/'metrics.json')
    values=[100*metrics['test'][f'{i/10:.1f}']['weighted_f1'] for i in range(8)]
    assert values==r['per_rate']
    assert abs(st.mean(values)-r['mean'])<1e-10
    hashes=[(h['train']['paired_history']['view1_mask_sha256'],h['train']['paired_history']['view2_mask_sha256']) for h in history]
    if r['seed'] in masks: assert masks[r['seed']]==hashes
    masks[r['seed']]=hashes
base=[r for r in rows if r['rho']==.05]
fm=st.mean(r['flat_mean'] for r in base);fh=st.mean(r['flat_high'] for r in base)
report=['# Three rho values, three seeds — complete','',
        'Seed66 reused; six new seed67/68 runs completed100epochs. Original per-rate BEST, not last epoch.',
        'All nine per-rate summaries verified; per-seed masks identical across rho values.',
        'INTERNAL TEST-ORACLE; candidates selected from a seed66 sweep, not a formal validation protocol.',
        'Mean ± sample standard deviation across seeds66/67/68. Units W-F1%, deltas percentage points.','',
        '|Model|8-rate mean ± SD|Delta|High missing mean ± SD|Delta|','|---|---:|---:|---:|---:|',
        f"|Flat|{fm:.3f} ± {st.stdev(r['flat_mean'] for r in base):.3f}|—|{fh:.3f} ± {st.stdev(r['flat_high'] for r in base):.3f}|—|"]
for rho in (.05,.1,.25):
    s=[r for r in rows if r['rho']==rho];m=st.mean(r['mean'] for r in s);h=st.mean(r['high'] for r in s)
    report.append(f"|{rho:.2f}|{m:.3f} ± {st.stdev(r['mean'] for r in s):.3f}|{m-fm:+.3f}|{h:.3f} ± {st.stdev(r['high'] for r in s):.3f}|{h-fh:+.3f}|")
report+=['','|rho|seed|8-rate|Delta vs seed-matched Flat|High missing|Delta vs Flat|','|---|---:|---:|---:|---:|---:|']
for r in rows:
    report.append(f"|{r['rho']:.2f}|{r['seed']}|{r['mean']:.3f}|{r['mean']-r['flat_mean']:+.3f}|{r['high']:.3f}|{r['high']-r['flat_high']:+.3f}|")
report+=['','All three rho values have lower mean eight-rate and high-missing W-F1 than Flat.',
         'The positive seed66 eight-rate result at rho.10 did not reproduce in seeds67/68.',
         'No significance claim; no new tuning, inference, or fixed-pattern experiments in this report.','']
(root/'THREE_SEED_RESULT.md').write_text('\n'.join(report))
print('\n'.join(report))
