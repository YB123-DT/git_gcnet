"""Report all fixed-budget probe arms and seeds, no best-arm selection."""
import csv
import json
from pathlib import Path
import statistics

from experiments.osram_nested_root_probe_20261011.probe import ARMS

def render(root):
    root=Path(root)
    state=json.loads((root/'STATUS.json').read_text())
    fits=json.loads((root/'PROBE_RESULTS.json').read_text())
    assert state['status']=='completed' and len(state['records'])==8 and len(fits)==144
    assert all(r['frozen_unchanged'] for r in state['records'])
    macro=[];per_rate=[]
    for r in fits:
        per_rate.append({'rate':r['rate'],'arm':r['arm'],'probe_seed':r['probe_seed'],
            'wf1':100*r['test']['weighted_f1_nonzero'],'acc':100*r['test']['acc_nonzero'],
            'mse':r['test']['mse'],'n':r['test']['n'],'nonneutral_n':r['test']['nonneutral_n']})
    for arm in ARMS:
        for seed in (66,67,68):
            rows=[r for r in per_rate if r['arm']==arm and r['probe_seed']==seed]
            assert len(rows)==8
            macro.append({'arm':arm,'probe_seed':seed,'mean8':statistics.mean(r['wf1'] for r in rows),
                'high':statistics.mean(r['wf1'] for r in rows if r['rate']>=.5)})
    with (root/'per_rate.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_rate[0]));writer.writeheader();writer.writerows(per_rate)
    lines=['# Frozen Nested rooted representation probes','','INTERNAL DIAGNOSTIC ONLY','',
        'Original frozen Nested seed66, original per-rate Test-oracle BEST100epoch checkpoints.',
        'Probe seeds66/67/68 do NOT represent three backbone seeds. Fixed100epochs, train-only',
        'MSE optimization, no test-based probe selection. Test W-F1 excludes zero labels; threshold0.',
        'Rooted=exact33x64 subgraph pooled states; PreDecoder=33x64 graph readout output.',
        'No head pooling or random projection. Flat=R_off hidden from SAME Nested checkpoint,',
        'still includes raw Local/Base/Gap. Not a separately trained Flat checkpoint.',
        'All probes3715->32->GELU->1,118945parameters. Missing input blocks zero. Train-only',
        'normalization; inactive slots re-zeroed. Donor is train-only, same availability/history,',
        'different conversation, no label matching; common eligible cohorts across arms.',
        'All model/checkpoint hashes unchanged; Full predictions match original labels/masks/archive.',
        'Zero-init refers to decoder initialization, NOT necessarily its trained weight values.','',
        '## W-F1 %, equal-rate mean (mean ± sample std over probe seeds)','',
        '| Probe | Mean8 | High .5/.6/.7 |','|---|---:|---:|']
    for arm in ARMS:
        rows=[r for r in macro if r['arm']==arm]
        lines.append('|'+arm+'|'+'|'.join(f'{statistics.mean(r[k] for r in rows):.6f} ± {statistics.stdev(r[k] for r in rows):.6f}' for k in ('mean8','high'))+'|')
    lines+=['','## Per probe seed','', '| Arm | Seed | Mean8 | High |','|---|---:|---:|---:|']
    for r in macro:lines.append(f"|{r['arm']}|{r['probe_seed']}|{r['mean8']:.6f}|{r['high']:.6f}|")
    lines+=['','## Paired changes vs Flat probe (pp)','',
        '| Seed | Flat+Rooted minus Flat | Flat+Rooted minus donor | Flat+PreDecoder minus Flat |',
        '|---|---:|---:|---:|']
    for seed in (66,67,68):
        values={r['arm']:r['mean8'] for r in macro if r['probe_seed']==seed}
        lines.append(f"|{seed}|{values['Flat+Rooted']-values['Flat']:+.6f}|{values['Flat+Rooted']-values['Flat+DonorRooted']:+.6f}|{values['Flat+PreDecoder']-values['Flat']:+.6f}|")
    lines+=['','## Limits','',
        'Positive joint gains above both Flat and donor support extra decodable information under',
        'this bounded probe class, not a deployment gain or identified emotion mechanism.',
        'Negative results do not prove information absence; capacity, optimization and the fixed',
        'budget constrain inference. Equal total parameters are not equal effective capacity.',
        'Inherited backbone test selection means all findings are INTERNAL DIAGNOSTIC ONLY.',
        'Per-rate outcomes in per_rate.csv; all observed-pattern metrics, coverage and train curves',
        'in PROBE_RESULTS.json. Frozen original-head baseline metrics/checkpoint hashes in STATUS.json.',
        'Caches, probe weights and predictions remain remote. No automatic new model training.','']
    (root/'RESULT.md').write_text('\n'.join(lines))
    (root/'MACRO.json').write_text(json.dumps(macro,indent=2))
    return macro

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(render(p.parse_args().output),indent=2))
