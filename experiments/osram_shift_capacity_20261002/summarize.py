"""Report all four completed depth/width experiments, seed66 per-rate BEST."""
import csv
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
assert read(root/'results/status.json')['status']=='complete'
rows=[]
for name in ('D2-W128','D2-W256','D3-W128','D3-W256'):
    p=root/'results'/name
    assert read(p/'PROVENANCE.json')['status']=='complete'
    assert len(read(p/'history.json'))==100
    config=read(p/'config.json')
    assert not config['paired_history_views']
    metrics=read(p/'metrics.json')
    assert metrics['selection_protocol']=='per-rate-test-oracle'
    values=[100*metrics['test'][f'{i/10:.1f}']['weighted_f1'] for i in range(8)]
    rows.append(dict(name=name,**{f'miss_{i/10:.1f}':v for i,v in enumerate(values)},
                     mean=statistics.mean(values),high=statistics.mean(values[5:])))
flat=81.06809539495711;flat_high=76.35225088637502
old=80.49675910868038;old_high=75.81263398697254
report=['# Scalar filter capacity — all four complete','',
        'Seed66,100epochs, eight per-rate BEST checkpoints each. Internal Test-oracle diagnostics only.',
        'GPU5 concurrent training; no model/data changes beyond declared scalar MLP depth/width.',
        'No paired-view training or InfoNCE. Relation128, scalar output1 and zero-init residual unchanged.','',
        '|Filter|8-rate W-F1|High .5/.6/.7|Delta eight-rate vs Flat|Delta high vs Flat|',
        '|---|---:|---:|---:|---:|',
        f'|Original Flat|{flat:.3f}|{flat_high:.3f}|—|—|',
        f'|Old D1-W128|{old:.3f}|{old_high:.3f}|{old-flat:+.3f}|{old_high-flat_high:+.3f}|']
for r in rows:
    report.append(f"|{r['name']}|{r['mean']:.3f}|{r['high']:.3f}|{r['mean']-flat:+.3f}|{r['high']-flat_high:+.3f}|")
report+=['','D3-W256 is best among these four and exceeds the old scalar residual by .435pp eight-rate',
         'and .388pp high-missing, but remains below original Flat by .136pp and .152pp.',
         'Depth/width effects are not monotonic. Single seed and four-config scan do not establish',
         'stable gains or statistical significance. No automatic additional runs were launched.','']
(root/'RESULT.md').write_text('\n'.join(report))
with (root/'per_rate.csv').open('w',newline='') as stream:
    w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('\n'.join(report))
