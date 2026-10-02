"""Summarize all authorized runs, never substitute last epoch for per-rate best."""
import csv
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
old=root.parent/'osram_paired_history_views_20261001/results'
reference=read(old/'SUMMARY.json')['flat']
rows=[]
baseline_masks=None
for index in range(1,11):
    folder=root/'results'/f'C{index:02d}'
    assert read(folder/'PROVENANCE.json')['status']=='complete'
    config=read(folder/'config.json')
    history=read(folder/'history.json')
    assert len(history)==100
    assert config['history_contrast_weight']==0
    masks=[(x['train']['paired_history']['view1_mask_sha256'],x['train']['paired_history']['view2_mask_sha256']) for x in history]
    if baseline_masks is None: baseline_masks=masks
    assert masks==baseline_masks
    metrics=read(folder/'metrics.json')
    assert metrics['selection_protocol']=='per-rate-test-oracle'
    scores=[100*metrics['test'][f'{r/10:.1f}']['weighted_f1'] for r in range(8)]
    rows.append(dict(run=f'C{index:02d}',rho=config['history_task_view2_weight'],
                     **{f'miss_{r/10:.1f}':v for r,v in enumerate(scores)},
                     mean_8rate=statistics.mean(scores),high_missing=statistics.mean(scores[5:])))
a_masks=[(x['train']['paired_history']['view1_mask_sha256'],x['train']['paired_history']['view2_mask_sha256']) for x in read(old/'control/history.json')]
assert baseline_masks==a_masks
a=read(old/'control/metrics.json')
a_scores=[100*a['test'][f'{i/10:.1f}']['weighted_f1'] for i in range(8)]
report=['# Ten rho configurations — completed', '',
        'All10 seed66 runs completed100epochs, eight per-rate best checkpoints each.',
        'INTERNAL Test-oracle sweep; not validation-selected paper results.',
        'All100epochs have identical View1/View2 mask hashes across all10runs and original A.', '',
        '|rho|8-rate W-F1|High .5/.6/.7|Delta vs Flat (8-rate)|', '|---|---:|---:|---:|']
flat=reference['mean_8rate']
high=reference['high_missing']
# Existing rate_scores reports percentages.
report.append(f'|Original Flat|{flat:.3f}|{high:.3f}|0.000|')
report.append(f'|0.50 (existing A)|{statistics.mean(a_scores):.3f}|{statistics.mean(a_scores[5:]):.3f}|{statistics.mean(a_scores)-flat:+.3f}|')
for r in rows:
    report.append(f"|{r['rho']:.2f}|{r['mean_8rate']:.3f}|{r['high_missing']:.3f}|{r['mean_8rate']-flat:+.3f}|")
report+=['','All scores %, deltas percentage points. Single seed;10 configurations were scanned.',
         'No statistical significance or generalization claim. Do not treat the test-best rho as a validated choice.',
         'No fixed-modality or history-deletion evaluation was added in this status task.','']
(root/'RESULT.md').write_text('\n'.join(report))
with (root/'per_rate.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print('\n'.join(report))
