"""Read-only analysis of saved epoch losses; no fitting or inference."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
rows, hashes = [], {}
for arm in ('control', 'contrastive'):
    source = root/'results'/arm
    config = json.loads((source/'config.json').read_text())
    weight = config['history_contrast_weight']
    history = json.loads((source/'history.json').read_text())
    assert len(history) == 100
    hashes[arm] = hashlib.sha256((source/'history.json').read_bytes()).hexdigest()
    for item in history:
        train = item['train']
        p = train['paired_history']
        task = .5*(p['task_view1']+p['task_view2'])
        auxiliary = weight*p['infonce_loss']
        assert abs(task-train['classification_loss']) < 1e-4
        assert abs(task+auxiliary-train['loss']) < 1e-4
        rows.append(dict(arm=arm, epoch=item['epoch'], view1=p['task_view1'],
                         view2=p['task_view2'], task=task, infonce=p['infonce_loss'],
                         weight=weight, weighted_infonce=auxiliary, total=train['loss'],
                         auxiliary_over_task_percent=100*auxiliary/task))
out=root/'loss_summary'
out.mkdir(exist_ok=True)
with (out/'epochs.csv').open('w', newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
report=['# Paired-view training loss audit', '',
        'Saved seed66 training logs only. A weight0, B weight0.1. No new inference/training.',
        'Task = .5*(View1 MSE + View2 MSE); total = task + weight*InfoNCE.',
        'All 200 epoch records pass reconstruction checks (absolute tolerance1e-4).', '',
        '|Arm|Epochs|View1 MSE|View2 MSE|Mean task|Raw InfoNCE|Weighted InfoNCE|Total|Weighted InfoNCE / task %|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in ('control','contrastive'):
    for lo,hi in [(1,1),(1,20),(21,40),(41,60),(61,80),(81,100),(100,100)]:
        selected=[r for r in rows if r['arm']==arm and lo<=r['epoch']<=hi]
        means={k:statistics.mean(r[k] for r in selected) for k in ('view1','view2','task','infonce','weighted_infonce','total')}
        ratio=100*means['weighted_infonce']/means['task']
        report.append('|'+('A' if arm=='control' else 'B')+f'|{lo}–{hi}|'+
                      '|'.join(f'{means[k]:.4f}' for k in means)+f'|{ratio:.2f}|')
report += ['', 'Window statistics average the logged epoch means; ratio is ratio of window means.',
           'These are training losses, not loss at each rate-specific best checkpoint.',
           'Cyclic missing rates vary across epochs; one epoch is not an eight-rate average.',
           'A computes raw InfoNCE for logging/equal-compute but multiplies it by0; its projector is not trained by InfoNCE.',
           'Therefore raw A/B InfoNCE is not a comparison of equally trained projectors.',
           'Task losses cover all valid utterances; InfoNCE covers only eligible history anchors.',
           'Scalar loss ratios do not measure gradient magnitude or gradient conflict.',
           'This analysis alone does not establish an oversized contrastive weight as the cause of W-F1 loss.', '']
(out/'RESULT.md').write_text('\n'.join(report))
(out/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('\n'.join(report))
