"""Paired write-minus-decay summaries; observations are not independent trials."""
import csv
import json
import gzip
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

root=Path(__file__).resolve().parent
rows=[]
for meta in sorted((root/'raw').glob('*/metadata.json')):
    info=json.loads(meta.read_text())
    grouped=defaultdict(list)
    raw=meta.parent/'retention.jsonl'
    handle=raw.open() if raw.exists() else gzip.open(str(raw)+'.gz','rt')
    for line in handle:
        x=json.loads(line)
        if x['status']=='retention': grouped[x['target_modality']].append(x)
    handle.close()
    for modality, records in grouped.items():
        paired=[x['write_damage']-x['decay_damage'] for x in records]
        conversations=defaultdict(list)
        for x,diff in zip(records,paired): conversations[x['sample_id']].append(diff)
        rows.append(dict(dataset=info['dataset'],seed=info['seed'],rate=info['rate'],
            modality=modality,head_records=len(records),conversations=len(conversations),
            mean_decay_damage=mean(x['decay_damage'] for x in records),
            mean_write_damage=mean(x['write_damage'] for x in records),
            mean_paired_difference=mean(paired),median_paired_difference=median(paired),
            fraction_paired_positive=mean(x>0 for x in paired),
            conversation_mean_paired=mean(mean(x) for x in conversations.values()),
            positive_conversation_count=sum(mean(x)>0 for x in conversations.values())))
if not rows: raise RuntimeError('No completed diagnostic metadata')
with (root/'paired_difference.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# Real-checkpoint association retention: paired analysis','',
       'Evaluation-only; no model changes, no training. Each retained head record pairs',
       'write_damage and decay_damage for the same query/probe. Difference = write - decay.',
       'Rows/heads/time points are dependent; no independent-sample significance claims.',
       'Conversation means are included in CSV to check domination by long conversations.',
       'err_decay is current read; err_post concerns future reads only. No F1 causality claim.',
       '', '| Dataset | Seed | Rate | Target | Decay | Write | Paired mean | Positive conversations |',
       '|---|---:|---:|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['dataset']} | {r['seed']} | {r['rate']} | {r['modality']} | {r['mean_decay_damage']:.6f} | {r['mean_write_damage']:.6f} | {r['mean_paired_difference']:.6f} | {r['positive_conversation_count']}/{r['conversations']} |")
(root/'RESULT.md').write_text('\n'.join(lines)+'\n')
print(f'{len(rows)} dataset/seed/rate/modality groups; '+
      f'{sum(x["mean_paired_difference"]>0 for x in rows)} positive paired means')
