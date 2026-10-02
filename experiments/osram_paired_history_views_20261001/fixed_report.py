"""Summarize all six requested patterns without selecting configurations."""
import csv
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
raw=json.loads((root/'fixed_results.json').read_text())
assert raw['status']=='complete' and len(raw['records'])==12
source=root.parent/'osram_cfg84_fixed_modality_ablation_20260923/results/per_seed_pattern.csv'
with source.open() as stream:
    flat={r['pattern'].replace('L','T'):r for r in csv.DictReader(stream) if r['seed']=='66'}
results={(r['arm'],r['pattern']):r for r in raw['records']}
assert len(results)==12
rows=[]
for pattern in ('A','T','V','AT','AV','TV'):
    f=float(flat[pattern]['weighted_f1'])
    a=100*results['control',pattern]['weighted_f1']
    b=100*results['contrastive',pattern]['weighted_f1']
    for arm in ('control','contrastive'):
        r=results[arm,pattern]
        assert r['checkpoint_rate']==float(flat[pattern]['checkpoint_rate'])
        assert r['frozen_hash_check'] and r['projector_calls']==0
    rows.append(dict(pattern=pattern,Flat=f,A=a,B=b,A_minus_Flat=a-f,B_minus_Flat=b-f,B_minus_A=b-a))
report=['# Fixed whole-conversation modality comparison', '',
        'Seed66; existing per-rate Test-oracle checkpoints. Evaluation-only, no retraining.',
        'A=paired task-only; B=paired task+InfoNCE. T=Text.',
        'Full MOSI test:686 utterances; original nonzero-label binary W-F1 uses656.',
        'Each conversation keeps only the named modalities throughout. Single-modality uses miss.7 best;',
        'pairs use miss.3 best. Flat reuses the original seed66 fixed-pattern record, not a multiseed mean.',
        'Same original evaluator; original task head, no projector calls, frozen checkpoint/state hashes.', '',
        '|Visible|Flat|A|B|A−Flat|B−Flat|B−A|', '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    report.append('|'+row['pattern']+'|'+'|'.join(f'{row[k]:.3f}' for k in ('Flat','A','B','A_minus_Flat','B_minus_Flat','B_minus_A'))+'|')
for name,patterns in [('No Text',('A','V','AV')),('Text present',('T','AT','TV')),('Six-pattern mean',tuple(r['pattern'] for r in rows))]:
    values=[r for r in rows if r['pattern'] in patterns]
    report.append('|'+name+'|'+'|'.join(f'{statistics.mean(r[k] for r in values):.3f}' for k in ('Flat','A','B','A_minus_Flat','B_minus_Flat','B_minus_A'))+'|')
report+=['','Units:% W-F1, differences in percentage points. Single seed; no significance claim.',
          'A smaller history-deletion flip rate is a different property from these persistent-missing scores.',
          'This comparison does not establish a causal training mechanism. No extra tuning or trials.','']
(root/'FIXED_RESULT.md').write_text('\n'.join(report))
print('\n'.join(report))
