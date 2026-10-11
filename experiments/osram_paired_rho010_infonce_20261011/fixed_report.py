"""Compare frozen rho.10 scores with previous seed-matched fixed-pattern results."""
import csv
import json
from pathlib import Path
import statistics


def render(root):
    raw=json.loads((root/'fixed_results.json').read_text())
    assert raw['status']=='complete' and len(raw['records'])==6
    old=json.loads((root.parent/'osram_paired_history_views_20261001/fixed_results.json').read_text())
    with (root.parent/'osram_cfg84_fixed_modality_ablation_20260923/results/per_seed_pattern.csv').open() as f:
        flat={r['pattern'].replace('L','T'):r for r in csv.DictReader(f) if r['seed']=='66'}
    values=[]
    for r in raw['records']:
        p=r['pattern']; prior=next(x for x in old['records'] if x['pattern']==p and x['arm']=='control')
        assert r['checkpoint_rate']==prior['checkpoint_rate']==float(flat[p]['checkpoint_rate'])
        assert r['frozen_hash_check'] and r['projector_calls']==0
        values.append(dict(pattern=p,flat=float(flat[p]['weighted_f1']),
            rho050=100*prior['weighted_f1'],rho010=100*r['weighted_f1']))
    lines=['# rho.10 task-only: fixed whole-conversation missing', '',
        'INTERNAL TEST-ORACLE DIAGNOSTIC ONLY', '',
        'MOSI seed66; frozen existing models, no training or new checkpoint selection.',
        'Each conversation keeps only the named modalities throughout. T=Text.',
        'Single modality: miss=.7 BEST; pairs: miss=.3 BEST. Original fixed evaluator.',
        '686 test utterances,656 nonneutral labels; threshold prediction>0, weighted binary F1.',
        'Historical backbone6e2a998 source hashes match original rho.10 C02. No training projector calls.',
        'Flat/rho.50 reuse seed66 records, not three-/five-seed averages. Current rho.10+InfoNCE training is separate.', '',
        '| Visible | Flat | rho.50 task-only | rho.10 task-only | rho.10−Flat pp |',
        '|---|---:|---:|---:|---:|']
    for r in values:
        lines.append(f"| {r['pattern']} | {r['flat']:.3f} | {r['rho050']:.3f} | {r['rho010']:.3f} | {r['rho010']-r['flat']:+.3f} |")
    for name,patterns in [('No Text',('A','V','AV')),('Text present',('T','AT','TV')),
        ('Six-pattern mean',tuple(r['pattern'] for r in values))]:
        rows=[r for r in values if r['pattern'] in patterns]
        means={k:statistics.mean(r[k] for r in rows) for k in ('flat','rho050','rho010')}
        lines.append(f"| {name} | {means['flat']:.3f} | {means['rho050']:.3f} | {means['rho010']:.3f} | {means['rho010']-means['flat']:+.3f} |")
    lines += ['', 'Units W-F1%; differences pp. Six patterns equally averaged, not a naturally weighted test population.',
        'Single seed. Different trained models and selected epochs; no causal mechanism or stable multi-seed benefit claim.',
        'No threshold tuning, synthetic modality completion, new masks for training or deployment performance claim.', '',
        f"Runner commit: {raw['runner_commit']}; server biggpu GPU6. Command:", '', '```bash',
        'python -m experiments.osram_paired_rho010_infonce_20261011.fixed \\',
        f"  --source {raw['source']} \\",
        f"  --output <new-output.json> --runner-commit {raw['runner_commit']}", '```','']
    (root/'FIXED_RESULT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    render(Path(__file__).resolve().parent)
