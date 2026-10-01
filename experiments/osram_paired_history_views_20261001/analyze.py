"""Summarize saved outputs only; never performs inference or checkpoint selection."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(root):
    summary=json.loads((root/'SUMMARY.json').read_text())
    assert not summary['smoke_only']
    assert summary['paired_mask_hashes_match']
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    children=json.loads((root/'children.json').read_text())
    assert len(children)==2 and all(c['returncode']==0 for c in children)
    variants=summary['variants']
    configs={}
    histories={}
    for name, row in variants.items():
        config=json.loads((root/name/'config.json').read_text())
        configs[name]=config
        assert config['epochs']==100 and config['seed']==66
        assert config['paired_history_views'] and config['history_drop_prob']==.2
        assert config['history_contrast_temperature']==.1
        assert config['history_contrast_weight']==(0. if name=='control' else .1)
        assert json.loads((root/name/'PROVENANCE.json').read_text())['status']=='complete'
        history=json.loads((root/name/'history.json').read_text())
        histories[name]=history
        assert [h['epoch'] for h in history]==list(range(1,101))
        metrics=json.loads((root/name/'metrics.json').read_text())
        assert metrics['selection_protocol']=='per-rate-test-oracle'
        scores=row['scores']
        for rate, value in scores['per_rate'].items():
            assert abs(metrics['test'][rate]['weighted_f1']*100-value)<1e-10
        assert abs(statistics.mean(scores['per_rate'].values())-scores['mean_8rate'])<1e-10
        stats=row['training_diagnostics']
        for key in ('anchor_count','eligible_anchor_count','dropped_observed_count',
                    'view1_observed_count','valid_utterance_count','batches_no_negatives'):
            assert sum(h['train']['paired_history'][key] for h in history)==row['counts'][key]
        assert len(stats)==100
    assert {key for key in configs['control'] if configs['control'][key]!=configs['contrastive'][key]}=={'history_contrast_weight'}
    for left,right in zip(histories['control'],histories['contrastive']):
        for key in ('view1_mask_sha256','view2_mask_sha256'):
            assert left['train']['paired_history'][key]==right['train']['paired_history'][key]
    flat=summary['flat']; control=variants['control']['scores']; contrast=variants['contrastive']['scores']
    lines=['# Paired-history results — seed66','',
        'INTERNAL TEST-ORACLE DIAGNOSTIC, not formal validation-selected paper results.',
        'Code32c6897; launch526d8bf. Both100epochs completed on biggpuGPU6.',
        'Original Flat reused; eight per-rate best checkpoints per arm.', '',
        'W-F1(%), deltas in percentage points. Single seed only; no significance claim.','',
        '|Missing rate|Flat|Paired task-only A|Paired InfoNCE B|A−Flat|B−Flat|B−A|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for rate in flat['per_rate']:
        f,a,b=(s['per_rate'][rate] for s in (flat,control,contrast))
        lines.append(f'|{rate}|{f:.3f}|{a:.3f}|{b:.3f}|{a-f:+.3f}|{b-f:+.3f}|{b-a:+.3f}|')
    for key,label in [('mean_8rate','8-rate mean'),('high_missing','High(.5/.6/.7)')]:
        f,a,b=(s[key] for s in (flat,control,contrast))
        lines.append(f'|{label}|{f:.3f}|{a:.3f}|{b:.3f}|{a-f:+.3f}|{b-f:+.3f}|{b-a:+.3f}|')
    lines+=['','## Training exposures','',
        '|Arm|Anchors|Usable anchors|Removed observed bits|View1 observed bits|Actual drop|Batches without negatives|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in variants.items():
        c=row['counts']
        lines.append(f"|{name}|{c['anchor_count']}|{c['eligible_anchor_count']}|{c['dropped_observed_count']}|{c['view1_observed_count']}|{100*c['actual_drop_ratio']:.4f}%|{c['batches_no_negatives']}|")
    lines+=['','Counts are summed training exposures over100epochs, not unique utterances.',
        'Both arms match View1 AND View2 mask hashes at every epoch. Evaluation masks',
        'verified against original Flat by the run verifier. Task loss is original MOSI MSE;',
        'W-F1 is the primary outcome. Fixed drop.2/lambda.1/temp.1; no hyperparameter search.',
        'No original model/backbone changes; projector is training-only. Same-conversation',
        'nonpositive anchors excluded. More compute than single-view Flat; A is the key',
        'control for attributing changes to contrastive training rather than two-view augmentation.',
        'No fixed-pattern inference or downstream module search was added.','',
        '## Interpretation','',
        f"Task-only versus Flat:8-rate{control['mean_8rate']-flat['mean_8rate']:+.3f}pp; high{control['high_missing']-flat['high_missing']:+.3f}pp.",
        f"InfoNCE versus task-only:8-rate{contrast['mean_8rate']-control['mean_8rate']:+.3f}pp; high{contrast['high_missing']-control['high_missing']:+.3f}pp.",
        'This fixed seed/configuration provides no overall improvement. It does not establish',
        'that all contrastive/history-invariance methods fail, nor identify the causal reason.',
        'No additional runs or hyperparameter changes were made after observing these scores.','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=analyze(args.root)
    if args.output:
        args.output.write_text(report)
    print(report)
