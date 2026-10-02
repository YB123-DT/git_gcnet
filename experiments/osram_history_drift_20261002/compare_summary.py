"""Compare existing models on identical saved Text-history intervention anchors."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from sklearn.metrics import f1_score


def score(rows):
    keep = [r for r in rows if float(r['label']) != 0]
    y = [float(r['label']) > 0 for r in keep]
    p = [float(r['pred1']) > 0 for r in keep]
    q = [float(r['pred2']) > 0 for r in keep]
    before = 100 * f1_score(y, p, average='weighted')
    after = 100 * f1_score(y, q, average='weighted')
    harm = sum(a == t and b != t for a, b, t in zip(p, q, y))
    fix = sum(a != t and b == t for a, b, t in zip(p, q, y))
    return dict(n=len(keep), anchors=len(rows), before_wf1=before, after_wf1=after,
                delta_wf1=after-before, correct_to_wrong=harm, wrong_to_correct=fix,
                flip_percent=100*(harm+fix)/len(keep),
                prediction_shift=statistics.mean(abs(float(r['pred1'])-float(r['pred2'])) for r in rows))


def identity(row):
    return str(row['conversation_id']), int(row['utterance_index'])


def load(path):
    with path.open() as stream:
        return [r for r in csv.DictReader(stream) if r.get('mode', 'T') == 'T']


def main():
    root = Path(__file__).resolve().parent
    output = root / 'paired_summary'
    output.mkdir(exist_ok=True)
    all_stats = []
    aggregates = {}
    evidence = {}
    for arm in ('control', 'contrastive'):
        folder = root/'paired_results'/arm
        status = json.loads((folder/'status.json').read_text())
        assert status['status'] == 'complete' and len(status['records']) == 16
        for record in status['records']:
            assert record['model_state_before_sha256'] == record['model_state_after_sha256']
            assert record['checkpoint_sha256'] == record['checkpoint_after_sha256']
            assert record['saved_mask_anchor_replay_exact'] and record['projector_calls'] == 0
            assert record['identity_repeat_exact'] and record['prefix_allclose']
            assert record['local_absolute_max_error'] <= 1e-5
            assert record['local_relative_max_error'] <= 1e-6
            if record['split'] == 'test':
                assert abs(record['baseline_weighted_f1']-record['reference_weighted_f1']) < 1e-10
            for name, digest in record['source_artifact_sha256'].items():
                assert hashlib.sha256((root/'results'/name).read_bytes()).hexdigest() == digest
            path = folder/record['anchors_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['anchors_sha256']
        evidence[arm] = hashlib.sha256((folder/'status.json').read_bytes()).hexdigest()
    for split in ('validation', 'test'):
        for model, folder in [('Flat', root/'results'),
                              ('A', root/'paired_results/control'),
                              ('B', root/'paired_results/contrastive')]:
            stats = []
            for index in range(8):
                name = f'{split}_miss_0p{index}_anchors.csv'
                reference = load(root/'results'/name)
                rows = load(folder/name)
                ref = {identity(r): r for r in reference}
                assert len(rows) == len(ref) == len({identity(r) for r in rows})
                for row in rows:
                    other = ref[identity(row)]
                    assert float(row['label']) == float(other['label'])
                    for field in ('pred1', 'pred2'):
                        assert math.isfinite(float(row[field]))
                cell = dict(split=split, model=model, rate=index/10, **score(rows))
                stats.append(cell)
                all_stats.append(cell)
            macro = {}
            for field in score(rows):
                values = [s[field] for s in stats]
                macro[field] = sum(values) if field in ('n', 'anchors', 'correct_to_wrong', 'wrong_to_correct') else statistics.mean(values)
            aggregates[f'{split}_{model}'] = macro
    original = json.loads((root/'summary/SUMMARY.json').read_text())
    for split in ('validation', 'test'):
        ref = original['splits'][split]['by_mode']['T']['rate_macro']
        actual = aggregates[f'{split}_Flat']
        for field in ('flip_percent', 'correct_to_wrong', 'wrong_to_correct', 'prediction_shift'):
            # Old shift was subtracted in float32; saved predictions parse as float64.
            tolerance = 1e-7 if field == 'prediction_shift' else 1e-10
            assert abs(actual[field]-ref[field]) < tolerance
    (output/'SUMMARY.json').write_text(json.dumps(aggregates, indent=2)+'\n')
    (output/'source_status_hashes.json').write_text(json.dumps(evidence, indent=2)+'\n')
    with (output/'by_rate.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_stats[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(all_stats)
    report = ['# Identical Text-history deletion: Flat vs paired-view A/B', '',
              'Seed66, existing per-rate Test-oracle checkpoints; INTERNAL diagnostic only.',
              'No training. Original saved masks and same-current/different-history anchor IDs reused.',
              'Both W-F1 columns evaluate the SAME anchors, excluding label0, threshold prediction>0.',
              'Means are unweighted over eight rates; counts sum repeated rate exposures, not unique samples.',
              'Prediction shift includes neutral anchors, matching the previous diagnostic.', '']
    for split in ('validation', 'test'):
        report += [f'## {split}', '',
                   '|Model|Original history W-F1|Text-deleted history W-F1|Delta pp|Flip %|Correct→wrong / wrong→correct|Prediction shift|',
                   '|---|---:|---:|---:|---:|---:|---:|']
        for model in ('Flat', 'A', 'B'):
            s = aggregates[f'{split}_{model}']
            report.append(f"|{model}|{s['before_wf1']:.3f}|{s['after_wf1']:.3f}|{s['delta_wf1']:+.3f}|{s['flip_percent']:.3f}|{s['correct_to_wrong']} / {s['wrong_to_correct']}|{s['prediction_shift']:.5f}|")
        report.append('')
    report += ['## Separate original full-test-set reference', '',
               'Normal full-test eight-rate W-F1: Flat81.068%, A80.386%, B80.267%.',
               'These use all eligible test utterances, NOT the anchor subset in the tables above.',
               'A=paired-view task-only; B=paired-view task+InfoNCE. All single seed.',
               'Checkpoint epochs may differ; this evaluates the existing selected systems, not a controlled epoch comparison.',
               'Lower sensitivity alone is not evidence of higher sentiment accuracy or causal training benefit.', '']
    (output/'RESULT.md').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
