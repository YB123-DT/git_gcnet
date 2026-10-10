"""Summarize final cached metrics without training or inference."""
import json
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).resolve().parent
    references = json.loads((root.parent / 'osram_nested_input_gradient_20261010/SUMMARY.json').read_text())['records']
    rows = []
    for seed in (66, 67, 68):
        metrics = json.loads((root / f'seed_{seed}_METRICS.json').read_text())
        provenance = json.loads((root / f'seed_{seed}_COMPLETED_PROVENANCE.json').read_text())
        assert provenance['status'] == 'complete' and provenance['outputs_verified']
        assert len(provenance['artifact_sha256']) == 20
        assert metrics['selection_protocol'] == 'per-rate-test-oracle'
        scores = [100 * metrics['selected_weighted_f1_by_rate'][str(i / 10)] for i in range(8)]
        old = [100 * next(r for r in references if r['model'] == 'nested' and r['seed'] == seed and r['rate'] == i / 10)['metric']['weighted_f1_nonzero'] for i in range(8)]
        rows.append(dict(seed=seed, per_rate=scores, selected_epochs=metrics['selected_epoch_by_rate'],
                         mean8=mean(scores), high=mean(scores[5:]), old_mean8=mean(old),
                         old_high=mean(old[5:]), delta_mean8=mean(scores)-mean(old),
                         delta_high=mean(scores[5:])-mean(old[5:])))
    macro = {key: mean(r[key] for r in rows) for key in ('mean8', 'high', 'old_mean8', 'old_high', 'delta_mean8', 'delta_high')}
    summary = dict(status='complete', label='INTERNAL DIAGNOSTIC ONLY', source_commit='bbad58a',
                   seeds=rows, macro=macro, additional_training_runs=0, additional_inference_runs=0)
    (root / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
