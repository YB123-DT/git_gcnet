"""Compare cached 100 versus 150 epoch BEST metrics without inference."""
import json
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).resolve().parent
    rows = []
    for seed in (66, 67, 68):
        old = json.loads((root / f'seed_{seed}_ORIGINAL_100_METRICS.json').read_text())
        new = json.loads((root / f'seed_{seed}_METRICS.json').read_text())
        prov = json.loads((root / f'seed_{seed}_COMPLETED_PROVENANCE.json').read_text())
        assert prov['status'] == 'complete' and prov['outputs_verified']
        assert len(prov['artifact_sha256']) == 20
        assert old['mask_sha256'] == new['mask_sha256']
        assert old['selection_protocol'] == new['selection_protocol'] == 'per-rate-test-oracle'
        old_scores = [100 * old['selected_weighted_f1_by_rate'][str(i / 10)] for i in range(8)]
        new_scores = [100 * new['selected_weighted_f1_by_rate'][str(i / 10)] for i in range(8)]
        assert all(n >= o for n, o in zip(new_scores, old_scores))
        rows.append(dict(seed=seed, old_mean8=mean(old_scores), new_mean8=mean(new_scores),
                         delta_mean8=mean(new_scores)-mean(old_scores),
                         old_high=mean(old_scores[5:]), new_high=mean(new_scores[5:]),
                         delta_high=mean(new_scores[5:])-mean(old_scores[5:]),
                         old_per_rate=old_scores, new_per_rate=new_scores,
                         old_epochs=old['selected_epoch_by_rate'], new_epochs=new['selected_epoch_by_rate']))
    macro = {key: mean(r[key] for r in rows) for key in
             ('old_mean8', 'new_mean8', 'delta_mean8', 'old_high', 'new_high', 'delta_high')}
    summary = dict(label='INTERNAL DIAGNOSTIC ONLY', status='complete', seeds=rows, macro=macro,
                   additional_summary_training=0, additional_summary_inference=0)
    (root / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
