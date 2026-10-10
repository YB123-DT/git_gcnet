"""Summarize completed cached metrics; no training or inference."""
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent
    original = json.loads((root.parent / 'osram_nested_input_gradient_20261010/SUMMARY.json').read_text())['records']
    seeds, rates = [], []
    for seed in (66, 67, 68):
        metric = json.loads((root / f'seed_{seed}_METRICS.json').read_text())
        provenance = json.loads((root / f'seed_{seed}_COMPLETED_PROVENANCE.json').read_text())
        assert provenance['status'] == 'complete' and provenance['outputs_verified']
        assert metric['selection_protocol'] == 'per-rate-test-oracle'
        assert len(provenance['artifact_sha256']) == 20
        old, new = [], []
        for i in range(8):
            rate = i / 10
            reference = next(r for r in original if r['model'] == 'nested' and r['seed'] == seed and r['rate'] == rate)
            f = 100 * reference['metric']['weighted_f1_nonzero']
            g = 100 * metric['selected_weighted_f1_by_rate'][str(rate)]
            old.append(f)
            new.append(g)
            rates.append(dict(seed=seed, rate=rate, old_nested_wf1=f, gated_nested_wf1=g,
                              delta_pp=g-f, selected_epoch=metric['selected_epoch_by_rate'][str(rate)]))
        seeds.append(dict(seed=seed, old_mean8=statistics.mean(old), gate_mean8=statistics.mean(new),
                          delta_mean8=statistics.mean(new)-statistics.mean(old),
                          old_high=statistics.mean(old[5:]), gate_high=statistics.mean(new[5:]),
                          delta_high=statistics.mean(new[5:])-statistics.mean(old[5:]),
                          finished_utc=provenance['finished_utc'], peak_allocated_mib=provenance['peak_allocated_mib']))
    macro = {k: statistics.mean(r[k] for r in seeds) for k in
             ('old_mean8', 'gate_mean8', 'delta_mean8', 'old_high', 'gate_high', 'delta_high')}
    value = dict(label='INTERNAL DIAGNOSTIC ONLY', status='complete',
                 source_commit='1db4bda', seeds=seeds, per_rate= rates, macro=macro,
                 additional_inference_runs=0, additional_training_runs=0)
    (root / 'SUMMARY.json').write_text(json.dumps(value, indent=2, allow_nan=False))
    print(json.dumps(dict(seeds=seeds, macro=macro), indent=2))


if __name__ == '__main__':
    main()
