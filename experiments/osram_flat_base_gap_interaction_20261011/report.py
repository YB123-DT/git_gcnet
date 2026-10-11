"""Render all paired descriptive results without selecting a model or threshold."""
import json
from pathlib import Path


def render(root):
    data = json.loads((root/'results/SUMMARY.json').read_text())
    status = json.loads((root/'results/STATUS.json').read_text())
    assert status['status'] == 'completed' and len(data['records']) == 8
    assert all(r['frozen_unchanged'] for r in data['records'])
    lines = ['# Flat Base–Gap output interaction', '', 'INTERNAL DIAGNOSTIC ONLY', '',
        'MOSI seed66; eight original per-rate Test-oracle BEST checkpoints. No training.',
        f"Code snapshot: `{status['code_commit']}`; server biggpu, physical GPU {status['gpu']} (`{status['gpu_uuid']}`).",
        f"Source: `{status['source']}`.", '',
        'One causal Memory scan per batch; replay original Flat, Local Skip, emotion_norm and task head.',
        'All parameters and checkpoint files unchanged. Original prediction archives, labels and availability masks verified.',
        'Only forward512 read values captured; backward512 reconstructed as exact zeros. Inactive Gap and padding safely masked.', '',
        '`I_BG=f11-f10-f01+f00`; diagnostic additive prediction `fadd=f10+f01-f00`.',
        'Full−additive differences measure removing this output interaction, not replacing Memory or evaluating a trained additive architecture.',
        'W-F1 excludes neutral labels y=0 and uses prediction>0; interaction statistics use all valid utterances.',
        'Macro means equally weight rates; subgroup means use nonempty rates. Counts sum repeated rate/sample evaluations, not unique utterances.', '',
        '| Rate | Full W-F1 | Additive W-F1 | Full−additive pp | Rescue | Harm | Mean abs I | p95 abs I |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['per_rate']:
        if r['group'] == 'all':
            lines.append(f"| {r['rate']:.1f} | {r['full_wf1']:.3f} | {r['additive_wf1']:.3f} | {r['delta_pp']:+.3f} | {r['corrections']} | {r['harms']} | {r['interaction_abs_mean']:.5f} | {r['interaction_abs_p95']:.5f} |")
    lines += ['', '## Rate-macro and availability/adjacent-polarity groups', '',
        '| Group | Scope | Rates | Full | Additive | Difference pp | Rescue | Harm | Mean abs I |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['macro']:
        if not r['nonempty_rates']:
            continue
        lines.append(f"| {r['group']} | {r['scope']} | {r['nonempty_rates']} | {r['full_wf1']:.3f} | {r['additive_wf1']:.3f} | {r['delta_pp']:+.3f} | {r['corrections']} | {r['harms']} | {r['interaction_abs_mean']:.5f} |")
    overall = next(r for r in data['macro'] if r['group']=='all' and r['scope']=='eight_rates')
    high = next(r for r in data['macro'] if r['group']=='all' and r['scope']=='high_missing')
    lines += ['', '## Descriptive finding', '',
        f"Retaining the output interaction changes rate-macro W-F1 by {overall['delta_pp']:+.3f} pp overall and {high['delta_pp']:+.3f} pp at high missing. Across repeated rate/sample evaluations: {overall['corrections']} corrections, {overall['harms']} harms.",
        'The output is not exactly additive on all samples; its classification effect is concentrated in higher missing rates. This does not support a claim that original Flat entirely fails to use Base–Gap joint effects.',
        'Both adjacent-polarity groups contain corrections and harms; opposite polarity alone is not evidence for suppressing the interaction.',
        'Availability groups are current masks within random-missing evaluations, NOT separately trained single-modality models or whole-conversation persistent missing. Small subgroup cohorts can have unstable W-F1; per-rate N is supplied in per_rate.csv.', '']
    lines += ['', '## Boundaries and interpretation', '',
        '- Same/opposite means exactly adjacent nonneutral utterance polarity; first turns/neutral pairs excluded, no cross-conversation pairing or backward skipping.',
        '- Gold labels define analysis groups only; no gold input, learned gate or test-label routing.',
        '- Adjacent polarity is not the full Memory sentiment state or a confirmed emotion transition. Same-availability rows are supplied to control current support.',
        '- Nonzero output interaction can arise in the adapter, normalization or head; this diagnostic does not localize it to a specific operation.',
        '- Zeroing active reads can be out-of-distribution. Synthetic additive prediction is not a deployable score or proof interaction is indispensable.',
        '- One backbone seed, descriptive paired analysis only; rates are not independent seeds. No significance or cross-seed stability claim.',
        '- No threshold tuning, training, new model, loss, query, memory write or second missing view.', '',
        'Complete distributions, all groups, per-utterance values and checkpoint hashes are in `results/`.', '',
        '## Reproduction', '', '```bash',
        f"CUDA_VISIBLE_DEVICES={status['gpu_uuid']} python -m experiments.osram_flat_base_gap_interaction_20261011.run \\",
        f"  --source {status['source']} \\",
        f"  --output <new-empty-output-directory> --code-commit {status['code_commit']} --gpu {status['gpu']}",
        '```', '']
    (root/'RESULT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    render(Path(__file__).resolve().parent)
