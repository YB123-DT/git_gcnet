"""Read saved predictions only: localize 57 early A-only positive harms.

No model import, feature loading, forward, training, mask sampling or GPU use.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path('/data2/yb/remote_experiments')
NESTED = ROOT / 'osram_mosei_flat128_nested_20261024/execution_v1/runs/n10'
FLAT = ROOT / 'osram_mosei_capacity_cmerc_20261024/execution_v1/runs/f128'
REPLAY = ROOT / 'osram_mosei_n10_scale_20261024'
IDENTITY = ('labels', 'availability', 'conversation_ids', 'utterance_indices')
CASES = ('all_off', 'all_half', 'Local_off', 'Base_off',
         'Gap-A_off', 'Gap-T_off', 'Gap-V_off')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prediction(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Prediction hash changed: ' + str(path))
    with np.load(path, allow_pickle=False) as saved:
        result = {k: saved[k].copy() for k in (*IDENTITY, 'prediction')}
        if 'speakers' in saved.files:
            result['speakers'] = saved['speakers'].copy()
    n = len(result['labels'])
    if result['prediction'].shape == (n, 1):
        result['prediction'] = result['prediction'][:, 0]
    if result['prediction'].shape != (n,) or not np.isfinite(result['prediction']).all():
        raise ValueError('Invalid scalar prediction')
    if not np.isfinite(result['labels']).all():
        raise ValueError('Invalid labels')
    availability = result['availability']
    if availability.shape != (n, 3) or not np.isin(availability, [0, 1]).all():
        raise ValueError('Invalid availability')
    if not (availability.sum(-1) >= 1).all():
        raise ValueError('Empty observed set')
    keys = list(zip(result['conversation_ids'].tolist(), result['utterance_indices'].tolist()))
    if len(set(keys)) != n:
        raise ValueError('Duplicate utterance within a rate')
    return result


def align(reference, other):
    for key in IDENTITY:
        np.testing.assert_array_equal(reference[key], other[key], err_msg=key)
    if 'speakers' in reference and 'speakers' in other:
        np.testing.assert_array_equal(reference['speakers'], other['speakers'])


def counts(labels, reference, candidate, mask):
    mask = mask & (labels != 0)
    good_ref = (reference > 0) == (labels > 0)
    good_new = (candidate > 0) == (labels > 0)
    return dict(N=int(mask.sum()),
                corrections=int((mask & ~good_ref & good_new).sum()),
                harms=int((mask & good_ref & ~good_new).sum()),
                both_correct=int((mask & good_ref & good_new).sum()),
                both_wrong=int((mask & ~good_ref & ~good_new).sum()))


def recovery(full, candidate, mask):
    n = int(mask.sum())
    return dict(N=n, restored_positive=int((mask & (candidate > 0)).sum()),
                remains_nonpositive=int((mask & (candidate <= 0)).sum()),
                prediction_increased=int((mask & (candidate > full)).sum()),
                prediction_decreased=int((mask & (candidate < full)).sum()),
                mean_prediction_shift=float(np.mean(candidate[mask] - full[mask])) if n else None)


def summarize(arrays):
    y, av, index = (arrays[k] for k in ('labels', 'availability', 'utterance_indices'))
    flat, full = arrays['flat'], arrays['full']
    early = (index >= 1) & (index <= 2)  # saved indices are zero-based
    a_only = (av == [1, 0, 0]).all(-1)
    cohort = early & a_only & (y > 0)
    harm57 = cohort & (flat > 0) & (full <= 0)
    groups = {'all_non_neutral': np.ones(len(y), dtype=bool),
              'early_A_only_non_neutral': early & a_only,
              'early_A_only_positive': cohort}
    result = dict(cohort=counts(y, flat, full, cohort),
                  target_records=int(harm57.sum()),
                  target_unique_utterances=len(set(zip(
                      arrays['conversation_ids'][harm57].tolist(), index[harm57].tolist()))),
                  cases={})
    for case in CASES:
        p = arrays[case]
        result['cases'][case] = dict(
            target=recovery(full, p, harm57),
            versus_full={k: counts(y, full, p, v) for k, v in groups.items()},
            versus_flat_early_positive=counts(y, flat, p, cohort))
    affected = np.stack([arrays[c] > 0 for c in CASES if c.endswith('_off')])
    result['target_restored_by_any_off_case'] = int((harm57 & affected.any(0)).sum())
    # Overlap is reported, not added: interventions pass through nonlinear Flat/LN.
    result['target_all_off_restored_and_slot_restored'] = {
        c: int((harm57 & (arrays['all_off'] > 0) & (arrays[c] > 0)).sum())
        for c in CASES if c not in ('all_off', 'all_half')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite: ' + str(args.output))
    paths = {'nested': NESTED / 'SUMMARY.json', 'flat': FLAT / 'SUMMARY.json',
             'global': REPLAY / 'replay_v2/SUMMARY.json',
             'slots': REPLAY / 'slot_replay_v1/SUMMARY.json'}
    summaries = {k: json.loads(p.read_text()) for k, p in paths.items()}
    for item in summaries.values():
        if item['status'] != 'complete':
            raise ValueError('Incomplete source')
    for name in ('global', 'slots'):
        if summaries[name]['source_summary_sha256'] != sha(paths['nested']):
            raise ValueError('Replay uses a different Nested run')
        if summaries[name]['flat_summary_sha256'] != sha(paths['flat']):
            raise ValueError('Replay uses a different Flat run')
    pieces = []
    per_rate, provenance = {}, []
    for rate in [f'{i / 10:.1f}' for i in range(8)]:
        selected = {k: summaries[k]['best_by_rate'][rate] for k in ('nested', 'flat')}
        loaded = {}
        for name, directory in (('nested', NESTED), ('flat', FLAT)):
            p = directory / selected[name]['predictions']
            loaded[name] = load_prediction(p, selected[name]['prediction_sha256'])
            provenance.append(dict(path=str(p), sha256=sha(p)))
        align(loaded['nested'], loaded['flat'])
        reference = loaded['nested']
        data = {k: reference[k] for k in IDENTITY}
        data.update(flat=loaded['flat']['prediction'], full=reference['prediction'])
        for name in ('global', 'slots'):
            record = summaries[name]['rates'][rate]
            if record['checkpoint_sha256'] != selected['nested']['checkpoint_sha256']:
                raise ValueError('Replay checkpoint mismatch')
            for scale, item in record['scales'].items():
                if name == 'global' and scale not in ('0.0', '0.5', '1.0'):
                    continue
                if name == 'slots' and scale != '1.0' and not scale.endswith('_0.0'):
                    continue
                filename = f'gamma_{scale.replace(".", "p")}_miss_{rate.replace(".", "p")}.npz'
                directory = REPLAY / ('replay_v2' if name == 'global' else 'slot_replay_v1')
                p = directory / filename
                other = load_prediction(p, item['prediction_sha256'])
                align(reference, other)
                provenance.append(dict(path=str(p), sha256=sha(p)))
                if scale == '1.0':
                    np.testing.assert_allclose(other['prediction'], data['full'], atol=1e-6, rtol=0)
                    np.testing.assert_array_equal(other['prediction'] > 0, data['full'] > 0)
                else:
                    case = {'0.0': 'all_off', '0.5': 'all_half'}.get(scale, scale.replace('_0.0', '_off'))
                    data[case] = other['prediction']
        if any(c not in data for c in CASES):
            raise ValueError('Missing intervention')
        # Gap-A is inactive in the cohort, so its intervention must preserve polarity.
        a_only = (data['availability'] == [1, 0, 0]).all(-1)
        np.testing.assert_array_equal(data['Gap-A_off'][a_only] > 0, data['full'][a_only] > 0)
        per_rate[rate] = dict(flat_epoch=selected['flat']['epoch'],
                              nested_epoch=selected['nested']['epoch'], **summarize(data))
        pieces.append(data)
    pooled = {k: np.concatenate([d[k] for d in pieces]) for k in pieces[0]}
    result = summarize(pooled)
    if (result['cohort']['N'], result['cohort']['corrections'], result['cohort']['harms']) != (581, 5, 57):
        raise ValueError('Does not reproduce the original 57-harm cohort')
    for rate in per_rate.values():
        c = rate['cohort']
        if sum(c[k] for k in ('corrections', 'harms', 'both_correct', 'both_wrong')) != c['N']:
            raise ValueError('Counts do not partition cohort')
    report = dict(label='INTERNAL DIAGNOSTIC ONLY', status='complete',
                  source_commit=args.source_commit,
                  analysis_sha256=sha(Path(__file__)), training_updates=0,
                  model_forwards=0, gpu_used=False,
                  protocol='Saved exact GCNet masks; original per-rate Test-best checkpoints',
                  definitions=dict(cohort='current A-only, y>0, zero-based indices1/2 (turns2/3)',
                      target='Flat>0 and full Nested<=0 inside the fixed cohort',
                      off='Bypass only extra Nested task-input corrections; original CHM retained',
                      inference='same Nested weights across off cases; Flat is separately trained',
                      counting='rate-utterance records, not independent observations'),
                  pooled=result, per_rate=per_rate,
                  manifests={k: dict(path=str(v), sha256=sha(v)) for k, v in paths.items()},
                  prediction_files=provenance)
    args.output.mkdir(parents=True)
    (args.output / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
