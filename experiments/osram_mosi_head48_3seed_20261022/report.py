"""Completed-only, same-protocol head48 versus original Flat seed comparison."""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import statistics

from experiments.osram_mosi_memory_grid_20261022.config import RATES, candidate_config


SEED66 = Path('/data2/yb/remote_experiments/osram_mosi_memory_grid_20261022/execution_v1/runs/head_48')
FLATS = {
    66: Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003'),
    **{seed: Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_3seed_20261021/execution_v1/runs/MOSI')
       / f'seed_{seed}' / 'flat' for seed in (67, 68)},
}
CORE = tuple('gcnet_missing_m3/' + name for name in
             ('osram.py', 'model.py', 'train_gcnet.py', 'mixed_rate.py')) + (
    'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
    'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
)
METRICS = ('ACC', 'WF1', 'MSE')
# These original ledgers omitted this file. Recover ONLY its Git-object bytes
# at the declared immutable source SHA, independently checked before launch.
LEGACY_MIXED_RATE = {
    '5877678ce3fac9358a94f1217e3dbac5509f6f58': '0d3e9af0b3c35c0fde4a148851390cf47850a50d9bdb0737a6997dc9438b22c1',
    '55a57718a40b92676d5f8dee885d4c64a873b74b': '0d3e9af0b3c35c0fde4a148851390cf47850a50d9bdb0737a6997dc9438b22c1',
}
PLAIN_CORE_TRANSITIONS = {
    'gcnet_missing_m3/osram.py': (
        'd02e9b5da370caf264c0a26496cda9e2dceb28c887f425bfe73d7f496c19a838',
        '84c09c1f586e35eed34e88a5d826df2ae18092b87407d02b51791e0d5016a746'),
    'gcnet_missing_m3/model.py': (
        'ce1afb7b98279cb1d62afc1a3ab1420d167580d02afdbd38d38f7e9e454b3ee2',
        '13cb139d83b41235fb92e4515a2a9db46f89bcc678ad9cf5a67bf7d5811316d9'),
    'gcnet_missing_m3/train_gcnet.py': (
        '12267fc232895296be802c3d8f673003190ad1c88e9a7dbe4216879a89b6751f',
        '162a72afdca933d02b7e959ccd1d32f1bba0aaddad71585fe339c57c8e647853'),
}


def three_seed_stats(values):
    if set(values) != {66, 67, 68}:
        return None
    return {scope: {metric: dict(
        mean=statistics.mean(values[seed][scope][metric] for seed in (66, 67, 68)),
        std=statistics.stdev(values[seed][scope][metric] for seed in (66, 67, 68)))
        for metric in METRICS} for scope in ('mean', 'high')}


def validate_completed(record):
    if (record.get('status') != 'complete' or record.get('epoch') != 150
            or set(record.get('evaluation', {})) != set(RATES)
            or set(record.get('best_by_rate', {})) != set(RATES)):
        raise ValueError('Require complete 150-epoch, eight-rate result')
    for rate in RATES:
        epoch = record['best_by_rate'][rate].get('epoch')
        if type(epoch) is not int or not 1 <= epoch <= 150:
            raise ValueError('Invalid selected epoch: ' + rate)
        metrics = record['evaluation'][rate].get('router', {})
        if any(not isinstance(metrics.get(key), Real) or not math.isfinite(metrics[key])
               for key in METRICS):
            raise ValueError('Missing or nonfinite metrics: ' + rate)


def test_masks(record):
    masks = record['identity']['mask_hashes']
    return masks.get('test', masks)


def core_hashes(record):
    hashes = dict(record['identity']['source_hashes'])
    commit = record['identity'].get('source_commit',record.get('source_commit'))
    if 'gcnet_missing_m3/mixed_rate.py' not in hashes and commit in LEGACY_MIXED_RATE:
        hashes['gcnet_missing_m3/mixed_rate.py'] = LEGACY_MIXED_RATE[commit]
    try:
        return {name: hashes[name] for name in CORE}
    except KeyError as error:
        raise ValueError('Missing numerical core provenance: ' + str(error)) from error


def require_paired_core(candidate,flat):
    """Only the audited inactive-module additions may bridge old/new Flat code."""
    new,old=core_hashes(candidate),core_hashes(flat)
    differences={path for path in CORE if new[path]!=old[path]}
    if not differences:
        return
    commit=flat['identity'].get('source_commit',flat.get('source_commit'))
    inactive=('osram_source_role_v3','osram_source_relation','osram_tasr',
              'osram_source_role','osram_source_tracking')
    if commit not in LEGACY_MIXED_RATE or any(
            record['identity']['config'].get(flag,False)
            for record in (candidate,flat) for flag in inactive):
        raise ValueError('Core bridge requires the audited original commit and inactive optional modules')
    if any(path not in PLAIN_CORE_TRANSITIONS or
           (old[path],new[path])!=PLAIN_CORE_TRANSITIONS[path] for path in differences):
        raise ValueError('Unaudited numerical core difference')


def validate_candidate(seed, candidate, flat):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    validate_completed(candidate)
    expected = candidate_config(replace(TrainConfig(**flat['identity']['config']), seed=66), 'head_48')
    actual = asdict(TrainConfig(**candidate['identity']['config']))
    if actual != asdict(replace(expected, seed=seed)):
        raise ValueError('Only K/V64→48 may change versus same-seed original Flat')
    if candidate.get('parameters', {}).get('total') != 11245857:
        raise ValueError('Head48 parameter count differs')
    if candidate.get('mode', 'head_48') != 'head_48' or candidate.get('seed', seed) != seed:
        raise ValueError('Unexpected candidate mode or seed')
    for key in ('split', 'seed_protocol', 'data_files'):
        if candidate['identity'][key] != flat['identity'][key]:
            raise ValueError('Same-seed reference differs: ' + key)
    if test_masks(candidate) != test_masks(flat):
        raise ValueError('Exact paired test masks differ')
    require_paired_core(candidate,flat)
    proof = candidate['identity'].get('initialization_proof', {})
    if (proof.get('same_shape_parameters_exact') is not True
            or f'seed{seed}' not in proof.get('source', '')
            or 'fresh' not in proof.get('source', '')):
        raise ValueError('Missing same-seed fresh common-shape initialization proof')
    original_digest = candidate['identity'].get('original_flat_initial_sha256')
    if original_digest is not None and original_digest != flat['initial_backbone_sha256']:
        raise ValueError('Fresh original initialization digest differs')


def summarize(output):
    output = Path(output)
    states, rows, original, candidate, paired, provenance = {}, [], {}, {}, {}, {}
    common_masks = common_core = common_candidate_core = None
    for seed in (66, 67, 68):
        flat_path = FLATS[seed] / 'SUMMARY.json'
        flat = json.loads(flat_path.read_text())
        validate_completed(flat)
        masks, core = test_masks(flat), core_hashes(flat)
        if ((common_masks is not None and masks != common_masks)
                or (common_core is not None and core != common_core)):
            raise ValueError('Matched originals differ in numerical core or common test masks')
        common_masks, common_core = masks, core
        folder = SEED66 if seed == 66 else output / 'runs' / f'seed_{seed}'
        path = folder / 'SUMMARY.json'
        record = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
        states[str(seed)] = dict(path=str(folder), reused=seed == 66,
            **{key: record.get(key) for key in ('status', 'epoch', 'error', 'parameters', 'seconds')})
        runs = [('original_flat', flat, flat_path)]
        if record.get('status') == 'complete':
            validate_candidate(seed, record, flat)
            current_core=core_hashes(record)
            if common_candidate_core is not None and current_core!=common_candidate_core:
                raise ValueError('Candidate numerical core differs across seeds')
            common_candidate_core=current_core
            runs.append(('head_48', record, path))
        for model, run, summary_path in runs:
            values = {scope: {key: statistics.mean(run['evaluation'][rate]['router'][key] for rate in rates)
                             for key in METRICS}
                      for scope, rates in (('mean', RATES), ('high', RATES[5:]))}
            (original if model == 'original_flat' else candidate)[seed] = values
            for rate in RATES:
                rows.append(dict(seed=seed, model=model, rate=rate,
                    epoch=run['best_by_rate'][rate]['epoch'],
                    **{key: run['evaluation'][rate]['router'][key] for key in METRICS}))
            identity = run['identity']
            provenance[f'{seed}/{model}'] = dict(
                summary_path=str(summary_path),
                summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                source_commit=identity.get('source_commit', run.get('source_commit')),
                source_hashes=identity['source_hashes'], config=identity['config'],
                mask_hashes=identity['mask_hashes'], split=identity['split'],
                seed_protocol=identity['seed_protocol'], data_files=identity['data_files'],
                environment=identity.get('environment', run.get('environment')),
                mixed_rate_hash_recovery=(dict(
                    declared_source_commit=identity.get('source_commit',run.get('source_commit')),
                    sha256=core_hashes(run)['gcnet_missing_m3/mixed_rate.py'],
                    source='Verified git show SOURCE_SHA:gcnet_missing_m3/mixed_rate.py bytes; original ledger omitted file')
                    if 'gcnet_missing_m3/mixed_rate.py' not in identity['source_hashes'] else None),
                initialization_proof=identity.get('initialization_proof'))
        if seed in candidate:
            paired[str(seed)] = {scope: {key: candidate[seed][scope][key] - original[seed][scope][key]
                                         for key in ('ACC', 'WF1')} for scope in ('mean', 'high')}
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states, per_rate=rows,
        original_flat=original, candidate=candidate, paired_difference_pp=paired,
        three_seed_original=three_seed_stats(original), three_seed_candidate=three_seed_stats(candidate),
        provenance=provenance)
    result['plain_core_bridge'] = dict(
        exact_audited_hash_pairs=PLAIN_CORE_TRANSITIONS,
        optional_modules_required_inactive=True,
        evidence='CORE_BRIDGE.json: original/current seed66 150 train MSE/masks/update ledgers and all eight selected prediction arrays exact',
        original_and_candidate_internal_core_hashes_strictly_checked=True)
    (output / 'REPORT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'PER_RATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seed', 'model', 'rate', 'epoch', *METRICS])
        writer.writeheader(); writer.writerows(rows)
    lines = ['# MOSI head48 three-seed supplement', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Reuse completed seed66; train only seeds67/68. 150epochs; LR3e-4 (1–100),3e-5 (101–150).',
        'GCNet random-mask-v1; cyclic0.0–0.7; per-rate Test-best W-F1, earliest tie.',
        'Eight per-rate checkpoints may differ: not single-checkpoint deployment performance.',
        'Head count8, write_step0.6, Flat1600; K/V dimensions48 instead of64.',
        '11,245,857 parameters. Forward context and Flat input width also shrink;',
        'this is not a parameter-matched storage-only ablation.',
        'Only completed150-epoch candidates enter scores. No two-seed mean is called three seeds.', '',
        '| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |',
        '|---:|---|---:|---:|---:|---:|']
    for seed in (66, 67, 68):
        for name, values in (('original_flat', original), ('head_48', candidate)):
            if seed in values:
                v = values[seed]
                lines.append(f"| {seed} | {name} | {v['mean']['ACC']:.3f} | {v['mean']['WF1']:.3f} | "
                             f"{v['high']['ACC']:.3f} | {v['high']['WF1']:.3f} |")
    lines += ['', '## Same-seed differences', '', '| Seed | ΔACC | ΔW-F1 | High ΔACC | High ΔW-F1 |',
              '|---:|---:|---:|---:|---:|']
    for seed, v in paired.items():
        lines.append(f"| {seed} | {v['mean']['ACC']:+.3f} | {v['mean']['WF1']:+.3f} | "
                     f"{v['high']['ACC']:+.3f} | {v['high']['WF1']:+.3f} |")
    for name in ('original', 'candidate'):
        stat = result['three_seed_' + name]
        if stat is not None:
            lines += ['', f'## Three-seed {name} (mean ± sample SD)', '']
            for scope in ('mean', 'high'):
                lines.append(scope + ': ' + '; '.join(
                    f"{key} {stat[scope][key]['mean']:.3f} ± {stat[scope][key]['std']:.3f}"
                    for key in ('ACC', 'WF1')))
    lines += ['', '## Per-rate Test-best', '', '| Seed | Model | Rate | Epoch | ACC | W-F1 | MSE |',
              '|---:|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['model']} | {r['rate']} | {r['epoch']} | {r['ACC']:.3f} | "
              f"{r['WF1']:.3f} | {r['MSE']:.6f} |" for r in rows]
    lines += ['', '## Run status', ''] + [f'- seed{seed}: {state}' for seed, state in states.items()]
    lines += ['', 'Head48 was selected after observing seed66 test results. Candidate-selection and',
        'Test-oracle bias remain; no independent generalization or causal mechanism claim.',
        'Worker source SHA may differ for the seed-routing extension; unchanged numerical core',
        'hashes, same-seed fresh initialization, data/split and exact masks are checked separately.',
        'Original Flat ledgers omitted mixed_rate.py: only the two audited declared Git SHAs',
        'can recover its identical byte hash; actual training mask/update/LR traces are checked by each worker.',
        'Full source/config/initialization provenance is retained in REPORT.json.']
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    return result


def write_report(output, queue):
    jobs = queue.get('jobs', [])
    if sorted(job.get('seed', -1) for job in jobs) != [67, 68]:
        raise ValueError('Final queue must contain exactly seeds67/68; reuse66, no additional jobs')
    return summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
