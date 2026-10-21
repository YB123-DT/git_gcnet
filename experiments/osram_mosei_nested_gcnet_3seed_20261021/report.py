"""Completed-only, same-protocol Flat/Nested tables; no model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

RATES = tuple(f'{i/10:.1f}' for i in range(8))
SEEDS = (66, 67, 68)
FLAT66_REFERENCE = Path('/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat')


def load_complete(path):
    if not path.exists():
        return {'status': 'not_started'}
    state = json.loads(path.read_text())
    if state.get('status') != 'complete':
        return state
    if set(state.get('evaluation', {})) != set(RATES) or set(state.get('best_by_rate', {})) != set(RATES):
        raise ValueError('Completed result missing required rate: '+str(path))
    for rate in RATES:
        values = state['evaluation'][rate]['router']
        if not all(math.isfinite(float(values[k])) for k in ('ACC', 'WF1')):
            raise ValueError('Nonfinite completed metric: '+str(path))
    return state


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pair(flat, nested):
    """Worker validates full config/data/masks; repeat available identity checks."""
    if nested.get('reference_pair_verified') is not True:
        raise ValueError('Worker reference verification is missing or failed')
    f, n = flat.get('identity', {}), nested.get('identity', {})
    for key in ('dataset_manifest_sha256', 'split', 'seed_protocol', 'seed'):
        if key not in f or key not in n or f[key] != n[key]:
            raise ValueError('Paired identity differs: '+key)
    for data in (f, n):
        if 'mask_hashes' not in data:
            raise ValueError('Missing paired mask hashes')
    fm, nm = f['mask_hashes'], n['mask_hashes']
    if fm.get('test', fm) != nm.get('test', nm):
        raise ValueError('Paired identity differs: mask_hashes')
    if (not flat.get('initial_backbone_sha256')
            or flat['initial_backbone_sha256'] != nested.get('initial_backbone_sha256')):
        raise ValueError('Paired original initialization differs')
    configs = []
    defaults = dict(osram_source_role=False, osram_source_role_mode='residual',
                    osram_source_relation=False, osram_source_relation_layout='source-mean')
    for item in (flat, nested):
        config = dict(item.get('config', item.get('identity', {}).get('config', {})))
        if not config:
            raise ValueError('Missing paired config')
        for key, value in defaults.items():
            config.setdefault(key, value)
        configs.append(config)
    if configs[0].get('osram_meaningful_block') != 'none' or configs[1].get('osram_meaningful_block') != 'nested_gnn_rooted_evidence':
        raise ValueError('Wrong Flat/Nested method switch')
    configs[1]['osram_meaningful_block'] = 'none'
    if configs[0] != configs[1]:
        raise ValueError('Paired config differs beyond Nested switch')


def summarize(root, flat66_reference=FLAT66_REFERENCE):
    root, flat66_reference = Path(root), Path(flat66_reference)
    states, per_rate, aggregate, paired, complete = {}, [], [], [], {}
    provenance = {}
    for seed in SEEDS:
        for mode in ('flat', 'nested'):
            directory = flat66_reference if (seed, mode) == (66, 'flat') else root/'runs/CMUMOSEI'/f'seed_{seed}'/mode
            path = directory/'SUMMARY.json'
            data = load_complete(path)
            name = f'{mode}_seed{seed}'
            states[name] = {k: data.get(k) for k in ('status', 'epoch', 'error', 'source_commit', 'seconds', 'peak_memory_mib')}
            provenance[name] = dict(path=str(path), sha256=sha256(path) if path.exists() else None,
                                    reused_reference=(seed, mode) == (66, 'flat'))
            if data.get('status') != 'complete':
                continue
            if (seed, mode) != (66, 'flat') and data.get('reference_pair_verified') is not True:
                raise ValueError('New completed run missing successful reference verification: '+str(path))
            complete[(seed, mode)] = data
            for rate in RATES:
                metric = data['evaluation'][rate]['router']
                per_rate.append(dict(seed=seed, mode=mode, rate=rate, epoch=data['best_by_rate'][rate]['epoch'],
                    ACC=float(metric['ACC']), WF1=float(metric['WF1']), MSE=metric.get('MSE')))
            for scope, selected_rates in [('mean', RATES), ('high', RATES[5:])]:
                aggregate.append(dict(seed=seed, mode=mode, scope=scope,
                    **{metric: statistics.mean(data['evaluation'][r]['router'][metric] for r in selected_rates)
                       for metric in ('ACC', 'WF1')}))
        if (seed, 'flat') in complete and (seed, 'nested') in complete:
            verify_pair(complete[(seed, 'flat')], complete[(seed, 'nested')])
            for scope in ('mean', 'high'):
                f = next(v for v in aggregate if v['seed'] == seed and v['mode'] == 'flat' and v['scope'] == scope)
                n = next(v for v in aggregate if v['seed'] == seed and v['mode'] == 'nested' and v['scope'] == scope)
                paired.append(dict(seed=seed, scope=scope, delta_ACC_pp=n['ACC']-f['ACC'], delta_WF1_pp=n['WF1']-f['WF1']))
    three_seed = []
    # Require all six completed results before publishing any 3-seed comparison.
    if len(complete) == 6:
        for mode in ('flat', 'nested'):
            for scope in ('mean', 'high'):
                rows = [v for v in aggregate if v['mode'] == mode and v['scope'] == scope]
                three_seed.append(dict(mode=mode, scope=scope, n=3,
                    **{metric+suffix: function(v[metric] for v in rows)
                       for metric in ('ACC', 'WF1')
                       for suffix, function in [('_mean', statistics.mean), ('_sample_std', statistics.stdev)]}))
    payload = dict(label='INTERNAL TEST-ORACLE SCREEN', dataset='CMUMOSEI', states=states,
        provenance=provenance, per_rate=per_rate, aggregate=aggregate,
        paired_aggregate=paired, three_seed=three_seed,
        protocol='gcnet-random-mask-v1;60epochs;constant1e-3;batch32;WD1e-5;per-rate TEST-best WF1')
    (root/'REPORT.json').write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n')
    for filename, rows in [('PER_RATE.csv', per_rate), ('AGGREGATE.csv', aggregate), ('PAIRED_DELTA.csv', paired), ('THREE_SEED.csv', three_seed)]:
        if rows:
            with (root/filename).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# Original Nested GNN / current GCNet-missing MOSEI, three seeds', '',
        'INTERNAL TEST-ORACLE SCREEN', '',
        'Eight-rate results combine independently selected TEST-best W-F1 epochs; not one deployment checkpoint.',
        'Only completed runs enter metric tables; no inference or test-label training is performed by this report.', '',
        '| Seed | Model | Scope | ACC | W-F1 |', '|---:|---|---|---:|---:|']
    for row in aggregate:
        lines.append(f"| {row['seed']} | {row['mode']} | {row['scope']} | {row['ACC']:.3f} | {row['WF1']:.3f} |")
    lines += ['', '## Paired differences (Nested minus Flat, percentage points)', '',
              '| Seed | Scope | ΔACC | ΔW-F1 |', '|---:|---|---:|---:|']
    for row in paired:
        lines.append(f"| {row['seed']} | {row['scope']} | {row['delta_ACC_pp']:+.3f} | {row['delta_WF1_pp']:+.3f} |")
    lines += ['', '## Three-seed summary (sample standard deviation)', '',
              '| Model | Scope | ACC mean ± SD | W-F1 mean ± SD |', '|---|---|---:|---:|']
    for row in three_seed:
        lines.append(f"| {row['mode']} | {row['scope']} | {row['ACC_mean']:.3f} ± {row['ACC_sample_std']:.3f} | {row['WF1_mean']:.3f} ± {row['WF1_sample_std']:.3f} |")
    if not three_seed:
        lines += ['', 'Three-seed comparison is pending; incomplete runs are not treated as final scores.']
    lines += ['', '## Per-rate results', '', '| Seed | Model | Missing rate | BEST epoch | ACC | W-F1 |',
              '|---:|---|---:|---:|---:|---:|']
    for row in per_rate:
        lines.append(f"| {row['seed']} | {row['mode']} | {row['rate']} | {row['epoch']} | {row['ACC']:.3f} | {row['WF1']:.3f} |")
    lines += ['', '## Provenance and state', '',
        'Flat66 is reused from the accepted existing constant1e-3 GCNet-random-mask-v1 run. '
        'Flat67/68 and Nested66/67/68 are five fresh runs from the recorded immutable source. '
        'Worker checks data/splits/masks/config and original initialization; different source SHAs are disclosed, not claimed byte-identical.',
        'The historical83.843-style conversation-mask-v2 result is not a matched baseline and is excluded.', '',
        *[f"- {name}: {json.dumps(value, sort_keys=True)}; summary={provenance[name]['path']}" for name, value in states.items()], '',
        'Full checkpoint, exact selected masks/predictions, environment and source hashes remain in each remote run; weights/data are not Git artifacts.',
        'Scientific limit: seed3 exploratory Test-oracle comparison, no independent generalization or module-effectiveness claim before completion.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
