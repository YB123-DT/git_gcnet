"""Completed-only three-seed report; no training or additional inference."""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics

from experiments.osram_mosi_v3_capacity_20261022.worker import candidate_config, RATES

SEED66 = Path('/data2/yb/remote_experiments/osram_mosi_v3_capacity_20261022/execution_v1/runs/f512_v128')
FLATS = {66: Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003'),
         **{seed: Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_3seed_20261021/execution_v1/runs/MOSI') / f'seed_{seed}' / 'flat'
            for seed in (67, 68)}}
CORE = tuple('gcnet_missing_m3/' + name for name in
             ('osram.py', 'model.py', 'source_role_v3.py', 'tasr.py', 'train_gcnet.py', 'mixed_rate.py')) + (
    'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
    'experiments/osram_text_path_router_joint_mosi_20261019/run.py')


def three_seed_stats(runs):
    if set(runs) != {66, 67, 68}:
        return None
    return {scope: {key: dict(mean=statistics.mean(runs[seed][scope][key] for seed in (66, 67, 68)),
                              std=statistics.stdev(runs[seed][scope][key] for seed in (66, 67, 68)))
                    for key in ('ACC', 'WF1')} for scope in ('mean', 'high')}


def summarize(output):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    output = Path(output)
    states, rows, flat_values, candidates, paired = {}, [], {}, {}, {}
    common_masks = common_core = None
    for seed in (66, 67, 68):
        fp = FLATS[seed] / 'SUMMARY.json'
        flat = json.loads(fp.read_text())
        if flat.get('status') != 'complete' or flat.get('epoch') != 150:
            raise ValueError('Need complete matched original Flat: ' + str(fp))
        folder = SEED66 if seed == 66 else output / 'runs' / f'seed_{seed}'
        p = folder / 'SUMMARY.json'
        s = json.loads(p.read_text()) if p.exists() else {'status': 'not_started'}
        states[str(seed)] = dict(path=str(folder), **{k: s.get(k) for k in ('status', 'epoch', 'error', 'parameters')})
        runs = [('original_flat', flat)]
        if s.get('status') == 'complete':
            if s.get('epoch') != 150 or set(s.get('evaluation', {})) != set(RATES):
                raise ValueError('Incomplete candidate marked complete')
            actual = TrainConfig(**s['identity']['config'])
            expected = candidate_config(replace(TrainConfig(**flat['identity']['config']), seed=66), 'f512_v128')
            expected = replace(expected, seed=seed)
            if asdict(actual) != asdict(expected) or s['parameters']['total'] != 7864639:
                raise ValueError('Candidate configuration or parameter budget differs')
            for key in ('split', 'seed_protocol', 'data_files'):
                if s['identity'][key] != flat['identity'][key]:
                    raise ValueError('Same-seed reference differs: ' + key)
            masks = s['identity']['mask_hashes']['test']
            fm = flat['identity']['mask_hashes']
            if masks != fm.get('test', fm) or (common_masks is not None and masks != common_masks):
                raise ValueError('Exact common evaluation masks differ')
            if s['identity']['original_large_flat_initial_sha256'] != flat['initial_backbone_sha256']:
                raise ValueError('Original same-seed initialization differs')
            core = {name: s['identity']['source_hashes'][name] for name in CORE}
            if common_core is not None and core != common_core:
                raise ValueError('Candidate computational core differs across seeds')
            common_masks, common_core = masks, core
            runs.append(('flat512_v3_128', s))
        for model, record in runs:
            values = {scope: {key: statistics.mean(record['evaluation'][rate]['router'][key] for rate in rr)
                              for key in ('ACC', 'WF1', 'MSE')}
                      for scope, rr in (('mean', RATES), ('high', RATES[5:]))}
            (flat_values if model == 'original_flat' else candidates)[seed] = values
            for rate in RATES:
                rows.append(dict(seed=seed, model=model, rate=rate,
                    epoch=record['best_by_rate'][rate]['epoch'], **record['evaluation'][rate]['router']))
        if seed in candidates:
            paired[str(seed)] = {scope: {key: candidates[seed][scope][key] - flat_values[seed][scope][key]
                                         for key in ('ACC', 'WF1')} for scope in ('mean', 'high')}
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states, per_rate=rows,
                  original_flat=flat_values, candidate=candidates, paired_difference_pp=paired,
                  three_seed_original=three_seed_stats(flat_values), three_seed_candidate=three_seed_stats(candidates))
    (output / 'REPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'PER_RATE.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# MOSI Flat512 + V3-128 three-seed verification', '', 'INTERNAL TEST-ORACLE SCREEN', '',
             'Reuse seed66; only67/68 newly trained.150epochs,3e-4→3e-5,new GCNet missing.',
             'Only completed candidates enter metrics; no two-seed mean presented as three seeds.', '',
             '| Seed | Model | ACC | W-F1 | High ACC | High W-F1 |', '|---:|---|---:|---:|---:|---:|']
    for seed in (66, 67, 68):
        for model, values in (('original_flat', flat_values), ('flat512_v3_128', candidates)):
            if seed in values:
                v = values[seed]
                lines.append(f"| {seed} | {model} | {v['mean']['ACC']:.3f} | {v['mean']['WF1']:.3f} | {v['high']['ACC']:.3f} | {v['high']['WF1']:.3f} |")
    for name in ('original', 'candidate'):
        stat = result['three_seed_' + name]
        if stat is not None:
            lines.append(f"\nThree-seed {name} WF1: {stat['mean']['WF1']['mean']:.3f} ± {stat['mean']['WF1']['std']:.3f}; high {stat['high']['WF1']['mean']:.3f} ± {stat['high']['WF1']['std']:.3f} (sample SD).")
    lines += ['', '## States', ''] + [f'- seed{seed}: {v}' for seed, v in states.items()]
    lines += ['', 'This candidate was selected using seed66 test scores. Test-oracle and candidate',
              'selection bias remain. No independent-generalization or structural-causality claim.',
              'The small-Flat-only67/68 controls are not trained in this supplement.']
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    return result
