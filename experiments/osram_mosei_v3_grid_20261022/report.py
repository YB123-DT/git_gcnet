"""Complete-only grid, clearly labeling old reused results and new arms."""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

from .worker import MODES, RATES, candidate_config
from experiments.osram_mosei_v3_capacity_20261022.worker import REFERENCE

OLD_ROOT = Path('/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1/runs')
REUSED = {'f1024_plain': OLD_ROOT / 'small_flat', 'f1024_v256': OLD_ROOT / 'wide_v3'}
CORE = tuple('gcnet_missing_m3/' + f for f in ('osram.py', 'model.py', 'train_gcnet.py',
             'source_role_v3.py', 'tasr.py', 'mixed_rate.py')) + (
    'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
    'experiments/osram_text_path_router_joint_mosi_20261019/run.py')


def summarize(output, *, include_reused=True):
    output = Path(output)
    paths = {mode: output / 'runs' / mode for mode in MODES}
    if include_reused:
        paths.update(REUSED); paths['original_flat'] = REFERENCE
    states, rows, aggregate, completed = {}, [], [], {}
    reference = json.loads((REFERENCE / 'SUMMARY.json').read_text()) if include_reused else None
    common_masks = common_core = None
    for name, folder in paths.items():
        p = folder / 'SUMMARY.json'
        s = json.loads(p.read_text()) if p.exists() else {'status': 'not_started'}
        reused = name in REUSED or name == 'original_flat'
        states[name] = dict(path=str(folder), reused=reused,
            **{k: s.get(k) for k in ('status', 'epoch', 'error', 'parameters', 'seconds')})
        if s.get('status') != 'complete':
            continue
        if s.get('epoch') != 60 or set(s.get('evaluation', {})) != set(RATES):
            raise ValueError('Incomplete formal MOSEI result: ' + name)
        if name != 'original_flat':
            cfg = s['identity']['config']
            if reference is not None:
                from gcnet_missing_m3.train_gcnet import TrainConfig
                original_cfg = TrainConfig(**reference['identity']['config'])
                if name in MODES:
                    expected = candidate_config(original_cfg, name)
                else:
                    width = 256 if name == 'f1024_v256' else 0
                    expected = replace(original_cfg, osram_adapter_hidden_dim=1024,
                        osram_source_role_v3=bool(width), osram_source_role_v3_dim=width or 64,
                        osram_source_role_v3_pair_hidden=512 if width else 128)
                if asdict(TrainConfig(**cfg)) != asdict(expected):
                    raise ValueError('Unexpected MOSEI grid config: ' + name)
                for key in ('split', 'seed_protocol', 'data_files'):
                    if s['identity'][key] != reference['identity'][key]:
                        raise ValueError('Original reference differs: ' + key)
            masks = s['identity']['mask_hashes']['test']
            if common_masks is not None and masks != common_masks:
                raise ValueError('Same exact evaluation masks required')
            if reference is not None:
                ref_masks = reference['identity']['mask_hashes']
                if masks != ref_masks.get('test', ref_masks):
                    raise ValueError('Original Flat masks differ')
            core = {k: s['identity']['source_hashes'][k] for k in CORE}
            if common_core is not None and core != common_core:
                raise ValueError('Grid/reused core computation differs')
            common_masks, common_core = masks, core
        completed[name] = s
        for rate in RATES:
            rows.append(dict(model=name, reused=reused, rate=rate,
                epoch=s['best_by_rate'][rate]['epoch'], **s['evaluation'][rate]['router']))
        for scope, rates in (('mean', RATES), ('high', RATES[5:])):
            aggregate.append(dict(model=name, reused=reused, scope=scope,
                **{k: sum(s['evaluation'][r]['router'][k] for r in rates) / len(rates)
                   for k in ('ACC', 'WF1', 'MSE')}))
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states,
                  per_rate=rows, aggregate=aggregate)
    (output / 'REPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    if rows:
        with (output / 'PER_RATE.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# MOSEI completed capacity grid', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Four fresh seed66 runs;two prior1024-width arms reused without retraining.',
        '60epochs,constant1e-3,newGCNetmissing,unchanged task/model computation.',
        'Model/sampler/task core and exact test masks verified across new/reused arms.', '',
        '| Model | Reused | Scope | ACC | W-F1 | MSE |', '|---|---|---|---:|---:|---:|']
    for a in aggregate:
        lines.append(f"| {a['model']} | {a['reused']} | {a['scope']} | {a['ACC']:.3f} | {a['WF1']:.3f} | {a['MSE']:.6f} |")
    lines += ['', '## State', ''] + [f'- {name}: {state}' for name, state in states.items()]
    lines += ['', 'Only completed results shown;missing runs are not zeros. Per-rate Test-best',
              'can select different epochs,not a single-checkpoint independent test score.',
              'One-seed screening only;no automatic HPO or multi-seed expansion.']
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    return result
