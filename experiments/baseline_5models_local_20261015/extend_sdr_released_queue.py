"""Add released-compatible SDR tasks without regenerating any existing config.

Apply only while the current dispatcher is stopped; training children continue.
The dispatcher must then be restarted to load the new smoke gate handling.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil

from configure import config
from runtime import ROOT, RUN_ROOT, atomic_json, digest, enforce_local, now


def apply_extension(paused_dispatcher_pid, validation_report=None):
    enforce_local()
    process = Path('/proc') / str(paused_dispatcher_pid)
    assert process.stat().st_uid == os.getuid()
    command = (process / 'cmdline').read_bytes().split(b'\0')
    # Resolve against this checkout: never migrate another checkout's queue.
    expected_dispatcher = (ROOT / 'dispatch.py').resolve()
    arguments = [item.decode() for item in command if item]
    command_cwd = (process / 'cwd').resolve()
    assert any((Path(item) if Path(item).is_absolute() else command_cwd / item).resolve() == expected_dispatcher
               for item in arguments), 'Unexpected dispatcher owner/root'
    state = (process / 'stat').read_text().split(') ', 1)[1].split()[0]
    assert state in ('T', 't'), 'Pause only the dispatcher before this additive registry migration'
    validation_path = Path(validation_report) if validation_report is not None else ROOT / 'audit/sdr_released_cpu_validation.json'
    validation = json.loads(validation_path.read_text())
    assert validation['passed'] is True
    assert all(digest(ROOT / rel) == sha for rel, sha in validation['adapter_source_hashes'].items())
    with (RUN_ROOT / 'registry_update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry_path = ROOT / 'experiment_registry.json'
        gates_path = ROOT / 'gates.json'
        registry = json.loads(registry_path.read_text())
        gates = json.loads(gates_path.read_text())
        tasks = registry['tasks']
        assert all(Path(t['run_dir']).resolve().is_relative_to(RUN_ROOT) for t in tasks), \
            'Registry outputs belong to a different BASELINE_RUN_ROOT'
        existing = {t['id']: t for t in tasks}
        assert len(existing) == len(tasks), 'Duplicate existing task identifiers'
        backup = RUN_ROOT / 'queue_transitions/sdr_released_20261016'
        backup.mkdir(parents=True, exist_ok=True)
        for path in (registry_path, gates_path):
            target = backup / path.name
            if not target.exists():
                shutil.copy2(path, target)
        before_configs = {t['id']: dict(t['config']) for t in tasks}
        added = []

        def add(task):
            path = ROOT / 'configs' / (task['id'].replace('/', '__') + '.json')
            task['config_path'] = str(path)
            if task['id'] in existing:
                old = existing[task['id']]
                assert old['config'] == task['config'] and old['run_dir'] == task['run_dir']
                return
            if path.exists():
                assert json.loads(path.read_text()) == task['config']
            atomic_json(path, task['config'])
            tasks.append(task)
            existing[task['id']] = task
            added.append(task['id'])

        for dataset, priority in (('MELD', 0), ('IEMOCAPFour', 20), ('IEMOCAPSix', 30)):
            gate = 'SDR-GNN_released_compatible_' + dataset
            gates.setdefault(gate, dict(ready=False, reason='Await two real GPU smoke results',
                                       source='released_compatible_protocol_20261016'))
            gates.setdefault(gate + '_smoke', dict(ready=True, source='sdr_released_cpu_validation.json',
                                                  verified_at=validation['verified_at']))
            for mr in (.3, .6):
                c = config('SDR-GNN', 'released_compatible', dataset, 66, mr, 0 if dataset == 'MELD' else 1)
                c.update(epochs=2, batch_size=4)
                ident = f'compatibility_smoke/{dataset}/SDR-GNN/released_compatible/seed_66/mr_{mr:.1f}'
                add(dict(id=ident, config=c, run_dir=str(RUN_ROOT / ident),
                         causality='Non-causal', phase='compatibility_smoke',
                         priority=-10 if dataset == 'MELD' else 10 if dataset == 'IEMOCAPFour' else 25,
                         gate=gate + '_smoke', status='MISSING'))
            for seed in (66, 67, 68):
                for rate in (3, 6, 0, 1, 2, 4, 5, 7):
                    for fold in ([0] if dataset == 'MELD' else range(1, 6)):
                        c = config('SDR-GNN', 'released_compatible', dataset, seed, rate / 10, fold)
                        ident = f'{dataset}/SDR-GNN/released_compatible/seed_{seed}/mr_{rate/10:.1f}/fold_{fold}'
                        add(dict(id=ident, config=c, run_dir=str(RUN_ROOT / ident),
                                 causality='Non-causal', phase='formal', priority=priority,
                                 table_role='main', gate=gate, status='MISSING'))
        for task in tasks:
            c = task['config']
            if c['model'] == 'SDR-GNN' and c['model_version'] == 'implementation_fixed':
                if 'priority_before_released_extension' not in task:
                    task['priority_before_released_extension'] = task['priority']
                    task['priority'] += 40
                task['table_role'] = 'implementation_fixed_comparison'
        assert all(existing[ident]['config'] == c for ident, c in before_configs.items()), 'Existing config mutation'
        assert len({t['run_dir'] for t in tasks if t['config']['model'] == 'SDR-GNN'}) == sum(t['config']['model'] == 'SDR-GNN' for t in tasks)
        formal = [t for t in tasks if t['phase'] == 'formal']
        registry['formal_fold_units'] = len(formal)
        registry['formal_seed_configs'] = len({tuple(t['config'][k] for k in ('dataset', 'model', 'model_version', 'seed', 'missing_rate')) for t in formal})
        registry['released_compatible_extension'] = dict(at=now(), added_tasks=len(added),
                                                        preserved_previous_configurations=len(before_configs),
                                                        backup=str(backup), main_sdr_version='released_compatible')
        atomic_json(gates_path, gates)
        atomic_json(registry_path, registry)
        report = dict(at=now(), added_ids=added, added_tasks=len(added),
                      formal_seed_configs=registry['formal_seed_configs'], formal_fold_units=len(formal),
                      phases=dict(Counter(t['phase'] for t in tasks)),
                      previous_configurations_unchanged=True, backup=str(backup))
        atomic_json(ROOT / 'audit/sdr_released_queue_extension_20261016.json', report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paused-dispatcher-pid', type=int, required=True)
    parser.add_argument('--validation-report', type=Path,
                        help='Local passed CPU validation with adapter_source_hashes; never copied from another checkout')
    args = parser.parse_args()
    print(json.dumps(apply_extension(args.paused_dispatcher_pid, args.validation_report), indent=2))
