"""Finite F2-only learning-rate screen; original backbone/protocol unchanged."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from experiments.osram_tasr_20261018.protocol import TRAINING_PROTOCOL, build_cases
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json

ROOT = Path(__file__).resolve().parents[2]
RATES = (3e-6, 1e-5, 3e-5)
GPU_UUID = 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'


def build_jobs(output):
    return [dict(id=f'{dataset}_lr_{rate:.0e}'.replace('-', 'm'), method='tasr_lr',
                 output=str(Path(output)/f'{dataset}_lr_{rate:.0e}'.replace('-', 'm')),
                 config=dict(dataset=dataset, mode='real',
                     training=dict(TRAINING_PROTOCOL, learning_rate=rate),
                     anchor_sha256=case['checkpoint_sha256']))
            for dataset, case in build_cases().items() for rate in RATES]


def summarize(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, states, rows = [], [], []
    for job in build_jobs(output):
        path = Path(job['output'])/'SUMMARY.json'
        state = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
        states.append(dict(id=job['id'], status=state['status'], epoch=state.get('epoch'),
                           error=state.get('error')))
        if state['status'] != 'complete':
            continue
        if (state['identity']['training'] != job['config']['training']
                or state['identity']['mode'] != 'real' or not state['backbone_unchanged']
                or state['identity']['case']['checkpoint_sha256'] != job['config']['anchor_sha256']):
            raise ValueError('Completed artifact does not match the immutable LR plan')
        results.append(dict(id=job['id'], summary=state))
        for group in ('random', 'persistent'):
            for condition, value in state['evaluation'][group]['conditions'].items():
                rows.append(dict(id=job['id'], dataset=job['config']['dataset'],
                    learning_rate=job['config']['training']['learning_rate'], group=group,
                    condition=condition, selected_epoch=state['selected_epoch'],
                    **{f'{tag}_{metric}': value[tag][metric]
                       for tag in ('F0', 'adapter', 'delta') for metric in ('ACC', 'UA', 'WF1', 'CE')},
                    **value['paired']))
    report = dict(protocol='INTERNAL DIAGNOSTIC ONLY; validation-selected adapter; historical Test-oracle backbone',
                  planned=6, complete=len(results), states=states, results=results)
    write_json(output/'LR_RESULTS.json', report)
    if rows:
        with (output/'LR_RESULTS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ['# TASR F2 Adapter LR screen', '', report['protocol'], '',
             f'Completed {len(results)}/6. Only LR changes:3e-6/1e-5/3e-5; previous3e-4 retained, not rerun.', '',
             'One fixed backbone per dataset; seed66, IEMOCAP6 fold5 only;30epochs.',
             'Checkpoint selection remains validation random8 ACC, then CE. No test-based selection.', '',
             '| Dataset | LR | Epoch | Random8 ACC | UA | W-F1 | Persistent6 ACC | UA | W-F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in results:
        state = item['summary']; evaluation = state['evaluation']
        random = evaluation['random']['adapter']['mean']
        persistent = evaluation['persistent']['six']['adapter']
        values = [state['dataset'], f"{state['identity']['training']['learning_rate']:g}", str(state['selected_epoch'])]
        values += [f'{scores[m]:.4f}' for scores in (random, persistent) for m in ('ACC', 'UA', 'WF1')]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['', 'Per-rate/pattern, high-missing metrics, diagnostics and paired corrections/harms are in JSON/CSV.',
              'Small single-seed LR exploration does not establish effectiveness; do not expand automatically.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output); return
    from experiments.osram_tasr_20261018.runner_helpers import sha
    for dataset in ('MELD', 'IEMOCAPSix'):
        smoke = json.loads((ROOT/'experiments/osram_tasr_20261018'/f'SMOKE_{dataset}.json').read_text())
        if (smoke['status'] != 'complete' or not smoke['original_backbone_unchanged']
                or smoke['real_and_neutral_zero_logit_max_abs_error'] != 0
                or any(smoke['source_files'][name] != sha(ROOT/name)
                       for name in ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py'))):
            raise RuntimeError('Previously verified TASR structure changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 5*2**30:
        raise RuntimeError('Need5GiB free; never delete historical checkpoints')
    args.output.mkdir(exist_ok=True)
    jobs = build_jobs(args.output)
    plan = dict(source_commit=args.source_commit, jobs=jobs, gpu=1, gpu_uuid=GPU_UUID,
                max_concurrent=4, no_expansion=True, parent_source='ff8ea45')
    manifest = args.output/'EXPERIMENT.json'
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError('Immutable LR plan changed')
    write_json(manifest, plan)

    class LRQueue(Scheduler):
        def command(self, job, config_path, gpu_uuid):
            config = job['config']
            command = [self.python, '-u', '-m', 'experiments.osram_tasr_20261018.run',
                       '--phase', 'train', '--dataset', config['dataset'], '--mode', 'real',
                       '--adapter-learning-rate', str(config['training']['learning_rate']),
                       '--output', job['output'], '--gpu-uuid', gpu_uuid]
            if (Path(job['output'])/'last.pt').exists(): command.append('--resume')
            return command

    scheduler = LRQueue(args.output/'queue', gpus=(1,), per_gpu=4, code_root=ROOT,
                        smoke_costs={'tasr_lr': dict(peak_memory_mib=3072, disk_bytes=256*2**20)})
    if scheduler.gpu_snapshot()[1]['uuid'] != GPU_UUID:
        raise RuntimeError('Physical GPU1 UUID changed')
    records = scheduler.run_jobs(jobs)
    write_json(args.output/'QUEUE_FINAL.json', records)
    summarize(args.output)
    if any(row['status'] != 'complete' for row in records):
        raise RuntimeError('Finite LR screen has incomplete/failed jobs; see queue logs')


if __name__ == '__main__':
    main()
