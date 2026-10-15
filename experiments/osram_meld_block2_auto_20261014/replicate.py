"""Replicate three frozen top-ACC configurations; reuse existing seed66 runs."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .methods import METHODS, validate_training_config
from .scheduler import Scheduler, write_json

ROOT = Path(__file__).resolve().parents[2]


class ReplicationScheduler(Scheduler):
    def command(self, job, config_path, gpu_uuid):
        return super().command(job, config_path, gpu_uuid) + [
            '--replication-seed', str(job['config']['seed'])]


def make_jobs(manifests, output):
    if len(manifests) > 3 or not manifests:
        raise ValueError('Replicate one through three frozen configurations only')
    jobs = []
    seen = set()
    for item in manifests:
        identifier, source = item['id'], item['summary']
        if (identifier in seen or not identifier or
                any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identifier)):
            raise ValueError('Configuration IDs must be unique safe names')
        seen.add(identifier)
        if source.get('status') != 'complete':
            raise ValueError('Only completed seed66 sources can be replicated')
        method, config = source['method'], source['config']
        if method not in METHODS or method == 'Flat':
            raise ValueError('Require a completed Block2 candidate')
        validate_training_config(type('Config', (), config)())
        if (config['dataset'] != 'MELD' or config['missing_mask_protocol'] != 'gcnet-random-mask-v1'
                or config['checkpoint_selection'] != 'test-oracle-per-rate'
                or config['checkpoint_selection_metric'] != 'accuracy'):
            raise ValueError('Replication cannot change the accepted MELD protocol')
        for seed in (67, 68):
            effective = deepcopy(config)
            effective['seed'] = seed
            job_id = f'{identifier}_seed{seed}'
            jobs.append(dict(id=job_id, method=method, config=effective,
                             output=str(Path(output) / 'runs' / job_id)))
    return jobs


def summarize(manifests, records, output):
    records = {row['id']: row for row in records}
    report = {'warning': 'INTERNAL TEST-ORACLE THREE-SEED REPLICATION; seed66 selected configurations',
              'configurations': {}, 'status': 'complete'}
    table = []
    for item in manifests:
        identifier = item['id']
        seeds = {'66': item['summary']}
        for seed in (67, 68):
            record = records[f'{identifier}_seed{seed}']
            if record['status'] == 'complete':
                seeds[str(seed)] = record['summary']
            else:
                report['status'] = 'incomplete_or_failed'
        result = {'seeds': seeds, 'source_seed66_summary': item['source_summary']}
        for seed, summary in seeds.items():
            for rate, metrics in summary['testbest']['rates'].items():
                table.append(dict(configuration=identifier, seed=seed, rate=rate, **metrics))
            for section in ('mean', 'high'):
                table.append(dict(configuration=identifier, seed=seed, rate=section,
                                  **summary['testbest'][section]))
        if len(seeds) == 3:
            result['testbest_3seed'] = {
                section: {metric: {'mean': statistics.mean(s['testbest'][section][metric] for s in seeds.values()),
                                  'std_ddof1': statistics.stdev(s['testbest'][section][metric] for s in seeds.values())}
                          for metric in ('ACC', 'UA', 'WF1', 'CE')}
                for section in ('mean', 'high')}
        report['configurations'][identifier] = result
    write_json(Path(output) / 'SUMMARY.json', report)
    fields = sorted({key for row in table for key in row})
    with (Path(output) / 'SUMMARY.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    lines = ['# MELD top Test-best ACC configurations: three seeds', '', report['warning'], '',
             '| Configuration | Seed | ACC | UA | W-F1 |', '|---|---:|---:|---:|---:|']
    for identifier, result in report['configurations'].items():
        for seed, summary in result['seeds'].items():
            values = summary['testbest']['mean']
            lines.append(f"| {identifier} | {seed} | {values['ACC']:.3f} | {values['UA']:.3f} | {values['WF1']:.3f} |")
        if 'testbest_3seed' in result:
            values = result['testbest_3seed']['mean']
            cells = [f"{values[m]['mean']:.3f} ± {values[m]['std_ddof1']:.3f}" for m in ('ACC', 'UA', 'WF1')]
            lines.append(f"| {identifier} | mean ± sample SD | {' | '.join(cells)} |")
    lines += ['', 'Per-rate and high-missing results and full source/configuration provenance are in SUMMARY.csv/JSON.',
              'No new seed66 training, architecture change, validation selection, or additional hyperparameter search.',
              'Seed66 was used for configuration search; this three-seed mean is not independent test evidence.',
              'Original seed66 source is 7abb855. Replication changes only an explicit seed admission guard and the launcher; numerical model/trainer/sampler source hashes must match.', '']
    (Path(output) / 'RESULT.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,2,3')
    parser.add_argument('--per-gpu', type=int, default=2)
    args = parser.parse_args()
    if args.per_gpu > 2:
        raise ValueError('At most two replications/card alongside the existing HPO queue')
    manifest = json.loads(args.manifest.read_text())
    for item in manifest['configurations']:
        for name, expected in item['summary']['provenance']['code_hashes'].items():
            if name.startswith(('gcnet_missing_m3/', 'gcnet_modality_jepa/')):
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                    raise ValueError('Numerical training source differs from seed66: ' + name)
    jobs = make_jobs(manifest['configurations'], args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'MANIFEST.json').exists():
        if json.loads((args.output / 'MANIFEST.json').read_text()) != manifest:
            raise ValueError('Frozen replication selection changed')
    write_json(args.output / 'MANIFEST.json', manifest)
    scheduler = ReplicationScheduler(args.output / 'queue', gpus=tuple(map(int, args.gpus.split(','))),
                                     per_gpu=args.per_gpu, smoke_costs=manifest['smoke_costs'])
    records = scheduler.run_jobs(jobs)
    summarize(manifest['configurations'], records, args.output)


if __name__ == '__main__':
    main()
