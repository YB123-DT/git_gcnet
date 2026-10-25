"""Finite Signature fixed-rate extension: MOSI67/68 and MOSEI66 only."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save
from experiments.osram_neumiss_fixed_rate_stage2_20261025.scheduler import capacity, digest

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
MODULE = 'experiments.osram_signature_stage2_extension_20261025.run'
OLD_ROOT = Path('/data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/execution_v1')


def jobs_for(phase):
    specifications = ([('CMUMOSI', 67, '0.7'), ('CMUMOSEI', 66, '0.7')] if phase == 'smoke'
                      else [('CMUMOSEI', 66, r) for r in RATES] +
                           [('CMUMOSI', seed, r) for seed in (67, 68) for r in RATES])
    return [dict(id=f'{dataset}_{seed}_{rate}', dataset=dataset, seed=seed, rate=rate,
                 status='pending') for dataset, seed, rate in specifications]


def launch(args, job, uuid, phase):
    output = args.root / phase / job['id']
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', '-m', MODULE, '--dataset', job['dataset'],
               '--seed', str(job['seed']), '--rate', job['rate'], '--gpu-uuid', uuid,
               '--output', str(output), '--source-commit', args.source_commit]
    if phase == 'smoke':
        command.append('--smoke')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    log_path = output / 'worker.log'
    log = log_path.open('a', buffering=1)
    try:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    except BaseException:
        log.close()
        raise
    job.update(status='running', pid=process.pid, command=command, gpu_uuid=uuid,
               output=str(output), log=str(log_path), started_at=now(),
               environment={key: environment.get(key) for key in (
                   'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS',
                   'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    return process, log, job


def finish(job, code, phase):
    job.update(returncode=code, finished_at=now())
    try:
        if code:
            raise RuntimeError(f'Worker exited {code}; see {job["log"]}')
        path = Path(job['output']) / 'SUMMARY.json'
        summary = json.loads(path.read_text())
        expected = 'smoke_passed' if phase == 'smoke' else 'complete'
        if summary.get('status') != expected:
            raise ValueError(f'Expected SUMMARY status {expected}')
        peak = float(summary['peak_memory_mib'])
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError('Missing positive measured GPU peak')
        job.update(status=expected, summary=str(path), peak_memory_mib=peak)
    except Exception as error:
        job.update(status='failed', error=str(error))


def run_phase(args, state, phase, peaks=None):
    jobs = state[phase] = jobs_for(phase)
    active = []
    uuids = [args.gpu0_uuid, args.gpu1_uuid, args.gpu3_uuid]
    try:
        while any(job['status'] in ('pending', 'running') for job in jobs):
            for process, log, job in active[:]:
                code = process.poll()
                if code is not None:
                    log.close()
                    finish(job, code, phase)
                    active.remove((process, log, job))
            if any(job['status'] == 'failed' for job in jobs):
                for job in jobs:
                    if job['status'] == 'pending':
                        job.update(status='blocked', error='Earlier queue job failed')
            else:
                snapshot = resources(args.root, uuids)
                state['latest_resources'] = snapshot
                capacity(snapshot)
                for job in jobs:
                    if job['status'] != 'pending':
                        continue
                    preferred = ([args.gpu0_uuid, args.gpu1_uuid, args.gpu3_uuid]
                                 if job['dataset'] == 'CMUMOSEI'
                                 else [args.gpu3_uuid, args.gpu0_uuid, args.gpu1_uuid])
                    candidates = [args.gpu3_uuid] if phase == 'smoke' else preferred
                    for uuid in candidates:
                        peers = [entry for _, _, entry in active if entry['gpu_uuid'] == uuid]
                        if len(peers) >= (2 if phase == 'smoke' else 4):
                            continue
                        if phase != 'smoke':
                            required = peaks[job['dataset']] + sum(peaks[p['dataset']] for p in peers) + 4096
                            if snapshot['gpus'][uuid]['free_mib'] < required:
                                continue
                        active.append(launch(args, job, uuid, phase))
                        save(args.root / 'QUEUE.json', state)
                        if phase != 'smoke':
                            snapshot['gpus'][uuid]['free_mib'] -= peaks[job['dataset']]
                        break
            state['updated_at'] = now()
            save(args.root / 'QUEUE.json', state)
            if not active and any(job['status'] == 'pending' for job in jobs):
                raise RuntimeError('No pending job fits measured capacity; protocol unchanged')
            if active:
                time.sleep(5)
    except Exception as error:
        # Admission/resource/launch failures also stop admission and drain children;
        # do not leave an unmonitored process group after a controller failure.
        state['admission_error'] = str(error)
        for job in jobs:
            if job['status'] == 'pending':
                job.update(status='blocked', error=str(error))
        while active:
            for process, log, job in active[:]:
                code = process.poll()
                if code is not None:
                    log.close()
                    finish(job, code, phase)
                    active.remove((process, log, job))
            state['updated_at'] = now()
            save(args.root / 'QUEUE.json', state)
            if active:
                time.sleep(5)
        raise
    finally:
        for _, log, _ in active:
            log.close()
    expected = 'smoke_passed' if phase == 'smoke' else 'complete'
    if any(job['status'] != expected for job in jobs):
        raise RuntimeError(f'{phase} has failed/blocked jobs; see QUEUE.json')


def validated_row(path, dataset, seed, rate, reused=False):
    summary = json.loads(path.read_text())
    identity = summary['identity']
    parent = identity['parent']
    epochs = 150 if dataset == 'CMUMOSI' else 60
    if summary.get('status') != 'complete' or summary['epoch'] != epochs:
        raise ValueError(f'{path}: expected complete epoch {epochs}')
    # Original MOSI66 worker omits dataset because its scope is MOSI-only.
    actual_dataset = identity.get('dataset', 'CMUMOSI' if reused else None)
    if actual_dataset != dataset or identity['seed'] != seed:
        raise ValueError(f'{path}: dataset/seed identity mismatch')
    if (identity['training_rate'] != rate or identity['evaluation_rate'] != rate
            or identity.get('cyclic_training') is not False or parent['rate'] != rate):
        raise ValueError(f'{path}: expected matching fixed training/evaluation/parent rate')
    expected = parent['checkpoint_sha256']
    if (summary['parent_id'] != expected or parent['parent_id'] != expected
            or digest(parent['checkpoint']) != expected):
        raise ValueError(f'{path}: loaded parent hash differs from recorded checkpoint')
    # The accepted parent ledger is independently checked instead of trusting a
    # self-consistent worker identity alone.
    checkpoint = Path(parent['checkpoint'])
    accepted = json.loads((checkpoint.parent / 'SUMMARY.json').read_text())
    selected = accepted['best_by_rate'][rate]
    if (selected['checkpoint_sha256'] != expected
            or (checkpoint.parent / selected['checkpoint']).resolve() != checkpoint.resolve()):
        raise ValueError(f'{path}: parent differs from accepted same-rate BEST')
    baseline, best = summary['baseline'], summary['best']['metrics']
    row = dict(dataset=dataset, seed=seed, rate=rate)
    for key in ('ACC', 'WF1', 'MSE'):
        row[key] = best[key]
        row['parent_'+key] = baseline[key]
        row['delta_'+key] = best[key] - baseline[key]
    return row, dict(summary=str(path), parent_id=expected, epoch=epochs, reused=reused)


def collect(args, state):
    rows, evidence = [], {}
    inputs = [(Path(job['summary']), job['dataset'], job['seed'], job['rate'], False)
              for job in state['train']]
    inputs += [(args.mosi66_root / 'train' / f'signature_{rate}' / 'SUMMARY.json',
                'CMUMOSI', 66, rate, True) for rate in RATES]
    for path, dataset, seed, rate, reused in inputs:
        row, proof = validated_row(path, dataset, seed, rate, reused)
        key = f'{dataset}_{seed}_{rate}'
        if key in evidence:
            raise ValueError('Duplicate logical result: ' + key)
        rows.append(row)
        evidence[key] = proof
    expected = {(d, s, r) for d, seeds in (('CMUMOSI', (66, 67, 68)), ('CMUMOSEI', (66,)))
                for s in seeds for r in RATES}
    if {(r['dataset'], r['seed'], r['rate']) for r in rows} != expected:
        raise ValueError('Expected exactly MOSI three seeds and MOSEI66 at eight rates')
    metric_keys = [key for key in rows[0] if key not in ('dataset', 'seed', 'rate')]
    seed_aggregates = []
    for dataset, seeds in (('CMUMOSI', (66, 67, 68)), ('CMUMOSEI', (66,))):
        for seed in seeds:
            for scope, rates in (('mean', RATES), ('high', RATES[5:])):
                selected = [r for r in rows if r['dataset'] == dataset and r['seed'] == seed and r['rate'] in rates]
                seed_aggregates.append(dict(dataset=dataset, seed=seed, rate=scope,
                    **{k: statistics.mean(r[k] for r in selected) for k in metric_keys}))
    groups = []
    for dataset in ('CMUMOSI', 'CMUMOSEI'):
        for scope in (*RATES, 'mean', 'high'):
            selected = [r for r in rows + seed_aggregates if r['dataset'] == dataset and r['rate'] == scope]
            groups.append(dict(dataset=dataset, rate=scope, n_seeds=len(selected),
                metrics={k: dict(mean=statistics.mean(r[k] for r in selected),
                    sample_sd=statistics.stdev(r[k] for r in selected) if len(selected) > 1 else None)
                    for k in metric_keys}))
    save(args.root / 'RESULT.json', dict(status='complete', source_commit=args.source_commit,
         completed_at=now(), protocol='Signature fixed-rate frozen-parent internal Test-oracle',
         per_rate=rows, seed_aggregates=seed_aggregates, across_seed=groups, evidence=evidence))
    with (args.root / 'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + seed_aggregates)
    with (args.root / 'GROUPS.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'rate', 'n_seeds', 'metric', 'mean', 'sample_sd'])
        for group in groups:
            for metric, value in group['metrics'].items():
                writer.writerow([group['dataset'], group['rate'], group['n_seeds'], metric,
                                 value['mean'], value['sample_sd']])
    lines = ['# Signature fixed-rate stage-two extension', '',
             f'Source: `{args.source_commit}`. Internal per-rate Test-oracle exploration.', '',
             'MOSI66 reuses completed runs; MOSI67/68 train 150 epochs; MOSEI66 trains 60 epochs.',
             'Each seed is averaged over rates first. Reported SD is sample SD across seeds, never rates; '
             'single-seed SD is unavailable.', '',
             '| Dataset | Rate/scope | Seeds | ACC mean ± SD | WF1 mean ± SD | MSE mean ± SD | Δ WF1 mean ± SD |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for group in groups:
        values = []
        for key in ('ACC', 'WF1', 'MSE', 'delta_WF1'):
            value = group['metrics'][key]
            sd = 'NA' if value['sample_sd'] is None else f'{value["sample_sd"]:.4f}'
            values.append(f'{value["mean"]:.4f} ± {sd}')
        lines.append(f'| {group["dataset"]} | {group["rate"]} | {group["n_seeds"]} | ' + ' | '.join(values) + ' |')
    lines += ['', '| Dataset | Seed | Rate/scope | ACC | WF1 | MSE | Δ WF1 |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in seed_aggregates + rows:
        lines.append(f'| {row["dataset"]} | {row["seed"]} | {row["rate"]} | '
                     f'{row["ACC"]:.4f} | {row["WF1"]:.4f} | {row["MSE"]:.4f} | {row["delta_WF1"]:.4f} |')
    (args.root / 'RESULT.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu0-uuid', required=True)
    parser.add_argument('--gpu1-uuid', required=True)
    parser.add_argument('--gpu3-uuid', required=True)
    parser.add_argument('--mosi66-root', type=Path, default=OLD_ROOT)
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.mosi66_root = args.mosi66_root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), started_at=now(), source_commit=args.source_commit,
                 command=[sys.executable]+sys.argv, python=sys.executable, cwd=os.getcwd(),
                 max_per_gpu=4, max_global_jobs=12, mosi66_root=str(args.mosi66_root),
                 gpu0_uuid=args.gpu0_uuid, gpu1_uuid=args.gpu1_uuid, gpu3_uuid=args.gpu3_uuid)
    save(args.root / 'QUEUE.json', state)
    try:
        previous = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            for index, uuid in ((0, args.gpu0_uuid), (1, args.gpu1_uuid), (3, args.gpu3_uuid)):
                os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                guard_gpu(uuid, index)
        finally:
            if previous is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = previous
        state['preflight'] = resources(args.root, [args.gpu0_uuid, args.gpu1_uuid, args.gpu3_uuid])
        capacity(state['preflight'])
        state['reused_mosi66_preflight'] = {}
        for rate in RATES:
            path = args.mosi66_root / 'train' / f'signature_{rate}' / 'SUMMARY.json'
            _, proof = validated_row(path, 'CMUMOSI', 66, rate, reused=True)
            state['reused_mosi66_preflight'][rate] = proof
        save(args.root / 'QUEUE.json', state)
        state['status'] = 'smoke'
        run_phase(args, state, 'smoke')
        peaks = {j['dataset']: j['peak_memory_mib'] for j in state['smoke']}
        state.update(status='train', admission_smoke_peaks_mib=peaks)
        run_phase(args, state, 'train', peaks)
        collect(args, state)
        state.update(status='complete', completed_at=now())
        save(args.root / 'QUEUE.json', state)
        return 0
    except Exception as error:
        state.update(status='failed', error=str(error), finished_at=now())
        save(args.root / 'QUEUE.json', state)
        print(f'Queue failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
