"""Eight finite MOSI68 cyclic-train/current-rate-test Signature runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save
from experiments.osram_neumiss_fixed_rate_stage2_20261025.scheduler import capacity, digest
from experiments.osram_signature_stage2_extension_20261025.scheduler import validated_row, finish

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
MODULE = 'experiments.osram_signature_cyclic68_20261025.run'
FIXED_ROOT = Path('/data2/yb/remote_experiments/osram_signature_stage2_extension_20261025/execution_v2')


def fixed_path(args, rate):
    return args.fixed_root / 'train' / f'CMUMOSI_68_{rate}' / 'SUMMARY.json'


def launch(args, job, uuid, phase):
    output = args.root / phase / job['rate']
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', '-m', MODULE, '--rate', job['rate'],
               '--gpu-uuid', uuid, '--output', str(output), '--source-commit', args.source_commit]
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
               environment={k: environment.get(k) for k in ('CUDA_VISIBLE_DEVICES',
                   'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    return process, log, job


def run_phase(args, state, phase, peak=None):
    jobs = state[phase] = [dict(rate=r, status='pending') for r in (('0.7',) if phase == 'smoke' else RATES)]
    active = []
    try:
        while any(j['status'] in ('pending', 'running') for j in jobs):
            for process, log, job in active[:]:
                code = process.poll()
                if code is not None:
                    log.close()
                    finish(job, code, phase)
                    active.remove((process, log, job))
            if any(j['status'] == 'failed' for j in jobs):
                for job in jobs:
                    if job['status'] == 'pending':
                        job.update(status='blocked', error='Earlier worker failed')
            else:
                snapshot = resources(args.root, [args.gpu2_uuid, args.gpu3_uuid])
                capacity(snapshot)
                state['latest_resources'] = snapshot
                for job in jobs:
                    if job['status'] != 'pending':
                        continue
                    for uuid in ([args.gpu3_uuid] if phase == 'smoke' else [args.gpu2_uuid, args.gpu3_uuid]):
                        peers = [j for _, _, j in active if j['gpu_uuid'] == uuid]
                        if len(peers) >= (1 if phase == 'smoke' else 4):
                            continue
                        if phase != 'smoke' and snapshot['gpus'][uuid]['free_mib'] < peak*(len(peers)+1)+4096:
                            continue
                        active.append(launch(args, job, uuid, phase))
                        save(args.root / 'QUEUE.json', state)
                        if phase != 'smoke':
                            snapshot['gpus'][uuid]['free_mib'] -= peak
                        break
            save(args.root / 'QUEUE.json', state)
            if not active and any(j['status'] == 'pending' for j in jobs):
                raise RuntimeError('No pending run fits measured capacity; protocol unchanged')
            if active:
                time.sleep(5)
    except Exception as error:
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
            save(args.root / 'QUEUE.json', state)
            if active:
                time.sleep(5)
        raise
    finally:
        for _, log, _ in active:
            log.close()
    expected = 'smoke_passed' if phase == 'smoke' else 'complete'
    if any(j['status'] != expected for j in jobs):
        raise RuntimeError(f'{phase} contains failed/blocked jobs')


def collect(args, state):
    from collections import Counter

    rows, proofs = [], {}
    reference_trace = None
    for job in state['train']:
        rate = job['rate']
        summary = json.loads(Path(job['summary']).read_text())
        fixed = json.loads(fixed_path(args, rate).read_text())
        validated_row(fixed_path(args, rate), 'CMUMOSI', 68, rate)
        identity, old_identity = summary['identity'], fixed['identity']
        parent = identity['parent']
        if (summary.get('status') != 'complete' or summary['epoch'] != 150
                or identity['dataset'] != 'CMUMOSI' or identity['seed'] != 68
                or identity['cyclic_training'] is not True
                or set(map(str, identity['training_rates'])) != set(RATES)
                or identity['evaluation_rate'] != rate or parent['rate'] != rate):
            raise ValueError(f'{rate}: expected completed MOSI68 cyclic-train/current-rate-test')
        expected_hash = old_identity['parent']['checkpoint_sha256']
        if (summary['parent_id'] != expected_hash or parent['checkpoint_sha256'] != expected_hash
                or digest(parent['checkpoint']) != expected_hash):
            raise ValueError(f'{rate}: cyclic and fixed runs must use the same accepted parent')
        if identity['test_mask_sha256'] != old_identity['test_mask_sha256']:
            raise ValueError(f'{rate}: cyclic/fixed test masks differ')
        block_source = 'osram_four_block_wiring/blocks.py'
        if identity['source_hashes'][block_source] != old_identity['source_hashes'][block_source]:
            raise ValueError(f'{rate}: Signature block source differs from fixed control')
        config = identity['effective_stage2_config']
        old_config = old_identity['effective_stage2_config']
        for key in ('learning_rate', 'weight_decay', 'batch_size'):
            if config[key] != old_config[key]:
                raise ValueError(f'{rate}: fixed/cyclic {key} differs')
        history = json.loads((Path(job['output']) / 'history.json').read_text())
        if len(history) != 150 or [r['epoch'] for r in history] != list(range(1, 151)):
            raise ValueError(f'{rate}: require complete contiguous 150-epoch history')
        steps = sum(row.get('optimizer_steps', row.get('batches', 0)) for row in history)
        if steps != 300:
            raise ValueError(f'{rate}: expected 300 optimizer steps, found {steps}')
        consumed = [float(r) for row in history for r in row['training_rates']]
        expected_consumed = [float(RATES[(epoch + batch) % len(RATES)])
                             for epoch in range(150) for batch in range(2)]
        if (set(consumed) != {float(r) for r in RATES}
                or consumed != expected_consumed
                or any(len(row['training_rates']) != row['optimizer_steps'] for row in history)):
            raise ValueError(f'{rate}: consumed training rates differ from BalancedBatchRateSchedule')
        rate_counts = Counter(consumed)
        # The original schedule advances by epoch, not global optimizer step:
        # 150 epochs x 2 batches produces spread 2, rather than spread 1.
        expected_counts = Counter(expected_consumed)
        if rate_counts != expected_counts:
            raise ValueError(f'{rate}: unexpected cyclic training-rate counts')
        mask_trace = [row['mask_trace_sha256'] for row in history]
        if any(not trace for trace in mask_trace):
            raise ValueError(f'{rate}: missing training mask trace')
        if reference_trace is None:
            reference_trace = mask_trace
        elif mask_trace != reference_trace:
            raise ValueError(f'{rate}: training mask traces differ across frozen parents')
        init_proof = {}
        for key in ('initial_block_sha256', 'block_init_sha256', 'module_init_sha256'):
            if key in old_identity:
                if identity.get(key) != old_identity[key]:
                    raise ValueError(f'{rate}: initial Signature module differs')
                init_proof[key] = identity[key]
        baseline, fixed_metrics, cyclic = summary['baseline'], fixed['best']['metrics'], summary['best']['metrics']
        for key in ('ACC', 'WF1', 'MSE'):
            if abs(baseline[key] - fixed['baseline'][key]) > 1e-6:
                raise ValueError(f'{rate}: frozen-parent replay metrics differ')
        row = dict(rate=rate)
        for key in ('ACC', 'WF1', 'MSE'):
            row.update({f'Flat_{key}':baseline[key], f'Fixed_{key}':fixed_metrics[key],
                        f'Cyclic_{key}':cyclic[key], f'Cyclic_minus_Flat_{key}':cyclic[key]-baseline[key],
                        f'Cyclic_minus_Fixed_{key}':cyclic[key]-fixed_metrics[key]})
        rows.append(row)
        proofs[rate] = dict(parent_id=expected_hash, test_mask_sha256=identity['test_mask_sha256'],
                           optimizer_steps=steps, module_initialization=init_proof or 'not recorded in fixed identity',
                           training_rate_counts={str(r):rate_counts[r] for r in sorted(rate_counts)},
                           training_rate_count_spread=max(rate_counts.values())-min(rate_counts.values()),
                           original_balanced_schedule_exact=True, matched_training_mask_traces=True,
                           block_source_sha256=identity['source_hashes'][block_source],
                           matched_config={key:config[key] for key in ('learning_rate', 'weight_decay', 'batch_size')},
                           cyclic_summary=job['summary'], fixed_summary=str(fixed_path(args, rate)))
    if {r['rate'] for r in rows} != set(RATES) or len(rows) != 8:
        raise ValueError('Exactly eight rates required')
    aggregates = [dict(rate=scope, **{k: statistics.mean(r[k] for r in rows if r['rate'] in rates)
                  for k in rows[0] if k != 'rate'}) for scope, rates in (('mean', RATES), ('high', RATES[5:]))]
    save(args.root / 'RESULT.json', dict(status='complete', source_commit=args.source_commit,
         protocol='MOSI68 Signature cyclic train; per-parent current-rate TEST oracle',
         per_rate=rows, aggregate=aggregates, evidence=proofs,
         matched_training_mask_trace_sha256=reference_trace))
    with (args.root / 'RESULT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows+aggregates)
    lines = ['# MOSI68 Signature cyclic training comparison', '',
             f'Source `{args.source_commit}`. Eight independent frozen per-rate parents; internal TEST-oracle.', '',
             '| Rate/scope | Method | ACC | WF1 | MSE | Δ WF1 vs Flat | Δ WF1 vs Fixed |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in aggregates+rows:
        for method in ('Flat', 'Fixed', 'Cyclic'):
            lines.append(f'| {row["rate"]} | {method} | {row[method+"_ACC"]:.4f} | '
                         f'{row[method+"_WF1"]:.4f} | {row[method+"_MSE"]:.4f} | '
                         f'{row[method+"_WF1"]-row["Flat_WF1"]:.4f} | '
                         f'{row[method+"_WF1"]-row["Fixed_WF1"]:.4f} |')
    (args.root / 'RESULT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu2-uuid', required=True)
    parser.add_argument('--gpu3-uuid', required=True)
    parser.add_argument('--fixed-root', type=Path, default=FIXED_ROOT)
    args = parser.parse_args(argv)
    args.root, args.fixed_root = args.root.resolve(), args.fixed_root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), started_at=now(), source_commit=args.source_commit,
                 command=[sys.executable]+sys.argv, cwd=os.getcwd(), max_per_gpu=4,
                 gpu2_uuid=args.gpu2_uuid, gpu3_uuid=args.gpu3_uuid, fixed_root=str(args.fixed_root))
    save(args.root/'QUEUE.json', state)
    try:
        previous = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            for index, uuid in ((2,args.gpu2_uuid),(3,args.gpu3_uuid)):
                os.environ['CUDA_VISIBLE_DEVICES']=uuid
                guard_gpu(uuid,index)
        finally:
            if previous is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES']=previous
        state['preflight']=resources(args.root,[args.gpu2_uuid,args.gpu3_uuid])
        capacity(state['preflight'])
        state['fixed_preflight']={}
        for rate in RATES:
            _,proof=validated_row(fixed_path(args,rate),'CMUMOSI',68,rate)
            state['fixed_preflight'][rate]=proof
        state['status']='smoke'
        run_phase(args,state,'smoke')
        peak=state['smoke'][0]['peak_memory_mib']
        state.update(status='train',admission_smoke_peak_mib=peak)
        run_phase(args,state,'train',peak)
        collect(args,state)
        state.update(status='complete',completed_at=now())
        save(args.root/'QUEUE.json',state)
        return 0
    except Exception as error:
        state.update(status='failed',error=str(error),finished_at=now())
        save(args.root/'QUEUE.json',state)
        print(f'Queue failed: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
