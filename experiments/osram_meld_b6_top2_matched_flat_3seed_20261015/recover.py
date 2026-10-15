"""CPU-only exact Test-best recovery, preserving original failed replay reports."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def recovery_action(raw, epochs, checkpoints):
    if raw['status'] == 'running':
        return 'wait'
    if epochs != 50 or checkpoints != 8:
        return 'failed'
    if raw['status'] == 'complete':
        return 'existing_complete'
    if raw['status'] == 'failed' and 'Arrays are not equal' in raw.get('error', ''):
        return 'recover'
    return 'failed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES empty is required; no new GPU inference')
    os.chdir(args.source_root)
    sys.path.insert(0, str(args.source_root))
    from experiments.osram_meld_block2_auto_20261014.recover_testbest import recover_run
    from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
    from gcnet_missing_m3.training_resume import _sha
    import numpy as np
    manifest = json.loads((args.root / 'MANIFEST.json').read_text())
    if _sha(args.launcher) != manifest['launcher_sha256']:
        raise ValueError('Do not mix summary launcher versions')
    spec = importlib.util.spec_from_file_location('frozen_top2_launcher', args.launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = args.root / 'testbest_recovery_v1'
    output.mkdir(parents=True, exist_ok=True)
    canonical = Path(manifest['sources']['B6_tuned']['source_summary']).parent / 'paired_miss_0p0.npz'
    jobs = module.make_jobs(manifest['sources'], args.root)
    if jobs != manifest['jobs']:
        raise ValueError('Seven-run manifest/configuration changed')
    with (args.root / '.testbest_recovery.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cached, previous = {}, None
        deadline = time.monotonic() + 12 * 3600
        while True:
            records = []
            for job in jobs:
                name, path = job['id'], Path(job['output'])
                f = path / 'SUMMARY.json'
                if name not in cached and f.exists():
                    try:
                        raw = json.loads(f.read_text())
                    except json.JSONDecodeError:
                        raw = dict(status='running')
                    h = path / 'history.json'
                    action = recovery_action(raw, len(json.loads(h.read_text())) if h.exists() else 0,
                                             len(list(path.glob('best_miss_*.pt'))))
                    if action == 'existing_complete':
                        cached[name] = dict(job, status='complete', summary=raw)
                    elif action == 'recover':
                        with np.load(canonical) as reference:
                            result = recover_run(path, reference, output / 'runs')
                        cached[name] = dict(job, status='complete', summary=result)
                        print(json.dumps(dict(recovered=name, testbest=result['testbest']['mean'])), flush=True)
                    elif action == 'failed':
                        cached[name] = dict(job, status='failed', failures=[dict(
                            reason='Not a verified50epoch/8checkpoint positional replay-only failure', error=raw.get('error'))])
                records.append(cached.get(name, dict(job, status='pending')))
            signature = [(r['id'], r['status']) for r in records]
            if signature != previous:
                report = module.summarize(manifest['sources'], records, output)
                write_json(output / 'RECOVERY.json', dict(status=report['status'],
                    recovered_or_complete=[r['id'] for r in records if r['status'] == 'complete'],
                    original_outputs_unchanged=True, new_training=False, new_inference=False,
                    canonical_identity_file=str(canonical), canonical_sha256=_sha(canonical),
                    recovery_launcher_sha256=_sha(Path(__file__)),
                    scope='Exact saved Test-best metrics only; cross-seed common-mask replay remains incomplete',
                    root_cause='Seed-derived test conversation order differs; positional seed66 mask replay is not cross-seed safe'))
                previous = signature
            if not args.wait or all(r['status'] in ('complete', 'failed') for r in records):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded twelve-hour recovery wait exhausted')
            time.sleep(15)


if __name__ == '__main__':
    main()
