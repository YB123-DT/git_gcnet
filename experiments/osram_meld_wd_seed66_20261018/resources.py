"""Audited resource-only override; never edit sealed training source or identity."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ORIGINAL_COMMIT = '3628d1e69d611f7d6a8a0d6896b435a63f805a9c'
FROZEN_DRIVER_SHA = '03c4d962586dd346d61008d5f14896ec11b9f32e680391937cfec055ed411d17'
DEFAULT_FROZEN = Path('/home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/source_3628d1e')
GPU_UUIDS = {0:'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
             2:'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}
sys.path.insert(0, str(DEFAULT_FROZEN if os.path.isdir(DEFAULT_FROZEN) else Path(__file__).resolve().parents[2]))
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json


def load_frozen(path):
    path = Path(path).resolve()
    driver = path / 'experiments/osram_meld_wd_seed66_20261018/run.py'
    if not driver.is_file() or hashlib.sha256(driver.read_bytes()).hexdigest() != FROZEN_DRIVER_SHA:
        raise ValueError('Require the unchanged original frozen seed66 driver')
    spec = importlib.util.spec_from_file_location('wd66_frozen_resource_target', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_source(ORIGINAL_COMMIT)
    return module


class ResourceScheduler(Scheduler):
    def __init__(self, root, args, python=sys.executable):
        self.args = args
        super().__init__(root, gpus=(0,2), per_gpu=4, python=python,
            code_root=args.frozen_source,
            smoke_costs={'Flat':dict(peak_memory_mib=4096,disk_bytes=2*2**30)})

    def command(self, job, config_path, gpu_uuid):
        return [self.python, '-u', str(Path(__file__).resolve()), '--worker', '--job', job['id'],
            '--config', str(config_path), '--output', job['output'], '--gpu-uuid', gpu_uuid,
            '--driver-commit', ORIGINAL_COMMIT, '--wrapper-commit', self.args.wrapper_commit,
            '--frozen-source', str(self.args.frozen_source)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frozen-source', type=Path, default=DEFAULT_FROZEN)
    p.add_argument('--driver-commit', default=ORIGINAL_COMMIT)
    p.add_argument('--wrapper-commit', required=True)
    p.add_argument('--worker', action='store_true'); p.add_argument('--job')
    p.add_argument('--config', type=Path); p.add_argument('--gpu-uuid')
    args = p.parse_args()
    if args.driver_commit != ORIGINAL_COMMIT:
        p.error('Do not change the preserved original training identity')
    manifest_path = Path(__file__).with_name('RESOURCE_MANIFEST.json')
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest['resources_sha256']:
        raise ValueError('Resource-only wrapper hash changed')
    original = load_frozen(args.frozen_source)
    original.GPU_UUIDS = dict(GPU_UUIDS)
    audit = dict(scope='resource-only; no numerical/training/identity changes',
        wrapper_commit=args.wrapper_commit, wrapper_sha256=manifest['resources_sha256'],
        frozen_source=str(args.frozen_source), training_commit=ORIGINAL_COMMIT,
        allowed_gpu_uuids=GPU_UUIDS, pid=os.getpid(), at_unix=time.time())
    if args.worker:
        if args.job not in original.JOBS or args.config is None or args.gpu_uuid not in GPU_UUIDS.values():
            p.error('Require approved job/config/healthy GPU UUID')
        write_json(args.output / 'RESOURCE_OVERRIDE.json', dict(audit,gpu_uuid=args.gpu_uuid))
        original.worker(args)
    else:
        def scheduler(target):
            return ResourceScheduler(target.output / 'queue', args)
        original.Seed66Scheduler = scheduler
        args.resume = True
        write_json(args.output / 'RESOURCE_CONTROLLER.json', audit)
        # Existing live worker commands are kept exactly as recorded; Scheduler
        # recognizes them by PID+birth+argv and never relaunches them. Only
        # pending jobs receive the new UUID-aware command above.
        original.controller(args)


if __name__ == '__main__':
    main()
