"""Eight independent MELD rate-specific Flat models; finite same-rate test queue."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json

BASE_CONFIG = ROOT / 'experiments/osram_meld_mmgcn_features_20261018/CONFIG.json'
DATA = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
DATA_SHA = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'
GPU_UUIDS = {0: 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
            5: 'GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def make_config(rate):
    if isinstance(rate, bool) or rate not in tuple(i / 10 for i in range(8)):
        raise ValueError('Only the eight official fixed rates are authorized')
    base = json.loads(BASE_CONFIG.read_text())
    base.update(train_rate_mode='fixed', fixed_missing_rate=rate, train_missing_rates=[rate])
    return base


def build_jobs(output):
    return [dict(id=f'R{i}', method='Flat', config=make_config(i / 10),
                 output=str(Path(output) / f'rate_0p{i}' / 'seed_66')) for i in range(8)]


def check_snapshot():
    manifest_path = ROOT / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('git_sha') or not manifest.get('files'):
        raise ValueError('Immutable source manifest is required')
    for name, digest in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError(f'Immutable source differs: {name}')
    return manifest


class FixedRateScheduler(Scheduler):
    def __init__(self, output):
        self.output_root = Path(output).resolve()
        super().__init__(self.output_root / 'queue', gpus=(0, 5), per_gpu=4,
                         code_root=ROOT,
                         smoke_costs={'Flat': {'peak_memory_mib': 4096, 'disk_bytes': 2*2**30}})

    def command(self, job, config_path, gpu_uuid):
        return [self.python, '-u', str(Path(__file__)), '--worker', '--rate',
                str(job['config']['fixed_missing_rate']), '--config', str(config_path),
                '--output', job['output'], '--gpu-uuid', gpu_uuid]

    def admit(self, job):
        first = self.output_root / 'rate_0p0/seed_66/EPOCH_STATUS.json'
        if job['id'] != 'R0' and not first.exists():
            if self.state['jobs'].get('R0', {}).get('status') == 'failed':
                self.failure(job, 'First fixed-rate integration epoch failed; remaining launches cancelled')
            return None
        if first.exists():
            peak = json.loads(first.read_text())['peak_memory_mib']
            self.smoke_costs['Flat']['peak_memory_mib'] = max(4096, peak * 1.25)
        return super().admit(job)


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def selected_result(output, rate):
    import numpy as np
    import torch
    from experiments.osram_meld_write_tuning_20261017.capture import _metrics
    key = format(rate, '.1f')
    path = output / f'best_miss_{key.replace(".", "p")}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    reference = checkpoint['selection_artifact']
    artifact = (output / reference['path']).resolve()
    if not artifact.is_relative_to(output.resolve()) or sha(artifact) != reference['sha256']:
        raise ValueError('Selected prediction hash/path mismatch')
    with np.load(artifact, allow_pickle=False) as saved:
        metrics = _metrics(saved)
        if (int(saved['selected_epoch']) != checkpoint['epoch'] or metrics['N'] != 2610
                or checkpoint['selection_rate'] != rate
                or abs(metrics['ACC']/100-checkpoint['selection_accuracy']) > 1e-10
                or sha_bytes(saved['availability'].tobytes()) != reference['availability_sha256']):
            raise ValueError('Selected prediction/epoch/rate/metric/mask mismatch')
    return dict(metrics, epoch=checkpoint['epoch'], checkpoint_sha256=sha(path), **reference)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def rate_state_class(base, output, rate):
    import torch
    from experiments.osram_meld_alternate_features_20261017.worker import compact_state_class
    compact = compact_state_class(base, torch)

    class FixedRateState(compact):
        def bind(self, model, optimizer, **kwargs):
            super().bind(model, optimizer, **kwargs)
            self.initial_hash = model_hash(model)
            initial = dict(sha256=self.initial_hash, parameters=sum(p.numel() for p in model.parameters()))
            if initial['parameters'] != 13290795:
                raise ValueError('Original ECERC Flat parameter count changed')
            path = output / 'INITIAL_STATE.json'
            if path.exists() and json.loads(path.read_text()) != initial:
                raise ValueError('Initialization changed on resume')
            write_json(path, initial)
            return self

        def save_best(self, *, epoch, rate, checkpoint):
            if float(rate) != self.identity['fixed_missing_rate']:
                raise ValueError('Unexpected other-rate checkpoint')
            return super().save_best(epoch=epoch, rate=rate, checkpoint=checkpoint)

        def commit_epoch(self, **kwargs):
            epoch = kwargs['next_epoch']
            row = kwargs['history'][-1]
            key = format(rate, '.1f')
            if set(row['test_oracle']) != {str(rate)}:
                raise ValueError('Fixed-rate selection evaluated another rate')
            counts = row['train']['rate_batch_counts']
            if (counts.get(key, 0) != 33 or sum(counts.values()) != 33
                    or row['train']['optimizer_steps'] != 33
                    or row['train']['source_conversation_count'] != 1038
                    or row['train']['jepa_loss'] != 0):
                raise ValueError('Expected exactly 33 same-rate emotion-only optimizer steps')
            super().commit_epoch(**kwargs)
            write_json(output / 'EPOCH_STATUS.json', dict(epoch=epoch, fixed_missing_rate=rate,
                initial_state_sha256=self.initial_hash, best=selected_result(output, rate),
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                train_rate_batch_counts=counts, optimizer_steps=33))
    return FixedRateState


def worker(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, GCNET_MELD_PICKLE=str(DATA),
                      GCNET_DATASET_ROOT=str(ROOT/'dataset'),
                      GCNET_CACHE_ROOT=str(args.output/'feature_cache'))
    manifest = check_snapshot()
    if sha(DATA) != DATA_SHA:
        raise ValueError('Accepted ECERC feature hash changed')
    from experiments.osram_meld_alternate_features_20261017.worker import guard_gpu, repair_uuid_enumeration
    import numpy as np
    import torch
    gpu = guard_gpu(args.gpu_uuid)
    if GPU_UUIDS.get(gpu['host_index']) != gpu['uuid']:
        raise ValueError('Only registered healthy GPU0/5 are allowed')
    repair = repair_uuid_enumeration(torch)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    trainer = importlib.import_module('gcnet_missing_m3.train_gcnet')
    from gcnet_missing_m3.training_resume import TrainingState
    if not Path(trainer.__file__).resolve().is_relative_to(ROOT):
        raise ValueError('Numerical imports escaped immutable source')
    raw = json.loads(args.config.read_text())
    if raw != make_config(args.rate):
        raise ValueError('Only the registered fixed-rate config is authorized')
    config = trainer.TrainConfig(**raw)
    if trainer._protocol_rates(config) != (args.rate,):
        raise ValueError('Fixed rate must be the only evaluation rate')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(git_sha=manifest['git_sha'], source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
                    config_sha256=sha(args.config), dataset_sha256=DATA_SHA, fixed_missing_rate=args.rate)
    path = args.output/'SUMMARY.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['identity'] != identity:
            raise ValueError('Cannot mix code/config/data within an output')
        if previous['status'] == 'complete':
            return
    summary = dict(status='running', identity=identity, config=asdict(config), pid=os.getpid(),
                   started_unix=time.time(), protocol='INTERNAL SINGLE-RATE TEST-ORACLE ACC',
                   provenance=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                                   numpy=np.__version__, gpu=gpu, command=sys.argv, uuid_rng=repair))
    write_json(path, summary)
    tick = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        state = rate_state_class(TrainingState, args.output, args.rate)(args.output, identity=identity)
        trainer.run_experiment(config, 'MELD-pickle', 'MELD-pickle', 'MELD-pickle',
                               args.output, training_state=state)
        progress = json.loads((args.output/'EPOCH_STATUS.json').read_text())
        if progress['epoch'] != 50:
            raise ValueError('Expected 50 fully committed epochs')
        summary.update(status='complete', testbest=selected_result(args.output, args.rate),
                       initial_state_sha256=state.initial_hash,
                       parameters=sum(p.numel() for p in state.model.parameters()),
                       peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        summary.update(seconds=time.monotonic()-tick, finished_unix=time.time())
        write_json(path, summary)


def summarize(completed):
    successes = [row for row in completed if row['status'] == 'complete']
    rates = {format(row['config']['fixed_missing_rate'], '.1f'): row['summary']['testbest'] for row in successes}
    result = dict(status='complete' if len(successes) == 8 else 'failed', rates=rates,
                  failures={row['id']: row['failures'] for row in completed if row['status'] != 'complete'},
                  protocol='EIGHT INDEPENDENT MODELS; TRAIN RATE = TEST RATE; INTERNAL TEST-ORACLE')
    if len(successes) == 8:
        for name, keys in [('mean',list(rates)), ('high',['0.5','0.6','0.7'])]:
            result[name] = {m: sum(rates[r][m] for r in keys)/len(keys) for m in ('ACC','UA','WF1','CE')}
        result['initialization_equal'] = len({row['summary']['initial_state_sha256'] for row in successes}) == 1
        if not result['initialization_equal']:
            raise ValueError('Rate-specific models unexpectedly differ at initialization')
    return result


def controller(args):
    manifest = check_snapshot()
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a new output; recovery requires --resume with unchanged code/config')
    if args.resume and not args.output.exists():
        raise FileNotFoundError('No queue exists to recover')
    if sha(DATA) != DATA_SHA or shutil.disk_usage(args.output.parent).free < 20*2**30:
        raise RuntimeError('Accepted data hash or 20GiB disk admission failed')
    args.output.mkdir(parents=True, exist_ok=True)
    scheduler = FixedRateScheduler(args.output)
    snapshot = scheduler.gpu_snapshot()
    if any(snapshot[g]['uuid'] != uuid for g,uuid in GPU_UUIDS.items()):
        raise ValueError('Physical GPU UUID mapping changed')
    runner = dict(status='running', pid=os.getpid(), started_unix=time.time(), git_sha=manifest['git_sha'],
                  jobs=8, seed=66, epochs=50, gpus=GPU_UUIDS, per_gpu=4, dataset_sha256=DATA_SHA)
    write_json(args.output/'RUNNER.json', runner)
    try:
        completed = scheduler.run_jobs(build_jobs(args.output))
        result = summarize(completed)
        write_json(args.output/'RESULTS.json', result)
        runner['status'] = result['status']
    except BaseException as error:
        runner.update(status='failed', error=repr(error))
        raise
    finally:
        runner['finished_unix'] = time.time()
        write_json(args.output/'RUNNER.json', runner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--rate', type=float)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    if args.worker and any(getattr(args,k) is None for k in ('rate','config','gpu_uuid')):
        parser.error('Worker requires rate/config/gpu-uuid')
    (worker if args.worker else controller)(args)


if __name__ == '__main__':
    main()
