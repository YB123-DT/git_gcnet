"""Five finite, matched MELD write-parameter training jobs. No later search."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json

REFERENCE = ROOT/'experiments/osram_meld_memory_sensitivity_20261017/FROZEN_SUMMARY.json'
PRESETS = ROOT/'experiments/osram_meld_memory_sensitivity_20261017/TUNING_PRESETS.json'
HISTORICAL = 'b4a313841f098487a6482d2712bdac94aa8e319e'
DATASET = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
DATA_SHA = 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'
WRITE_KEYS = {'osram_write_step', 'osram_write_ridge'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def shared_protocol_hash(config):
    return hashlib.sha256(json.dumps({k:v for k,v in config.items() if k not in WRITE_KEYS},
                                    sort_keys=True).encode()).hexdigest()


def make_config(base, variant):
    from experiments.osram_meld_write_retention_20261014.task_replay import validate_source
    validate_source(base)
    expected = dict(epochs=50, batch_size=32, learning_rate=.001, weight_decay=1e-5,
        dropout=.5, train_rate_mode='cyclic', checkpoint_selection='test-oracle-per-rate',
        osram_write_step=.6, osram_write_ridge=.001, beta_mode='embedded')
    if any(base.get(key) != value for key,value in expected.items()):
        raise ValueError('Original MELD training reference changed')
    if base.get('initial_backbone_checkpoint') is not None:
        raise ValueError('From-scratch training required')
    presets = {row['id']:row for row in json.loads(PRESETS.read_text())['first_round']}
    if variant not in presets:
        raise ValueError('Only W0..W4 are authorized')
    config = deepcopy(base)
    config.update({key:value for key,value in presets[variant].items() if key != 'id'})
    return config


def aggregate(scores):
    if set(scores) != {str(i/10) for i in range(8)}:
        raise ValueError('Require exactly eight selected rates')
    return dict(rates=scores,
        mean={k:sum(v[k] for v in scores.values())/8 for k in ('ACC','UA','WF1','CE')},
        high={k:sum(scores[str(i/10)][k] for i in (5,6,7))/3 for k in ('ACC','UA','WF1','CE')})


class WriteScheduler(Scheduler):
    def __init__(self, root, *, model_source, control):
        self.model_source, self.control = Path(model_source), Path(control)
        super().__init__(root, gpus=(5,0), per_gpu=4, code_root=ROOT,
            smoke_costs={'Flat':{'peak_memory_mib':6144, 'disk_bytes':2*2**30}})

    def command(self, job, config_path, gpu_uuid):
        return [self.python, '-u', str(Path(__file__)), '--worker', '--variant', job['id'],
            '--config', str(config_path), '--output', job['output'],
            '--model-source', str(self.model_source), '--control', str(self.control),
            '--gpu-uuid', gpu_uuid]

    def admit(self, job):
        # The first actual control epoch, not a second training condition, is
        # also the real-batch forward/backward/optimizer and peak-memory check.
        path = self.control/'EPOCH_STATUS.json'
        if job['id'] != 'W0' and not path.is_file():
            if self.state['jobs'].get('W0',{}).get('status') == 'failed':
                self.failure(job,'Control failed before first committed epoch; dependent launch cancelled')
            return None
        if path.is_file():
            progress = json.loads(path.read_text())
            if progress['epoch'] >= 1:
                self.smoke_costs['Flat']['peak_memory_mib'] = max(4096., progress['peak_memory_mib']*1.25)
        return super().admit(job)


def load_historical(path):
    if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip() != HISTORICAL:
        raise ValueError('Require accepted original b4a3138 numerical source')
    subprocess.run(['git','-C',str(path),'diff','--exit-code','HEAD','--',
        'gcnet_missing_m3','gcnet_modality_jepa','config.py'], check=True, capture_output=True)
    for prefix in ('gcnet_missing_m3','gcnet_modality_jepa'):
        if any(name == prefix or name.startswith(prefix+'.') for name in sys.modules):
            raise RuntimeError('Worker must start in a fresh process before numerical imports')
    sys.path.insert(0,str(path))
    trainer = importlib.import_module('gcnet_missing_m3.train_gcnet')
    if not Path(trainer.__file__).resolve().is_relative_to(path.resolve()):
        raise RuntimeError('Historical model import escaped the pinned snapshot')
    return trainer


def state_class(base_class, capture, output, variant, control):
    """Extend the existing epoch-state utility, never the numerical trainer."""
    import torch

    class CapturedState(base_class):
        def bind(self, model, optimizer, **kwargs):
            super().bind(model,optimizer,**kwargs)
            capture.bind(model)
            initial = dict(sha256=capture.initial_state_sha256, parameters=capture.modelparameters)
            path = output/'INITIAL_STATE.json'
            if path.exists() and json.loads(path.read_text()) != initial:
                raise ValueError('Fresh initialization differs on restart')
            if variant != 'W0' and json.loads((control/'INITIAL_STATE.json').read_text()) != initial:
                raise ValueError('Variant initialization differs from matched W0')
            write_json(path,initial)
            return self

        def restore(self):
            restored = super().restore()
            if restored is not None:
                capture.resume_trace(restored['next_epoch'])
                for rate,ref in self.references.items():
                    metadata = json.loads(self._checked(ref['selection']).read_text())['checkpoint']
                    capture.selected_scores[rate] = metadata['selected_metrics']
            return restored

        def save_best(self, *, epoch, rate, checkpoint):
            reference = capture.selection_reference(epoch,rate)
            scores = capture.selected_scores[rate]
            if abs(scores['ACC']/100-checkpoint['selection_accuracy']) > 1e-10:
                raise ValueError('Selected checkpoint ACC differs from same-forward predictions')
            if abs(scores['WF1']/100-checkpoint['selection_weighted_f1']) > 1e-10:
                raise ValueError('Selected checkpoint WF1 differs from same-forward predictions')
            payload = dict(checkpoint, selection_artifact=reference, selected_metrics=scores)
            super().save_best(epoch=epoch,rate=rate,checkpoint=payload)

        def commit_epoch(self, **kwargs):
            # Trace can be ahead on interruption; resume_trace trims it back to
            # the atomically committed last_training epoch, never the reverse.
            capture.write_trace()
            super().commit_epoch(**kwargs)
            keep = {self._checked(ref[key]).resolve() for ref in self.references.values()
                    for key in ('model','selection')}
            versions = output/'training_versions'
            if versions.is_dir():
                for path in versions.iterdir():
                    if path.is_file() and path.resolve() not in keep:
                        path.unlink()  # ONLY this new run's unreferenced cache, never BEST/last.
            epoch = kwargs['next_epoch']
            write_json(output/'EPOCH_STATUS.json', dict(epoch=epoch,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                initial_state_sha256=capture.initial_state_sha256,
                mask_trace_sha256=sha(output/'MASK_TRACE.json'),
                best=aggregate(capture.selected_scores),
                alpha=self.model.osram.alpha_logits.sigmoid().detach().cpu().tolist(),
                beta=self.model.osram.beta_logits.sigmoid().detach().cpu().tolist()))
    return CapturedState


def worker(args):
    os.environ.update(GCNET_MELD_PICKLE=str(DATASET),
        GCNET_DATASET_ROOT=str(args.model_source/'dataset'),
        GCNET_CACHE_ROOT=str(args.output.parent/'feature_cache'))
    from experiments.osram_meld_write_tuning_20261017.capture import TrainingCapture, _metrics
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    import numpy as np
    import torch
    gpu = guard_gpu(args.gpu_uuid)
    count = torch._C._cuda_getDeviceCount()
    if count != 1:
        raise ValueError('UUID binding must expose exactly one CUDA runtime device')
    torch.cuda.device_count = lambda:count
    torch.cuda.random.device_count = torch.cuda.device_count
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    trainer = load_historical(args.model_source)
    from gcnet_missing_m3.training_resume import TrainingState
    config_dict = json.loads(args.config.read_text())
    expected = make_config(json.loads(REFERENCE.read_text())['source_config'], args.variant)
    if config_dict != expected or sha(DATASET) != DATA_SHA:
        raise ValueError('Immutable variant config or ECERC data hash differs')
    config = trainer.TrainConfig(**config_dict)
    args.output.mkdir(parents=True,exist_ok=True)
    code_sha = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    identity = dict(driver_commit=code_sha, numerical_commit=HISTORICAL, data_sha256=DATA_SHA,
        config_sha256=sha(args.config), shared_protocol_hash=shared_protocol_hash(config_dict))
    source_files = [p for folder in ('gcnet_missing_m3','gcnet_modality_jepa')
                    for p in (args.model_source/folder).glob('*.py')]
    source_files += [args.model_source/'config.py', Path(__file__), Path(__file__).with_name('capture.py')]
    summary = dict(status='running', variant=args.variant, config=config_dict,
        provenance=dict(**identity, gpu=gpu, command=sys.argv, source_files={str(p):sha(p) for p in source_files},
            python=sys.version, torch=torch.__version__, numpy=np.__version__, uuid_rng_alias_fix=True),
        protocol='INTERNAL TEST-ORACLE SCREEN; per-rate ACC; UA from exact selected forward',
        pid=os.getpid(), started_unix=time.time())
    previous = args.output/'SUMMARY.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if old['provenance']['config_sha256'] != identity['config_sha256'] or old['provenance']['driver_commit'] != code_sha:
            raise ValueError('Do not mix sources/configurations within a run')
        if old['status'] == 'complete':
            return
    write_json(previous,summary)
    capture = TrainingCapture(trainer,args.output)
    state = state_class(TrainingState,capture,args.output,args.variant,args.control)(args.output,identity=identity)
    tick = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        with capture:
            trainer.run_experiment(config,'MELD-pickle','MELD-pickle','MELD-pickle',args.output,training_state=state)
            capture.write_trace()
        selected = {}
        for i in range(8):
            rate = str(i/10)
            path = args.output/f'best_miss_0p{i}.pt'
            checkpoint = torch.load(path,map_location='cpu',weights_only=False)
            ref = checkpoint['selection_artifact']
            artifact = args.output/ref['path']
            if sha(artifact) != ref['sha256']:
                raise ValueError('Selected artifact integrity check failed')
            with np.load(artifact,allow_pickle=False) as data:
                scores = _metrics(data)
                if int(data['selected_epoch']) != checkpoint['epoch'] or scores['N'] != 2610:
                    raise ValueError('Incomplete/mismatched selected evaluation')
                if abs(scores['ACC']/100-checkpoint['selection_accuracy']) > 1e-10:
                    raise ValueError('Final selected score parity failed')
            selected[rate] = dict(scores,epoch=checkpoint['epoch'],checkpoint_sha256=sha(path), **ref)
        summary.update(status='complete',testbest=aggregate(selected),
            parameters=capture.modelparameters, initial_state_sha256=capture.initial_state_sha256,
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
            training_mask_trace_sha256=sha(args.output/'MASK_TRACE.json'))
    except BaseException as error:
        summary.update(status='failed',error=repr(error))
        raise
    finally:
        summary.update(seconds=time.monotonic()-tick,finished_unix=time.time())
        write_json(previous,summary)


def controller(args):
    base = json.loads(REFERENCE.read_text())['source_config']
    if args.output.exists():
        raise FileExistsError('Use a new run directory; recovery uses the same queue explicitly')
    if sha(DATASET) != DATA_SHA or shutil.disk_usage(args.output.parent).free < 12*2**30:
        raise RuntimeError('Feature hash or 12GiB disk admission failed')
    args.output.mkdir(parents=True)
    scheduler = WriteScheduler(args.output/'queue',model_source=args.model_source,control=args.output/'W0')
    snapshot = scheduler.gpu_snapshot()
    if snapshot[5]['uuid'] != 'GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62' or snapshot[0]['uuid'] != 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45':
        raise ValueError('Healthy physical GPU0/5 identity changed')
    jobs = [dict(id=f'W{i}',method='Flat',config=make_config(base,f'W{i}'),output=str(args.output/f'W{i}')) for i in range(5)]
    meta = dict(status='running',pid=os.getpid(),started_unix=time.time(),gpus=snapshot,
        numerical_commit=HISTORICAL,driver_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        authorized_jobs=5, later_rounds=False, selection=base['checkpoint_selection'], jobs=[x['id'] for x in jobs])
    write_json(args.output/'RUNNER.json',meta)
    try:
        completed = scheduler.run_jobs(jobs)
        successes = [row for row in completed if row['status']=='complete']
        paired = dict(initialization_equal=False,mask_stream_equal=False)
        if len(successes) == 5:
            traces = [json.loads((Path(row['output'])/'MASK_TRACE.json').read_text()) for row in completed]
            paired['initialization_equal'] = len({x['initial_state_sha256'] for x in traces}) == 1
            def masks(trace):
                return {phase:{epoch:({rate:row['sha256'] for rate,row in value.items()} if phase=='selection' else value['sha256'])
                    for epoch,value in trace[phase].items()} for phase in ('train','selection')}
            paired['mask_stream_equal'] = all(masks(x)==masks(traces[0]) for x in traces[1:])
            if not all(paired.values()):
                raise ValueError('Completed runs do not share initialization/mask stream')
        write_json(args.output/'RESULTS.json',dict(status='complete' if len(successes)==5 else 'failed',
            paired_checks=paired, results={row['id']:row.get('summary',{}) for row in completed},
            failures={row['id']:row['failures'] for row in completed if row['status']!='complete'}))
        meta['status'] = 'complete' if len(successes)==5 else 'failed'
    except BaseException as error:
        meta.update(status='failed',error=repr(error))
        raise
    finally:
        meta['finished_unix'] = time.time()
        write_json(args.output/'RUNNER.json',meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-source',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--variant',choices=[f'W{i}' for i in range(5)])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--control',type=Path)
    args = parser.parse_args()
    if args.worker and any(getattr(args,key) is None for key in ('variant','config','gpu_uuid','control')):
        parser.error('Worker needs variant/config/gpu-uuid/control')
    (worker if args.worker else controller)(args)


if __name__ == '__main__':
    main()
