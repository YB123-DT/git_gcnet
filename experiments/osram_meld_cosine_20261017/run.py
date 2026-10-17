"""One MELD Flat cosine run, reusing the sealed numerical trainer and capture."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.osram_meld_write_tuning_20261017 import run as original
from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json

GPU_UUID = 'GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62'
_original_config = original.make_config
_original_state = original.state_class


def make_config(base):
    config = _original_config(base, 'W0')
    if config['lr_schedule'] != 'constant' or config['warmup_ratio'] != .05:
        raise ValueError('Expected original constant schedule with dormant .05 warmup')
    return dict(config, lr_schedule='cosine')


def expected_lrs(base_lr, epochs, warmup_ratio):
    warmup = max(1, int(round(warmup_ratio*epochs)))
    return [base_lr*(step/warmup if step <= warmup else
            .5*(1+math.cos(math.pi*(step-warmup)/max(1, epochs-warmup))))
            for step in range(1, epochs+1)]


class UASelection:
    """Independent UA-best views, anchored in the main atomic epoch state.

    Reuse TrainingState's hashed model/metadata versions, without a second
    optimizer, training pass, evaluation pass or complete-state checkpoint.
    """
    def __init__(self, state_class, output, identity):
        self.state = state_class(output, identity=identity)
        self.scores, self.rows = {}, []

    def bind(self, model, optimizer):
        self.state.bind(model, optimizer)

    def update(self, capture, epoch, config):
        import numpy as np
        from experiments.osram_meld_write_tuning_20261017.capture import _atomic
        rates = {str(i/10) for i in range(8)}
        if set(capture.latest) != rates or any(v['epoch'] != epoch for v in capture.latest.values()):
            raise ValueError('UA selection needs all eight evaluations from the same epoch')
        if epoch != len(self.rows)+1:
            raise ValueError('UA history must continue the committed epoch sequence')
        self.rows.append(dict(epoch=epoch, rates={key:deepcopy(row['scores']) for key,row in capture.latest.items()}))
        for rate, row in capture.latest.items():
            if row['scores']['UA'] <= self.scores.get(rate, {}).get('UA', -math.inf):
                continue  # Same first-tie policy as original ACC selection.
            relative = Path('selection_versions')/f'epoch_{epoch:06d}_{rate.replace(".", "p")}.npz'
            artifact = self.state.output/relative
            _atomic(artifact, lambda stream: np.savez_compressed(stream, **row['artifacts'], selected_epoch=np.asarray(epoch)))
            reference = dict(path=str(relative), sha256=original.sha(artifact),
                availability_sha256=hashlib.sha256(row['artifacts']['availability'].tobytes()).hexdigest())
            scores = dict(row['scores'], epoch=epoch)
            checkpoint = dict(model=self.state.model.state_dict(), config=config, epoch=epoch,
                selection_split='test', selection_protocol='per-rate-test-oracle',
                selection_rate=float(rate), selection_metric='ua', selection_score=scores['UA']/100,
                selection_accuracy=scores['ACC']/100, selection_weighted_f1=scores['WF1']/100,
                selected_metrics=scores, selection_artifact=reference)
            self.state.save_best(epoch=epoch, rate=rate, checkpoint=checkpoint)
            self.scores[rate] = scores

    def state_dict(self):
        return deepcopy(dict(scores=self.scores, rows=self.rows, references=self.state.references))

    def restore(self, saved):
        self.scores, self.rows = deepcopy(saved['scores']), deepcopy(saved['rows'])
        self.state.references = deepcopy(saved['references'])
        for rate, ref in self.state.references.items():
            self.state._checked(ref['model'])
            metadata = json.loads(self.state._checked(ref['selection']).read_text())['checkpoint']
            if metadata['selected_metrics'] != self.scores[rate]:
                raise ValueError('UA metadata differs from committed UA scores')
            artifact = self.state.output/metadata['selection_artifact']['path']
            if original.sha(artifact) != metadata['selection_artifact']['sha256']:
                raise ValueError('UA selection artifact hash changed')
            self.state._canonical(rate, ref)

    def prune(self):
        # Only this new observer's unreferenced cache; no historical artifacts.
        keep = {self.state._checked(ref[key]).resolve() for ref in self.state.references.values()
                for key in ('model', 'selection')}
        versions = self.state.output/'training_versions'
        if versions.is_dir():
            for path in versions.iterdir():
                if path.is_file() and path.resolve() not in keep:
                    path.unlink()
        self.state._epoch_models = {}


def state_factory(config):
    def factory(base, capture, output, variant, control):
        Parent = _original_state(base, capture, output, variant, control)

        class ObservedState(Parent):
            def bind(self, model, optimizer, **kwargs):
                super().bind(model, optimizer, **kwargs)
                self.ua = UASelection(base, output/'ua_best', self.identity)
                self.ua.bind(model, optimizer)
                self.lr_rows = []
                return self

            def restore(self):
                restored = super().restore()
                if restored is not None:
                    observer = restored['auxiliary_state']['cosine_observer']
                    self.ua.restore(observer['ua'])
                    self.lr_rows = observer['learning_rates']
                    if len(self.lr_rows) != restored['next_epoch'] or len(self.ua.rows) != restored['next_epoch']:
                        raise ValueError('Observer recovery epoch mismatch')
                return restored

            def commit_epoch(self, **kwargs):
                epoch = kwargs['next_epoch']
                self.ua.update(capture, epoch, config)
                rates = [float(g['lr']) for g in self.optimizer.param_groups]
                expected = expected_lrs(config['learning_rate'], config['epochs'], config['warmup_ratio'])[epoch-1]
                multipliers = {'backbone':config['backbone_lr_multiplier'],
                    'projector':config['projector_lr_multiplier'], 'classifier':config['classifier_lr_multiplier']}
                for group, actual in zip(self.optimizer.param_groups, rates):
                    multiplier = multipliers.get(group.get('name'), 1.)
                    if not math.isclose(actual, expected*multiplier, abs_tol=1e-15, rel_tol=1e-12):
                        raise ValueError('Actual optimizer LR differs from predeclared curve')
                self.lr_rows.append(dict(epoch=epoch, rates=rates))
                auxiliary = dict(kwargs.get('auxiliary_state') or {})
                auxiliary['cosine_observer'] = dict(ua=self.ua.state_dict(), learning_rates=self.lr_rows)
                kwargs['auxiliary_state'] = auxiliary
                super().commit_epoch(**kwargs)
                self.ua.prune()
                trace = json.loads((output/'MASK_TRACE.json').read_text())
                reference = json.loads((control/'MASK_TRACE.json').read_text())
                for phase in ('train','selection'):
                    left, right = trace[phase][str(epoch)], reference[phase][str(epoch)]
                    hashes = lambda x: x['sha256'] if phase == 'train' else {k:v['sha256'] for k,v in x.items()}
                    if hashes(left) != hashes(right):
                        raise ValueError(f'Paired {phase} masks differ at epoch {epoch}')
                progress = json.loads((output/'EPOCH_STATUS.json').read_text())
                progress.update(best_ua=original.aggregate(self.ua.scores), learning_rates=rates,
                    initialization_equal=True, paired_masks_equal_through_epoch=epoch)
                write_json(output/'EPOCH_STATUS.json', progress)
                write_json(output/'EVALUATION_HISTORY.json', self.ua.rows)
                write_json(output/'LEARNING_RATES.json', self.lr_rows)
        return ObservedState
    return factory


def worker(args):
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads(args.config.read_text())
        baseline = json.loads((args.control/'SUMMARY.json').read_text())
        if baseline['status'] != 'complete' or config != make_config(baseline['config']):
            raise ValueError('Require completed matched W0 and schedule-only config difference')
        original.make_config = lambda base, variant: make_config(base)
        original.state_class = state_factory(config)
        args.variant = 'C0'
        write_json(args.output/'ENTRY_PROVENANCE.json',dict(
            source_sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            wrapper_sha256=original.sha(Path(__file__)), control=str(args.control),
            config=config, authorized_runs=1, from_scratch=True))
        try:
            original.worker(args)
            path = args.output/'SUMMARY.json'
            summary = json.loads(path.read_text())
            progress = json.loads((args.output/'EPOCH_STATUS.json').read_text())
            if progress['epoch'] != 50 or progress['paired_masks_equal_through_epoch'] != 50:
                raise ValueError('Incomplete paired mask / learning-rate verification')
            summary.update(testbest_ua=progress['best_ua'],
                paired_masks_equal_through_epoch=50, initialization_equal=True,
                wrapper_sha256=original.sha(Path(__file__)),
                note='ACC-best and UA-best are separate selection tables, never mixed.')
            write_json(path,summary)
        except BaseException as error:
            path = args.output/'SUMMARY.json'
            summary = json.loads(path.read_text()) if path.exists() else {}
            summary.update(status='failed', error=repr(error))
            write_json(path,summary)
            raise


class CosineScheduler(Scheduler):
    def __init__(self, root, *, source, control):
        self.source, self.control = Path(source), Path(control)
        super().__init__(root,gpus=(5,),per_gpu=1,code_root=ROOT,
            smoke_costs={'Flat':dict(peak_memory_mib=4096,disk_bytes=3*2**30)})

    def command(self, job, config_path, gpu_uuid):
        return [self.python,'-u',str(Path(__file__)),'--worker','--config',str(config_path),
            '--output',job['output'],'--model-source',str(self.source),
            '--control',str(self.control),'--gpu-uuid',gpu_uuid]


def controller(args):
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Use --resume for this exact immutable queue, never overwrite')
    elif args.resume:
        raise FileNotFoundError('No existing queue to resume')
    baseline = json.loads((args.control/'SUMMARY.json').read_text())
    if baseline['status'] != 'complete' or original.sha(original.DATASET) != original.DATA_SHA:
        raise ValueError('Require completed W0 and accepted ECERC feature hash')
    if shutil.disk_usage(args.output.parent).free < 6*2**30:
        raise RuntimeError('Need 6GiB free disk for run and atomic save margin')
    args.output.mkdir(parents=True,exist_ok=True)
    scheduler = CosineScheduler(args.output/'queue',source=args.model_source,control=args.control)
    snapshot = scheduler.gpu_snapshot()
    if snapshot[5]['uuid'] != GPU_UUID:
        raise ValueError('Physical GPU5 UUID changed')
    config = make_config(baseline['config'])
    jobs = [dict(id='C0',method='Flat',config=config,output=str(args.output/'C0'))]
    write_json(args.output/'RUNNER.json',dict(status='running',pid=os.getpid(),started_unix=time.time(),
        authorized_jobs=1, gpu=snapshot[5], numerical_commit=original.HISTORICAL,
        control=str(args.control),selection='INTERNAL TEST-ORACLE SCREEN; ACC-best and UA-best separate'))
    rows = scheduler.run_jobs(jobs)
    results = {row['id']:row.get('summary',{}) for row in rows}
    ok = all(row['status'] == 'complete' for row in rows)
    write_json(args.output/'RESULTS.json',dict(status='complete' if ok else 'failed',
        results=results,constant_control=baseline,failures=[row.get('failures') for row in rows if row['status']!='complete']))
    meta = json.loads((args.output/'RUNNER.json').read_text())
    meta.update(status='complete' if ok else 'failed',finished_unix=time.time())
    write_json(args.output/'RUNNER.json',meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-source',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.worker and (args.config is None or args.gpu_uuid != GPU_UUID or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID):
        parser.error('Worker requires a config and UUID-bound healthy physical GPU5')
    (worker if args.worker else controller)(args)


if __name__ == '__main__':
    main()
