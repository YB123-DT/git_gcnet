"""Frozen IEMOCAP4/6 information probes, never OSRAM/backbone training.

Replay original ACC-selected GCNet masks exactly. All model numerics come from
the immutable training checkout785a012; this experiment only fits small heads.
"""
from __future__ import annotations

import argparse
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
TRAINING_SHA = '785a01242ef1a82b97d5337acbb33ba963df5537'
BACKBONE_SEED = 66
PROBE_SEEDS = (66, 67, 68)
SOURCE = Path('/home/qinjinghui/remote_experiments/osram_iemocap_b6_source_785a012')
WEIGHTS = Path('/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016/runs')
DATA = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
GPUS = {0: 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
        5: 'GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62'}
LABEL_HASHES = {4: '072cffa9c528dd0536792de7ac85d5a0efaf25d4a9e3ed01f7e45453547cc9fd',
               6: 'e35143e9f087d436779cccd650cb0327e9caa55178d326330f4a8a914e91e023'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
    write_json(path, value)


def check_snapshot():
    path = ROOT/'SOURCE_MANIFEST.json'
    manifest = json.loads(path.read_text())
    for name, digest in manifest['files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError('Experiment source changed: '+name)
    return manifest


def validate_source_config(raw, classes, fold):
    expected = dict(dataset={4:'IEMOCAPFour', 6:'IEMOCAPSix'}[classes], seed=66,
        fold=fold, epochs=100, missing_mask_protocol='gcnet-random-mask-v1',
        training_objective='emotion-only', osram_output_dim=1600,
        osram_readout_fusion='flat', osram_disable_local_skip=False,
        osram_meaningful_block='none', osram_bidirectional=False, train_rate_mode='cyclic')
    for key, value in expected.items():
        if raw.get(key, False if key in ('osram_disable_local_skip','osram_bidirectional') else None) != value:
            raise ValueError(f'Original Flat config mismatch: {key}')


def activate_source(source):
    """Bind numerical packages before importing new diagnostic-only helpers."""
    source = source.resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'], cwd=source, text=True).strip() != TRAINING_SHA:
        raise ValueError('Require original immutable numerical source785a012')
    if subprocess.check_output(['git','diff','--name-only','HEAD','--',
                               'gcnet_missing_m3','gcnet_modality_jepa','config.py'],
                               cwd=source, text=True).strip():
        raise ValueError('Training source has tracked numerical edits')
    if any(name in sys.modules for name in ('config','gcnet_missing_m3','gcnet_modality_jepa')):
        raise ValueError('Numerical modules imported before source binding')
    sys.path.insert(0, str(source))
    try:
        for name in ('config','gcnet_missing_m3','gcnet_modality_jepa'):
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError('Numerical import escaped original checkout: '+name)
    finally:
        sys.path.remove(str(source))
    # Other historical top-level dependencies may live only in this checkout;
    # diagnostic helpers still resolve from ROOT ahead of this fallback.
    sys.path.append(str(source))
    return {str(p.relative_to(source)): sha(p) for folder in ('gcnet_missing_m3','gcnet_modality_jepa')
            for p in sorted((source/folder).glob('*.py'))}


def pack_capture(view, capture, logits, *, classes):
    """Return source-loader order; never sort before exact-selection checks."""
    import numpy as np
    import torch
    from experiments.osram_meld_information_availability_20261014.features import mask_forward_memory
    valid = view['umask'].bool()
    if ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError('Padding must follow the valid conversation prefix')
    local, flat = capture.local, capture.flat_input
    if local.shape != (*logits.shape[:2],256) or flat.shape != (*logits.shape[:2],4352):
        raise ValueError('Original cfg84 Flat widths changed')
    if logits.shape[-1] != classes:
        raise ValueError('Wrong number of IEMOCAP classes')
    torch.testing.assert_close(flat[..., :256],local,rtol=0,atol=0)
    def packed(tensor):
        return tensor.transpose(0,1)[valid].detach().cpu().numpy()
    base = packed(flat[...,256:1280])
    gap = packed(flat[...,1280:].reshape(*flat.shape[:2],3,1024))
    if np.any(base[...,512:]) or np.any(gap[...,512:]):
        raise ValueError('Unexpected nonzero backward Memory half')
    availability = packed(view['availability']).astype(np.int64)
    base, gap = mask_forward_memory(base,gap,availability)
    mask = valid.cpu().numpy()
    ids = np.asarray([str(v) for v in view['conversation_ids']])
    positions = np.broadcast_to(np.arange(valid.shape[1]),valid.shape)[mask]
    data = dict(local=packed(local), base=base, gap=gap, prediction=packed(logits),
        labels=view['labels'][valid].cpu().numpy().astype(np.int64), availability=availability,
        conversation_ids=np.broadcast_to(ids[:,None],valid.shape)[mask],
        utterance_indices=positions, speaker=view['qmask'][valid].cpu().numpy().astype(np.int64))
    if any(not np.isfinite(data[k]).all() for k in ('local','base','gap','prediction')):
        raise ValueError('Nonfinite frozen features')
    if np.any(base[positions==0]) or np.any(gap[positions==0]):
        raise ValueError('Read-before-write first utterance has nonzero Memory')
    return data


def assert_exact_replay(data, saved):
    import numpy as np
    for left, right in (('labels','labels'), ('availability','availability'),
                        ('conversation_ids','conversation_ids'),
                        ('utterance_indices','utterance_indices'), ('speaker','speakers')):
        np.testing.assert_array_equal(data[left],saved[right])
    np.testing.assert_array_equal(data['prediction'].argmax(1),saved['predictions'])
    np.testing.assert_allclose(data['prediction'],saved['logits'],rtol=2e-5,atol=2e-5)


def extract(model, loader, config, dimensions, rate, split, *, saved=None):
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view_from_primary_masks
    from gcnet_modality_jepa.gcnet_legacy_random_mask import random_mask
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import FlatReadoutCapture
    from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
        scatter_saved_availability, replay_view_from_availability)
    from experiments.osram_meld_information_availability_20261014.features import strict_past_summaries
    if split == 'test' and saved is None:
        raise ValueError('Exact selected masks are mandatory on test')
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Backbone must remain frozen in eval mode')
    # Separate offline training-view RNG. No test labels affect masking/donors.
    rng = np.random.RandomState(66000+100*config.fold+int(rate*10))
    parts, cursor = [], 0
    with torch.no_grad():
        for raw in loader:
            data = _move_batch(raw,next(model.parameters()).device)
            valid = data[7].bool()
            if saved is not None:
                availability, cursor = scatter_saved_availability(saved['availability'],cursor,data[7])
                view = replay_view_from_availability(data,availability,dimensions)
            else:
                batch,length = valid.shape
                masks = []
                for _ in ('host','guest'):
                    sampled = random_mask(3,batch*length,rate,rng=rng)
                    side = torch.as_tensor(sampled.reshape(batch,length,3).transpose(1,0,2),
                                           dtype=torch.float32,device=valid.device)
                    masks.append(torch.where(valid.T[...,None],side,torch.zeros_like(side)))
                view = _prepare_view_from_primary_masks(data,*masks,dimensions)
            with FlatReadoutCapture(model.osram) as capture:
                logits,_,_,completion = model([view['incomplete']],view['availability'],
                    view['qmask'],view['umask'],view['lengths'],predict_missing=False)
            if completion is not None or capture.local_calls != 1 or capture.adapter_calls != 1:
                raise ValueError('Require exactly one unchanged no-completion Flat scan')
            parts.append(pack_capture(view,capture,logits,classes=logits.shape[-1]))
    result = {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if saved is not None:
        if cursor != len(saved['labels']):
            raise ValueError('Exact selected masks not fully consumed')
        assert_exact_replay(result,saved)
    # Restore conversation order only AFTER original-loader identity parity.
    order = np.lexsort((result['utterance_indices'],result['conversation_ids']))
    result = {key:value[order] for key,value in result.items()}
    if len(set(zip(result['conversation_ids'],result['utterance_indices']))) != len(result['labels']):
        raise ValueError('Duplicate utterance identity')
    result.update(strict_past_summaries(result))
    return result


def fit_rate(train, test, output, classes):
    import numpy as np
    from experiments.osram_meld_information_availability_20261014.features import match_donors
    from experiments.osram_meld_information_availability_20261014.probe import (
        PROBE_PAIRS,build_probe_inputs,fit_probe)
    from experiments.osram_iemocap_b6_transfer_20261015.worker import class_metrics
    output.mkdir(parents=True,exist_ok=True)
    fits = []
    for seed in PROBE_SEEDS:
        donors = {name:match_donors(values,seed+1009*i)
                  for i,(name,values) in enumerate((('train',train),('test',test)))}
        inputs = {name:build_probe_inputs(values,donors[name]) for name,values in (('train',train),('test',test))}
        write(output/f'donors_seed{seed}.json',{name:dict(indices=donor.indices.tolist(),
            included=donor.included.tolist(), target_conversation=values['conversation_ids'].tolist())
            for name,donor,values in [('train',donors['train'],train),('test',donors['test'],test)]})
        cache = {}
        for variant in ('Local',*[name for pair in PROBE_PAIRS.items() for name in pair]):
            destination = output/f'seed_{seed}'/variant
            if not (destination/'RESULT.json').exists():
                fit_probe(inputs['train'][variant][donors['train'].included],
                    train['labels'][donors['train'].included],
                    inputs['test'][variant][donors['test'].included],
                    test['labels'][donors['test'].included],seed=seed,epochs=30,
                    destination=destination,device='cuda:0',batch_size=256,classes=classes)
            result = json.loads((destination/'RESULT.json').read_text())
            with np.load(destination/'predictions.npz',allow_pickle=False) as stored:
                values = {key:stored[key] for key in stored.files}
            np.testing.assert_array_equal(values['labels'],test['labels'][donors['test'].included])
            np.testing.assert_array_equal(values['predictions'],values['logits'].argmax(1))
            metrics = class_metrics(values['labels'],values['logits'],classes)
            if abs(metrics['ACC']/100-result['metrics']['accuracy']) > 1e-12:
                raise ValueError('Stored probe selection metrics disagree')
            # Include explicit IDs in each prediction artifact's sidecar; masks
            # and complete identities also remain in test_features.npz.
            identity = dict(conversation_ids=test['conversation_ids'][donors['test'].included].tolist(),
                utterance_indices=test['utterance_indices'][donors['test'].included].tolist(),
                train_N=int(donors['train'].included.sum()),test_N=metrics['N'],
                common_cohort=True,probe_seed=seed,classes=classes)
            write(destination/'IDENTITY.json',identity)
            row = dict(seed=seed,variant=variant,metrics=metrics,
                best_epoch=result['best_epoch'],parameters=result['parameter_count'],
                cross_vs_local=paired_counts(values['labels'],cache.get('Local',values['logits']),values['logits']))
            cache[variant] = values['logits']
            fits.append(row)
        for row in [r for r in fits if r['seed']==seed and r['variant'] in PROBE_PAIRS]:
            row['cross_vs_donor'] = paired_counts(test['labels'][donors['test'].included],
                cache[PROBE_PAIRS[row['variant']]],cache[row['variant']])
    result = dict(status='complete',fits=fits,probe_epochs=30,
        probe_seeds=list(PROBE_SEEDS),cohort='all variants use the same donor-included train/test rows')
    write(output/'PROBES.json',result)
    return result


def paired_counts(labels, before, after):
    import numpy as np
    left, right = np.asarray(before).argmax(1), np.asarray(after).argmax(1)
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or left.shape != y.shape or right.shape != y.shape:
        raise ValueError('Require aligned nonempty labels and prediction rows')
    a,b = left==y,right==y
    return dict(N=len(y),corrections=int((~a & b).sum()),harms=int((a & ~b).sum()),
                both_correct=int((a & b).sum()),both_wrong=int((~a & ~b).sum()))


def worker(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,GCNET_DATASET_ROOT=str(DATA),
                      GCNET_CACHE_ROOT=str(args.output/'feature_cache'))
    manifest = check_snapshot()
    numerical_hashes = activate_source(args.source_root)
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment,repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu,state_digest
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    environment = require_replication_environment(torch)
    gpu = guard_gpu(args.gpu_uuid)
    if GPUS.get(gpu['host_index']) != gpu['uuid']:
        raise ValueError('This queue allows only verified healthy GPU0/5')
    repair_uuid_rng(torch)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    classes,fold = args.classes,args.fold
    source = WEIGHTS/f'iemocap{classes}/seed_66/fold_{fold}'
    raw = json.loads((source/'config.json').read_text())
    validate_source_config(raw,classes,fold)
    source_summary = json.loads((source/'SUMMARY.json').read_text())
    if source_summary['status'] != 'complete' or source_summary['identity']['source_git_sha'] != TRAINING_SHA:
        raise ValueError('Incomplete or wrong source backbone')
    label = DATA/'IEMOCAP'/f'IEMOCAP_features_raw_{classes}way.pkl'
    if sha(label) != LABEL_HASHES[classes]:
        raise ValueError('IEMOCAP label source changed')
    config = TrainConfig(**raw)
    roots = [str(DATA/'IEMOCAP/features'/name)
             for name in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    identity = dict(source_git_sha=TRAINING_SHA,experiment_git_sha=manifest['git_sha'],
        experiment_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        numerical_hashes=numerical_hashes,label_sha256=LABEL_HASHES[classes],
        source_summary_sha256=sha(source/'SUMMARY.json'),source_config_sha256=sha(source/'config.json'))
    args.output.mkdir(parents=True,exist_ok=True)
    summary_path = args.output/'SUMMARY.json'
    summary = dict(status='running',dataset=config.dataset,fold=fold,backbone_seed=66,
        probe_seeds=list(PROBE_SEEDS),identity=identity,environment=environment,gpu=gpu,
        source=str(source),rates={},started_unix=time.time(),pid=os.getpid())
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous['identity'] != identity:
            raise ValueError('Cannot resume a different source or experiment')
        if previous['status']=='complete':
            return
        summary['rates'] = previous['rates']
    write(summary_path,summary)
    tick = time.monotonic()
    try:
        train_loaders,_,test_loaders,*dimensions = get_loaders(*roots,num_folder=5,dataset=config.dataset,
            batch_size=32,num_workers=0,seed=66,validation_fraction=config.validation_fraction,
            evaluation_protocol=config.evaluation_protocol)
        train_loader,test_loader = train_loaders[fold-1],test_loaders[fold-1]
        train_ids = {train_loader.dataset.vids[i] for i in train_loader.sampler.indices}
        test_ids = {test_loader.dataset.vids[i] for i in test_loader.sampler.indices}
        if train_ids & test_ids:
            raise ValueError('LOSO train/test conversations overlap')
        model = _build_model(config,dimensions,torch.device('cuda:0'))
        torch.cuda.reset_peak_memory_stats()
        for index in range(8):
            rate = index/10
            key = format(rate,'.1f')
            if summary['rates'].get(key,{}).get('status') == 'complete':
                continue
            destination = args.output/f'rate_0p{index}'
            destination.mkdir(parents=True,exist_ok=True)
            checkpoint_path = source/f'best_miss_0p{index}.pt'
            checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            reference = checkpoint['selection_artifact']
            selected_path = (source/reference['path']).resolve()
            if not selected_path.is_relative_to(source) or sha(selected_path) != reference['sha256']:
                raise ValueError('Selected artifact hash mismatch')
            with np.load(selected_path,allow_pickle=False) as saved:
                selected = {name:saved[name] for name in saved.files}
            if (int(selected['selected_epoch']) != checkpoint['epoch']
                    or hashlib.sha256(selected['availability'].tobytes()).hexdigest()!=reference['availability_sha256']):
                raise ValueError('Wrong selected epoch or exact mask hash')
            model.load_state_dict(checkpoint['model'],strict=True)
            before = state_digest(model)
            del checkpoint
            caches = {}
            for split,loader in (('train',train_loader),('test',test_loader)):
                path = destination/f'{split}_features.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as cached:
                        caches[split] = {name:cached[name] for name in cached.files}
                else:
                    caches[split] = extract(model,loader,config,dimensions,rate,split,
                        saved=selected if split=='test' else None)
                    np.savez_compressed(path,**caches[split])
            after = state_digest(model)
            if before != after or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError('Source model was modified during extraction')
            print(f'EXTRACTED classes={classes} fold={fold} rate={rate} exact-selected-mask',flush=True)
            fitted = fit_rate(caches['train'],caches['test'],destination/'probes',classes)
            if state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
                raise ValueError('Probe training modified the frozen backbone')
            summary['rates'][key] = dict(fitted,selected_epoch=int(selected['selected_epoch']),
                checkpoint_sha256=sha(checkpoint_path),selection_artifact=reference,
                features={split:sha(destination/f'{split}_features.npz') for split in caches},
                model_hash_before=before,model_hash_after=state_digest(model))
            write(summary_path,summary)
            print(f'COMPLETE classes={classes} fold={fold} rate={rate} fits={len(fitted["fits"])}',flush=True)
        summary['status']='complete'
    except BaseException as error:
        summary.update(status='failed',error=repr(error))
        raise
    finally:
        summary.update(seconds=time.monotonic()-tick,finished_unix=time.time(),
                       peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        write(summary_path,summary)


def controller(args):
    from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler
    from experiments.osram_iemocap_information_availability_20261018.analysis import aggregate,render_markdown
    manifest = check_snapshot()
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing output requires explicit --resume; never overwrite historical runs')
    if shutil.disk_usage(args.output.parent).free < 20*2**30:
        raise ValueError('Need20GiB output headroom')
    args.output.mkdir(parents=True,exist_ok=True)
    class ProbeScheduler(Scheduler):
        def command(self,job,config_path,gpu_uuid):
            return [self.python,'-u',str(Path(__file__)),'--worker',
                '--classes',str(job['config']['classes']),'--fold',str(job['config']['fold']),
                '--gpu-uuid',gpu_uuid,'--output',job['output'],'--source-root',str(args.source_root)]
    scheduler = ProbeScheduler(args.output/'queue',gpus=(0,5),per_gpu=2,code_root=ROOT,
        smoke_costs={'Probe':dict(peak_memory_mib=4096,disk_bytes=2*2**30)})
    devices = scheduler.gpu_snapshot()
    if any(devices[g]['uuid'] != uuid for g,uuid in GPUS.items()):
        raise ValueError('Healthy GPU identity changed')
    jobs = [dict(id=f'I{c}_fold{f}',method='Probe',config=dict(classes=c,fold=f,
        experiment_sha=manifest['git_sha']),output=str(args.output/f'iemocap{c}/fold_{f}'))
        for f in (5,1,2,3,4) for c in (4,6)]
    runner = dict(status='running',pid=os.getpid(),jobs=10,source_git_sha=TRAINING_SHA,
        experiment_git_sha=manifest['git_sha'],gpus=GPUS,per_gpu=2,started_unix=time.time(),
        scope='frozen backbone66;4/6classes;5folds;8rates;13probes;3probe initializations;30probe epochs')
    write(args.output/'RUNNER.json',runner)
    try:
        completed = scheduler.run_jobs(jobs)
        failures = {r['id']:r['failures'] for r in completed if r['status']!='complete'}
        if failures:
            runner.update(status='failed',failures=failures)
        else:
            report = aggregate([r['summary'] for r in completed])
            write(args.output/'FINAL_SUMMARY.json',report)
            (args.output/'RESULT.md').write_text(render_markdown(report))
            runner['status']='complete'
    except BaseException as error:
        runner.update(status='failed',error=repr(error))
        raise
    finally:
        runner['finished_unix']=time.time()
        write(args.output/'RUNNER.json',runner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=SOURCE)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--classes',type=int,choices=(4,6))
    parser.add_argument('--fold',type=int,choices=range(1,6))
    args=parser.parse_args()
    if args.worker and any(getattr(args,k) is None for k in ('gpu_uuid','classes','fold')):
        parser.error('Worker requires gpu-uuid/classes/fold')
    (worker if args.worker else controller)(args)


if __name__=='__main__':
    main()
