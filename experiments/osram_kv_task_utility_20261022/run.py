"""Frozen MOSI66: joint KV fit reduction versus skip-source-write task utility."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

RATES = tuple(str(i/10) for i in range(8))
DISTANCES = (1, 2, 4, 8)


def pattern(bits):
    return ''.join(name for name, active in zip('ATV', bits) if active)


def validate_mask_bank(banks, expected):
    actual = {rate:hashlib.sha256(b''.join(row['availability'].contiguous().numpy().tobytes()
              for row in rows)).hexdigest() for rate,rows in banks.items()}
    if actual != expected:
        raise ValueError('Original exact test mask bank changed')
    return actual


def setup(reference, gpu_uuid, output):
    """Load current protected Flat directly; no dependency on deleted legacy weights."""
    import torch
    import sklearn
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_router_3seed_20261019.worker import load_manifest
    from experiments.osram_mosi_recent_history_20261018 import run as data_source
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_mosi_memory_grid_20261022.config import candidate_config
    if (torch.__version__, np.__version__, sklearn.__version__) != ('2.2.2+cu121','1.24.3','1.3.2'):
        raise ValueError('Use the unchanged MOSI s0 environment')
    gpu = guard_gpu(gpu_uuid)
    summary = json.loads((reference/'SUMMARY.json').read_text())
    cfg = trainer.TrainConfig(**summary['identity']['config'])
    if (cfg.dataset != 'CMUMOSI' or cfg.seed != 66 or
            asdict(candidate_config(cfg,'baseline')) != asdict(cfg)):
        raise ValueError('Require original seed66 LargeFlat,step0.6,head64')
    selected = summary['best_by_rate']['0.3']
    checkpoint = reference/selected['checkpoint']
    if h.sha(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Protected checkpoint hash changed')
    if h.sha(reference/selected['predictions']) != selected['prediction_sha256']:
        raise ValueError('Protected original prediction hash changed')
    manifest, proof = load_manifest(data_source.DATA, data_source.DATA_SHA)
    for name,digest in manifest['files'].items():
        if h.sha(name) != digest:
            raise ValueError('MOSI data hash changed: '+name)
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(output.parent/'feature_cache'/'CMUMOSI')
    train,dev,test,*dims = trainer.get_loaders(*manifest['feature_roots'],dataset='CMUMOSI',
        num_folder=1,batch_size=cfg.batch_size,num_workers=0,seed=66,
        validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
    splits = {name:loader.protocol_metadata for name,loader in
              [('train',train[0]),('validation',dev[0]),('test',test[0])]}
    if splits != summary['identity']['split']:
        raise ValueError('Data split metadata differs from protected reference')
    model = _build_model(cfg,tuple(dims),torch.device('cuda:0'))
    saved = torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(saved['model'],strict=True)
    model.eval().requires_grad_(False)
    masks = torch.load(reference/'fixed_masks.pt',map_location='cpu')['test']
    mask_hashes = validate_mask_bank(masks,summary['identity']['mask_hashes']['test'])
    return dict(model=model,trainer=trainer,loader=test[0],dimensions=tuple(dims),cfg=cfg,
        masks=masks,summary=summary,reference=reference,identity=dict(
        checkpoint=str(checkpoint),checkpoint_sha256=h.sha(checkpoint),selected_rate='0.3',
        selected_epoch=selected['epoch'],fixed_masks_sha256=h.sha(reference/'fixed_masks.pt'),
        test_mask_hashes=mask_hashes,prediction_sha256=selected['prediction_sha256'],
        data_files=manifest['files'],data_manifest=proof,split=splits,config=asdict(cfg),gpu=gpu,
        environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,
                         sklearn=sklearn.__version__)))


def run_rate(bundle, rate, stream, *, max_batches=None):
    import torch
    from .core import kv_quality, omit_write_forks
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018.runner_helpers import forward, isolated_rng
    from experiments.osram_meld_write_retention_20261014.run import WriteCapture
    from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture, replay_task_logits
    model,trainer,loader = (bundle[k] for k in ('model','trainer','loader'))
    loader.sampler.set_epoch(0)
    selected = bundle['summary']['best_by_rate']['0.3']
    with np.load(bundle['reference']/selected['predictions'],allow_pickle=False) as saved:
        saved = {k:saved[k].copy() for k in saved.files}
    count = cursor = batches = 0
    zero_energy = 0
    max_state_error = max_prediction_error = max_direct_omit_error = 0.
    with isolated_rng(66+99193),torch.no_grad():
        for index,raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            data = trainer._move_batch(raw,next(model.parameters()).device)
            meta = metadata(data)
            bank = bundle['masks'][rate][index]
            for k,value in meta.items():
                np.testing.assert_array_equal(value,bank[k])
            a = bank['availability'].to(data[7].device)
            view = trainer._prepare_view_from_primary_masks(data,a,a,bundle['dimensions'])
            with WriteCapture(model.osram,view['umask']) as capture,TaskCapture(model.osram) as target:
                prediction = forward(model,view)
            if target.calls != 1 or len(capture.snapshots) != len(capture.times):
                raise RuntimeError('Expected one complete factual causal scan')
            valid = view['umask'].T.bool()
            if bool(a[~valid].any()) or bool((a.sum(-1)[valid]<1).any()):
                raise ValueError('Saved mask violates padding/nonempty observed-set rules')
            # A normal uncaptured model forward also guards hook transparency.
            torch.testing.assert_close(forward(model,view),prediction,rtol=0,atol=0)
            take = prediction.transpose(0,1)[view['umask'].bool()].cpu().numpy().reshape(-1)
            if rate == '0.3':
                n = len(take)
                for k,value in meta.items():
                    np.testing.assert_array_equal(value,saved[k][cursor:cursor+n])
                np.testing.assert_array_equal(a.transpose(0,1)[view['umask'].bool()].cpu().numpy(),
                                             saved['availability'][cursor:cursor+n])
                np.testing.assert_allclose(take,saved['prediction'][cursor:cursor+n].reshape(-1),rtol=1e-6,atol=1e-6)
                np.testing.assert_array_equal(take>0,saved['prediction'][cursor:cursor+n].reshape(-1)>0)
            cursor += len(take)
            mask_hash = hashlib.sha256(a.cpu().contiguous().numpy().tobytes()).hexdigest()
            av = a.bool().cpu().numpy();labels = view['labels'].cpu().numpy()
            snapshots = capture.snapshots; time_index = {v[0]:i for i,v in enumerate(snapshots)}
            # One real GPU full-model omit scan checks state replay independently.
            direct = None
            if index == 0:
                original = model.osram.block_write
                calls = [0]
                def skip_first(memory,*args,**kwargs):
                    t = calls[0];calls[0] += 1
                    return memory if t == 0 else original(memory,*args,**kwargs)
                model.osram.block_write = skip_first
                try:
                    direct = forward(model,view)
                finally:
                    del model.osram.block_write
                if not torch.equal(prediction[0],direct[0]):
                    raise RuntimeError('Source read changed when its subsequent write was omitted')
            for source_index,(t,pre,k,v,observed,post) in enumerate(snapshots):
                scores = kv_quality(model.osram,pre,post,k,v,observed)
                zero_energy += int(((scores['value_energy'] <= 1e-12)&valid[t]).sum())
                score_arrays = {name:val.cpu().numpy() for name,val in scores.items()}
                for distance,states,state_error in omit_write_forks(model.osram,snapshots,source_index):
                    j = t+distance
                    current_k = snapshots[time_index[j]][2]
                    keep,omit = [replay_task_logits(model,target.local[j],states[name],target.queries[j],
                        current_k,a[j],valid[j]).squeeze(-1) for name in ('keep','omit')]
                    actual = prediction[j,:,0]
                    torch.testing.assert_close(keep[valid[j]],actual[valid[j]],rtol=2e-5,atol=2e-5)
                    if not torch.equal(keep[valid[j]]>0,actual[valid[j]]>0):
                        raise RuntimeError('Factual replay changed a polarity')
                    max_state_error = max(max_state_error,state_error)
                    max_prediction_error = max(max_prediction_error,float((keep[valid[j]]-actual[valid[j]]).abs().max()))
                    if source_index == 0 and direct is not None:
                        torch.testing.assert_close(omit[valid[j]],direct[j,valid[j],0],rtol=2e-5,atol=2e-5)
                        max_direct_omit_error = max(max_direct_omit_error,float((omit[valid[j]]-direct[j,valid[j],0]).abs().max()))
                    lp = torch.stack((keep,omit),-1).cpu().numpy()
                    for b in (valid[t]&valid[j]&(scores['value_energy']>1e-12)).nonzero().flatten().tolist():
                        y = float(labels[b,j]);pk,po = map(float,lp[b])
                        lk,lo = (pk-y)**2,(po-y)**2
                        goodk,goodo = (pk>0)==(y>0),(po>0)==(y>0)
                        row = dict(conversation_id=str(view['conversation_ids'][b]),rate=rate,
                            source_t=t,target_t=j,distance=distance,source_pattern=pattern(av[t,b]),
                            target_pattern=pattern(av[j,b]),label=y,prediction_keep=pk,prediction_omit=po,
                            loss_keep=lk,loss_omit=lo,utility=lo-lk,polarity_eligible=y!=0,
                            correction=bool(y!=0 and goodk and not goodo),
                            harm=bool(y!=0 and goodo and not goodk),
                            mask_sha256=mask_hash,**{name:float(value[b]) for name,value in score_arrays.items()})
                        stream.write(json.dumps(row,allow_nan=False)+'\n');count += 1
            stream.flush()
            batches += 1
            print(f'rate={rate} batch={index+1} valid={int(valid.sum())} rows={count} '
                  f'state_parity={max_state_error:.3g} prediction_parity={max_prediction_error:.3g}',flush=True)
    if max_batches is None and (cursor != len(saved['labels']) or batches != len(bundle['masks'][rate])):
        raise RuntimeError('Full test saved predictions/mask bank not completely consumed')
    return dict(rows=count,utterances=cursor,batches=batches,full_test_coverage=max_batches is None,
        zero_energy_writes=zero_energy,
        max_state_parity_error=max_state_error,max_prediction_parity_error=max_prediction_error,
        max_direct_omit_parity_error=max_direct_omit_error,
        source_current_prediction_unchanged=True,exact_mask_metadata=True,
        saved_miss03_prediction_parity=rate=='0.3')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_tasr_20261018.runner_helpers import backbone_digest,write_json,sha
    if args.output.exists():
        raise FileExistsError('Use a fresh diagnostic directory')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__),Path(__file__).with_name('core.py'),Path(__file__).with_name('report.py')]
    sources += [root/name for name in ('gcnet_missing_m3/osram.py','gcnet_missing_m3/model.py',
        'gcnet_missing_m3/train_gcnet.py','gcnet_modality_jepa/gcnet_legacy_random_mask.py',
        'experiments/osram_meld_write_retention_20261014/run.py',
        'experiments/osram_meld_write_retention_20261014/task_replay.py',
        'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py',
        'experiments/osram_tasr_20261018/runner_helpers.py')]
    state = dict(status='preflight',source_commit=args.source_commit,pid=os.getpid(),
        server='biggpu',new_training_steps=0,distances=DISTANCES,smoke=args.smoke,
        label='INTERNAL DIAGNOSTIC ONLY',source_hashes={str(path.relative_to(root)):sha(path) for path in sources})
    write_json(args.output/'STATUS.json',state);tick=time.monotonic()
    try:
        torch.set_num_threads(1)
        bundle = setup(args.reference,args.gpu_uuid,args.output)
        model = bundle['model'];before = backbone_digest(model)
        state.update(status='running',identity=bundle['identity'],model_digest=before)
        write_json(args.output/'STATUS.json',state)
        torch.cuda.reset_peak_memory_stats()
        with gzip.open(args.output/'ROWS.jsonl.gz','wt') as stream:
            state['rates'] = {}
            for rate in (('0.3',) if args.smoke else RATES):
                state['rates'][rate] = run_rate(bundle,rate,stream,max_batches=1 if args.smoke else None)
                write_json(args.output/'STATUS.json',state)
        if backbone_digest(model)!=before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen diagnostic changed model parameters or gradients')
        from .report import main as report_main
        # Report API is also exposed through its standalone CLI.
        report_main(['--rows',str(args.output/'ROWS.jsonl.gz'),'--output',str(args.output)])
        state.update(status='complete',weights_unchanged_exact=True,
            peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
            seconds=time.monotonic()-tick,rows_sha256=sha(args.output/'ROWS.jsonl.gz'))
    except BaseException as error:
        state.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        write_json(args.output/'STATUS.json',state)


if __name__=='__main__':
    main()
