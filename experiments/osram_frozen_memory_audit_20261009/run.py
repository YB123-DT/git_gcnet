"""Frozen original OSRAM: train-only probe fitting and test-only interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABEL = 'INTERNAL DIAGNOSTIC ONLY; backbone and probes Test-oracle selected; not independent generalization'
GPUS = {2: 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4',
        3: 'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_cache(data, expected_ids):
    n = len(data['labels'])
    for name, shape in (('local', (n,256)), ('memory', (n,4,512)), ('availability', (n,3)),
                        ('labels',(n,)), ('prediction',(n,)), ('speaker',(n,)),
                        ('utterance_indices',(n,)), ('conversation_ids',(n,))):
        if data[name].shape != shape:
            raise ValueError(f'bad cache shape {name}: {data[name].shape}')
        if name != 'conversation_ids' and not np.isfinite(data[name]).all():
            raise ValueError(f'nonfinite cache {name}')
    ids = data['conversation_ids'].astype(str)
    indices = data['utterance_indices']
    if len(set(zip(ids,indices))) != n:
        raise ValueError('duplicate sample IDs')
    if set(ids) != set(expected_ids):
        raise ValueError('split coverage mismatch')
    a = data['availability']
    if not np.isin(a, (0,1)).all() or (a.sum(-1) == 0).any():
        raise ValueError('invalid availability')
    if np.count_nonzero(data['memory'][indices == 0]):
        raise ValueError('first utterance has historical reads')
    if np.count_nonzero(data['memory'][:,1:][a.astype(bool)]):
        raise ValueError('inactive Gap leakage')
    return n


def extract(model, loader, schedule, dimensions, device, expected_ids):
    import torch
    from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view
    from experiments.osram_history_drift_20261002.diagnostic import capture
    keys = ('local','memory','availability','labels','prediction','speaker',
            'utterance_indices','conversation_ids')
    pieces = {key: [] for key in keys}
    views = []
    digest = hashlib.sha256()
    loader.sampler.set_epoch(0)
    for raw in loader:
        view = _prepare_view(_move_batch(raw,device), schedule,0,dimensions)
        snapshot = capture(model, lambda: model([view['incomplete']],view['availability'],
            view['qmask'],view['umask'],view['lengths'],predict_missing=False))
        valid = view['umask'].T.bool()
        positions = valid.nonzero().cpu().tolist()
        def array(tensor):
            return tensor.detach().cpu().numpy()
        values = dict(local=array(snapshot['local'][valid]),
            memory=array(torch.cat((snapshot['base'].unsqueeze(-2),snapshot['gap']),dim=-2)[valid]),
            availability=array(view['availability'][valid]), labels=array(view['labels'].T[valid]),
            prediction=array(snapshot['prediction'][valid]).reshape(-1),speaker=array(view['qmask'].T[valid]),
            utterance_indices=np.asarray([t for t,b in positions],dtype=np.int64),
            conversation_ids=np.asarray([str(view['conversation_ids'][b]) for t,b in positions],dtype=str))
        for key in keys:
            pieces[key].append(values[key])
        for conversation, position, mask in zip(values['conversation_ids'],values['utterance_indices'],values['availability']):
            digest.update(json.dumps([conversation,int(position),mask.tolist()]).encode())
        # CPU copies avoid retaining all datasets on GPU while offline probes fit.
        views.append(({k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in view.items()},
                      {k:v.detach().cpu() for k,v in snapshot.items()}))
    result = {key: np.concatenate(value,axis=0) for key,value in pieces.items()}
    validate_cache(result,expected_ids)
    return result, views, digest.hexdigest()


def trim_check_view(view):
    """A real-checkpoint safety check only; never used by formal diagnostics."""
    n = min(2,len(view['lengths']))
    t = min(6,view['incomplete'].shape[0])
    output = dict(view)
    for k in ('complete','incomplete','availability'):
        output[k] = view[k][:t,:n]
    for k in ('labels','umask','qmask'):
        output[k] = view[k][:n,:t]
    output['lengths'] = [min(int(x),t) for x in view['lengths'][:n]]
    output['conversation_ids'] = view['conversation_ids'][:n]
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rates',type=float,nargs='+',default=[i/10 for i in range(8)])
    p.add_argument('--gpu',type=int,choices=tuple(GPUS),default=2)
    p.add_argument('--reference',type=Path,default=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66'))
    p.add_argument('--dataset-root',type=Path,default=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'))
    p.add_argument('--probe-epochs',type=int,default=100)
    p.add_argument('--check-only',action='store_true')
    args = p.parse_args()
    if len(set(args.rates)) != len(args.rates) or any(r not in [i/10 for i in range(8)] for r in args.rates):
        p.error('unique rates must lie on the original eight-rate grid')
    uuid = subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if uuid != GPUS[args.gpu] or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('explicit healthy host GPU UUID required; GPU4 forbidden')
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    args.output.mkdir(parents=True,exist_ok=False)
    state = dict(label=LABEL,status='running',pid=os.getpid(),started_utc=now(),gpu=args.gpu,
                 gpu_uuid=uuid,rates=args.rates,records=[],check_only=args.check_only,command=sys.argv)
    write(args.output/'STATUS.json',state)
    try:
        import torch
        from gcnet_modality_jepa.train_gcnet import get_loaders
        from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        from experiments.osram_readout_paths_20260930.run import state_hash, sha, split_ids, configuration_dict
        from experiments.osram_history_drift_20261002.diagnostic import capture
        from experiments.osram_frozen_memory_audit_20261009.probe import run_probes, metrics
        from experiments.osram_frozen_memory_audit_20261009.intervention import evaluate_interventions
        torch.set_num_threads(2)
        torch.manual_seed(66)
        configuration_dict(66,args.reference.parent)  # validation only; do not use its query-enabled return
        raw = json.loads((args.reference/'config.json').read_text())
        cfg = TrainConfig(**raw)
        if cfg.mosi_task_mode != 'regression' or cfg.task_regression_loss != 'mse':
            raise ValueError('original MOSI regression required')
        features = [str(args.dataset_root/'CMUMOSI/features'/name) for name in
                    ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        train,val,test,*dimensions = get_loaders(audio_root=features[0],text_root=features[1],
            video_root=features[2],num_folder=1,dataset='CMUMOSI',batch_size=cfg.batch_size,
            num_workers=0,seed=66,validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
        loaders = dict(train=train[0],test=test[0])
        ids = split_ids(dict(train=train[0],validation=val[0],test=test[0]))
        write(args.output/'SPLITS.json',dict(ids=ids,validation='metadata only; never iterated or fitted'))
        device = torch.device('cuda:0')
        model = _build_model(cfg,tuple(dimensions)).to(device).requires_grad_(False).eval()
        write(args.output/'SOURCE_CONFIG.json',raw)
        provenance = dict(label=LABEL,host=platform.node(),python=sys.version,torch=torch.__version__,
            source_config_sha256=sha(args.reference/'config.json'),gpu_uuid=uuid,
            source_metrics_sha256=sha(args.reference/'metrics.json'),numpy=np.__version__,
            data_manifest='/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json',
            data_manifest_sha256=sha(Path('/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json')),
            source_snapshot=json.loads((REPO/'SNAPSHOT.json').read_text())['code_commit'] if (REPO/'SNAPSHOT.json').exists() else None,
            label_sha256=sha(args.dataset_root/'CMUMOSI/CMUMOSI_features_raw_2way.pkl'),
            code_sha256={str(f.relative_to(REPO)):sha(f) for f in [*Path(__file__).parent.glob('*.py'),
                REPO/'gcnet_missing_m3/model.py',REPO/'gcnet_missing_m3/osram.py',REPO/'gcnet_missing_m3/train_gcnet.py']},
            model_frozen=True,mask_epoch=0,probe_gradient_split='train',probe_selection_split='test')
        write(args.output/'PROVENANCE.json',provenance)
        reference_metrics=json.loads((args.reference/'metrics.json').read_text())
        for rate in args.rates:
            destination=args.output/f'rate_{rate:.1f}'.replace('.','p')
            destination.mkdir()
            checkpoint=args.reference/f'best_miss_{rate:.1f}.pt'.replace('.','p',1)
            checkpoint_hash=sha(checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            checkpoint_epoch=int(saved['epoch'])
            model.load_state_dict(saved['model'],strict=True)
            model.eval()
            frozen_hash=state_hash(model)
            split_data,views,masks={},{},{}
            for split,loader in loaders.items():
                split_data[split],views[split],masks[split]=extract(model,loader,_schedules(cfg,split)[rate],tuple(dimensions),device,ids[split])
                np.savez_compressed(destination/f'{split}_features.npz',**split_data[split])
            original=metrics(split_data['test']['labels'],split_data['test']['prediction'])
            reference=reference_metrics['test'][str(rate)]['weighted_f1']
            if abs(original['weighted_f1_nonzero']-reference)>1e-10:
                raise AssertionError(f'original test W-F1 parity failed: {original} vs {reference}')
            del saved
            print(json.dumps(dict(rate=rate,phase='extraction_complete',original=original)),flush=True)
            probe=run_probes(split_data,destination/'probes',device='cpu',
                seeds=(66,) if args.check_only else (66,67,68),epochs=2 if args.check_only else args.probe_epochs)
            print(json.dumps(dict(rate=rate,phase='probes_complete',fits=len(probe['runs']))),flush=True)
            intervention_rows,coverage=[],[]
            for cpu_view,cpu_snapshot in views['test']:
                view={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in cpu_view.items()}
                snap={k:v.to(device) for k,v in cpu_snapshot.items()}
                if args.check_only:
                    view=trim_check_view(view)
                    snap=capture(model,lambda:model([view['incomplete']],view['availability'],view['qmask'],
                        view['umask'],view['lengths'],predict_missing=False))
                result=evaluate_interventions(model,view,tuple(dimensions),snap,rate,'test')
                intervention_rows.extend(result['rows'])
                coverage.append(result['coverage'])
                if args.check_only:
                    break
            write(destination/'interventions.json',dict(rows=intervention_rows,coverage=coverage))
            if state_hash(model)!=frozen_hash or sha(checkpoint)!=checkpoint_hash or any(p.requires_grad for p in model.parameters()):
                raise AssertionError('frozen model/source checkpoint changed')
            record=dict(rate=rate,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
                checkpoint_epoch=checkpoint_epoch,
                frozen_state_sha256=frozen_hash,frozen_unchanged=True,original_test=original,
                baseline_reference_weighted_f1=reference,baseline_parity_verified=True,
                model_frozen=True,backbone_seed=66,probe_selection_split='test',
                mask_sha256=masks,probe_fits=len(probe['runs']),intervention_pairs=len(intervention_rows),
                artifacts={str(f.relative_to(destination)):sha(f) for f in destination.rglob('*') if f.is_file()})
            write(destination/'RESULT.json',record)
            state['records'].append(record)
            write(args.output/'STATUS.json',state)
            print(json.dumps(dict(rate=rate,phase='complete',pairs=len(intervention_rows))),flush=True)
        state.update(status='complete',finished_utc=now())
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),finished_utc=now())
        raise
    finally:
        write(args.output/'STATUS.json',state)


if __name__=='__main__':
    main()
