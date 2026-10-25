"""Frozen N10 input-correction scale replay, no training."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/data2/yb/remote_experiments/osram_mosei_flat128_nested_20261024/execution_v1/runs/n10')
FLAT = Path('/data2/yb/remote_experiments/osram_mosei_capacity_cmerc_20261024/execution_v1/runs/f128')
GPU = 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'
SCALES = (0., .5, 1.)
FINE_SCALES = (.1, .2, 1.)
SLOTS = ('Local', 'Base', 'Gap-A', 'Gap-T', 'Gap-V')
RATES = tuple(f'{i/10:.1f}' for i in range(8))


def scaled_inputs(raw, corrected, gamma):
    if gamma not in SCALES or len(raw) != 3 or len(corrected) != 3:
        raise ValueError('Only the approved three-slot gamma0/.5/1 replay is allowed')
    if any(a.shape != b.shape for a,b in zip(raw, corrected)):
        raise ValueError('Correction cannot change a slot shape')
    if gamma == 0.:
        return raw
    if gamma == 1.:
        return corrected
    return tuple(a+gamma*(b-a) for a,b in zip(raw, corrected))


def make_cases(per_slot=False, fine=False):
    if fine and not per_slot:
        raise ValueError('Fine strengths are only approved one slot at a time')
    if not per_slot:
        return {str(g):(None,g) for g in SCALES}
    scales = FINE_SCALES if fine else SCALES
    return {'1.0':(None,1.), **{f'{slot}_{g}':(slot,g)
        for slot in SLOTS for g in scales if g!=1.}}


def scale_one_slot(raw, corrected, slot, gamma):
    if slot not in SLOTS or gamma not in (*SCALES, *FINE_SCALES):
        raise ValueError('Unknown slot or coefficient outside the approved grid')
    if len(raw)!=3 or len(corrected)!=3 or any(a.shape!=b.shape for a,b in zip(raw,corrected)):
        raise ValueError('Correction cannot change a slot shape')
    if gamma==1.:
        return corrected
    result=list(corrected)
    if slot in ('Local','Base'):
        index=SLOTS.index(slot)
        result[index]=raw[index] if gamma==0. else raw[index]+gamma*(corrected[index]-raw[index])
    else:
        index=SLOTS.index(slot)-2
        result[2]=corrected[2].clone() if hasattr(corrected[2],'clone') else corrected[2].copy()
        before,after=raw[2][:,:,index],corrected[2][:,:,index]
        result[2][:,:,index]=before if gamma==0. else before+gamma*(after-before)
    return tuple(result)


def paired_counts(labels, original, prediction, mask=None):
    valid = labels != 0
    if mask is not None:
        valid &= mask
    old = (original[valid]>0) == (labels[valid]>0)
    new = (prediction[valid]>0) == (labels[valid]>0)
    return dict(N=int(valid.sum()), corrections=int((~old&new).sum()),
        harms=int((old&~new).sum()), both_correct=int((old&new).sum()),
        both_wrong=int((~old&~new).sum()), N_positive=int((labels[valid]>0).sum()),
        N_negative=int((labels[valid]<0).sum()))


def aggregate(rates):
    if set(rates) != set(RATES):
        raise ValueError('Require all eight missing rates')
    return {scope:{m:float(np.mean([rates[r][m] for r in rs])) for m in ('ACC','WF1','MSE')}
            for scope,rs in (('mean',RATES),('high',RATES[5:]))}


def validate_checkpoint_identity(state, selected, expected_config):
    # torch.save retains dataclass tuples; the protected SUMMARY is JSON.
    actual = json.loads(json.dumps(state['identity']['config']))
    if state['epoch'] != selected['epoch'] or actual != expected_config:
        raise ValueError('Wrong selected checkpoint identity')


def report(output, summary):
    if summary.get('mode')=='per-slot':
        return report_slots(output,summary)
    lines = ['# Frozen N10 correction strength', '', 'INTERNAL DIAGNOSTIC ONLY',
        'Same completed N10 checkpoints, exact accepted masks, no training. Gamma0 is NOT separately-trained Flat.',
        'Original LocalSkip and CHM stay unchanged; gamma scales all five task-input correction slots together.', '',
        '| Gamma | ACC | W-F1 | High W-F1 | Delta WF1 vs gamma1 |', '|---|---:|---:|---:|---:|']
    if 'aggregate' in summary:
        ref=summary['aggregate']['1.0']['mean']['WF1']
        for gamma in SCALES:
            a=summary['aggregate'][str(gamma)]
            lines.append(f"| {gamma} | {a['mean']['ACC']:.6f} | {a['mean']['WF1']:.6f} | {a['high']['WF1']:.6f} | {a['mean']['WF1']-ref:+.6f} |")
    lines += ['', 'State: '+summary['status'], 'Completed rates: '+', '.join(summary['rates']),
        'Per-rate TEST-best W-F1 checkpoints are held fixed; no coefficient-dependent epoch selection.',
        'Test coefficient comparison is exploratory, not independent deployable performance evidence.',
        'Paired counts across rates repeat utterances and are not weighted-F1 differences.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    with (output/'PER_RATE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['rate','gamma','ACC','WF1','MSE'])
        writer.writeheader()
        writer.writerows(dict(rate=r,gamma=g,**v['metrics']) for r,item in summary['rates'].items()
                         for g,v in item['scales'].items())


def report_slots(output,summary):
    lines=['# Frozen N10 one-slot correction strength','','INTERNAL DIAGNOSTIC ONLY',
        'Exactly one slot changes; all other corrections stay at1. LocalSkip and CHM are untouched.',
        'Eleven unique readouts per scan:one common full plus five slots times two non-full strengths.','',
        '| Slot | Gamma | ACC | W-F1 | High W-F1 | Delta WF1 vs full |',
        '|---|---:|---:|---:|---:|---:|']
    if 'aggregate' in summary:
        reference=summary['aggregate']['1.0']['mean']['WF1']
        for slot in SLOTS:
            for g in summary.get('scales',SCALES):
                case='1.0' if g==1. else f'{slot}_{g}'
                a=summary['aggregate'][case]
                lines.append(f"| {slot} | {g} | {a['mean']['ACC']:.6f} | {a['mean']['WF1']:.6f} | {a['high']['WF1']:.6f} | {a['mean']['WF1']-reference:+.6f} |")
    lines+=['','State:'+summary['status'],'Completed rates:'+','.join(summary['rates']),
        'Fixed per-rate Test-best checkpoints/exact masks. No retraining or combined-coefficient oracle.',
        'Any test-selected preferred coefficient is exploratory, not independent deployment evidence.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    cases=summary.get('cases') or {k:dict(slot=s,gamma=g) for k,(s,g) in make_cases(True).items()}
    with (output/'PER_RATE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['rate','case','slot','gamma','ACC','WF1','MSE'])
        writer.writeheader()
        writer.writerows(dict(rate=r,case=case,slot=cases[case]['slot'] or 'all_full',
            gamma=cases[case]['gamma'],**v['metrics']) for r,item in summary['rates'].items()
            for case,v in item['scales'].items())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--per-slot',action='store_true',help='Change one correction slot at a time, others fixed at1')
    parser.add_argument('--fine-slots',action='store_true',help='With --per-slot, replay gamma0.1/0.2 instead of0/0.5')
    args=parser.parse_args(argv)
    if args.fine_slots and not args.per_slot:
        parser.error('--fine-slots requires --per-slot')
    cases=make_cases(args.per_slot,fine=args.fine_slots)
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_nested_cmerc_tuning_20261023.run import verify_source_snapshot
    from experiments.osram_mosei_flat128_nested_20261024.worker import setup
    from experiments.osram_nested_delta_off_missing_text_20261010.readout import flat_prediction
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required; preserve prior evidence')
    manifest=verify_source_snapshot(ROOT,args.source_commit)
    free=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'],text=True).strip())
    if free<6144:
        raise RuntimeError('Insufficient free GPU1 memory; do not disturb training')
    source=json.loads((SOURCE/'SUMMARY.json').read_text())
    baseline=json.loads((FLAT/'SUMMARY.json').read_text())
    if source['status']!='complete' or source['epoch']!=60 or source['method']!='mosei_f128_n10':
        raise ValueError('Require the completed original N10 seed66 result')
    for name,digest in source['identity']['source_hashes'].items():
        if name.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/')) and h.sha(ROOT/name)!=digest:
            raise ValueError('Checkpoint numerical source differs: '+name)
    args.output.mkdir(parents=True)
    tick=time.monotonic()
    summary=dict(status='preflight',label='INTERNAL DIAGNOSTIC ONLY',source_commit=args.source_commit,
        source_manifest_sha256=manifest['manifest_sha256'],pid=os.getpid(),gpu_uuid=GPU,
        source=str(SOURCE),source_summary_sha256=h.sha(SOURCE/'SUMMARY.json'),
        flat_reference=str(FLAT),flat_summary_sha256=h.sha(FLAT/'SUMMARY.json'),
        original_flat_aggregate=baseline['aggregate']['router'],scales=FINE_SCALES if args.fine_slots else SCALES,rates={},training_updates=0,
        mode='per-slot' if args.per_slot else 'joint',
        cases={key:dict(slot=slot,gamma=g) for key,(slot,g) in cases.items()})
    h.write_json(args.output/'SUMMARY.json',summary)
    try:
        torch.set_num_threads(1)
        bundle,plain=setup('CMUMOSEI','mosei_f128_n10',GPU,args.output/'data',seed=66,fold=1,variant='N10')
        del plain
        model=bundle['model'].eval().requires_grad_(False)
        trainer,loader=bundle['trainer'],bundle['test_loader']
        cfg=json.loads(json.dumps(bundle['cfg'].__dict__))
        if cfg!=source['config'] or bundle['environment']!=source['environment']:
            raise ValueError('Effective configuration/environment differs from completed N10')
        summary.update(status='running',environment=bundle['environment'],config=cfg,
                       gpu=bundle['gpu'],data=bundle['data'],mask_reference=bundle['mask_reference'])
        scans=[0]
        handle=model.osram.register_forward_hook(lambda *unused:scans.__setitem__(0,scans[0]+1))
        try:
            for rate in RATES:
                selected=source['best_by_rate'][rate]
                ckpt=SOURCE/selected['checkpoint'];prediction=SOURCE/selected['predictions']
                fp=FLAT/baseline['best_by_rate'][rate]['predictions']
                if h.sha(ckpt)!=selected['checkpoint_sha256'] or h.sha(prediction)!=selected['prediction_sha256']:
                    raise ValueError('Protected N10 checkpoint/prediction hash changed')
                if h.sha(fp)!=baseline['best_by_rate'][rate]['prediction_sha256']:
                    raise ValueError('Original Flat prediction hash changed')
                state=torch.load(ckpt,map_location='cpu',weights_only=False)
                validate_checkpoint_identity(state,selected,source['config'])
                model.load_state_dict(state['model'],strict=True);del state
                before=h.backbone_digest(model)
                with np.load(prediction,allow_pickle=False) as stored:
                    expected={k:stored[k].copy() for k in stored.files}
                with np.load(fp,allow_pickle=False) as stored:
                    flat={k:stored[k].copy() for k in ('labels','availability','conversation_ids','utterance_indices','prediction')}
                for key in ('labels','availability','conversation_ids','utterance_indices'):
                    np.testing.assert_array_equal(flat[key],expected[key])
                loader.sampler.set_epoch(0)
                pieces={key:[] for key in cases};cursor=0;start_scans=scans[0];max_error=0.
                with h.isolated_rng(66+99193),torch.no_grad():
                    for index,raw in enumerate(loader):
                        data=trainer._move_batch(raw,torch.device('cuda:0'));meta=metadata(data);n=len(meta['labels'])
                        for key,value in meta.items():
                            np.testing.assert_array_equal(value,expected[key][cursor:cursor+n])
                        record=bundle['masks'][rate][index]
                        for key,value in meta.items():np.testing.assert_array_equal(value,record[key])
                        av=record['availability'].to(data[7].device)
                        view=trainer._prepare_view_from_primary_masks(data,av,av,bundle['dimensions'])
                        np.testing.assert_array_equal(av.transpose(0,1)[data[7].bool()].cpu().numpy(),expected['availability'][cursor:cursor+n])
                        capture={}
                        def pre(_module,inputs):
                            if capture:raise RuntimeError('More than one Nested call per scan')
                            capture['raw']=tuple(x.detach().clone() for x in inputs)
                        def post(_module,_inputs,outputs):capture['corrected']=tuple(x.detach().clone() for x in outputs)
                        hooks=(model.osram.meaningful_block.register_forward_pre_hook(pre),
                               model.osram.meaningful_block.register_forward_hook(post))
                        try:
                            direct=h.forward(model,view).squeeze(-1)
                        finally:
                            for hook in hooks:hook.remove()
                        original=capture['raw'][:3];corrected=capture['corrected']
                        for x,y in zip(original[1:],corrected[1:]):
                            if not torch.equal(x[...,512:],y[...,512:]):raise RuntimeError('Backward half changed')
                        rng=torch.cuda.get_rng_state().clone()
                        valid=data[7].bool()
                        for case,(slot,gamma) in cases.items():
                            readout=(scaled_inputs(original,corrected,gamma) if slot is None else
                                     scale_one_slot(original,corrected,slot,gamma))
                            current=flat_prediction(model,*capture['raw'],readout=readout)
                            values=current.T[valid].cpu().numpy()
                            if not np.isfinite(values).all():raise RuntimeError('Nonfinite replay')
                            if case=='1.0':
                                reference=direct.T[valid].cpu().numpy()
                                np.testing.assert_allclose(values,reference,atol=1e-6,rtol=1e-6)
                                np.testing.assert_array_equal(values>0,reference>0)
                                max_error=max(max_error,float(abs(values-reference).max()))
                            pieces[case].append(values)
                        if not torch.equal(rng,torch.cuda.get_rng_state()):raise RuntimeError('Readout replay changed CUDA RNG')
                        cursor+=n
                if cursor!=len(expected['labels']) or scans[0]-start_scans!=len(loader):
                    raise RuntimeError('Incomplete samples or extra Memory scans')
                arrays={g:np.concatenate(v) for g,v in pieces.items()}
                np.testing.assert_allclose(arrays['1.0'],expected['prediction'].reshape(-1),atol=1e-6,rtol=1e-6)
                np.testing.assert_array_equal(arrays['1.0']>0,expected['prediction'].reshape(-1)>0)
                y=expected['labels'];flat_pred=flat['prediction'].reshape(-1);av=expected['availability'];aonly=(av==[1,0,0]).all(-1)
                cohorts=dict(all=np.ones(len(y),bool),A_only=aonly,A_only_positive=aonly&(y>0),
                    A_only_negative=aonly&(y<0),A_only_positive_near_flat=aonly&(y>0)&(abs(flat_pred)<=.25))
                results={}
                for g,pred in arrays.items():
                    m=trainer._metrics('CMUMOSEI',y,pred,'regression')
                    metrics=dict(ACC=100*m['accuracy'],WF1=100*m['weighted_f1'],MSE=float(np.mean((pred-y)**2)))
                    file=args.output/f'gamma_{g.replace(".","p")}_miss_{rate.replace(".","p")}.npz'
                    np.savez_compressed(file,labels=y,prediction=pred,availability=av,
                        conversation_ids=expected['conversation_ids'],utterance_indices=expected['utterance_indices'])
                    results[g]=dict(metrics=metrics,prediction_sha256=h.sha(file),
                        paired_vs_gamma1={name:paired_counts(y,arrays['1.0'],pred,mask) for name,mask in cohorts.items()},
                        paired_vs_flat66={name:paired_counts(y,flat_pred,pred,mask) for name,mask in cohorts.items()})
                if h.backbone_digest(model)!=before or h.sha(ckpt)!=selected['checkpoint_sha256']:
                    raise RuntimeError('Frozen parameters or original checkpoint changed')
                summary['rates'][rate]=dict(selected_epoch=selected['epoch'],checkpoint_sha256=selected['checkpoint_sha256'],
                    saved_prediction_sha256=selected['prediction_sha256'],scales=results,
                    memory_scans=scans[0]-start_scans,max_gamma1_direct_error=max_error)
                summary.update(seconds=time.monotonic()-tick,total_memory_scans=scans[0])
                h.write_json(args.output/'SUMMARY.json',summary);report(args.output,summary)
                print(json.dumps(dict(rate=rate,scales={g:v['metrics'] for g,v in results.items()})),flush=True)
        finally:
            handle.remove()
        summary['aggregate']={case:aggregate({r:v['scales'][case]['metrics'] for r,v in summary['rates'].items()}) for case in cases}
        summary.update(status='complete',seconds=time.monotonic()-tick,all_parameters_unchanged=True,
                       original_predictions_reproduced=True,one_memory_scan_per_batch=True)
        if verify_source_snapshot(ROOT,args.source_commit)!=manifest:raise RuntimeError('Immutable source changed')
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(args.output/'SUMMARY.json',summary);report(args.output,summary)


if __name__=='__main__':
    main()
