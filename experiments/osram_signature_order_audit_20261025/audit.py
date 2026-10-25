"""Frozen MOSI Signature mechanism interventions; no training or new selection."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


RATES = tuple(f'{i/10:.1f}' for i in range(8))


def validate_rates(values):
    if not values or len(set(values)) != len(values) or any(v not in RATES for v in values):
        raise ValueError('Distinct rates0.0 through0.7 are required')
    return list(values)


def paired_metrics(labels, candidate, reference):
    from sklearn.metrics import accuracy_score, f1_score
    y, p, r = [np.asarray(v).reshape(-1) for v in (labels, candidate, reference)]
    if not (len(y) == len(p) == len(r)) or not all(np.isfinite(v).all() for v in (y,p,r)):
        raise ValueError('Aligned finite labels/predictions required')
    valid = y != 0
    n = int(valid.sum())
    result = dict(N=len(y), N_polarity=n, ACC=None, WF1=None, MSE=None,
                  delta_MSE=None, corrections_vs_reference=0, harms_vs_reference=0,
                  polarity_flips=0, mean_abs_prediction_change=None)
    if len(y):
        result.update(MSE=float(np.mean((y-p)**2)),
            delta_MSE=float(np.mean((y-p)**2-(y-r)**2)),
            mean_abs_prediction_change=float(np.mean(np.abs(p-r))))
    if n:
        truth, new, old = y[valid]>0, p[valid]>0, r[valid]>0
        result.update(ACC=100*float(accuracy_score(truth,new)),
            WF1=100*float(f1_score(truth,new,average='weighted',zero_division=0)),
            corrections_vs_reference=int(((new==truth)&(old!=truth)).sum()),
            harms_vs_reference=int(((new!=truth)&(old==truth)).sum()),
            polarity_flips=int((new!=old).sum()))
    return result


def mse_bootstrap(labels, control, normal, ids, repeats=1000):
    """Paired conversation-cluster bootstrap, fixed learned model/mask only."""
    y, c, n = [np.asarray(v).reshape(-1) for v in (labels,control,normal)]
    ids = np.asarray(ids).reshape(-1)
    if not (len(y)==len(c)==len(n)==len(ids)):
        raise ValueError('Bootstrap records must align')
    if not len(y):
        return dict(unit='conversation', lower=None, upper=None, conversations=0)
    _, inverse = np.unique(ids, return_inverse=True)
    count = np.bincount(inverse)
    diff = np.bincount(inverse, weights=(y-c)**2-(y-n)**2)
    rng = np.random.default_rng(66025)
    samples = rng.integers(0,len(count),size=(repeats,len(count)))
    means = diff[samples].sum(1)/count[samples].sum(1)
    return dict(unit='conversation', lower=float(np.quantile(means,.025)),
                upper=float(np.quantile(means,.975)), conversations=len(count), repeats=repeats)


def analyze(arrays, names):
    y, normal = arrays['labels'], arrays['normal']
    groups = dict(all=np.ones(len(y),dtype=bool), shuffle_eligible=arrays['eligible'],
        no_history=~arrays['has_history'], T_present=arrays['availability'][:,1].astype(bool),
        T_missing_seen=~arrays['availability'][:,1].astype(bool)&arrays['seen_T'],
        T_missing_never=~arrays['availability'][:,1].astype(bool)&~arrays['seen_T'])
    results = {}
    for group, keep in groups.items():
        rows = {}
        for name in names:
            rows[name] = paired_metrics(y[keep],arrays[name][keep],normal[keep])
            if name not in ('flat','normal'):
                rows[name]['delta_MSE_bootstrap95'] = mse_bootstrap(
                    y[keep],arrays[name][keep],normal[keep],arrays['conversation_ids'][keep])
        results[group] = rows
    return results


def report(output, records, identity):
    from experiments.osram_tasr_20261018.runner_helpers import write_json
    write_json(output/'RESULT.json',dict(status='complete',identity=identity,per_rate=records))
    lines = ['# Frozen Signature order / second-level area audit','',
        'MOSI seed66; eight original per-rate Test-best checkpoints; INTERNAL mechanism audit.',
        'No training, new selection, masks, CHM updates, Local/Base/Gap changes or labels as inputs.',
        'Controls change only the extra Signature descriptor of each fixed frame.','',
        '| Rate | Mode | N | ACC | WF1 | MSE | ΔMSE vs normal | Corrections / harms vs normal |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    csv_rows = []
    for rate, item in records.items():
        for group, modes in item['groups'].items():
            for mode, score in modes.items():
                csv_rows.append(dict(rate=rate,group=group,mode=mode,
                    **{k:v for k,v in score.items() if not isinstance(v,dict)}))
                if group=='all':
                    lines.append(f'| {rate} | {mode} | {score["N"]} | {score["ACC"]:.4f} | '
                        f'{score["WF1"]:.4f} | {score["MSE"]:.6f} | {score["delta_MSE"]:+.6f} | '
                        f'{score["corrections_vs_reference"]} / {score["harms_vs_reference"]} |')
    modes = list(next(iter(records.values()))['groups']['all'])
    lines += ['', '## Eight-rate arithmetic means','',
        '| Mode | ACC | WF1 | MSE |','| --- | ---: | ---: | ---: |']
    for name in modes:
        values=[r['groups']['all'][name] for r in records.values()]
        lines.append('| '+name+' | '+' | '.join(f'{np.mean([v[k] for v in values]):.6f}'
                                                for k in ('ACC','WF1','MSE'))+' |')
    lines += ['', '## Interpretation limits','',
        '- CHM is already order-sensitive. These interventions test incremental path information, not an order-blind baseline.',
        '- First-level signature under the fixed zero basepoint is an endpoint descriptor, not a temporal mean.',
        '- Interior shuffle keeps observed tokens, counts, endpoints and fixed time coordinates, but changes their interior content/time correspondence.',
        '- Removing antisymmetric second-level area preserves first level and symmetric outer-product terms. It can be out-of-distribution.',
        '- Same-checkpoint interference is mechanism sensitivity, not proof of necessity or superiority to an equally trained order-free branch.',
        '- Test-oracle and seed66 limitations remain; shuffled draws are not independent training seeds.',
        '- Prediction bootstrap intervals condition on the trained checkpoint and test mask; they do not correct selection bias.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    import csv
    with (output/'RESULT.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--rates',nargs='+',default=list(RATES))
    args=parser.parse_args(argv)
    rates=validate_rates(args.rates)
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_signature_cyclic_multi_20261025 import prepare
    from experiments.osram_signature_cyclic_multi_20261025.summarize import selected_row
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from .controls import descriptor_control
    from osram_four_block_wiring import SignatureBlock
    torch.set_num_threads(1)
    repair_uuid_rng(torch)
    modes=['flat','normal','degree2_no_area']+[f'interior_shuffle_{s}' for s in (6601,6602,6603)]
    identity=dict(audit_source_commit=args.source_commit,numerical_source=str(prepare.ROOT),
        dataset='CMUMOSI',seed=66,gpu_uuid=args.gpu_uuid,python=sys.executable,
        torch=torch.__version__,numpy=np.__version__,mode='frozen inference only',
        mask_protocol='gcnet-random-mask-v1',shuffle_seeds=[6601,6602,6603],
        source_hashes={p.name:h.sha(p) for p in Path(__file__).parent.glob('*.py')})
    h.write_json(args.output/'STATUS.json',dict(status='running',identity=identity,rates=rates,pid=os.getpid()))
    records={}; tick=time.monotonic()
    try:
        for rate in rates:
            source=args.selected_root/f'CMUMOSI_66_{rate}'
            selected_row(dict(dataset='CMUMOSI',seed=66,rate=rate,output=str(source)))
            summary=json.loads((source/'SUMMARY.json').read_text())
            saved=torch.load(source/summary['best']['checkpoint'],map_location='cpu',weights_only=False)
            canonical=lambda value: json.loads(json.dumps(value))
            if canonical(saved['identity']) != summary['identity'] or saved['epoch'] != summary['best']['epoch']:
                raise ValueError('Selected Signature identity/epoch mismatch')
            for name,digest in summary['identity']['source_hashes'].items():
                if h.sha(prepare.ROOT/name)!=digest:
                    raise ValueError('Original numerical source changed: '+name)
            bundle,reader,parent=prepare.setup('CMUMOSI',66,args.gpu_uuid,args.output/f'rate_{rate}',rate)
            if parent['parent_id']!=summary['parent_id']:
                raise ValueError('Different frozen Flat parent')
            cached=prepare.test_cache(bundle,reader,rate)
            block=SignatureBlock().to('cuda:0').eval().requires_grad_(False)
            block.load_state_dict(saved['block'],strict=True)
            parent_digest=h.backbone_digest(reader.parent)
            block_digest=h.backbone_digest(block)
            pieces=[]
            with torch.no_grad():
                for batch_index,row in enumerate(cached):
                    frame=row['frame'].to('cuda:0'); valid=frame.valid.T
                    observed=frame.availability&frame.valid[...,None]
                    seen=observed.long().cumsum(0)-observed.long()>0
                    history=frame.valid.long().cumsum(0)-frame.valid.long()>0
                    normal,eligible,_=descriptor_control(block,frame,'normal')
                    torch.testing.assert_close(normal,block.descriptor(frame),rtol=0,atol=0)
                    piece=dict(row['metadata'],availability=row['availability'],
                        eligible=eligible.T[valid].cpu().numpy(),
                        has_history=history.T[valid].cpu().numpy(),seen_T=seen[...,1].T[valid].cpu().numpy(),
                        flat=frame.prediction.transpose(0,1)[valid].cpu().numpy().reshape(-1))
                    for mode in modes[1:]:
                        descriptor=normal
                        if mode!='normal':
                            kind='interior_shuffle' if mode.startswith('interior_shuffle') else mode
                            seed=int(mode.rsplit('_',1)[-1]) if kind=='interior_shuffle' else 66
                            descriptor,_,_=descriptor_control(block,frame,kind,shuffle_seed=seed+batch_index*100003)
                        delta=block.out(descriptor)*(frame.valid&history)[...,None]
                        prediction=reader.decode_delta(frame,delta)
                        piece[mode]=prediction.transpose(0,1)[valid].cpu().numpy().reshape(-1)
                    pieces.append(piece)
            arrays={key:np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
            with np.load(source/summary['best']['predictions'],allow_pickle=False) as old:
                for key in ('labels','conversation_ids','utterance_indices','availability'):
                    np.testing.assert_array_equal(arrays[key],old[key])
                np.testing.assert_allclose(arrays['normal'],old['prediction'].reshape(-1),rtol=1e-5,atol=1e-6)
                np.testing.assert_array_equal(arrays['normal']>0,old['prediction'].reshape(-1)>0)
                replay_error=float(np.max(np.abs(arrays['normal']-old['prediction'].reshape(-1))))
            if parent_digest!=h.backbone_digest(reader.parent) or block_digest!=h.backbone_digest(block):
                raise RuntimeError('Read-only audit changed model weights')
            if any(p.grad is not None for p in list(reader.parent.parameters())+list(block.parameters())):
                raise RuntimeError('Inference audit unexpectedly created gradients')
            np.savez_compressed(args.output/f'rate_{rate}_predictions.npz',**arrays)
            mask_sha=hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
            if mask_sha!=summary['identity']['test_mask_sha256']:
                raise ValueError('Signature selected test mask mismatch')
            records[rate]=dict(selected_summary_sha256=h.sha(source/'SUMMARY.json'),
                selected_checkpoint_sha256=summary['best']['checkpoint_sha256'],
                selected_prediction_sha256=summary['best']['prediction_sha256'],
                parent_id=summary['parent_id'],selected_epoch=summary['best']['epoch'],
                test_mask_sha256=mask_sha,replay_max_absolute_error=replay_error,
                eligible_count=int(arrays['eligible'].sum()),groups=analyze(arrays,modes))
            h.write_json(args.output/f'rate_{rate}.json',records[rate])
            print(json.dumps(dict(rate=rate,N=len(arrays['labels']),eligible=records[rate]['eligible_count'],
                scores=records[rate]['groups']['all'],seconds=time.monotonic()-tick)),flush=True)
            del bundle,reader,block,cached,saved
            gc.collect();torch.cuda.empty_cache()
        identity.update(seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        report(args.output,records,identity)
        h.write_json(args.output/'STATUS.json',dict(status='complete',rates=rates,identity=identity))
    except BaseException as error:
        h.write_json(args.output/'STATUS.json',dict(status='failed',error=repr(error),identity=identity,
                                                    completed_rates=list(records)))
        raise


if __name__=='__main__':
    main()
