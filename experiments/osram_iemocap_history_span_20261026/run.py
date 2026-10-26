"""Frozen IEMOCAP-6 seed66/fold5, one protected anchor and eight saved masks."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

NUMERIC = Path('/data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/source_21e46f2')
WINDOWS = (0,1,2,4,8,16,32,None)
GPU = 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'


def write(path,payload):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def summarize(arrays):
    import numpy as np
    from experiments.osram_persistent_patterns_20261018.evaluate import classification_metrics
    labels,full = arrays['labels'],arrays['logits_full']
    patterns = np.asarray([''.join(m for m,on in zip('ATV',a) if on) for a in arrays['availability']])
    cohorts = dict(all=np.ones(len(labels),dtype=bool),common_past_ge32=arrays['past_count']>=32)
    cohorts.update({f'current_{p}':patterns==p for p in ('A','T','V','AT','AV','TV','ATV')})
    rows = {}
    for window in WINDOWS:
        name = 'full' if window is None else str(window)
        prediction = arrays['logits_'+name]
        groups = {}
        for group,include in cohorts.items():
            if not include.any():
                groups[group] = dict(N=0,status='N/A')
                continue
            y,p,ref = labels[include],prediction[include],full[include]
            metrics = classification_metrics(y,p,6)
            reference = classification_metrics(y,ref,6)
            correct,was = p.argmax(-1)==y,ref.argmax(-1)==y
            metrics.update(delta_vs_full={k:metrics[k]-reference[k] for k in ('ACC','UA','WF1','CE')},
                corrections=int((correct&~was).sum()),harms=int((~correct&was).sum()),
                changed=int((p.argmax(-1)!=ref.argmax(-1)).sum()))
            groups[group] = metrics
        rows[name] = groups
    return rows


def report(output,result):
    import numpy as np
    rates = result['rates']
    lines = ['# Frozen IEMOCAP-6 history-span audit','',
        'INTERNAL DIAGNOSTIC; seed66/fold5 only. No training or selection.',
        'All conditions use the SAME protected original miss0.3 checkpoint.',
        'Eight original selected GCNet masks are replayed; the per-rate source checkpoints only identify masks, not model weights.',
        'K counts strictly prior valid turns. Rebuild from zero on the most recent min(K,n) turns, then apply target decay and original read.',
        'Original Local, Query, current input, zero backward slots, joint writes, Flat and head are unchanged.','',
        '## Equal-rate average (not a deployment/test-best score)','',
        '| Past turns | All ACC | All UA | All W-F1 | Past≥32 ACC | Past≥32 UA | Past≥32 W-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    means = {}
    for window in WINDOWS:
        name = 'full' if window is None else str(window)
        means[name] = {group:{k:float(np.mean([r['metrics'][name][group][k] for r in rates.values()]))
            for k in ('ACC','UA','WF1','CE')} for group in ('all','common_past_ge32')}
        values = [means[name][g][k] for g in ('all','common_past_ge32') for k in ('ACC','UA','WF1')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    result['equal_rate_mean'] = means
    lines += ['',f"All-target support/rate: {next(iter(rates.values()))['metrics']['full']['all']['N']}; common past≥32 support/rate: {next(iter(rates.values()))['metrics']['full']['common_past_ge32']['N']}.",
        '', '## Per-rate scores','', '| Rate | K | ACC | UA | W-F1 | CE | Corrections vs full | Harms vs full |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    table = []
    for rate,row in rates.items():
        for window in WINDOWS:
            name = 'full' if window is None else str(window)
            m = row['metrics'][name]['all']
            lines.append(f"| {rate} | {name} | {m['ACC']:.3f} | {m['UA']:.3f} | {m['WF1']:.3f} | {m['CE']:.5f} | {m['corrections']} | {m['harms']} |")
            for group,values in row['metrics'][name].items():
                table.append(dict(rate=rate,window=name,cohort=group,**{k:v for k,v in values.items() if not isinstance(v,(dict,list))}))
    lines += ['', '## Correctness / costs','', '```json',json.dumps(result['checks'],indent=2), '```','',
        f"Wall seconds: {result['wall_seconds']:.2f}; GPU peak MiB: {result['peak_gpu_mib']:.2f}.",
        '', '## Limits','',
        'Truncation is a frozen state intervention and can be distribution-shifting. A score plateau does not prove a finite storage capacity.',
        'All-target curves include short dialogues; the shared past≥32 cohort controls history availability, not label/speaker confounding.',
        'Only one seed and one held-out session are measured. Do not generalize to five-fold performance or tune windows on these test outcomes.',
        'CSV/JSON include each current modality pattern; raw aligned logits/masks stay remote. No protected checkpoint or artifact was modified.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    fields = sorted({k for r in table for k in r})
    with (output/'RESULT.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fields)
        writer.writeheader();writer.writerows(table)
    write(output/'RESULT.json',result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != NUMERIC:
        raise ValueError('Require protected IEMOCAP numerical snapshot as cwd')
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output; never overwrite evidence')
    args.output.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare
    prepare.configure_environment(args.output)
    import numpy as np
    import torch
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from .replay import ScanCapture,replay_reads,decode
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    start = time.monotonic()
    result = dict(status='running',source_commit=args.source_commit,numerical_source=str(NUMERIC),
        numerical_commit='21e46f2ecd092345818659694db8ff372672c35e',dataset='IEMOCAPSix',seed=66,fold=5,
        checkpoint_rate='0.3',training=False,windows=['full' if k is None else k for k in WINDOWS],
        rates={},checks=dict(rng_repair=repair),source_hashes={p.name:h.sha(p) for p in Path(__file__).parent.glob('*.py')})
    write(args.output/'STATUS.json',result)
    try:
        bundle,reader,proof = prepare.setup(66,5,GPU,args.output,'0.3')
        model,trainer = bundle['model'],bundle['trainer']
        before = h.backbone_digest(model)
        result.update(parent=proof,environment=bundle['environment'],gpu=bundle['gpu'])
        case = prepare.registered_parent(66,5)
        selected,refs = {},{}
        for rate,ref in case['references'].items():
            if h.sha(ref['checkpoint']) != ref['checkpoint_sha256']:
                raise ValueError('Protected mask-reference checkpoint changed')
            state = torch.load(ref['checkpoint'],map_location='cpu',weights_only=False)
            ref = dict(ref,selection=str(h.resolve_selection(state,case['output'],ref['selection_sha256'])))
            selected[rate] = prepare.selected_arrays(ref)
            refs[rate] = ref
            del state
        checks = result['checks']
        checks.update(max_full_base_abs=0.,max_full_gap_abs=0.,max_full_logits_abs=0.,
            max_full_logits_relative=0.,full_classes_exact=True,anchor_selected_logits_exact=False,
            replay_rng_unchanged=True,original_scans=0,backbone_unchanged=False)
        torch.cuda.reset_peak_memory_stats()
        for rate in prepare.RATES:
            rate_start = time.monotonic()
            saved,pieces,cursor = selected[rate],[],0
            loader = bundle['test_loader']
            with h.isolated_rng(66+99193),torch.no_grad():
                loader.sampler.set_epoch(0)
                for raw in loader:
                    data = trainer._move_batch(raw,torch.device('cuda:0'))
                    meta = h.metadata(data);n = len(meta['labels'])
                    for key,v in meta.items():
                        np.testing.assert_array_equal(v,saved[key][cursor:cursor+n])
                    a = torch.zeros((*data[7].shape,3),device='cuda:0')
                    a[data[7].bool()] = torch.as_tensor(saved['availability'][cursor:cursor+n],device='cuda:0',dtype=a.dtype)
                    a = a.transpose(0,1)
                    view = trainer._prepare_view_from_primary_masks(data,a,a,bundle['dimensions'])
                    with ScanCapture(model.osram) as trace:
                        frame = reader(view['incomplete'],a,view['qmask'],view['umask'],view['lengths'])
                    checks['original_scans'] += trace.calls
                    assert trace.calls == 1
                    cpu_rng,cuda_rng = torch.random.get_rng_state(),torch.cuda.get_rng_state()
                    logits = {}
                    for window in (None,0,1,2,4,8,16,32):
                        name = 'full' if window is None else str(window)
                        base,gap = replay_reads(model.osram,trace,window)
                        p = decode(model,frame,base,gap)
                        assert torch.isfinite(p).all()
                        if window is None:
                            for label,value,ref in (('base',base,trace.base),('gap',gap,trace.gap),('logits',p,frame.prediction)):
                                error = float((value-ref).abs().max())
                                checks['max_full_'+label+'_abs'] = max(checks['max_full_'+label+'_abs'],error)
                                torch.testing.assert_close(value,ref,rtol=2e-5,atol=2e-5)
                            checks['max_full_logits_relative'] = max(checks['max_full_logits_relative'],float(((p-frame.prediction).abs()/frame.prediction.abs().clamp_min(1e-6)).max()))
                            assert torch.equal(p.argmax(-1)[frame.valid],frame.prediction.argmax(-1)[frame.valid])
                            logits['logits_original'] = frame.prediction.transpose(0,1)[data[7].bool()].cpu().numpy()
                        if window == 0:
                            assert torch.count_nonzero(base) == torch.count_nonzero(gap) == 0
                        logits['logits_'+name] = p.transpose(0,1)[data[7].bool()].cpu().numpy()
                    assert torch.equal(cpu_rng,torch.random.get_rng_state())
                    assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
                    prior = frame.valid.long().cumsum(0)-frame.valid.long()
                    pieces.append(dict(meta,availability=saved['availability'][cursor:cursor+n],
                        past_count=prior.T[data[7].bool()].cpu().numpy(),**logits))
                    cursor += n
            assert cursor == len(saved['labels'])
            arrays = {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            if rate == '0.3':
                np.testing.assert_array_equal(arrays['logits_original'],saved['logits'])
                checks['anchor_selected_logits_exact'] = True
            np.savez_compressed(args.output/f'predictions_rate_{rate}.npz',**arrays)
            import hashlib
            result['rates'][rate] = dict(metrics=summarize(arrays),mask_reference=refs[rate],
                availability_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest(),
                prediction_artifact_sha256=h.sha(args.output/f'predictions_rate_{rate}.npz'),seconds=time.monotonic()-rate_start)
            if result['rates'][rate]['availability_sha256'] != refs[rate]['mask_sha256']:
                raise ValueError('Replay changed protected availability')
            write(args.output/'STATUS.json',result)
            print(json.dumps(dict(rate=rate,seconds=result['rates'][rate]['seconds'],full=result['rates'][rate]['metrics']['full']['all'])),flush=True)
        if before != h.backbone_digest(model) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen model changed or acquired gradients')
        checks['backbone_unchanged'] = True
        result.update(status='complete',wall_seconds=time.monotonic()-start,
            peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        report(args.output,result)
        write(args.output/'STATUS.json',result)
    except Exception as error:
        result.update(status='failed',failure=repr(error),wall_seconds=time.monotonic()-start)
        write(args.output/'STATUS.json',result)
        raise


if __name__ == '__main__':
    main()
