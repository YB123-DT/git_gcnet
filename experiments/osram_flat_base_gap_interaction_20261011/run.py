"""Frozen Flat output interaction, not a learned or deployable additive model."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def interaction(f00, f10, f01, f11):
    return f11-f10-f01+f00


def adjacent(previous, current):
    if previous is None or previous == 0 or current == 0:
        return 'excluded'
    return 'same' if (previous > 0) == (current > 0) else 'opposite'


def summarize(rows):
    import numpy as np
    from sklearn.metrics import f1_score
    y = np.asarray([r['label'] for r in rows])
    full = np.asarray([r['f11'] for r in rows])
    additive = np.asarray([r['f10']+r['f01']-r['f00'] for r in rows])
    i = np.asarray([r['interaction'] for r in rows])
    keep = y != 0
    ac, fc = (additive[keep] > 0) == (y[keep] > 0), (full[keep] > 0) == (y[keep] > 0)
    def score(p):
        return 100*float(f1_score(y[keep] > 0, p[keep] > 0, average='weighted')) if keep.any() else None
    f, a = score(full), score(additive)
    main = np.asarray([abs(r['f10']-r['f00'])+abs(r['f01']-r['f00']) for r in rows])
    return dict(n=len(rows), nonneutral_n=int(keep.sum()), full_wf1=f, additive_wf1=a,
        delta_pp=f-a if f is not None else None,
        corrections=int((~ac & fc).sum()), harms=int((ac & ~fc).sum()),
        sign_flips=int((ac != fc).sum()),
        interaction_mean=float(i.mean()) if len(i) else None,
        interaction_abs_mean=float(abs(i).mean()) if len(i) else None,
        interaction_abs_p50=float(np.quantile(abs(i), .5)) if len(i) else None,
        interaction_abs_p95=float(np.quantile(abs(i), .95)) if len(i) else None,
        interaction_abs_max=float(abs(i).max()) if len(i) else None,
        main_effect_abs_sum_mean=float(main.mean()) if len(main) else None,
        interaction_to_main_mean_ratio=float(abs(i).mean()/main.mean()) if len(main) and main.mean() > 0 else None)


def grouped(rows):
    patterns = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
    groups = {'all':rows, 'no_Text':[r for r in rows if 'T' not in r['pattern']],
              'Text_present':[r for r in rows if 'T' in r['pattern']]}
    groups.update({p:[r for r in rows if r['pattern'] == p] for p in patterns})
    for relation in ('same', 'opposite'):
        groups[relation] = [r for r in rows if r['adjacent'] == relation]
        for p in patterns:
            groups[p+'/'+relation] = [r for r in rows if r['pattern'] == p and r['adjacent'] == relation]
    return [dict(group=k, **summarize(v)) for k,v in groups.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--code-commit', required=True)
    p.add_argument('--gpu', type=int, default=6)
    args = p.parse_args()
    assert args.gpu in (0,1,2,3,5,6,7)
    uuid = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True).strip()
    assert uuid == os.environ.get('CUDA_VISIBLE_DEVICES')
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, _move_batch, _prepare_view
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    from experiments.osram_history_drift_20261002.diagnostic import capture
    from experiments.osram_nested_delta_off_missing_text_20261010.readout import flat_prediction
    from experiments.osram_nested_delta_off_missing_text_20261010.run import sha, dump
    from experiments.osram_nested_diagnostics_20261007.evaluate import state_hash
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    config = TrainConfig(**json.loads((args.source/'config.json').read_text()))
    assert config.seed == 66 and config.training_objective == 'emotion-only'
    assert config.osram_meaningful_block in ('none', '', None)
    assert config.mosi_task_mode == 'regression'
    dataset = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')
    os.environ['GCNET_DATASET_ROOT'] = str(dataset)
    features = [str(dataset/'CMUMOSI/features'/n) for n in
        ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    train, val, test, *dims = get_loaders(audio_root=features[0], text_root=features[1],
        video_root=features[2], num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size,
        num_workers=0, seed=66, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    del train, val
    loader, device = test[0], torch.device('cuda:0')
    model = _build_model(config, tuple(dims)).to(device).eval().requires_grad_(False)
    state = dict(status='running', pid=os.getpid(), server='biggpu', gpu=args.gpu,
        gpu_uuid=uuid, code_commit=args.code_commit, source=str(args.source),
        label='INTERNAL DIAGNOSTIC ONLY', training_runs=0, records=[])
    dump(args.output/'STATUS.json', state)
    rows, table = [], []
    for rate in [i/10 for i in range(8)]:
        suffix = f'{rate:.1f}'.replace('.', 'p')
        checkpoint = args.source/f'best_miss_{suffix}.pt'
        checksum = sha(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        epoch = int(saved['epoch'])
        del saved
        before = state_hash(model)
        current, predictions, labels, masks = [], [], [], []
        replay_error, inactive_interaction = 0., 0.
        loader.sampler.set_epoch(0)
        with torch.no_grad():
            for raw in loader:
                v = _prepare_view(_move_batch(raw, device), _schedules(config,'test')[rate],0,tuple(dims))
                c = capture(model, lambda:model([v['incomplete']], v['availability'],v['qmask'],
                    v['umask'],v['lengths'],predict_missing=False))
                base = torch.cat((c['base'], torch.zeros_like(c['base'])), -1)
                gap = torch.cat((c['gap'], torch.zeros_like(c['gap'])), -1)
                outputs = {}
                for name, b, g in [('f00', torch.zeros_like(base),torch.zeros_like(gap)),
                    ('f10',base,torch.zeros_like(gap)), ('f01',torch.zeros_like(base),gap), ('f11',base,gap)]:
                    outputs[name] = flat_prediction(model,c['local'],b,g,v['availability'],v['umask'])
                valid = v['umask'].T.bool()
                replay_error = max(replay_error,float((outputs['f11'][valid]-c['prediction'].squeeze(-1)[valid]).abs().max()))
                assert replay_error <= 1e-5
                for output in outputs.values():
                    assert torch.isfinite(output).all() and not output[~valid].count_nonzero()
                ibg = interaction(*(outputs[k].double() for k in ('f00','f10','f01','f11')))
                no_gap = valid & (v['availability'].sum(-1) == 3)
                if no_gap.any():
                    inactive_interaction = max(inactive_interaction,float(ibg[no_gap].abs().max()))
                assert inactive_interaction <= 1e-6
                bv = v['umask'].bool()
                predictions.extend(outputs['f11'].T[bv].cpu().tolist())
                labels.extend(v['labels'][bv].cpu().tolist())
                masks.extend(v['availability'].transpose(0,1)[bv].cpu().tolist())
                for t,b in valid.nonzero().cpu().tolist():
                    y = float(v['labels'][b,t])
                    previous = float(v['labels'][b,t-1]) if t > 0 else None
                    current.append(dict(seed=66,rate=rate,conversation=str(v['conversation_ids'][b]),
                        utterance=t,pattern=''.join(n for n,a in zip('ATV',v['availability'][t,b].tolist()) if a),
                        label=y,previous_label=previous,adjacent=adjacent(previous,y),
                        **{k:float(value[t,b]) for k,value in outputs.items()}, interaction=float(ibg[t,b])))
        archive = np.load(args.source/f'predictions_miss_{suffix}.npz')
        assert np.array_equal(labels, archive['labels']) and np.array_equal(masks,archive['availability'])
        archive_error = float(abs(np.asarray(predictions)-archive['predictions'].reshape(-1)).max())
        assert archive_error <= 1e-5
        assert before == state_hash(model) and checksum == sha(checkpoint)
        assert all(not x.requires_grad and x.grad is None for x in model.parameters())
        table.extend(dict(rate=rate,**r) for r in grouped(current))
        rows.extend(current)
        state['records'].append(dict(rate=rate,epoch=epoch,checkpoint_sha256=checksum,
            state_sha256=before,frozen_unchanged=True,replay_error=replay_error,
            archive_error=archive_error,no_active_gap_interaction_max=inactive_interaction))
        dump(args.output/'STATUS.json',state)
        print(json.dumps(dict(rate=rate,**summarize(current))),flush=True)
    macro = []
    for group in dict.fromkeys(r['group'] for r in table):
        for scope in ('eight_rates','high_missing'):
            selected = [r for r in table if r['group']==group and r['full_wf1'] is not None
                and (scope != 'high_missing' or r['rate']>=.5)]
            macro.append(dict(group=group,scope=scope,nonempty_rates=len(selected),
                **{k:statistics.mean(r[k] for r in selected) if selected else None
                    for k in ('full_wf1','additive_wf1','delta_pp','interaction_abs_mean')},
                corrections=sum(r['corrections'] for r in selected),harms=sum(r['harms'] for r in selected)))
    for name,data in [('utterances',rows),('per_rate',table),('macro',macro)]:
        with (args.output/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    dump(args.output/'SUMMARY.json',dict(label=state['label'],per_rate=table,macro=macro,records=state['records']))
    state['status']='completed'
    dump(args.output/'STATUS.json',state)


if __name__ == '__main__':
    main()
