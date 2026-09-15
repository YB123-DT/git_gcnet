"""Frozen Text task-direction prediction: MMoE versus observed-input ridge."""
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.pretrained_teacher import state_sha256, file_sha256
from experiments.osram_supervised_teacher_20260914.smoke import build
from experiments.osram_supervised_teacher_20260914.run import ROOT, runner
from experiments.osram_supervised_teacher_20260914.information_audit import sentiment_metrics
from experiments.osram_supervised_teacher_20260914.text_transfer_audit import fit_text_probe, PATTERNS


def raw_direction(probe):
    scaler, ridge = probe.steps[0][1], probe.steps[1][1]
    w = ridge.coef_ / scaler.scale_
    b = float(ridge.intercept_ - scaler.mean_ @ w)
    return w, b


def correlation(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def direction_metrics(real, predicted, labels, intercept, seed):
    corr = correlation(real, predicted)
    mae = float(np.abs(real-predicted).mean())
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(real) for _ in range(8)]
    cs = [correlation(s, predicted) for s in shuffled]
    return dict(correlation=corr, mae=mae, bias=float((predicted-real).mean()),
        real_std=float(real.std()), predicted_std=float(predicted.std()),
        real_minus_shuffle_correlation=(corr-statistics.mean(cs) if corr is not None and all(c is not None for c in cs) else None),
        shuffle_minus_real_mae=statistics.mean(float(np.abs(s-predicted).mean()) for s in shuffled)-mae,
        teacher_sign_agreement=float(np.mean((real+intercept>0)==(predicted+intercept>0))),
        sentiment=sentiment_metrics(labels, predicted+intercept))


@torch.no_grad()
def audit(seed, output):
    previous = json.loads((ROOT/'text_transfer_audit'/f'seed_{seed}.json').read_text())
    path = Path(previous['checkpoint'])
    assert file_sha256(path) == previous['checkpoint_sha256']
    cp = torch.load(path, map_location='cpu', weights_only=False)
    cfg = tr.TrainConfig(**cp['config'])
    tr.set_random_seed(seed)
    roots = [str(runner.FEATURES/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    train, val, _, *dims = tr.get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=cfg.dataset, batch_size=cfg.batch_size, num_workers=0, seed=seed,
        evaluation_protocol=cfg.evaluation_protocol, validation_fraction=cfg.validation_fraction)
    model = build(cfg, dims)
    model.load_state_dict(cp['model'], strict=True)
    model.requires_grad_(False).eval()
    assert model.teacher_integrity() == previous['teacher_sha256']
    before = state_sha256(model.state_dict())
    groups, ids = {}, {}
    train_text, train_y = [], []
    for split, loader in [('train',train[cfg.fold-1]),('validation',val[cfg.fold-1])]:
        groups[split] = {p:{k:[] for k in ('input','prediction','target','labels')} for p in PATTERNS}
        ids[split] = []
        schedule = tr._build_schedule(cfg,split,.5)
        for raw in loader:
            view = tr._prepare_view(tr._move_batch(raw,torch.device(cfg.device)),schedule,0,tuple(dims))
            a, valid = view['availability'], view['umask'].T.bool()
            logits, _, latents, pred = model([view['incomplete']],a,view['qmask'],view['umask'],view['lengths'],predict_missing=True)
            context = model.last_osram_context
            assert torch.equal(logits,model([view['incomplete']],a,view['qmask'],view['umask'],view['lengths'])[0])
            # Only real observed sources and pre-write contextual reads enter X.
            inputs = {name:torch.cat([latents[m] for m,bit in zip(('audio','text','visual'),bits) if bit]
                       +[context['base'],context['gap'][...,1,:]],dim=-1) for name,bits in PATTERNS.items()}
            target = model.encode_teacher_targets([view['complete']])['text']
            if split=='train':
                train_text.append(target[valid].cpu().numpy())
                train_y.append(view['labels'].T[valid].cpu().numpy())
            ids[split].extend(map(str,view['conversation_ids']))
            for name,bits in PATTERNS.items():
                sel = valid & (a==a.new_tensor(bits)).all(-1)
                assert pred.target_mask[...,1][sel].all()
                for key,value in dict(input=inputs[name],prediction=pred.reg_predictions[...,1,:],
                                      target=target,labels=view['labels'].T).items():
                    groups[split][name][key].append(value[sel].cpu().numpy())
    assert not set(ids['train']).intersection(ids['validation'])
    assert set(ids['validation']) == set(previous['validation_ids'])
    assert state_sha256(model.state_dict()) == before
    q = fit_text_probe(np.concatenate(train_text),np.concatenate(train_y))
    w,b = raw_direction(q)
    rows = []
    for name in PATTERNS:
        a,c = ({k:np.concatenate(v) for k,v in groups[split][name].items()} for split in ('train','validation'))
        rtrain, real, predicted = a['target']@w, c['target']@w, c['prediction']@w
        np.testing.assert_allclose(real+b,q.predict(c['target']),atol=1e-5,rtol=1e-5)
        with np.load(ROOT/'text_transfer_audit'/f'seed_{seed}_{name}.npz') as old:
            np.testing.assert_array_equal(c['labels'],old['labels'])
            np.testing.assert_allclose(predicted+b,old['pred_score'],atol=1e-5,rtol=1e-5)
            np.testing.assert_allclose(real+b,old['real_score'],atol=1e-5,rtol=1e-5)
        accessible = fit_text_probe(a['input'],rtrain)
        oracle = accessible.predict(c['input'])
        rows.append(dict(pattern=name,train_count=len(rtrain),validation_count=len(real),input_dimension=a['input'].shape[1],
            teacher=sentiment_metrics(c['labels'],real+b),
            mmoe=direction_metrics(real,predicted,c['labels'],b,seed),
            input_ridge=direction_metrics(real,oracle,c['labels'],b,seed),
            train_input_ridge=direction_metrics(rtrain,accessible.predict(a['input']),a['labels'],b,seed)))
        np.savez_compressed(output/f'seed_{seed}_{name}.npz',teacher_r=real,mmoe_r=predicted,input_ridge_r=oracle,
                            labels=c['labels'],raw_weight=w,intercept=b)
    result = dict(seed=seed,epoch=cp['epoch'],checkpoint_sha256=previous['checkpoint_sha256'],
        teacher_sha256=previous['teacher_sha256'],state_unchanged=True,logits_unchanged=True,
        protocol='Natural miss=.5; alpha10 train-only per-pattern ridge; no new labels in scalar fit; validation only',
        direction='w_raw=coef/scale; b_raw=intercept-mean@w_raw; r=z@w_raw; sentiment decision uses r+b_raw',
        scope='Observed-input ridge is an empirical linear accessibility reference, not a mathematical upper bound',rows=rows)
    (output/f'seed_{seed}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    torch.set_num_threads(2)
    output = ROOT/'task_direction_audit'
    output.mkdir(exist_ok=False)
    results = [audit(s,output) for s in range(66,71)]
    summary = []
    for name in PATTERNS:
        rows = [next(r for r in result['rows'] if r['pattern']==name) for result in results]
        record = dict(pattern=name,teacher_wf1=statistics.mean(r['teacher']['weighted_f1'] for r in rows))
        for method in ('mmoe','input_ridge','train_input_ridge'):
            keys=('correlation','mae','bias','real_std','predicted_std','real_minus_shuffle_correlation','shuffle_minus_real_mae','teacher_sign_agreement')
            record[method] = {k:statistics.mean(r[method][k] for r in rows) if all(r[method][k] is not None for r in rows) else None for k in keys}
            record[method]['wf1']=statistics.mean(r[method]['sentiment']['weighted_f1'] for r in rows)
            record[method]['wf1_sd']=statistics.stdev(r[method]['sentiment']['weighted_f1'] for r in rows)
        summary.append(record)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)
