"""Frozen Stage1 Teacher exits: train-fit, validation-only sentiment ridge probes."""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.pretrained_teacher import read_source, file_sha256, state_sha256
from experiments.osram_supervised_teacher_20260914.smoke import build
from experiments.osram_supervised_teacher_20260914.run import ROOT, runner

NAMES = ('audio', 'text', 'visual', 'fused_node', 'final_hidden')


def sentiment_metrics(y, scores):
    keep = y != 0
    return dict(accuracy=float(accuracy_score(y[keep] > 0, scores[keep] > 0)),
                weighted_f1=float(f1_score(y[keep] > 0, scores[keep] > 0, average='weighted')),
                mae=float(np.abs(y - scores).mean()), n_binary=int(keep.sum()))


def probe(x, y, validation, labels):
    # Same fixed alpha as the existing complete-state audit. No validation tuning.
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=10., solver='cholesky').fit(scaler.transform(x), y)
    scores = ridge.predict(scaler.transform(validation))
    return dict(dimension=x.shape[1], alpha=10., **sentiment_metrics(labels, scores)), scores


@torch.no_grad()
def extract(model, loader, cfg, dims, split):
    arrays = {k: [] for k in (*NAMES, 'labels', 'emotion_score')}
    ids = []
    captured = {}
    def capture(_module, _args, output):
        captured['node'] = output[0]
    hook = model.observed_set.register_forward_hook(capture)
    schedule = tr._build_schedule(cfg, split, 0.)
    try:
        for raw in loader:
            view = tr._prepare_view(tr._move_batch(raw, torch.device(cfg.device)), schedule, 0, tuple(dims))
            valid = view['umask'].T.bool()
            assert torch.all(view['availability'][valid] == 1)
            scores, hidden, latents, predictions = model(
                [view['complete']], view['availability'], view['qmask'], view['umask'], view['lengths'])
            assert predictions is None
            assert torch.equal(model.smax_fc(hidden)[valid], scores[valid])
            values = dict(latents, fused_node=captured['node'], final_hidden=hidden,
                          labels=view['labels'].T, emotion_score=scores.squeeze(-1))
            for name in arrays:
                arrays[name].append(values[name][valid].cpu().numpy())
            ids.extend(map(str, view['conversation_ids']))
    finally:
        hook.remove()
    return {k: np.concatenate(v) for k, v in arrays.items()}, ids


def audit(seed, output):
    path = ROOT / 'teacher' / f'seed_{seed}' / 'best.pt'
    cp, projectors, projector_hash = read_source(path)
    cfg = tr.TrainConfig(**cp['config'])
    tr.set_random_seed(seed)
    roots = [str(runner.FEATURES / n) for n in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    train, val, _, *dims = tr.get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=cfg.dataset, batch_size=cfg.batch_size, num_workers=0, seed=seed,
        evaluation_protocol=cfg.evaluation_protocol, validation_fraction=cfg.validation_fraction)
    model = build(cfg, dims)
    model.load_state_dict(cp['model'], strict=True)
    model.requires_grad_(False).eval()
    assert state_sha256(model.observed_set.projectors.state_dict()) == projector_hash
    before = state_sha256(model.state_dict())
    # Guard against accidentally invoking the dormant EMA bank or MMoE.
    from unittest.mock import patch
    with patch.object(model.teacher, 'forward', side_effect=AssertionError('No EMA targets in audit')), \
         patch.object(model.missing_predictor, 'forward', side_effect=AssertionError('No predictor in audit')):
        a, train_ids = extract(model, train[cfg.fold-1], cfg, dims, 'train')
        b, val_ids = extract(model, val[cfg.fold-1], cfg, dims, 'validation')
    assert not set(train_ids).intersection(val_ids)
    assert before == state_sha256(model.state_dict())
    assert all(p.grad is None for p in model.parameters())
    native = sentiment_metrics(b['labels'], b['emotion_score'])
    assert abs(native['weighted_f1'] - cp['validation_mean_weighted_f1']) < 1e-7
    rows, predictions = [], {}
    for name in NAMES:
        metrics, scores = probe(a[name], a['labels'], b[name], b['labels'])
        rows.append(dict(representation=name, **metrics))
        predictions[name] = scores
    result = dict(seed=seed, checkpoint=str(path), checkpoint_sha256=file_sha256(path),
        epoch=cp['epoch'], projector_sha256=projector_hash, source_prefix='observed_set.projectors.',
        protocol='Complete view; fixed alpha10; train-fit scaler/ridge; validation evaluation; no test',
        checkpoint_selection='validation; reused for diagnostic evaluation, not independent holdout',
        state_unchanged=True, predictor_and_ema_not_called=True, native_teacher=native, rows=rows,
        train_count=len(a['labels']), validation_count=len(b['labels']),
        train_conversation_ids=train_ids, validation_conversation_ids=val_ids)
    (output / f'seed_{seed}.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(output / f'seed_{seed}_validation_scores.npz', labels=b['labels'],
                        native_teacher=b['emotion_score'], **predictions)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    output = ROOT / 'information_audit'
    output.mkdir(exist_ok=False)
    results = [audit(seed, output) for seed in range(66, 71)]
    summary = []
    for name in NAMES:
        rows = [next(x for x in r['rows'] if x['representation'] == name) for r in results]
        summary.append(dict(representation=name, dimension=rows[0]['dimension'],
            mean_wf1=statistics.mean(r['weighted_f1'] for r in rows),
            sd_wf1=statistics.stdev(r['weighted_f1'] for r in rows),
            mean_accuracy=statistics.mean(r['accuracy'] for r in rows)))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)
