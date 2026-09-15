"""Sample quality and fixed Text-probe transfer on natural validation miss=.5."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.pretrained_teacher import state_sha256, file_sha256
from experiments.osram_supervised_teacher_20260914.smoke import build
from experiments.osram_supervised_teacher_20260914.run import ROOT, runner
from experiments.osram_supervised_teacher_20260914.information_audit import sentiment_metrics
from experiments.osram_predictor_reaudit_20260909.audit import old, PATTERNS


def fit_text_probe(x, labels):
    return make_pipeline(StandardScaler(), Ridge(alpha=10., solver='cholesky')).fit(x, labels)


def transfer(probe, real, predicted, labels):
    truth, estimate = probe.predict(real), probe.predict(predicted)
    return dict(real_text=sentiment_metrics(labels, truth),
                predicted_text=sentiment_metrics(labels, estimate),
                score_mae_to_real=float(np.abs(truth-estimate).mean()))


@torch.no_grad()
def audit(seed, output):
    path = ROOT / 'student' / f'seed_{seed}' / 'best_miss_0p5.pt'
    cp = torch.load(path, map_location='cpu', weights_only=False)
    cfg = tr.TrainConfig(**cp['config'])
    assert cfg.teacher_mode == 'pretrained-frozen' and cfg.training_objective == 'joint'
    tr.set_random_seed(seed)
    roots = [str(runner.FEATURES/n) for n in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    train, val, _, *dims = tr.get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=cfg.dataset, batch_size=cfg.batch_size, num_workers=0, seed=seed,
        evaluation_protocol=cfg.evaluation_protocol, validation_fraction=cfg.validation_fraction)
    model = build(cfg, dims)
    model.load_state_dict(cp['model'], strict=True)
    model.requires_grad_(False).eval()
    assert model.teacher_integrity() == model.teacher_provenance['projector_sha256']
    before = state_sha256(model.state_dict())
    device = torch.device(cfg.device)
    xs, ys, train_ids = [], [], []
    for raw in train[cfg.fold-1]:
        view = tr._prepare_view(tr._move_batch(raw, device), tr._build_schedule(cfg, 'train', 0.), 0, tuple(dims))
        valid = view['umask'].T.bool()
        xs.append(model.encode_teacher_targets([view['complete']])['text'][valid].cpu().numpy())
        ys.append(view['labels'].T[valid].cpu().numpy())
        train_ids.extend(map(str, view['conversation_ids']))
    x, y = np.concatenate(xs), np.concatenate(ys)
    probe = fit_text_probe(x, y)
    prototype = x.mean(0)
    groups = {name: {k: [] for k in ('prediction', 'target', 'label')} for name in PATTERNS}
    val_ids, all_real, all_labels = [], [], []
    digest = hashlib.sha256()
    schedule = tr._build_schedule(cfg, 'validation', .5)
    for raw in val[cfg.fold-1]:
        view = tr._prepare_view(tr._move_batch(raw, device), schedule, 0, tuple(dims))
        a, valid = view['availability'], view['umask'].T.bool()
        logits, _, _, pred = model([view['incomplete']], a, view['qmask'], view['umask'],
                                   view['lengths'], predict_missing=True)
        plain = model([view['incomplete']], a, view['qmask'], view['umask'], view['lengths'])[0]
        assert torch.equal(logits, plain)
        target = model.encode_teacher_targets([view['complete']])['text']
        all_real.append(target[valid].cpu().numpy())
        all_labels.append(view['labels'].T[valid].cpu().numpy())
        digest.update(a[valid].cpu().numpy().tobytes())
        val_ids.extend(map(str, view['conversation_ids']))
        for name, bits in PATTERNS.items():
            sel = valid & (a == a.new_tensor(bits)).all(-1)
            assert pred.target_mask[..., 1][sel].all()
            for key, value in dict(prediction=pred.reg_predictions[..., 1, :], target=target,
                                   label=view['labels'].T).items():
                groups[name][key].append(value[sel].cpu())
    assert not set(train_ids).intersection(val_ids)
    assert before == state_sha256(model.state_dict())
    rows = []
    for name, arrays in groups.items():
        p, t, labels = (torch.cat(arrays[k]) for k in ('prediction', 'target', 'label'))
        assert len(p) >= 2 and torch.isfinite(p).all() and torch.isfinite(t).all()
        quality = old._metrics(p, t, cfg.temperature, seed+100)
        quality['std_ratio'] = quality['channel_std']/max(quality['target_channel_std'], 1e-12)
        quality['retrieval_over_chance'] = quality['retrieval_top1']/quality['chance']
        metrics = transfer(probe, t.numpy(), p.numpy(), labels.numpy())
        metrics['prototype'] = sentiment_metrics(labels.numpy(), probe.predict(np.repeat(prototype[None],len(p),0)))
        rng = np.random.default_rng(seed)
        metrics['shuffled_prediction_wf1'] = statistics.mean(
            sentiment_metrics(labels.numpy(), probe.predict(rng.permutation(p.numpy())))['weighted_f1']
            for _ in range(8))
        rows.append(dict(pattern=name, quality=quality, transfer=metrics))
        np.savez_compressed(output/f'seed_{seed}_{name}.npz', prediction=p.numpy(), target=t.numpy(),
                            labels=labels.numpy(), real_score=probe.predict(t.numpy()), pred_score=probe.predict(p.numpy()))
    result = dict(seed=seed, checkpoint=str(path), checkpoint_sha256=file_sha256(path), epoch=cp['epoch'],
        selection='Student previously selected by per-rate Test oracle; no re-selection',
        teacher_sha256=model.teacher_integrity(), state_unchanged=True, logits_unchanged=True,
        protocol='natural validation miss=.5; original contextual reg_predictions; one Text probe train-fit alpha10',
        train_count=len(y), validation_count=sum(len(v) for v in all_labels),
        all_validation_real_text=sentiment_metrics(np.concatenate(all_labels),probe.predict(np.concatenate(all_real))),
        mask_sha256=digest.hexdigest(), train_ids=train_ids, validation_ids=val_ids, rows=rows)
    (output/f'seed_{seed}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(seed=seed, rows=rows)), flush=True)
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    output = ROOT/'text_transfer_audit'
    output.mkdir(exist_ok=False)
    results = [audit(seed, output) for seed in range(66,71)]
    summary = []
    for name in PATTERNS:
        rows = [next(row for row in result['rows'] if row['pattern']==name) for result in results]
        quality = {k: statistics.mean(row['quality'][k] for row in rows) for k in rows[0]['quality']}
        scores = {k: statistics.mean(row['transfer'][k]['weighted_f1'] for row in rows)
                  for k in ('real_text','predicted_text','prototype')}
        scores['shuffled_prediction'] = statistics.mean(row['transfer']['shuffled_prediction_wf1'] for row in rows)
        summary.append(dict(pattern=name, counts=[row['quality']['count'] for row in rows],
                            quality_mean=quality, wf1_mean=scores,
                            wf1_sd={k:statistics.stdev(row['transfer'][k]['weighted_f1'] for row in rows)
                                    for k in ('real_text','predicted_text')},
                            real_better_seeds=sum(row['transfer']['real_text']['weighted_f1'] > row['transfer']['predicted_text']['weighted_f1'] for row in rows)))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)
