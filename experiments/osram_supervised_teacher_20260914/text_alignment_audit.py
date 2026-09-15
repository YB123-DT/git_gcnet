"""Two train-fit diagnostics: predicted-Text probe and label-free affine alignment."""
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
from experiments.osram_supervised_teacher_20260914.text_transfer_audit import fit_text_probe, PATTERNS


def fit_alignment(prediction, target):
    # Multi-output affine map. No sentiment labels or validation inputs accepted.
    return make_pipeline(StandardScaler(), Ridge(alpha=10., solver='cholesky')).fit(prediction, target)


@torch.no_grad()
def audit(seed, output):
    previous_dir = ROOT/'text_transfer_audit'
    previous = json.loads((previous_dir/f'seed_{seed}.json').read_text())
    path = Path(previous['checkpoint'])
    assert file_sha256(path) == previous['checkpoint_sha256']
    cp = torch.load(path, map_location='cpu', weights_only=False)
    cfg = tr.TrainConfig(**cp['config'])
    assert cfg.teacher_mode == 'pretrained-frozen'
    tr.set_random_seed(seed)
    roots = [str(runner.FEATURES/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    train, _, _, *dims = tr.get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=cfg.dataset, batch_size=cfg.batch_size, num_workers=0, seed=seed,
        evaluation_protocol=cfg.evaluation_protocol, validation_fraction=cfg.validation_fraction)
    model = build(cfg, dims)
    model.load_state_dict(cp['model'], strict=True)
    model.requires_grad_(False).eval()
    assert model.teacher_integrity() == previous['teacher_sha256']
    before = state_sha256(model.state_dict())
    groups = {name: {k: [] for k in ('prediction','target','labels')} for name in PATTERNS}
    all_text, all_y, ids = [], [], []
    digest = hashlib.sha256()
    schedule = tr._build_schedule(cfg, 'train', .5)
    for raw in train[cfg.fold-1]:
        view = tr._prepare_view(tr._move_batch(raw, torch.device(cfg.device)), schedule, 0, tuple(dims))
        a, valid = view['availability'], view['umask'].T.bool()
        logits, _, _, pred = model([view['incomplete']], a, view['qmask'], view['umask'],
                                   view['lengths'], predict_missing=True)
        assert torch.equal(logits, model([view['incomplete']], a, view['qmask'], view['umask'], view['lengths'])[0])
        target = model.encode_teacher_targets([view['complete']])['text']
        all_text.append(target[valid].cpu().numpy())
        all_y.append(view['labels'].T[valid].cpu().numpy())
        ids.extend(map(str, view['conversation_ids']))
        digest.update(a[valid].cpu().numpy().tobytes())
        for name, bits in PATTERNS.items():
            sel = valid & (a == a.new_tensor(bits)).all(-1)
            assert pred.target_mask[...,1][sel].all()
            for key, value in dict(prediction=pred.reg_predictions[...,1,:], target=target,
                                   labels=view['labels'].T).items():
                groups[name][key].append(value[sel].cpu().numpy())
    assert set(ids) == set(previous['train_ids'])
    assert not set(ids).intersection(previous['validation_ids'])
    assert before == state_sha256(model.state_dict())
    q_text = fit_text_probe(np.concatenate(all_text), np.concatenate(all_y))
    rows = []
    for name, arrays in groups.items():
        p, t, y = (np.concatenate(arrays[k]) for k in ('prediction','target','labels'))
        with np.load(previous_dir/f'seed_{seed}_{name}.npz') as data:
            vp, vt, vy = data['prediction'], data['target'], data['labels']
            np.testing.assert_allclose(q_text.predict(vt), data['real_score'], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(q_text.predict(vp), data['pred_score'], atol=1e-5, rtol=1e-5)
        prediction_probe = fit_text_probe(p, y)
        align = fit_alignment(p, t)
        scores = dict(real_text=q_text.predict(vt), direct_transfer=q_text.predict(vp),
                      predicted_probe=prediction_probe.predict(vp), aligned_transfer=q_text.predict(align.predict(vp)))
        row = dict(pattern=name, train_count=len(y), validation_count=len(vy),
                   metrics={k:sentiment_metrics(vy,s) for k,s in scores.items()},
                   train_prediction_probe=sentiment_metrics(y,prediction_probe.predict(p)),
                   validation_source_sha256=file_sha256(previous_dir/f'seed_{seed}_{name}.npz'))
        rows.append(row)
        np.savez_compressed(output/f'seed_{seed}_{name}.npz', labels=vy, **scores)
    result = dict(seed=seed, epoch=cp['epoch'], checkpoint_sha256=previous['checkpoint_sha256'],
        teacher_sha256=model.teacher_integrity(), state_unchanged=True, logits_unchanged=True,
        train_mask_sha256=digest.hexdigest(), validation_mask_sha256=previous['mask_sha256'],
        protocol='Per-pattern train-fit alpha10; natural miss=.5 epoch0; cached unchanged validation cohorts',
        alignment_uses_labels=False, teacher_probe_unchanged=True,
        selection='Previously Test-oracle-selected Student; no new checkpoint selection', rows=rows)
    (output/f'seed_{seed}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    output = ROOT/'text_alignment_audit'
    output.mkdir(exist_ok=False)
    results = [audit(seed, output) for seed in range(66,71)]
    summary = []
    for name in PATTERNS:
        rows = [next(row for row in result['rows'] if row['pattern']==name) for result in results]
        summary.append(dict(pattern=name, train_counts=[r['train_count'] for r in rows],
            validation_counts=[r['validation_count'] for r in rows],
            mean={key:statistics.mean(r['metrics'][key]['weighted_f1'] for r in rows) for key in rows[0]['metrics']},
            sd={key:statistics.stdev(r['metrics'][key]['weighted_f1'] for r in rows) for key in rows[0]['metrics']},
            train_probe_mean=statistics.mean(r['train_prediction_probe']['weighted_f1'] for r in rows)))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)
